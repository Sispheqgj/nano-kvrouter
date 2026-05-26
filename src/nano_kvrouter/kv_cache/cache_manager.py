"""Unified KV-cache interface over RadixTree (capacity-counter design).

CacheManager is the single entry point schedulers use to query prefix-hit
details and the entry point the simulation engine uses to materialise new
KV cache entries after prefill completes.

Architecture (v2 capacity-counter, GPU-only)
--------------------------------------------
* One :class:`~nano_kvrouter.kv_cache.radix_tree.RadixTree` per node.
* Physical block accounting is derived directly from the tree state via
  ceiling arithmetic: ``_used[node_id]["gpu"] == ceil(total_token_count /
  block_size)``.  BlockPool is **not** called in v1 — it is retained as a
  module for future P2 multi-tier accounting.
* This eliminates the v1 ``_pool_ids`` mapping, which leaked blocks whenever
  a RadixTree split produced sub-block-sized nodes (Codex review regression).
* The fundamental invariant after every admit/evict:
      ``_used[nid]["gpu"] == _tree_ceiling_blocks(nid)``
  ``free_blocks`` is always derived from ``_capacity - _used``, so lookup
  and capacity signals are always consistent.
* ``transfer_cost_ms`` is always ``0.0`` in v1 — no CPU/Disk tier management.
* Tier promotion / demotion and cross-node transfer cost are deferred to P2.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.kv_cache.radix_tree import RadixTree
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheLookup

logger = logging.getLogger(__name__)

__all__ = ["CacheManager"]


class CacheManager:
    """Unified read/write interface over per-node RadixTrees.

    Satisfies the :class:`~nano_kvrouter.scheduler.base.CacheQuery` Protocol
    via structural subtyping so schedulers can query it without importing this
    module directly.

    The four public methods are intentionally minimal:

    * :meth:`lookup` / :meth:`lookup_all` — read-only; called by schedulers.
    * :meth:`free_blocks` — read-only; called by schedulers.
    * :meth:`admit` — write; called by the simulation engine after prefill.
    """

    def __init__(
        self,
        node_ids: Sequence[str],
        model_config: ModelConfig,
        node_config: NodeConfig,
        bandwidth_config: BandwidthConfig,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialise per-node trees and capacity counters.

        Args:
            node_ids: Stable sequence of node identifiers. Order is not
                significant; IDs must be unique.
            model_config: Source of ``block_size`` used for alignment.
            node_config: Source of ``gpu_blocks`` / ``cpu_blocks`` /
                ``disk_blocks`` for capacity limits.
            bandwidth_config: Stored for future P2 transfer-cost calculations;
                not used in v1.
            clock: Optional simulated-time callable forwarded to each node's
                :class:`RadixTree`. Defaults to ``time.time`` when ``None``.
        """
        self._block_size: int = model_config.block_size
        self._trees: dict[str, RadixTree] = {
            nid: RadixTree(clock=clock) for nid in node_ids
        }
        self._capacity: dict[str, dict[str, int]] = {
            nid: {
                "gpu": node_config.gpu_blocks,
                "cpu": node_config.cpu_blocks,
                "disk": node_config.disk_blocks,
            }
            for nid in node_ids
        }
        self._used: dict[str, dict[str, int]] = {
            nid: {"gpu": 0, "cpu": 0, "disk": 0} for nid in node_ids
        }
        # BlockPool is not used in v1; retained as a module for P2 multi-tier.
        self._model_cfg = model_config
        self._bw_cfg = bandwidth_config

    # ------------------------------------------------------------------
    # CacheQuery Protocol — read-only
    # ------------------------------------------------------------------

    def lookup(self, request: Request, node_id: str) -> CacheLookup:
        """Return prefix-hit details for *request* on *node_id*.

        Matched token count is aligned DOWN to the nearest ``block_size``
        boundary: a partial trailing block cannot be reused because the
        simulator stores one KV block per ``block_size`` tokens.

        ``transfer_cost_ms`` is always ``0.0`` in v1 (GPU-only; no
        cross-tier or cross-node transfer accounting).

        Args:
            request: The request whose prompt is matched against the tree.
            node_id: Node to query.

        Returns:
            :class:`~nano_kvrouter.scheduler.base.CacheLookup` with
            ``matched_blocks_by_tier`` containing only the ``"gpu"`` key
            (omitted when zero) and ``transfer_cost_ms == 0.0``.

        Raises:
            KeyError: If *node_id* is unknown.
        """
        if node_id not in self._trees:
            raise KeyError(node_id)

        matched_raw, _ = self._trees[node_id].match_prefix(request.token_ids)
        matched_blocks = matched_raw // self._block_size
        matched_tokens = matched_blocks * self._block_size

        return CacheLookup(
            matched_tokens=matched_tokens,
            matched_blocks_by_tier={"gpu": matched_blocks} if matched_blocks > 0 else {},
            transfer_cost_ms=0.0,
        )

    def lookup_all(self, request: Request) -> dict[str, CacheLookup]:
        """Return prefix-hit details for *request* on every known node.

        Args:
            request: The request to look up.

        Returns:
            Mapping ``node_id → CacheLookup`` for every node. Nodes with
            zero matched tokens are still present (``matched_tokens=0``).
        """
        return {nid: self.lookup(request, nid) for nid in self._trees}

    def free_blocks(self, node_id: str, tier: str) -> int:
        """How many free blocks *node_id* has on *tier*.

        Derived from ``_capacity - _used``. Always consistent with
        :meth:`lookup` because both come from the same tree state.

        Args:
            node_id: Node to inspect.
            tier: Storage tier (``"gpu"`` / ``"cpu"`` / ``"disk"``).

        Returns:
            Non-negative count of free blocks.

        Raises:
            KeyError: If *node_id* is unknown or *tier* is not one of the
                three valid tiers.
        """
        if node_id not in self._capacity:
            raise KeyError(node_id)
        if tier not in self._capacity[node_id]:
            raise KeyError(tier)
        return self._capacity[node_id][tier] - self._used[node_id][tier]

    # ------------------------------------------------------------------
    # Write path — called by SimulationEngine after PREFILL_COMPLETE
    # ------------------------------------------------------------------

    def admit(self, token_ids: list[int], node_id: str) -> None:
        """Materialise the KV cache for *token_ids* on *node_id*.

        Called by the simulation engine's ``PREFILL_COMPLETE`` handler once
        a prefill is done. Inserts the block-aligned prefix into the
        RadixTree and updates the GPU capacity counter, evicting LRU entries
        first when capacity is exhausted.

        Physical block accounting uses ceiling semantics:
        ``used = ceil(total_tree_tokens / block_size)``.
        This prevents the orphan leak that occurred in v1 when split created
        sub-block-sized nodes whose floor-capacity was 0.

        Only full ``block_size``-aligned blocks are stored; a trailing
        partial block is discarded.

        Args:
            token_ids: Full prompt token sequence.
            node_id: Node where the KV cache should be stored.

        Raises:
            KeyError: If *node_id* is unknown.
            MemoryError: If (a) the GPU tier is full and all remaining blocks
                are pinned (``ref_count > 0``), making eviction impossible; or
                (b) the prompt requires more blocks than the node's total GPU
                capacity.

        Notes:
            If the aligned prompt is already fully present in the tree (a
            scheduler re-routes to a node that already cached this prefix),
            admit is a no-op — no eviction, no allocation. This is essential
            for cache-aware schedulers to behave correctly under pinned-leaf
            full-pool conditions.
        """
        if node_id not in self._trees:
            raise KeyError(node_id)

        tree = self._trees[node_id]
        bs = self._block_size

        total_blocks = len(token_ids) // bs
        if total_blocks == 0:
            logger.debug("admit: token_ids too short for even one block, no-op")
            return
        aligned_tokens = token_ids[: total_blocks * bs]

        # Fast path: if every block of the aligned prompt is already cached,
        # admit is a no-op — no allocation needed even if pool is full or
        # all leaves are pinned.
        matched_raw, _ = tree.match_prefix(aligned_tokens)
        already_blocks = matched_raw // bs
        if already_blocks >= total_blocks:
            logger.debug(
                "admit: node %s already has %d/%d blocks cached, no-op",
                node_id, already_blocks, total_blocks,
            )
            return

        # Pre-check: a prompt requiring more blocks than total capacity can
        # never fit. Fail fast without evicting existing cache.
        worst_case_new = total_blocks
        capacity_gpu = self._capacity[node_id]["gpu"]
        if worst_case_new > capacity_gpu:
            raise MemoryError(
                f"node {node_id!r}: prompt requires {worst_case_new} blocks, "
                f"exceeds total GPU capacity {capacity_gpu}"
            )

        while capacity_gpu - self._used[node_id]["gpu"] < worst_case_new:
            evicted_lens = tree.evict_lru_with_lengths(1)
            if not evicted_lens:
                raise MemoryError(
                    f"node {node_id!r} GPU pool full and no evictable cache; "
                    f"need {worst_case_new}, "
                    f"free {capacity_gpu - self._used[node_id]['gpu']}"
                )
            for klen in evicted_lens:
                freed = (klen + bs - 1) // bs
                self._used[node_id]["gpu"] = max(
                    0, self._used[node_id]["gpu"] - freed
                )
            # Re-sync to ground truth after each eviction round.
            self._used[node_id]["gpu"] = self._tree_ceiling_blocks(node_id)

        pre = self._tree_ceiling_blocks(node_id)
        tree.insert(aligned_tokens)
        post = self._tree_ceiling_blocks(node_id)
        self._used[node_id]["gpu"] = post

        logger.debug(
            "admit: node %s blocks %d→%d (total_blocks=%d)",
            node_id,
            pre,
            post,
            total_blocks,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tree_ceiling_blocks(self, node_id: str) -> int:
        """Total physical GPU blocks needed to store all tokens in node_id's tree.

        Uses ceiling semantics matching vLLM PagedAttention block allocation:
        a partial block at the tail still occupies one physical block slot.
        """
        bs = self._block_size
        total_tokens = sum(len(n.key) for n in self._trees[node_id]._nodes.values())
        return (total_tokens + bs - 1) // bs

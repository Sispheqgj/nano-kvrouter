"""Unified KV-cache interface over RadixTree + BlockPool (M4: pool-backed capacity).

CacheManager is the single entry point schedulers use to query prefix-hit
details and the entry point the simulation engine uses to materialise new
KV cache entries after prefill completes.

Architecture (M4 pool-backed, GPU-only)
----------------------------------------
* One :class:`~nano_kvrouter.kv_cache.radix_tree.RadixTree` per node, wired
  to a per-node :class:`~nano_kvrouter.kv_cache.block_pool.BlockPool` via
  callbacks so every block allocation/free goes through the pool.
* ``RadixNode.block_ids`` has length ``ceil(len(key) / block_size)`` — the
  pool is the ground truth for how many physical GPU blocks are in use.
* ``pool.used("gpu")`` replaces the former ``_tree_ceiling_blocks`` derived
  counter: it directly reflects pool state without re-computing over tree nodes.
* ``free_blocks()`` is derived from ``pool._tiers["gpu"].free``, keeping
  lookup and capacity signals always consistent.
* ``transfer_cost_ms`` is always ``0.0`` in v1 — no CPU/Disk tier management.
* Tier promotion / demotion and cross-node transfer cost are deferred to P3.

Capacity eviction (admit):
    The eviction loop ensures
    ``pool.used("gpu") + worst_case_new_blocks <= capacity``
    before calling ``tree.insert()``.  "worst_case" includes a +1 margin
    for the mid-node that a radix-tree split can create; this prevents a
    MemoryError surfacing inside the tree's mint callback when a split
    occurs near the capacity limit.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.kv_cache.block_pool import BlockPool
from nano_kvrouter.kv_cache.radix_tree import RadixTree
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheLookup

logger = logging.getLogger(__name__)

__all__ = ["CacheManager"]


class CacheManager:
    """Unified read/write interface over per-node RadixTrees + BlockPools.

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
        """Initialise per-node trees, pools and capacity limits.

        Args:
            node_ids: Stable sequence of node identifiers. Order is not
                significant; IDs must be unique.
            model_config: Source of ``block_size`` used for alignment.
            node_config: Source of ``gpu_blocks`` / ``cpu_blocks`` /
                ``disk_blocks`` for capacity limits.
            bandwidth_config: Stored for future P3 transfer-cost calculations;
                not used in v1.
            clock: Optional simulated-time callable forwarded to each node's
                :class:`RadixTree`. Defaults to ``time.time`` when ``None``.
        """
        self._block_size: int = model_config.block_size
        self._capacity: dict[str, dict[str, int]] = {
            nid: {
                "gpu": node_config.gpu_blocks,
                "cpu": node_config.cpu_blocks,
                "disk": node_config.disk_blocks,
            }
            for nid in node_ids
        }
        # Per-node BlockPool — ground truth for block accounting.
        self._pools: dict[str, BlockPool] = {
            nid: BlockPool(node_config) for nid in node_ids
        }
        # Per-node RadixTree wired to its pool via mint/free callbacks.
        self._trees: dict[str, RadixTree] = {}
        for nid in node_ids:
            pool = self._pools[nid]
            self._trees[nid] = RadixTree(
                clock=clock,
                block_size=self._block_size,
                mint=lambda n, _pool=pool: _pool.allocate(n, "gpu"),
                free_blocks=lambda ids, _pool=pool: _pool.free(ids),
            )
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

        Reads directly from the pool, which is the ground truth.  Always
        consistent with :meth:`lookup` because both derive from the same
        tree/pool state.

        Args:
            node_id: Node to inspect.
            tier: Storage tier (``"gpu"`` / ``"cpu"`` / ``"disk"``).

        Returns:
            Non-negative count of free blocks.

        Raises:
            KeyError: If *node_id* is unknown or *tier* is not one of the
                three valid tiers.
        """
        if node_id not in self._pools:
            raise KeyError(node_id)
        pool = self._pools[node_id]
        if tier not in pool._tiers:
            raise KeyError(tier)
        return pool._tiers[tier].free

    # ------------------------------------------------------------------
    # Write path — called by SimulationEngine after PREFILL_COMPLETE
    # ------------------------------------------------------------------

    def admit(self, token_ids: list[int], node_id: str) -> None:
        """Materialise the KV cache for *token_ids* on *node_id*.

        Called by the simulation engine's ``PREFILL_COMPLETE`` handler once
        a prefill is done. Inserts the block-aligned prefix into the
        RadixTree (which allocates blocks from the pool via the ``mint``
        callback) and evicts LRU entries first when capacity is exhausted.

        Physical block accounting uses per-node ceiling semantics:
        each RadixNode's ``block_ids`` has length
        ``ceil(len(node.key) / block_size)``.  ``pool.used("gpu")`` is the
        authoritative capacity counter.

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
            admit is a no-op — no eviction, no allocation.
        """
        if node_id not in self._trees:
            raise KeyError(node_id)

        tree = self._trees[node_id]
        pool = self._pools[node_id]
        bs = self._block_size

        total_blocks = len(token_ids) // bs
        if total_blocks == 0:
            logger.debug("admit: token_ids too short for even one block, no-op")
            return
        aligned_tokens = token_ids[: total_blocks * bs]

        # Fast path: if every block of the aligned prompt is already cached,
        # admit is a no-op — no allocation needed even if pool is full.
        matched_raw, _ = tree.match_prefix(aligned_tokens)
        already_blocks = matched_raw // bs
        if already_blocks >= total_blocks:
            logger.debug(
                "admit: node %s already has %d/%d blocks cached, no-op",
                node_id, already_blocks, total_blocks,
            )
            return

        capacity_gpu = self._capacity[node_id]["gpu"]

        # Pre-check: a prompt requiring more blocks than total capacity can
        # never fit. Fail fast without evicting existing cache.
        if total_blocks > capacity_gpu:
            raise MemoryError(
                f"node {node_id!r}: prompt requires {total_blocks} blocks, "
                f"exceeds total GPU capacity {capacity_gpu}"
            )

        # worst_case_new: total blocks needed + 1 margin for the mid-node that a
        # radix-tree split can create, clamped to capacity_gpu.  When the prompt
        # fills the pool exactly (total_blocks == capacity_gpu) we set
        # worst_case_new = capacity_gpu so the loop evicts until the pool is
        # completely empty — at that point no existing edge can cause a split.
        worst_case_new = min(total_blocks + 1, capacity_gpu)

        while pool.used("gpu") + worst_case_new > capacity_gpu:
            evicted = tree.evict_lru(1)
            if not evicted:
                raise MemoryError(
                    f"node {node_id!r} GPU pool full and no evictable cache; "
                    f"need {worst_case_new}, "
                    f"free {capacity_gpu - pool.used('gpu')}"
                )
            for bid_list in evicted:
                pool.free(bid_list)

        pre = pool.used("gpu")
        tree.insert(aligned_tokens)
        post = pool.used("gpu")

        logger.debug(
            "admit: node %s pool blocks %d→%d (total_blocks=%d)",
            node_id,
            pre,
            post,
            total_blocks,
        )

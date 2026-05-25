"""Unified KV-cache interface over RadixTree + BlockPool.

CacheManager is the single entry point schedulers use to query prefix-hit
details and the entry point the simulation engine uses to materialise new
KV cache entries after prefill completes.

Architecture (v1 GPU-only)
--------------------------
* One :class:`~nano_kvrouter.kv_cache.radix_tree.RadixTree` per node — used
  exclusively for prefix matching.
* One :class:`~nano_kvrouter.kv_cache.block_pool.BlockPool` per node — used
  exclusively for GPU block capacity accounting.
* The two data structures are **parallel and independent**: RadixTree block
  IDs (generated internally by the tree) never match BlockPool block IDs. A
  flat per-node pool-ID list ``_pool_ids`` bridges the two: when we allocate
  N blocks from the pool we append N IDs to the list; when we evict a tree
  node we free ``key_len // block_size`` IDs from the front of the same list.
* ``pool.used`` reflects physical block allocation conservatively: it may
  exceed the tree's logical block coverage when admit triggers a
  non-block-aligned split. This is intentional — scheduler sees
  ``free_blocks`` slightly lower than maximum, but lookup and pool stay
  consistent (pool never under-counts allocated blocks).
* ``transfer_cost_ms`` is always 0.0 in v1 — no CPU/Disk tier management.
* Tier promotion / demotion and cross-node transfer cost are deferred to P1.

Design notes
------------
* ``matched_tokens`` returned by :meth:`lookup` is always aligned down to a
  ``block_size`` boundary because the simulator stores one KV block per
  ``block_size`` tokens; a partial last block cannot be reused.
* :meth:`admit` calls ``pool.allocate`` **before** ``tree.insert`` so that a
  ``MemoryError`` (pool full, nothing evictable) aborts without modifying the
  tree.
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
    """Unified read/write interface over per-node RadixTree + BlockPool pairs.

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
        """Initialise per-node trees and pools.

        Args:
            node_ids: Stable sequence of node identifiers. Order is not
                significant; IDs must be unique.
            model_config: Source of ``block_size`` used for alignment.
            node_config: Source of ``gpu_blocks`` / ``cpu_blocks`` /
                ``disk_blocks`` forwarded to each node's :class:`BlockPool`.
            bandwidth_config: Stored for future P1 transfer-cost calculations;
                not used in v1.
            clock: Optional simulated-time callable forwarded to each node's
                :class:`RadixTree`. Defaults to ``time.time`` when ``None``.
        """
        self._block_size: int = model_config.block_size
        self._trees: dict[str, RadixTree] = {
            nid: RadixTree(clock=clock) for nid in node_ids
        }
        self._pools: dict[str, BlockPool] = {
            nid: BlockPool(node_config) for nid in node_ids
        }
        self._model_cfg = model_config
        self._bw_cfg = bandwidth_config
        # Flat ordered list of currently-allocated pool block IDs per node.
        # Invariant: len(_pool_ids[nid]) == pools[nid].stats()["gpu"]["used"]
        # (subject to the v1 simplification described in the module docstring).
        self._pool_ids: dict[str, list[str]] = {nid: [] for nid in node_ids}

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
        # pool.stats() returns only known tiers; unknown tier → KeyError naturally.
        return self._pools[node_id].stats()[tier]["free"]

    # ------------------------------------------------------------------
    # Write path — called by SimulationEngine after PREFILL_COMPLETE
    # ------------------------------------------------------------------

    def admit(self, token_ids: list[int], node_id: str) -> None:
        """Materialise the KV cache for *token_ids* on *node_id*.

        Called by the simulation engine's ``PREFILL_COMPLETE`` handler once
        a prefill is done. Inserts the prompt prefix into the RadixTree and
        allocates the corresponding GPU blocks from the BlockPool, evicting
        LRU entries first when capacity is exhausted.

        Only full ``block_size``-aligned blocks are stored; a trailing
        partial block is discarded.

        Args:
            token_ids: Full prompt token sequence.
            node_id: Node where the KV cache should be stored.

        Raises:
            KeyError: If *node_id* is unknown.
            MemoryError: If the GPU tier is full and all remaining blocks are
                pinned (``ref_count > 0``), making eviction impossible.
        """
        if node_id not in self._trees:
            raise KeyError(node_id)

        tree = self._trees[node_id]
        pool = self._pools[node_id]

        total_blocks = len(token_ids) // self._block_size
        if total_blocks == 0:
            logger.debug("admit: token_ids too short for even one block, no-op")
            return

        already_matched, _ = tree.match_prefix(token_ids)
        already_blocks = already_matched // self._block_size
        new_blocks_needed = total_blocks - already_blocks

        if new_blocks_needed <= 0:
            logger.debug("admit: node %s already has full coverage, no-op", node_id)
            return

        # Evict LRU leaves until we have enough free GPU slots.
        # We inspect each candidate BEFORE eviction to know how many pool
        # blocks its key represents (eviction removes the node from _nodes).
        while pool.stats()["gpu"]["free"] < new_blocks_needed:
            candidates = [
                n for n in tree._nodes.values()
                if not n.children and n.ref_count == 0
            ]
            if not candidates:
                # All remaining nodes are pinned or tree is empty; let
                # pool.allocate raise MemoryError below.
                break

            lru_node = min(candidates, key=lambda n: n.last_access_time)
            lru_blocks = len(lru_node.key) // self._block_size

            tree.evict_lru(1)  # removes lru_node; same LRU selection as above
            logger.debug(
                "admit: evicted %d-block node from node %s", lru_blocks, node_id
            )

            if lru_blocks > 0:
                pid_list = self._pool_ids[node_id]
                to_free = pid_list[:lru_blocks]
                self._pool_ids[node_id] = pid_list[lru_blocks:]
                pool.free(to_free)

        # Raises MemoryError if pool is still full after exhausting evictions.
        new_pool_ids = pool.allocate(new_blocks_needed, "gpu")
        self._pool_ids[node_id].extend(new_pool_ids)

        # Insert the block-aligned prefix; the tail (< block_size tokens) is
        # discarded because we cannot cache a partial block.
        tree.insert(token_ids[: total_blocks * self._block_size])

        logger.debug(
            "admit: node %s +%d blocks (total_blocks=%d, already=%d)",
            node_id,
            new_blocks_needed,
            total_blocks,
            already_blocks,
        )

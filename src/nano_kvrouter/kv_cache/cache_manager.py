"""KV-cache interface over RadixTree + BlockPool (M6: multi-tier demotion chain).

M6 adds three capabilities on top of the M4 GPU-only baseline:

1. **Demotion chain (GPU → CPU → Disk → free)**: When the GPU tier is full,
   ``admit`` demotes LRU leaf nodes to slower tiers rather than evicting them
   from the tree.  Tree nodes are preserved so future ``lookup`` calls can
   still find the prefix (with a non-zero ``transfer_cost_ms`` for non-GPU
   blocks).  Zombie nodes (all blocks freed) are cleaned up automatically
   via ``tree.demote_lru``'s tier_of probe.

2. **Tier-aware lookup**: ``lookup`` / ``lookup_all`` call
   ``tree.match_prefix_path`` to get the full block list on the matched
   path, then ``pool.tiers_of`` to get per-block tier labels, then compute
   ``transfer_cost_ms`` from the per-tier bandwidth config.

3. **Dead config fields now live**: ``cpu_blocks``, ``disk_blocks``,
   ``bandwidth.gpu_to_cpu``, and ``bandwidth.cpu_to_disk`` are all read and
   used.  The only remaining dead field is anything scheduled for M7+.

``transfer_cost_ms`` semantics (per block, additive across blocks; serial
hops within one block — disk blocks pay both legs):
    - GPU hit:  0 ms (already in HBM)
    - CPU hit:  block_bytes / bandwidth.gpu_to_cpu   (CPU → GPU DMA)
    - Disk hit: block_bytes * (1/cpu_to_disk + 1/gpu_to_cpu)
                                                     (Disk → CPU → GPU,
                                                      both hops in series)

M6.fix2 — Important #2: previously the disk cost only counted the
``cpu_to_disk`` leg, undercounting the load. Both legs must be paid
because the block must traverse Disk → CPU → GPU before prefill can
re-use it.

The scheduler (``compute_est_ttft``) adds this ``transfer_cost_ms`` to the
TTFT estimate so cache-tier routing decisions are reflected in SLO gating.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.kv_cache.block_table import RequestBlockTable
from nano_kvrouter.kv_cache.block_pool import BlockPool
from nano_kvrouter.kv_cache.radix_tree import RadixTree
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheLookup

logger = logging.getLogger(__name__)

__all__ = ["CacheManager", "RequestBlockTable"]


def _demote_block(pool: BlockPool, tree: RadixTree, bid: str) -> None:
    """Cascade one block down the tier hierarchy until it finds a free slot.

    GPU → CPU → Disk → free (no-op if block already freed / None).

    M6.fix2 (Important #1): whenever a block is *truly* freed (no
    downstream tier with capacity), call ``tree.purge_block_ids`` so the
    tree drops the zombie leaf node. Without this the tree would still
    advertise the prefix in ``match_prefix`` even though the pool has
    forgotten the blocks, tripping the fast-path bug Critical #1.

    Args:
        pool: The node's BlockPool.
        tree: The node's RadixTree (used for zombie GC).
        bid: Block ID to demote.
    """
    current = pool.tier_of(bid)
    if current is None:
        return
    if pool.pin_count(bid) > 0:
        raise MemoryError(f"block {bid!r} is pinned and cannot be demoted")
    s = pool.stats()
    if current == "gpu":
        if s["cpu"]["free"] > 0:
            pool.move(bid, "gpu", "cpu")
        elif s["disk"]["free"] > 0:
            pool.move(bid, "gpu", "disk")
        else:
            pool.free([bid])
            tree.purge_block_ids([bid], pool.tier_of)
    elif current == "cpu":
        if pool.stats()["disk"]["free"] > 0:
            pool.move(bid, "cpu", "disk")
        else:
            pool.free([bid])
            tree.purge_block_ids([bid], pool.tier_of)
    elif current == "disk":
        pool.free([bid])
        tree.purge_block_ids([bid], pool.tier_of)


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
            model_config: Source of ``block_size`` and ``kv_bytes_per_token``
                used for block alignment and transfer-cost calculation.
            node_config: Source of ``gpu_blocks`` / ``cpu_blocks`` /
                ``disk_blocks`` for capacity limits.
            bandwidth_config: Source of ``gpu_to_cpu`` and ``cpu_to_disk``
                for transfer-cost calculation (M6 live).
            clock: Optional simulated-time callable forwarded to each node's
                :class:`RadixTree`. Defaults to ``time.time`` when ``None``.
        """
        self._block_size: int = model_config.block_size
        self._model_cfg = model_config
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
        self._block_tables: dict[tuple[str, str], RequestBlockTable] = {}

    # ------------------------------------------------------------------
    # CacheQuery Protocol — read-only
    # ------------------------------------------------------------------

    def lookup(self, request: Request, node_id: str) -> CacheLookup:
        """Return prefix-hit details for *request* on *node_id*.

        M6 tier-aware: uses ``match_prefix_path`` to collect all block_ids
        on the matched path, then ``pool.tiers_of`` to get per-block tier
        labels, then computes ``transfer_cost_ms`` from bandwidth config.

        Matched token count is aligned DOWN to the nearest ``block_size``
        boundary (same as M4).

        Args:
            request: The request whose prompt is matched against the tree.
            node_id: Node to query.

        Returns:
            :class:`~nano_kvrouter.scheduler.base.CacheLookup` with
            ``matched_blocks_by_tier`` broken down by tier and a non-zero
            ``transfer_cost_ms`` when CPU or Disk blocks are matched.

        Raises:
            KeyError: If *node_id* is unknown.
        """
        if node_id not in self._trees:
            raise KeyError(node_id)

        tree = self._trees[node_id]
        pool = self._pools[node_id]
        bs = self._block_size

        matched_raw, path_block_ids = tree.match_prefix_path(request.token_ids)
        matched_blocks = matched_raw // bs
        matched_tokens = matched_blocks * bs

        if matched_blocks == 0:
            return CacheLookup(
                matched_tokens=0,
                matched_blocks_by_tier={},
                transfer_cost_ms=0.0,
            )

        # Take only the leading blocks that correspond to matched_tokens.
        relevant_bids = path_block_ids[:matched_blocks]

        # Tier distribution (only live blocks — freed ones don't count).
        tiers = pool.tiers_of(relevant_bids)
        tier_counts: dict[str, int] = {}
        for bid in relevant_bids:
            t = tiers.get(bid)
            if t is None:
                continue  # freed (zombie block)
            tier_counts[t] = tier_counts.get(t, 0) + 1

        # Only count live (non-zombie) blocks toward matched_tokens.
        live_block_count = sum(tier_counts.values())
        matched_tokens = live_block_count * bs

        if live_block_count == 0:
            return CacheLookup(
                matched_tokens=0,
                matched_blocks_by_tier={},
                transfer_cost_ms=0.0,
            )

        # Transfer cost: bytes to load matched non-GPU blocks into GPU HBM.
        kv_bytes_per_block = bs * self._model_cfg.kv_bytes_per_token
        cpu_n = tier_counts.get("cpu", 0)
        disk_n = tier_counts.get("disk", 0)

        cpu_load_ms = (
            cpu_n * kv_bytes_per_block / self._bw_cfg.gpu_to_cpu * 1000.0
            if cpu_n > 0 else 0.0
        )
        # Disk hit pays both serial hops: Disk→CPU (1/cpu_to_disk) and
        # CPU→GPU (1/gpu_to_cpu). Conservative extrapolation from Mooncake's
        # storage hierarchy (FAST'25 §3 describes prefix-cache loading from
        # DRAM→GPU; SSD-resident prefix hit cost is not directly defined in the
        # paper). Formula: disk_blocks * block_bytes * (1/cpu_to_disk +
        # 1/gpu_to_cpu) * 1000. M6 v1 simulator extrapolation, not paper verbatim.
        disk_load_ms = (
            disk_n
            * kv_bytes_per_block
            * (1.0 / self._bw_cfg.cpu_to_disk + 1.0 / self._bw_cfg.gpu_to_cpu)
            * 1000.0
            if disk_n > 0 else 0.0
        )
        transfer_cost_ms = cpu_load_ms + disk_load_ms

        return CacheLookup(
            matched_tokens=matched_tokens,
            matched_blocks_by_tier=tier_counts,
            transfer_cost_ms=transfer_cost_ms,
        )

    def lookup_all(self, request: Request) -> dict[str, CacheLookup]:
        """Return prefix-hit details for *request* on every known node.

        Args:
            request: The request to look up.

        Returns:
            Mapping ``node_id -> CacheLookup`` for every node. Nodes with
            zero matched tokens are still present (``matched_tokens=0``).
        """
        return {nid: self.lookup(request, nid) for nid in self._trees}

    def free_blocks(self, node_id: str, tier: str) -> int:
        """How many free blocks *node_id* has on *tier*.

        Reads directly from the pool, which is the ground truth.

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
    # Active request BlockTables
    # ------------------------------------------------------------------

    def _table_key(self, request_id: str, node_id: str) -> tuple[str, str]:
        return (node_id, request_id)

    def _aligned_tokens(self, token_ids: list[int]) -> tuple[list[int], int]:
        total_blocks = len(token_ids) // self._block_size
        return token_ids[: total_blocks * self._block_size], total_blocks

    def _live_prefix_block_ids(
        self,
        token_ids: list[int],
        node_id: str,
        total_blocks: int,
    ) -> list[str]:
        tree = self._trees[node_id]
        pool = self._pools[node_id]

        matched_raw, path_block_ids = tree.match_prefix_path(token_ids)
        matched_blocks = min(matched_raw // self._block_size, total_blocks)
        prefix_bids: list[str] = []
        for bid in path_block_ids[:matched_blocks]:
            if pool.tier_of(bid) is None:
                break
            prefix_bids.append(bid)
        return prefix_bids

    def _retain_blocks(self, node_id: str, block_ids: list[str]) -> None:
        if not block_ids:
            return
        pool = self._pools[node_id]
        pool.pin(block_ids)

    def _release_blocks(self, node_id: str, block_ids: list[str]) -> None:
        if not block_ids:
            return
        pool = self._pools[node_id]
        pool.unpin(block_ids)

    def materialize_request(
        self,
        request_id: str,
        token_ids: list[int],
        node_id: str,
    ) -> RequestBlockTable:
        """Create and pin the logical KV block table for an active request.

        Existing prefix-hit physical blocks are shared and pinned before any
        tail allocation so the demotion chain cannot move/free blocks that the
        request will read. Missing tail blocks are materialised through the
        existing :meth:`admit` path, preserving RadixTree and BlockPool
        semantics.

        Duplicate materialisation for the same ``(node_id, request_id)`` is a
        caller error and raises ``ValueError``. Unknown nodes raise ``KeyError``.
        """
        if node_id not in self._trees:
            raise KeyError(node_id)
        key = self._table_key(request_id, node_id)
        if key in self._block_tables:
            raise ValueError(f"request {request_id!r} already materialized on {node_id!r}")

        aligned_tokens, total_blocks = self._aligned_tokens(token_ids)
        if total_blocks == 0:
            table = RequestBlockTable(
                request_id=request_id,
                node_id=node_id,
                block_ids=[],
                matched_blocks=0,
                new_blocks=0,
            )
            self._block_tables[key] = table
            return table

        retained: list[str] = []
        try:
            prefix_bids = self._live_prefix_block_ids(aligned_tokens, node_id, total_blocks)
            self._retain_blocks(node_id, prefix_bids)
            retained.extend(prefix_bids)

            if len(prefix_bids) < total_blocks:
                self.admit(aligned_tokens, node_id)

            matched_after, path_block_ids = self._trees[node_id].match_prefix_path(aligned_tokens)
            if matched_after // self._block_size < total_blocks:
                raise MemoryError(
                    f"node {node_id!r}: failed to materialize {total_blocks} blocks "
                    f"for request {request_id!r}"
                )

            block_ids = path_block_ids[:total_blocks]
            live_tiers = self._pools[node_id].tiers_of(block_ids)
            if len(live_tiers) != total_blocks:
                raise MemoryError(
                    f"node {node_id!r}: materialized path contains freed blocks "
                    f"for request {request_id!r}"
                )

            already_retained = set(retained)
            new_retains = [bid for bid in block_ids if bid not in already_retained]
            self._retain_blocks(node_id, new_retains)
            retained.extend(new_retains)

            table = RequestBlockTable(
                request_id=request_id,
                node_id=node_id,
                block_ids=block_ids,
                matched_blocks=len(prefix_bids),
                new_blocks=total_blocks - len(prefix_bids),
            )
            self._block_tables[key] = table
            return table
        except Exception:
            self._release_blocks(node_id, retained)
            raise

    def release_request(self, request_id: str, node_id: str) -> None:
        """Release an active request's pinned BlockTable.

        Unknown request/node pairs raise ``KeyError`` so lifecycle leaks and
        double releases surface during tests.
        """
        if node_id not in self._trees:
            raise KeyError(node_id)
        key = self._table_key(request_id, node_id)
        table = self._block_tables.pop(key)
        self._release_blocks(node_id, table.block_ids)

    # ------------------------------------------------------------------
    # Write path — called by SimulationEngine after PREFILL_COMPLETE
    # ------------------------------------------------------------------

    def admit(self, token_ids: list[int], node_id: str) -> None:
        """Materialise the KV cache for *token_ids* on *node_id*.

        M6 eviction policy (demotion chain):
            When the GPU tier is full, the LRU leaf node's blocks are
            demoted tier-by-tier (GPU→CPU→Disk→free) instead of being
            evicted from the tree.  The tree node is preserved so future
            ``lookup`` calls can still match the prefix (with transfer
            cost).  Zombie nodes (all blocks freed) are cleaned up by
            ``tree.demote_lru``'s built-in zombie probe.

        Args:
            token_ids: Full prompt token sequence.
            node_id: Node where the KV cache should be stored.

        Raises:
            KeyError: If *node_id* is unknown.
            MemoryError: If the GPU tier is full and no evictable / demotion
                candidates remain (all blocks pinned).
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

        # Fast path: already fully cached → no-op.
        # M6.fix2 (Critical #1): cannot trust tree.match_prefix alone — after
        # GPU→CPU→Disk→free chain, tree nodes may be zombies (block_ids point
        # at freed pool slots). Take the path block IDs, ask the pool which
        # are still alive, and only no-op when EVERY relevant block is alive.
        # Otherwise we must purge the zombie sub-tree before evict+insert
        # because tree.insert would otherwise reuse / split the zombie nodes
        # and produce stale block_ids.
        matched_raw, path_block_ids = tree.match_prefix_path(aligned_tokens)
        already_blocks = matched_raw // bs
        if already_blocks >= total_blocks:
            relevant_bids = path_block_ids[:total_blocks]
            live_tiers = pool.tiers_of(relevant_bids)
            if len(live_tiers) == total_blocks:
                logger.debug(
                    "admit: node %s already has %d/%d blocks cached (all live), no-op",
                    node_id, already_blocks, total_blocks,
                )
                return
            # Partial or full zombie detected: purge the stale path so that
            # tree.insert re-creates it with fresh GPU blocks. Also frees any
            # alive CPU/Disk blocks to reclaim their tier capacity (avoids leak).
            # purge_block_ids intentionally skips mixed-state nodes; use
            # purge_path_with_any_dead which handles them correctly.
            purged = tree.purge_path_with_any_dead(
                aligned_tokens,
                tier_of=pool.tier_of,
                free_blocks=lambda bids: pool.free(bids),
            )
            logger.debug(
                "admit: node %s partial/full zombie on path; purged %d nodes",
                node_id, purged,
            )

        capacity_gpu = self._capacity[node_id]["gpu"]

        if total_blocks > capacity_gpu:
            raise MemoryError(
                f"node {node_id!r}: prompt requires {total_blocks} blocks, "
                f"exceeds total GPU capacity {capacity_gpu}"
            )

        worst_case_new = min(total_blocks + 1, capacity_gpu)

        while pool.used("gpu") + worst_case_new > capacity_gpu:
            # Try tree-preserving demotion first.
            demoted = tree.demote_lru(
                1,
                tier_of=pool.tier_of,
                is_pinned=lambda bid, _pool=pool: _pool.pin_count(bid) > 0,
            )
            if not demoted:
                raise MemoryError(
                    f"node {node_id!r} GPU pool full and no demotion candidates; "
                    f"need {worst_case_new}, "
                    f"free {capacity_gpu - pool.used('gpu')}"
                )
            for bid_list in demoted:
                for bid in bid_list:
                    _demote_block(pool, tree, bid)

        pre = pool.used("gpu")
        tree.insert(aligned_tokens)
        post = pool.used("gpu")

        logger.debug(
            "admit: node %s pool blocks %d->%d (total_blocks=%d)",
            node_id, pre, post, total_blocks,
        )

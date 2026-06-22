"""Scheduling policy interface and supporting types.

This module defines the contract that every concrete scheduler
(RoundRobin, LeastLoaded, PrefixGreedy, E2Policy, MooncakeConductor)
must satisfy. Implementations rely on structural subtyping — they
expose the right methods rather than inheriting from anything.

Design notes
------------
* ``CacheQuery`` lives here, alongside ``SchedulingPolicy``, so the
  scheduler module owns the *input contract* it depends on (dependency
  inversion). ``CacheManager`` (P0 task B) provides the concrete
  implementation in ``kv_cache.cache_manager``.
* ``CacheLookup`` ships a precomputed ``transfer_cost_ms`` so schedulers
  do not need to know about :class:`BandwidthConfig` or block-size
  arithmetic. All bandwidth / tier accounting happens inside
  ``CacheManager``.
* Both ``lookup`` and ``lookup_all`` are provided. Single-point
  ``lookup`` is enough for greedy strategies that probe one candidate
  at a time; ``lookup_all`` lets Conductor-style strategies score every
  node in one cheap call without re-running prefix matching N times.
* M5a split P/D: :meth:`SchedulingPolicy.schedule` now accepts separate
  ``prefill_nodes`` and ``decode_nodes`` pools so a single scheduler can
  route the prefill phase to one pool and the decode phase to another.
  In combined deployments the caller passes the same list as both args
  and the scheduler is free to return ``prefill_node == decode_node``
  (RoundRobin / LeastLoaded do; cache-aware policies may still split
  because cache state lives on the decode pool).
* P4-B M1: ``TransferBacklogView`` is a narrow read-only Protocol that
  exposes only ``peek_backlog``; it is satisfied structurally by both
  ``NoopTransferModel`` and ``PerNodeLaneTransferModel`` without any
  import of ``simulator.transfer_model`` from scheduler code.
  ``compute_est_ttft`` now takes ``backlog_view`` + ``now`` so the
  KV-transfer estimate includes current lane queue wait. Under
  ``NoopTransferModel`` peek_backlog always returns zeros → estimate is
  byte-identical to pre-P4-B.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request

__all__ = [
    "CacheLookup",
    "CacheQuery",
    "SchedulingDecision",
    "SchedulingPolicy",
    "TransferBacklogView",
    "compute_est_ttft",
    "compute_est_tbt",
]


@dataclass(slots=True, frozen=True)
class CacheLookup:
    """Per-node cache hit summary for one request.

    Computed by :class:`CacheManager` and consumed by schedulers to
    weigh ``cache_benefit`` against ``transfer_penalty`` (Mooncake §4).

    Attributes:
        matched_tokens: How many of the request's prompt tokens are
            already cached and reusable when the request runs on this
            node. Zero means a full cold prefill is needed.
        matched_blocks_by_tier: Number of matched blocks broken down by
            storage tier (``"gpu"`` / ``"cpu"`` / ``"disk"``). Tiers
            with zero matched blocks may be omitted from the mapping.
        transfer_cost_ms: Estimated time to bring every matched block
            into this node's GPU HBM before prefill starts. Already
            accounts for cross-tier hops (Disk → CPU → GPU) and
            cross-node hops (other_node.GPU → this_node.GPU over RDMA).
            Zero when every matched block is already in this node's
            GPU HBM.
    """

    matched_tokens: int
    matched_blocks_by_tier: dict[str, int]
    transfer_cost_ms: float


@runtime_checkable
class CacheQuery(Protocol):
    """Read-only view over cluster cache state, consumed by schedulers.

    Schedulers MUST NOT import :class:`RadixTree` or :class:`BlockPool`
    directly — they go through this Protocol. The concrete implementation
    lives in ``kv_cache.cache_manager.CacheManager`` (P0 task B), which
    decides internally whether to maintain one shared RadixTree with a
    ``node_id`` mapping or one tree per node.

    Use :meth:`lookup` when the scheduler already has a specific
    candidate node in mind. Use :meth:`lookup_all` when the scheduler
    wants to scan every node (PrefixGreedy / E2Policy / MooncakeConductor).

    M5a: only decode_pool nodes are tracked. Prefill_pool nodes are not
    registered (KV cache lives on the decode side only, Mooncake §3).
    """

    def lookup(self, request: Request, node_id: str) -> CacheLookup:
        """Return cache hit details for ``request`` if it runs on ``node_id``.

        Args:
            request: The request whose prefix should be matched.
            node_id: Candidate node to evaluate.

        Returns:
            A :class:`CacheLookup` describing matched tokens, tier
            distribution and transfer cost.

        Raises:
            KeyError: If ``node_id`` is unknown.
        """
        ...

    def lookup_all(self, request: Request) -> dict[str, CacheLookup]:
        """Return per-node CacheLookup for every known node.

        Args:
            request: The request whose prefix should be matched.

        Returns:
            Mapping ``node_id → CacheLookup``. Nodes with zero matched
            tokens are still present (with ``matched_tokens=0`` and an
            empty / zero-sum ``matched_blocks_by_tier``) so callers can
            use the dict keys as the authoritative node set.
        """
        ...

    def free_blocks(self, node_id: str, tier: str) -> int:
        """How many free blocks ``node_id`` has on ``tier``.

        Args:
            node_id: The node to inspect.
            tier: Storage tier name (``"gpu"`` / ``"cpu"`` / ``"disk"``).

        Returns:
            Non-negative count of free blocks.

        Raises:
            KeyError: If ``node_id`` or ``tier`` is unknown.
        """
        ...


@runtime_checkable
class TransferBacklogView(Protocol):
    """Narrow read-only view of KV-transfer lane backlog for schedulers.

    Exposes only ``peek_backlog`` — schedulers must NOT call
    ``request_transfer``. Both ``NoopTransferModel`` and
    ``PerNodeLaneTransferModel`` satisfy this Protocol structurally.

    P4-B M1: used by ``compute_est_ttft`` to include current lane queue
    wait in the KV-transfer cost estimate. Under ``NoopTransferModel``
    peek_backlog always returns ``{"egress": 0.0, "ingress": 0.0}`` so
    the estimate remains byte-identical to pre-P4-B under the default
    ``contention_model: "none"`` config.
    """

    def peek_backlog(self, node_id: str) -> dict[str, float]:
        """Return current lane backlog timestamps for ``node_id``.

        Args:
            node_id: The node to inspect.

        Returns:
            Dict with ``"egress"`` and ``"ingress"`` keys, each holding
            the wall-clock time (ms) at which the corresponding lane will
            be free. ``0.0`` means idle.
        """
        ...


@dataclass(slots=True)
class SchedulingDecision:
    """A scheduler's verdict for one request.

    A non-rejected decision MUST have both ``prefill_node`` and
    ``decode_node`` set. They are equal in combined deployments and
    differ under M5a split P/D (Mooncake-style).

    Attributes:
        prefill_node: Node ID assigned for prefill. ``None`` means the
            request was rejected before prefill started (early rejection).
        decode_node: Node ID assigned for decode. Required when
            ``prefill_node`` is set; may equal or differ from
            ``prefill_node``. Should be ``None`` when ``prefill_node``
            is ``None``.
        estimated_ttft_ms: Predicted Time-To-First-Token (ms). Used by
            :class:`MetricsCollector` for SLO accounting.
        estimated_tbt_ms: Predicted Time-Between-Tokens (ms), one
            decode-step interval at the expected concurrent batch.
        reject_reason: Free-text reason for rejection. ``None`` iff the
            request was accepted; set to a human-readable string when
            rejected (e.g. ``"ttft_slo_exceeded"``).
        routing_score: Optional Bidaw routing score. ``None`` for schedulers
            that do not expose a routing-aware score.
        affinity_hit: True when Bidaw session affinity pinned the request to
            the returned decode node.
    """

    prefill_node: str | None
    decode_node: str | None
    estimated_ttft_ms: float
    estimated_tbt_ms: float
    reject_reason: str | None = None
    routing_score: float | None = None
    affinity_hit: bool = False

    @property
    def is_rejected(self) -> bool:
        """True when the request was rejected (no prefill node assigned)."""
        return self.prefill_node is None


# ----------------------------------------------------------------------
# Shared cost helpers — used by every scheduler so the formula is
# defined exactly once.  In combined deployments callers pass the same
# MockEngineNode object as both ``prefill_node`` and ``decode_node`` and
# the KV-transfer term collapses to a self-transfer that is still
# computed by the formula (small but non-zero); intentional so that
# bandwidth.gpu_to_gpu sensitivity sweeps work in either mode.
# ----------------------------------------------------------------------


def compute_est_ttft(
    prefill_node: MockEngineNode,
    decode_node: MockEngineNode,
    request: Request,
    decode_cache_match: CacheLookup,
    *,
    kv_bytes_per_token: int,
    bandwidth_bytes_per_s: float,
    backlog_view: TransferBacklogView,
    now: float,
) -> float:
    """Estimated TTFT (ms) for one request under split P/D.

    Formula (Mooncake §3 + Sarathi chunked-prefill + P4-B lane backlog):

        est_ttft = prefill_phase + queue_wait + kv_transfer + first_decode_tick

    where:

    * ``prefill_phase`` uses the chunked-pipeline formula on prefill_node,
      taking ``decode_cache_match.matched_tokens`` as the cache hit (KV
      lives on decode_node side — the prefill_node simulates skipping
      cached tokens as Mooncake does).  The batch-size hint is
      ``len(prefill_node.decoding) + 1`` — the prefill_node's own
      concurrent decode streams, NOT decode_node's.  In split P/D the
      prefill_node has an empty ``decoding`` set (it never runs decode
      streams), so ``prefill_bs = 1``.  In combined deployments
      ``prefill_node is decode_node`` so both values are equal and
      behaviour is unchanged.
    * ``queue_wait`` is measured on prefill_node because that is where
      the request is admitted first.
    * ``kv_transfer`` accounts for both service time and lane queue wait
      (P4-B): service = (prompt_len × kv_bytes_per_token) /
      bandwidth_bytes_per_s × 1000. Lane queue wait = max(src_egress_wait,
      dst_ingress_wait) where each is max(0, backlog_view.peek_backlog(
      node_id)[lane] − now). max-not-sum because both lanes are occupied
      simultaneously for [start, finish). Under NoopTransferModel, all
      backlog values are 0.0 so queue_wait == 0 and kv_transfer is
      byte-identical to the pre-P4-B formula.
    * ``first_decode_tick`` is the first decode step time on decode_node
      using ``len(decode_node.decoding) + 1`` as the batch size hint
      (the decode_node's own concurrent streams).
    * ``cache_load_ms`` is ``decode_cache_match.transfer_cost_ms`` — the
      estimated time to load matched prefix blocks from CPU/Disk into GPU
      HBM before prefill can start (M6 HiCache).  Zero in GPU-only configs.
      **This is the within-node tier-migration cost**, distinct from the
      prefill_node→decode_node KV transfer (``kv_transfer`` term above).

    Args:
        prefill_node: Node assigned for the prefill phase.
        decode_node: Node assigned for the decode phase.
        request: Incoming request; only ``token_ids`` and
            ``expected_output_len`` are read.
        decode_cache_match: CacheLookup for ``decode_node`` — used both
            for the prefill phase cache-hit term (we assume the prefill
            side can skip cached tokens) and to know that KV is already
            present on decode side.
        kv_bytes_per_token: From ``model.kv_bytes_per_token``.
        bandwidth_bytes_per_s: From ``bandwidth.gpu_to_gpu``.
        backlog_view: Read-only view of KV-transfer lane backlog
            (P4-B). ``NoopTransferModel`` always returns 0.0 → no change
            under default config. ``PerNodeLaneTransferModel`` returns
            real backlog timestamps → queue wait is included.
        now: Current simulated time (ms). Required to compute how much
            of any existing backlog is still in the future.

    Returns:
        Estimated TTFT in milliseconds.
    """
    prompt_len = len(request.token_ids)
    matched = decode_cache_match.matched_tokens
    # prefill_bs: concurrent decode streams on the prefill_node itself.
    # In split P/D this is 0+1=1 (prefill nodes never run decode).
    # In combined deployment prefill_node is decode_node so equal to decoding_bs.
    prefill_bs = len(prefill_node.decoding) + 1
    decoding_bs = len(decode_node.decoding) + 1  # for first_decode_tick only

    prefill_phase = prefill_node.estimate_prefill_time(
        prompt_len, cached_tokens=matched, batch_size_hint=prefill_bs
    )
    queue_wait = prefill_node.queue_wait_time(prompt_len, request.expected_output_len)
    kv_bytes = prompt_len * kv_bytes_per_token
    # bandwidth_bytes_per_s is validated > 0 by Pydantic at config load.
    service = (kv_bytes / bandwidth_bytes_per_s) * 1000.0
    # P4-B: include lane queue wait. max-not-sum because egress and ingress
    # lanes are reserved simultaneously for the same transfer window.
    src_egress_wait = max(0, backlog_view.peek_backlog(prefill_node.node_id)["egress"] - now)
    dst_ingress_wait = max(0, backlog_view.peek_backlog(decode_node.node_id)["ingress"] - now)
    lane_queue_wait = max(src_egress_wait, dst_ingress_wait)
    kv_transfer = service + lane_queue_wait
    # M6: CPU/Disk prefix blocks need loading into GPU HBM before prefill.
    cache_load_ms = decode_cache_match.transfer_cost_ms
    first_decode_tick = decode_node.estimate_decode_time(decoding_bs)
    return prefill_phase + queue_wait + kv_transfer + cache_load_ms + first_decode_tick


def compute_est_tbt(decode_node: MockEngineNode) -> float:
    """Estimated TBT (ms) for one decode step on ``decode_node``.

    TBT depends only on the decode side: ``decode_base + bs * marginal``
    where ``bs = len(decoding) + 1`` (the new request joins the batch).

    Args:
        decode_node: Node assigned for decode.

    Returns:
        One decode-step time in milliseconds.
    """
    return decode_node.estimate_decode_time(len(decode_node.decoding) + 1)


@runtime_checkable
class SchedulingPolicy(Protocol):
    """Pluggable scheduling strategy.

    Concrete implementations (``RoundRobinPolicy``, ``LeastLoadedPolicy``,
    ``PrefixGreedyPolicy``, ``E2Policy``, ``MooncakeConductor``) satisfy
    this Protocol via structural subtyping — no explicit inheritance.

    A policy is constructed once and reused across the simulation. It
    may keep internal state (round-robin cursor, historical load
    windows, etc.) between :meth:`schedule` calls.
    """

    def schedule(
        self,
        request: Request,
        prefill_nodes: Sequence[MockEngineNode],
        decode_nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
        *,
        now: float,
    ) -> SchedulingDecision:
        """Decide where ``request`` should run under split P/D.

        Args:
            request: The incoming request.
            prefill_nodes: Live prefill-pool nodes in stable order so
                deterministic strategies (e.g. RoundRobin) can index
                into it without sorting.
            decode_nodes: Live decode-pool nodes in stable order. KV
                cache and ``cache.lookup`` only know about decode-pool
                nodes; cache-aware policies must therefore evaluate
                ``matched_tokens`` against ``decode_nodes`` (not prefill).
            cache: Read-only handle for per-node prefix-hit details
                and free-block counts. Tracks decode-pool nodes only.
            now: Current simulated time (ms). Required; passed to
                ``compute_est_ttft`` for lane backlog wait calculation.

        Returns:
            A :class:`SchedulingDecision`. To reject, return one with
            ``prefill_node=None`` and a non-empty ``reject_reason``.
        """
        ...

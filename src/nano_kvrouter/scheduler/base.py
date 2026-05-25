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


@dataclass(slots=True)
class SchedulingDecision:
    """A scheduler's verdict for one request.

    A non-rejected decision MUST have both ``prefill_node`` and
    ``decode_node`` set. They are equal in combined deployments and
    differ under P/D split (Mooncake).

    Attributes:
        prefill_node: Node ID assigned for prefill. ``None`` means the
            request was rejected before prefill started (early rejection).
        decode_node: Node ID assigned for decode. Required when
            ``prefill_node`` is set; pass the same value for combined
            deployments. Should be ``None`` when ``prefill_node`` is
            ``None``.
        estimated_ttft_ms: Predicted Time-To-First-Token (ms). Used by
            :class:`MetricsCollector` for SLO accounting.
        estimated_tbt_ms: Predicted Time-Between-Tokens (ms), one
            decode-step interval at the expected concurrent batch.
        reject_reason: Free-text reason for rejection. ``None`` iff the
            request was accepted; set to a human-readable string when
            rejected (e.g. ``"ttft_slo_exceeded"``).
    """

    prefill_node: str | None
    decode_node: str | None
    estimated_ttft_ms: float
    estimated_tbt_ms: float
    reject_reason: str | None = None

    @property
    def is_rejected(self) -> bool:
        """True when the request was rejected (no prefill node assigned)."""
        return self.prefill_node is None


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
        nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> SchedulingDecision:
        """Decide where ``request`` should run.

        Args:
            request: The incoming request.
            nodes: Live cluster nodes. Provided in stable order so
                deterministic strategies (e.g. RoundRobin) can index
                into it without sorting.
            cache: Read-only handle for per-node prefix-hit details
                and free-block counts.

        Returns:
            A :class:`SchedulingDecision`. To reject, return one with
            ``prefill_node=None`` and a non-empty ``reject_reason``.
        """
        ...

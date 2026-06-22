"""Bidaw I/O-aware scheduling policy (FAST'26).

BidawPolicy implements the request-routing layer of Bidaw's dual-queue
I/O-aware scheduling. With M3 flags disabled, node selection mirrors
LeastLoadedPolicy (decode, lowest current_load) + round-robin (prefill,
deterministic). Optional M3 routing, TTFT SLO gating, and session affinity
read from a narrow BidawControllerView; mutating dual-queue state remains in
BidawAdmissionController (simulator/bidaw_controller.py), not here.

Double-charging guard:
BidawPolicy.schedule subtracts only the disk portion of transfer_cost_ms
before passing the CacheLookup to compute_est_ttft. The KV_LOAD event path
pays the real disk load latency in simulated wall-clock time; leaving the
disk component in transfer_cost_ms would double-count it. CPU reload cost
(cpu_load_ms) is preserved because KV_LOAD events only replay disk loads,
not CPU-to-GPU reloads. Formula mirrors cache_manager.py:241-247 exactly:
  disk_load_ms = disk_n * block_bytes * (1/cpu_to_disk + 1/gpu_to_cpu) * 1000
max(0.0, ...) guards against floating-point underflow. The 5 existing
schedulers are unaffected.

Metadata boundary:
KV_LOAD_COMPLETE gates this request's PREFILL_START and the cli wiring
attempts metadata-only disk→CPU promotion via CacheManager. No real tensor
copy exists; CPU/GPU are both treated as Bidaw's ready/performance layer.
"""
from __future__ import annotations

import logging
from typing import Sequence

from nano_kvrouter.config import BandwidthConfig, ModelConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import (
    CacheLookup,
    CacheQuery,
    SchedulingDecision,
    TransferBacklogView,
    compute_est_tbt,
    compute_est_ttft,
)
from nano_kvrouter.scheduler.bidaw_view import BidawControllerView

logger = logging.getLogger(__name__)

__all__ = ["BidawPolicy", "hrrn_priority"]


def hrrn_priority(waiting_ms: float, kv_size_blocks: int) -> float:
    """Bidaw disk-HRRN response ratio.

    response_ratio = 1 + waiting_ms / max(1, kv_size_blocks)

    Smaller KV first (denominator is larger → lower ratio), but
    starvation-bounded: waiting_ms grows for whoever waits longest.
    Controller picks MAX ratio (Highest Response Ratio Next).

    kv_size_blocks is the matched disk block count — not bytes, not tokens.
    This is a deliberate simplification; swap one line to use tokens if
    the paper's exact normalization is required.

    Args:
        waiting_ms: How long this request has been in the preparing queue.
        kv_size_blocks: Number of matched disk blocks for this request.

    Returns:
        Response ratio ≥ 1.0.
    """
    return 1.0 + waiting_ms / max(1, kv_size_blocks)


class BidawPolicy:
    """Bidaw I/O-aware scheduling policy (FAST'26 routing layer).

    Default routing: lowest current_load (tie-break by node_id) on
    decode + round-robin on prefill. The dual-queue admission controller
    lives in BidawAdmissionController.

    Satisfies SchedulingPolicy via structural subtyping. M3 adds
    Bidaw-only kwargs (controller_view, 3 enable flags, 4 routing
    weights, 2 affinity threshold knobs) — all optional, defaulted so
    `_build_scheduler` can construct without them.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
        backlog_view: TransferBacklogView,
        controller_view: BidawControllerView | None = None,
        enable_routing_aware: bool = False,
        enable_ttft_slo_gate: bool = False,
        enable_session_affinity: bool = False,
        routing_weight_matched_blocks: float = 1.0,
        routing_weight_load: float = 1.0,
        routing_weight_preparing: float = 1.0,
        routing_weight_in_flight: float = 2.0,
        affinity_overload_factor: float = 1.5,
        affinity_overload_abs_floor: float = 2.0,
    ) -> None:
        if (enable_routing_aware or enable_ttft_slo_gate or enable_session_affinity) and (
            controller_view is None
        ):
            raise ValueError("Bidaw M3 flags require controller_view")
        weights = (
            routing_weight_matched_blocks,
            routing_weight_load,
            routing_weight_preparing,
            routing_weight_in_flight,
            affinity_overload_factor,
            affinity_overload_abs_floor,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Bidaw M3 weights and affinity thresholds must be non-negative")
        self._model_cfg = model_config if model_config is not None else ModelConfig()
        self._bw_cfg = bandwidth_config if bandwidth_config is not None else BandwidthConfig()
        self._backlog_view = backlog_view
        self._controller_view = controller_view
        self._enable_routing_aware = enable_routing_aware
        self._enable_ttft_slo_gate = enable_ttft_slo_gate
        self._enable_session_affinity = enable_session_affinity
        self._routing_weight_matched_blocks = routing_weight_matched_blocks
        self._routing_weight_load = routing_weight_load
        self._routing_weight_preparing = routing_weight_preparing
        self._routing_weight_in_flight = routing_weight_in_flight
        self._affinity_overload_factor = affinity_overload_factor
        self._affinity_overload_abs_floor = affinity_overload_abs_floor
        self._prefill_rr_idx: int = 0

    def _routing_score(self, decode: MockEngineNode, match: CacheLookup) -> float:
        """Return lower-is-better M3 routing cost for one decode node.

        Preparing/in-flight terms use *disk block counts*, not queue depth
        or slot count — block-weighted is the actual I/O backlog measure
        (a queue of one 50-block request is worse than five 1-block ones).
        """
        matched_blocks = sum(match.matched_blocks_by_tier.values())
        preparing_blocks = 0
        in_flight_blocks = 0
        if self._controller_view is not None:
            preparing_blocks = self._controller_view.peek_preparing_disk_blocks(decode.node_id)
            in_flight_blocks = self._controller_view.peek_in_flight_disk_blocks(decode.node_id)
        return (
            -self._routing_weight_matched_blocks * matched_blocks
            + self._routing_weight_load * decode.current_load()
            + self._routing_weight_preparing * preparing_blocks
            + self._routing_weight_in_flight * in_flight_blocks
        )

    def _affinity_overloaded(
        self,
        pinned: MockEngineNode,
        decode_nodes: Sequence[MockEngineNode],
    ) -> bool:
        """Fall back from affinity when a meaningfully-better alternative exists.

        Threshold is anchored on the best alternative (min_load), not the
        cluster average — answers "is there a node clearly less loaded?"
        directly. `factor * min_load` dominates at high load (proportional
        margin); `min_load + abs_floor` dominates at low load (absolute
        margin prevents oscillation when min_load is small).
        """
        if len(decode_nodes) <= 1:
            return False
        min_load = min(node.current_load() for node in decode_nodes)
        threshold = max(
            min_load * self._affinity_overload_factor,
            min_load + self._affinity_overload_abs_floor,
        )
        return pinned.current_load() > threshold

    def _select_decode(
        self,
        request: Request,
        decode_nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> tuple[MockEngineNode, CacheLookup, float | None, bool]:
        """Select decode node and return match, routing score, affinity flag."""
        fallback_match = CacheLookup(0, {}, 0.0)
        lookups: dict[str, CacheLookup] | None = None
        if self._enable_routing_aware:
            lookups = cache.lookup_all(request)

        def match_for(node_id: str) -> CacheLookup:
            if lookups is not None:
                return lookups.get(node_id, fallback_match)
            try:
                return cache.lookup(request, node_id)
            except KeyError:
                return fallback_match

        candidates = list(decode_nodes)
        if (
            self._enable_session_affinity
            and self._controller_view is not None
            and request.session_id is not None
        ):
            pinned_id = self._controller_view.peek_session_affinity(request.session_id)
            pinned = next((node for node in candidates if node.node_id == pinned_id), None)
            if pinned is not None:
                if not self._affinity_overloaded(pinned, candidates):
                    match = match_for(pinned.node_id)
                    routing_score = (
                        self._routing_score(pinned, match)
                        if self._enable_routing_aware else None
                    )
                    return pinned, match, routing_score, True
                alternatives = [node for node in candidates if node.node_id != pinned.node_id]
                if alternatives:
                    candidates = alternatives

        if self._enable_routing_aware:
            scored: list[tuple[float, str, MockEngineNode, CacheLookup]] = []
            for node in candidates:
                match = match_for(node.node_id)
                scored.append((self._routing_score(node, match), node.node_id, node, match))
            routing_score, _, decode, decode_match = min(
                scored,
                key=lambda item: (item[0], item[1]),
            )
            return decode, decode_match, routing_score, False

        decode = min(candidates, key=lambda n: (n.current_load(), n.node_id))
        return decode, match_for(decode.node_id), None, False

    def schedule(
        self,
        request: Request,
        prefill_nodes: Sequence[MockEngineNode],
        decode_nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
        *,
        now: float,
    ) -> SchedulingDecision:
        """Pick decode node (least-loaded) and prefill node (round-robin).

        Double-charging guard: only the disk portion of transfer_cost_ms is
        subtracted before passing the CacheLookup to compute_est_ttft.
        CPU reload cost is preserved. See module docstring for rationale.

        Args:
            request: Incoming request.
            prefill_nodes: Prefill pool in stable order.
            decode_nodes: Decode pool in stable order.
            cache: Decode-pool cache view.
            now: Current simulated time (ms).

        Returns:
            SchedulingDecision. Rejected with reason "no_nodes_available"
            when either pool is empty.
        """
        if not prefill_nodes or not decode_nodes:
            return SchedulingDecision(
                prefill_node=None,
                decode_node=None,
                estimated_ttft_ms=0.0,
                estimated_tbt_ms=0.0,
                reject_reason="no_nodes_available",
            )

        # Prefill: round-robin for determinism.
        prefill = prefill_nodes[self._prefill_rr_idx % len(prefill_nodes)]
        self._prefill_rr_idx += 1

        decode, decode_match, routing_score, affinity_hit = self._select_decode(
            request,
            decode_nodes,
            cache,
        )

        # Double-charging guard: subtract only the disk portion of transfer_cost_ms.
        # CPU reload cost is kept intact — KV_LOAD events only replay disk loads.
        # Mirror cache_manager.py:241-247 for disk_load_ms formula.
        disk_n = decode_match.matched_blocks_by_tier.get("disk", 0)
        if disk_n > 0:
            block_bytes = self._model_cfg.block_size * self._model_cfg.kv_bytes_per_token
            disk_load_ms = (
                disk_n
                * block_bytes
                * (1.0 / self._bw_cfg.cpu_to_disk + 1.0 / self._bw_cfg.gpu_to_cpu)
                * 1000.0
            )
            decode_match_for_ttft = CacheLookup(
                matched_tokens=decode_match.matched_tokens,
                matched_blocks_by_tier=decode_match.matched_blocks_by_tier,
                transfer_cost_ms=max(0.0, decode_match.transfer_cost_ms - disk_load_ms),
            )
        else:
            decode_match_for_ttft = decode_match

        ttft_ms = compute_est_ttft(
            prefill,
            decode,
            request,
            decode_match_for_ttft,
            kv_bytes_per_token=self._model_cfg.kv_bytes_per_token,
            bandwidth_bytes_per_s=self._bw_cfg.gpu_to_gpu,
            backlog_view=self._backlog_view,
            now=now,
        )
        if self._enable_ttft_slo_gate and self._controller_view is not None:
            ttft_ms += self._controller_view.peek_projected_preparing_wait_ms(
                decode.node_id,
                disk_n,
                now,
            )
        tbt_ms = compute_est_tbt(decode)

        if self._enable_ttft_slo_gate and ttft_ms > request.slo_ttft:
            logger.debug(
                "Bidaw: reject %s ttft=%.1f > slo=%.1f",
                request.request_id,
                ttft_ms,
                request.slo_ttft,
            )
            return SchedulingDecision(
                prefill_node=None,
                decode_node=None,
                estimated_ttft_ms=ttft_ms,
                estimated_tbt_ms=tbt_ms,
                reject_reason="ttft_slo_exceeded",
                routing_score=routing_score,
                affinity_hit=affinity_hit,
            )

        logger.debug(
            "Bidaw: request %s → prefill=%s decode=%s(load=%.3f) disk_blocks=%d",
            request.request_id,
            prefill.node_id,
            decode.node_id,
            decode.current_load(),
            disk_n,
        )

        return SchedulingDecision(
            prefill_node=prefill.node_id,
            decode_node=decode.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
            routing_score=routing_score,
            affinity_hit=affinity_hit,
        )

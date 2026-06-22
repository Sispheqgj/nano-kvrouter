"""Bidaw I/O-aware scheduling policy (FAST'26).

BidawPolicy implements the request-routing layer of Bidaw's dual-queue
I/O-aware scheduling. Node selection mirrors LeastLoadedPolicy (decode,
lowest current_load) + round-robin (prefill, deterministic). The
dual-queue admission logic (ready vs preparing) lives in
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

    Selects decode_node by lowest current_load (ties broken by node_id)
    and prefill_node by round-robin for determinism. The dual-queue
    admission controller logic lives in BidawAdmissionController.

    Satisfies the SchedulingPolicy Protocol via structural subtyping
    (same __init__ signature as the 5 existing schedulers so
    _build_scheduler factory stays clean).
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
        backlog_view: TransferBacklogView,
    ) -> None:
        self._model_cfg = model_config if model_config is not None else ModelConfig()
        self._bw_cfg = bandwidth_config if bandwidth_config is not None else BandwidthConfig()
        self._backlog_view = backlog_view
        self._prefill_rr_idx: int = 0

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

        # Decode: least-loaded, tie-break by node_id for reproducibility.
        decode = min(decode_nodes, key=lambda n: (n.current_load(), n.node_id))

        # Prefill: round-robin for determinism.
        prefill = prefill_nodes[self._prefill_rr_idx % len(prefill_nodes)]
        self._prefill_rr_idx += 1

        try:
            decode_match = cache.lookup(request, decode.node_id)
        except KeyError:
            decode_match = CacheLookup(
                matched_tokens=0, matched_blocks_by_tier={}, transfer_cost_ms=0.0
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
        tbt_ms = compute_est_tbt(decode)

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
        )

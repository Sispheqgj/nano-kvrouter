from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

from nano_kvrouter.simulator.event import Event, EventType

if TYPE_CHECKING:
    from nano_kvrouter.simulator.engine import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["MetricsCollector"]


class MetricsCollector:
    """Passive observer: subscribes to simulation events and accumulates metrics.

    TTFT / TBT definitions (v2, aligned with Mooncake §3 and Sarathi-Serve):
        TTFT = DECODE_STEP[step_index=0].time - ARRIVE.time
        TBT  = avg(DECODE_STEP[i+1].time - DECODE_STEP[i].time) for i >= 0

    PREFILL_COMPLETE no longer participates in TTFT/TBT — its handler is
    retained as a no-op extension point for future prefill-stage metrics.

    Event payload contract:
        REQUEST_ARRIVE:   {"request": Request}
        SCHEDULED:        {"request_id": str, "decision": SchedulingDecision,
                           "matched_tokens": int}
        REQUEST_REJECTED: {"request_id": str, "reason": str}
        PREFILL_COMPLETE: {"request_id": str}                       # no-op
        DECODE_STEP:      {"request_id": str, "step_index": int}    # REQUIRED
        DECODE_COMPLETE:  {"request_id": str}
    """

    def __init__(self) -> None:
        self._requests: dict[str, dict] = {}
        self._ttft_per_request: list[float] = []
        self._tbt_per_request: list[float] = []
        self._e2e_per_request: list[float] = []
        self._rejected_count: int = 0
        self._total_arrived: int = 0
        self._completed_count: int = 0
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._last_decode_step_time: dict[str, float] = {}
        self._tbt_samples: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self, engine: SimulationEngine) -> None:
        """Register all handlers with *engine*."""
        engine.on(EventType.REQUEST_ARRIVE, self._on_arrive)
        engine.on(EventType.SCHEDULED, self._on_scheduled)
        engine.on(EventType.REQUEST_REJECTED, self._on_rejected)
        engine.on(EventType.PREFILL_COMPLETE, self._on_prefill_complete)
        engine.on(EventType.DECODE_STEP, self._on_decode_step)
        engine.on(EventType.DECODE_COMPLETE, self._on_decode_complete)

    def summary(self) -> dict:
        """Return aggregated metrics dict.  Safe to call at any point."""
        total = self._total_arrived
        return {
            "total_arrived": total,
            "completed": self._completed_count,
            "rejected": self._rejected_count,
            "rejection_rate": (
                self._rejected_count / total if total > 0 else None
            ),
            "ttft_p50_ms": self._median(self._ttft_per_request),
            "ttft_p99_ms": self._p99(self._ttft_per_request),
            "ttft_avg_ms": self._mean(self._ttft_per_request),
            "tbt_p50_ms": self._median(self._flat_tbt()),
            "tbt_avg_ms": self._mean(self._flat_tbt()),
            "e2e_p50_ms": self._median(self._e2e_per_request),
            "e2e_avg_ms": self._mean(self._e2e_per_request),
            "slo_ttft_hit_rate": self._slo_ttft_hit_rate(),
            "cache_hit_ratio": self._cache_hit_ratio(),
            "throughput_req_per_s": self._throughput(),
        }

    # ------------------------------------------------------------------
    # Private handlers — (event, engine) -> None
    # ------------------------------------------------------------------

    def _on_arrive(self, event: Event, engine: SimulationEngine) -> None:
        req = event.payload.get("request")
        if req is None:
            logger.warning("REQUEST_ARRIVE payload missing 'request'")
            return

        self._total_arrived += 1
        if self._start_time is None:
            self._start_time = event.time

        self._requests[req.request_id] = {
            "arrival_time": event.time,
            "token_count": len(req.token_ids),
            "slo_ttft": req.slo_ttft,
            "slo_tbt": req.slo_tbt,
            "matched_tokens": None,
            "prefill_node": None,
            "ttft": None,
        }

    def _on_scheduled(self, event: Event, engine: SimulationEngine) -> None:
        request_id = event.payload.get("request_id")
        if request_id is None:
            return
        rec = self._requests.get(request_id)
        if rec is None:
            return

        rec["matched_tokens"] = event.payload.get("matched_tokens", 0)
        decision = event.payload.get("decision")
        if decision is not None:
            rec["prefill_node"] = getattr(decision, "prefill_node", None)

    def _on_rejected(self, event: Event, engine: SimulationEngine) -> None:
        request_id = event.payload.get("request_id")
        if request_id is None:
            logger.warning("REQUEST_REJECTED payload missing 'request_id', skipping")
            return
        self._rejected_count += 1

    def _on_prefill_complete(self, event: Event, engine: SimulationEngine) -> None:
        """No-op in v2 — TTFT is measured at first DECODE_STEP, not at
        prefill completion. Handler retained for future prefill-stage metrics
        (e.g. prefill duration histogram) without forcing an attach() change.
        """
        logger.debug(
            "prefill_complete: request_id=%s (no-op in v2)",
            event.payload.get("request_id"),
        )

    def _on_decode_step(self, event: Event, engine: SimulationEngine) -> None:
        request_id = event.payload.get("request_id")
        if request_id is None:
            return
        rec = self._requests.get(request_id)
        if rec is None:
            return

        step_index = event.payload.get("step_index")
        if step_index is None:
            logger.warning(
                "DECODE_STEP payload missing 'step_index' for request_id=%s, skipping",
                request_id,
            )
            return

        if step_index == 0:
            if rec.get("ttft") is not None:
                # Duplicate step 0 — entire branch skipped (TTFT not double-counted,
                # TBT anchor not reset, prevents post-duplicate TBT distortion).
                logger.debug("duplicate step_index=0 for request_id=%s, skipping", request_id)
                return
            ttft = event.time - rec["arrival_time"]
            rec["ttft"] = ttft
            self._ttft_per_request.append(ttft)
            # Seed TBT reference; subsequent steps compute (current - last).
            self._last_decode_step_time[request_id] = event.time
            self._tbt_samples.setdefault(request_id, [])
            return

        # step_index >= 1: TBT = gap since previous step.
        last = self._last_decode_step_time.get(request_id)
        if last is None:
            # step_index >= 1 arrived before step 0 — defensive fallback.
            logger.warning(
                "DECODE_STEP request_id=%s step_index=%d arrived before step 0; "
                "TBT skipped",
                request_id,
                step_index,
            )
            self._last_decode_step_time[request_id] = event.time
            return

        tbt = event.time - last
        self._tbt_samples.setdefault(request_id, []).append(tbt)
        self._last_decode_step_time[request_id] = event.time

    def _on_decode_complete(self, event: Event, engine: SimulationEngine) -> None:
        request_id = event.payload.get("request_id")
        if request_id is None:
            return
        rec = self._requests.get(request_id)
        if rec is None:
            return

        self._completed_count += 1
        self._end_time = event.time

        self._e2e_per_request.append(event.time - rec["arrival_time"])


    # ------------------------------------------------------------------
    # Private stat helpers
    # ------------------------------------------------------------------

    def _flat_tbt(self) -> list[float]:
        return [t for samples in self._tbt_samples.values() for t in samples]

    @staticmethod
    def _median(data: list[float]) -> float | None:
        return statistics.median(data) if data else None

    @staticmethod
    def _mean(data: list[float]) -> float | None:
        return statistics.mean(data) if data else None

    @staticmethod
    def _p99(data: list[float]) -> float | None:
        if not data:
            return None
        if len(data) < 2:
            return data[0]
        return statistics.quantiles(data, n=100)[98]

    def _slo_ttft_hit_rate(self) -> float | None:
        eligible = [
            (rec["ttft"], rec["slo_ttft"])
            for rec in self._requests.values()
            if rec.get("ttft") is not None and rec.get("slo_ttft") is not None
        ]
        if not eligible:
            return None
        hits = sum(1 for ttft, slo in eligible if ttft <= slo)
        return hits / len(eligible)

    def _cache_hit_ratio(self) -> float | None:
        total_tokens = matched_tokens = 0
        for rec in self._requests.values():
            if rec.get("matched_tokens") is not None:
                total_tokens += rec.get("token_count", 0)
                matched_tokens += rec["matched_tokens"]
        if total_tokens == 0:
            return None
        return matched_tokens / total_tokens

    def _throughput(self) -> float | None:
        if self._start_time is None or self._end_time is None:
            return None
        duration_s = (self._end_time - self._start_time) / 1000.0
        if duration_s <= 0:
            return None
        return self._completed_count / duration_s

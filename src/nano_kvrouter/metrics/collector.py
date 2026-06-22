from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING, Mapping

from nano_kvrouter.simulator.event import Event, EventType

if TYPE_CHECKING:
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.simulator.engine import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["MetricsCollector"]


class MetricsCollector:
    """Passive observer: subscribes to simulation events and accumulates metrics.

    TTFT / TBT definitions (v2, aligned with Mooncake §3 and Sarathi-Serve):
        TTFT = TOKEN_GENERATED[step_index=0].time - ARRIVE.time
        TBT  = avg(TOKEN_GENERATED[i+1].time - TOKEN_GENERATED[i].time) for i >= 0

    PREFILL_COMPLETE no longer participates in TTFT/TBT — it now marks the
    end of the prefill phase for the M5a ``dual_phase_tick_count`` counter.

    M5a metrics
    -----------
    * ``kv_transfer_time_avg_ms``: mean of the ``cost_ms`` carried by every
      ``KV_TRANSFER_COMPLETE`` event. Under per_node_lane contention model,
      ``cost_ms`` == service_cost_ms + queued_cost_ms (includes queue wait);
      under ``none``, equals pure transfer time. Drives ``bandwidth.gpu_to_gpu`` /
      ``model.kv_bytes_per_token`` sensitivity analysis.
    * ``kv_transfer_queued_avg_ms``: mean of the ``queued_cost_ms`` component
      from KV_TRANSFER_COMPLETE events. Zero under ``none`` contention model;
      positive under ``per_node_lane`` when lanes back up. P4-B M1.
    * ``dual_phase_tick_count``: number of ``DECODE_BATCH_STEP`` events
      that fired while at least one request anywhere in the cluster was in
      the prefill phase AND at least one was in the decode phase. Replaces
      the M3 ``prefill_decode_interleave_step_count`` (which was per-tick
      same-node and went to 0 under always-split P/D).

    Event payload contract:
        REQUEST_ARRIVE:    {"request": Request}
        SCHEDULED:         {"request_id": str, "decision": SchedulingDecision,
                            "matched_tokens": int}
        REQUEST_REJECTED:  {"request_id": str, "reason": str}
        PREFILL_START:     {"request_id": str, "n_chunks": int}
        PREFILL_COMPLETE:  {"request_id": str}
        KV_TRANSFER_START: {"transfer_id": str, "service_cost_ms": float,
                            "queued_cost_ms": float, "cost_ms": float, ...}  # debug only
        KV_TRANSFER_COMPLETE: {"transfer_id": str, "service_cost_ms": float,
                               "queued_cost_ms": float, "cost_ms": float, ...}
        TOKEN_GENERATED:   {"request_id": str, "step_index": int}
        DECODE_BATCH_STEP: {"node_id": str, "batch_size": int}
        DECODE_COMPLETE:   {"request_id": str}
        KV_LOAD_START:     {"request_id": str, "decode_node_id": str,
                            "disk_blocks": int, "load_service_ms": float,
                            "preparing_wait_ms": float}
        KV_LOAD_COMPLETE:  {"request_id": str, "decode_node_id": str,
                            "promoted_count": int, "skipped_count": int}

    Metric semantics
    ----------------
    - ``throughput_req_per_s`` / ``decode_throughput_tokens_per_s``:
      Computed as count / makespan, where makespan = last_complete_time −
      first_arrival_time (simulated ms). Not wall-clock, not workload.duration_s.
    - ``cache_hit_ratio``: Computed over ALL SCHEDULED requests, including
      those later rejected at B1 (decode capacity exhausted). Measures
      "scheduling-time cache affinity", not "completed-requests cache hit".
    - ``dual_phase_tick_count``: Counts batch ticks where both prefill and
      decode are active cluster-wide. Driven by PREFILL_START/COMPLETE
      (``_active_prefills``) and first TOKEN_GENERATED/DECODE_COMPLETE
      (``_active_decodes``). Known short window: in split P/D, decode
      actually starts at KV_TRANSFER_COMPLETE but ``_active_decodes``
      joins only at first TOKEN_GENERATED — single-token decode requests
      with prefill overlap may undercount. Deferred to P3.
    - ``kv_transfer_time_avg_ms``: Sample-mean of ``cost_ms`` from
      KV_TRANSFER_COMPLETE events (= service + queue wait under per_node_lane;
      = service only under none). Stale (unknown ``transfer_id``) events are
      silently dropped via ``_seen_transfer_ids`` guard.
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
        # M2 batch-decode metrics
        self._batch_size_samples: list[int] = []
        self._total_decode_tokens: int = 0
        # Optional node registry for execute-time batch_size fallback (order-independent).
        self._nodes: dict[str, MockEngineNode] = {}
        # M3 chunked-prefill metrics
        self._chunked_prefill_steps: list[int] = []   # n_chunks per completed request
        self._request_n_chunks: dict[str, int] = {}

        # M5a metrics
        # KV transfer cost samples (one per KV_TRANSFER_COMPLETE event).
        self._kv_transfer_cost_samples: list[float] = []
        # P4-B: queued cost samples (lane wait component of KV transfer).
        self._kv_transfer_queued_samples: list[float] = []
        # M6 tier-hit counters: accumulated from SCHEDULED matched_blocks_by_tier.
        self._cache_hit_by_tier: dict[str, int] = {"gpu": 0, "cpu": 0, "disk": 0}
        # transfer_ids seen via KV_TRANSFER_START; guards against stale
        # KV_TRANSFER_COMPLETE events that arrive after cli drops the transfer.
        self._seen_transfer_ids: set[str] = set()
        # Lifecycle counters for cluster-wide dual-phase detection. A
        # request enters _active_prefills at PREFILL_START and leaves at
        # PREFILL_COMPLETE; it enters _active_decodes at first
        # TOKEN_GENERATED (step_index=0) and leaves at DECODE_COMPLETE.
        # ``dual_phase_tick_count`` is incremented on every
        # DECODE_BATCH_STEP fired while both sets are non-empty.
        self._active_prefills: set[str] = set()
        self._active_decodes: set[str] = set()
        self._dual_phase_tick_count: int = 0

        # P5-Bidaw M1 metrics (all default to empty/0 so non-Bidaw runs
        # see zeroed fields rather than None in summary()).
        self._bidaw_preparing_wait_samples: list[float] = []
        self._bidaw_load_service_samples: list[float] = []
        self._bidaw_promotions_count: int = 0
        self._bidaw_physical_promoted_blocks: int = 0
        self._bidaw_physical_skipped_blocks: int = 0
        # Stale-guard: mirrors _seen_transfer_ids for KV_LOAD events.
        self._seen_load_req_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(
        self,
        engine: SimulationEngine,
        nodes: Mapping[str, MockEngineNode] | None = None,
    ) -> None:
        """Register all handlers with *engine*.

        Args:
            engine: The simulation engine to subscribe to.
            nodes: Optional node registry used as a fallback in
                ``_on_decode_batch_step`` when the event payload carries no
                ``batch_size`` (reversed attach order).
        """
        if nodes is not None:
            self._nodes = dict(nodes)
        engine.on(EventType.REQUEST_ARRIVE, self._on_arrive)
        engine.on(EventType.SCHEDULED, self._on_scheduled)
        engine.on(EventType.REQUEST_REJECTED, self._on_rejected)
        engine.on(EventType.PREFILL_START, self._on_prefill_start)
        engine.on(EventType.PREFILL_COMPLETE, self._on_prefill_complete)
        engine.on(EventType.KV_TRANSFER_START, self._on_kv_transfer_start)
        engine.on(EventType.KV_TRANSFER_COMPLETE, self._on_kv_transfer_complete)
        engine.on(EventType.TOKEN_GENERATED, self._on_decode_step)
        engine.on(EventType.DECODE_BATCH_STEP, self._on_decode_batch_step)
        engine.on(EventType.DECODE_COMPLETE, self._on_decode_complete)
        engine.on(EventType.KV_LOAD_START, self._on_kv_load_start)
        engine.on(EventType.KV_LOAD_COMPLETE, self._on_kv_load_complete)

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
            "avg_batch_size": self._avg_batch_size(),
            "decode_throughput_tokens_per_s": self._decode_throughput(),
            "avg_chunked_prefill_steps_per_request": (
                statistics.mean(self._chunked_prefill_steps)
                if self._chunked_prefill_steps else None
            ),
            # M5a metrics
            "kv_transfer_time_avg_ms": (
                statistics.mean(self._kv_transfer_cost_samples)
                if self._kv_transfer_cost_samples else None
            ),
            "dual_phase_tick_count": self._dual_phase_tick_count,
            # P4-B queued cost metric
            "kv_transfer_queued_avg_ms": (
                statistics.mean(self._kv_transfer_queued_samples)
                if self._kv_transfer_queued_samples else 0.0
            ),
            # M6 tier-hit metrics
            "cache_hit_by_tier_blocks": dict(self._cache_hit_by_tier),
            "cache_hit_by_tier_ratio": self._cache_hit_by_tier_ratio(),
            # P5-Bidaw M1 metrics (0.0 / 0 on non-Bidaw runs, never None)
            "bidaw_preparing_wait_avg_ms": (
                statistics.mean(self._bidaw_preparing_wait_samples)
                if self._bidaw_preparing_wait_samples else 0.0
            ),
            "bidaw_preparing_wait_p99_ms": (
                (self._p99(self._bidaw_preparing_wait_samples) or 0.0)
                if self._bidaw_preparing_wait_samples else 0.0
            ),
            "bidaw_disk_load_service_avg_ms": (
                statistics.mean(self._bidaw_load_service_samples)
                if self._bidaw_load_service_samples else 0.0
            ),
            "bidaw_preparing_promotions": self._bidaw_promotions_count,
            "bidaw_physical_promoted_blocks": self._bidaw_physical_promoted_blocks,
            "bidaw_physical_skipped_blocks": self._bidaw_physical_skipped_blocks,
            "bidaw_answer_eviction_count": 0,
            "bidaw_answer_evicted_blocks": 0,
            "bidaw_answer_eviction_cpu_saved_blocks": 0,
            "bidaw_answer_eviction_hit_potential_avg": 0.0,
            "bidaw_answer_eviction_cpu_hit_rate": 0.0,
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
            "decode_node": None,
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
            rec["decode_node"] = getattr(decision, "decode_node", None)
        # M6: accumulate per-tier hit block counts.
        by_tier = event.payload.get("matched_blocks_by_tier") or {}
        for tier, n in by_tier.items():
            if tier in self._cache_hit_by_tier:
                self._cache_hit_by_tier[tier] += n

    def _on_rejected(self, event: Event, engine: SimulationEngine) -> None:
        request_id = event.payload.get("request_id")
        if request_id is None:
            logger.warning("REQUEST_REJECTED payload missing 'request_id', skipping")
            return
        self._rejected_count += 1
        # Clean up lifecycle state set by PREFILL_START (B2 mid-prefill reject
        # path). No-ops for B1 rejects that fire before PREFILL_START.
        self._request_n_chunks.pop(request_id, None)
        self._active_prefills.discard(request_id)

    def _on_prefill_start(self, event: Event, engine: SimulationEngine) -> None:
        """Cache n_chunks + enter the active-prefills set (M5a lifecycle counter).

        The n_chunks field is injected by cli at event-creation time so this
        handler reads the correct value regardless of attach order.
        """
        request_id = event.payload.get("request_id")
        if request_id is None:
            return
        n_chunks = event.payload.get("n_chunks", 0)
        self._request_n_chunks[request_id] = int(n_chunks)
        self._active_prefills.add(request_id)

    def _on_prefill_complete(self, event: Event, engine: SimulationEngine) -> None:
        """Exit the active-prefills set (M5a lifecycle counter)."""
        request_id = event.payload.get("request_id")
        if request_id is None:
            return
        self._active_prefills.discard(request_id)

    def _on_kv_transfer_start(self, event: Event, engine: SimulationEngine) -> None:
        """Track transfer_id so KV_TRANSFER_COMPLETE can detect stale events."""
        tid = event.payload.get("transfer_id")
        if tid is not None:
            self._seen_transfer_ids.add(tid)
        logger.debug(
            "KV_TRANSFER_START transfer_id=%s cost=%.3fms",
            tid,
            event.payload.get("cost_ms", 0.0),
        )

    def _on_kv_transfer_complete(self, event: Event, engine: SimulationEngine) -> None:
        """Sample one KV transfer cost — drives kv_transfer_time_avg_ms."""
        tid = event.payload.get("transfer_id")
        if tid is None or tid not in self._seen_transfer_ids:
            return  # stale / unknown, silently drop
        self._seen_transfer_ids.discard(tid)
        cost_ms = event.payload.get("cost_ms")
        if cost_ms is None:
            return
        self._kv_transfer_cost_samples.append(float(cost_ms))
        queued_cost_ms = event.payload.get("queued_cost_ms", 0.0)
        self._kv_transfer_queued_samples.append(float(queued_cost_ms))

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
                "TOKEN_GENERATED payload missing 'step_index' for request_id=%s, skipping",
                request_id,
            )
            return

        if step_index == 0:
            if rec.get("ttft") is not None:
                logger.debug("duplicate step_index=0 for request_id=%s, skipping", request_id)
                return
            ttft = event.time - rec["arrival_time"]
            rec["ttft"] = ttft
            self._ttft_per_request.append(ttft)
            self._last_decode_step_time[request_id] = event.time
            self._tbt_samples.setdefault(request_id, [])
            self._active_decodes.add(request_id)
            return

        # step_index >= 1: TBT = gap since previous step.
        last = self._last_decode_step_time.get(request_id)
        if last is None:
            logger.warning(
                "TOKEN_GENERATED request_id=%s step_index=%d arrived before step 0; "
                "TBT skipped",
                request_id,
                step_index,
            )
            self._last_decode_step_time[request_id] = event.time
            return

        tbt = event.time - last
        self._tbt_samples.setdefault(request_id, []).append(tbt)
        self._last_decode_step_time[request_id] = event.time

    def _on_decode_batch_step(self, event: Event, engine: SimulationEngine) -> None:
        """Accumulate batch_size + cluster dual-phase counter (M5a).

        ``batch_size`` uses the payload-with-fallback pattern (M2.fix4) for
        attach-order independence.

        ``dual_phase_tick_count`` is incremented whenever this tick fires
        while there is at least one request in the prefill phase AND at
        least one in the decode phase anywhere in the cluster. Replaces
        the M3 per-tick same-node ``interleave`` semantic (which went to
        zero under always-split P/D since a prefill_node never has decode
        and a decode_node never has prefill).
        """
        node_id = event.payload.get("node_id")
        node = self._nodes.get(node_id) if node_id else None

        batch_size = event.payload.get("batch_size")
        if batch_size is None and node is not None:
            batch_size = len(node.decoding)
        if batch_size is None:
            return
        self._batch_size_samples.append(int(batch_size))
        self._total_decode_tokens += int(batch_size)

        if self._active_prefills and self._active_decodes:
            self._dual_phase_tick_count += 1

    def _on_kv_load_start(self, event: Event, engine: SimulationEngine) -> None:
        """Sample preparing_wait_ms and load_service_ms; register stale guard."""
        req_id = event.payload.get("request_id")
        if req_id is not None:
            self._seen_load_req_ids.add(req_id)
        wait_ms = event.payload.get("preparing_wait_ms")
        if wait_ms is not None:
            self._bidaw_preparing_wait_samples.append(float(wait_ms))
        service_ms = event.payload.get("load_service_ms")
        if service_ms is not None:
            self._bidaw_load_service_samples.append(float(service_ms))
        logger.debug(
            "KV_LOAD_START request_id=%s disk_blocks=%d service_ms=%.3f wait_ms=%.3f",
            req_id,
            event.payload.get("disk_blocks", 0),
            service_ms or 0.0,
            wait_ms or 0.0,
        )

    def _on_kv_load_complete(self, event: Event, engine: SimulationEngine) -> None:
        """Count promotions; stale guard mirrors KV_TRANSFER_COMPLETE pattern."""
        req_id = event.payload.get("request_id")
        if req_id is None or req_id not in self._seen_load_req_ids:
            return  # stale / unknown — silently drop
        self._seen_load_req_ids.discard(req_id)
        self._bidaw_promotions_count += 1
        self._bidaw_physical_promoted_blocks += int(event.payload.get("promoted_count", 0))
        self._bidaw_physical_skipped_blocks += int(event.payload.get("skipped_count", 0))

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

        n_chunks = self._request_n_chunks.pop(request_id, None)
        if n_chunks is not None:
            self._chunked_prefill_steps.append(n_chunks)

        # Exit the active-decodes set (M5a lifecycle counter).
        self._active_decodes.discard(request_id)

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

    def _avg_batch_size(self) -> float | None:
        return statistics.mean(self._batch_size_samples) if self._batch_size_samples else None

    def _decode_throughput(self) -> float | None:
        if self._start_time is None or self._end_time is None:
            return None
        duration_s = (self._end_time - self._start_time) / 1000.0
        if duration_s <= 0:
            return None
        return self._total_decode_tokens / duration_s
    def _cache_hit_by_tier_ratio(self) -> dict[str, float] | None:
        """Per-tier hit ratio relative to total matched blocks across all tiers."""
        total = sum(self._cache_hit_by_tier.values())
        if total == 0:
            return None
        return {tier: n / total for tier, n in self._cache_hit_by_tier.items() if n > 0}

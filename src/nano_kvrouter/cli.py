"""CLI entry point — wires generator + scheduler + cache + metrics into SimEngine.

M5a split P/D:
    * Two MockEngineNode pools: ``p{i}`` (prefill) and ``d{i}`` (decode).
    * CacheManager only tracks decode-pool nodes (KV cache lives there).
    * After PREFILL_COMPLETE on prefill_node a Mooncake-style one-shot
      KV transfer fires (KV_TRANSFER_START → KV_TRANSFER_COMPLETE), at
      which point KV is admitted into decode_node's cache and the
      request enters the decode batch.
    * Decode-side back-pressure: if decode_node is at capacity when
      KV_TRANSFER_COMPLETE fires, the request is rejected (M5a B1) —
      KV is NOT admitted into cache, decode queue stays empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Mapping
from numbers import Number
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from nano_kvrouter.config import (
    BandwidthConfig,
    ModelConfig,
    NanoKVConfig,
    SensitivityExperiment,
    load_config,
    load_sensitivity_config,
)
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.metrics.collector import MetricsCollector
from nano_kvrouter.scheduler.base import SchedulingPolicy
from nano_kvrouter.scheduler.conductor import MooncakeConductor
from nano_kvrouter.scheduler.e2_policy import E2Policy
from nano_kvrouter.scheduler.least_loaded import LeastLoadedPolicy
from nano_kvrouter.scheduler.prefix_greedy import PrefixGreedyPolicy
from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import Event, EventType
from nano_kvrouter.request import Request
from nano_kvrouter.simulator.generator import RequestGenerator
from nano_kvrouter.simulator.trace_generator import TraceGenerator

logger = logging.getLogger(__name__)
console = Console()

SCHEDULER_NAMES = ["round_robin", "least_loaded", "prefix_greedy", "e2_policy", "conductor"]


# ----------------- scheduler factory -----------------

def _build_scheduler(
    name: str,
    params: dict[str, Any],
    model_cfg: ModelConfig,
    bandwidth_cfg: BandwidthConfig | None = None,
) -> SchedulingPolicy:
    """Construct a scheduler by name.

    Numeric params are coerced through ``float()`` so YAML string values
    (e.g. ``"2.0"`` from environment overrides) are accepted without error.

    Args:
        name: Scheduler name; must be in :data:`SCHEDULER_NAMES`.
        params: Scheduler-specific keyword params from config.
        model_cfg: Forwarded to ``compute_est_ttft`` (needs ``kv_bytes_per_token``).
        bandwidth_cfg: Forwarded to ``compute_est_ttft`` (needs ``gpu_to_gpu``).
            Defaults to ``BandwidthConfig()`` if omitted.

    Returns:
        A concrete scheduler satisfying :class:`SchedulingPolicy`.

    Raises:
        ValueError: If *name* is not recognised.
    """
    bw = bandwidth_cfg if bandwidth_cfg is not None else BandwidthConfig()
    if name == "round_robin":
        return RoundRobinPolicy(model_config=model_cfg, bandwidth_config=bw)
    if name == "least_loaded":
        return LeastLoadedPolicy(model_config=model_cfg, bandwidth_config=bw)
    if name == "prefix_greedy":
        min_hit_ratio = float(params.get("min_hit_ratio", 0.25))
        return PrefixGreedyPolicy(
            min_hit_ratio=min_hit_ratio,
            model_config=model_cfg,
            bandwidth_config=bw,
        )
    if name == "e2_policy":
        return E2Policy(
            w_historical=float(params.get("w_historical", 1.0)),
            w_eviction=float(params.get("w_eviction", 1.0)),
            w_run=float(params.get("w_run", 1.0)),
            model_config=model_cfg,
            bandwidth_config=bw,
        )
    if name == "conductor":
        return MooncakeConductor(
            alpha=float(params.get("alpha", 1.0)),
            beta=float(params.get("beta", 1.0)),
            gamma=float(params.get("gamma", 1.0)),
            model_config=model_cfg,
            bandwidth_config=bw,
        )
    raise ValueError(f"Unknown scheduler {name!r}; valid: {SCHEDULER_NAMES}")


# ----------------- event wiring -----------------

def _wire_simulator(
    eng: SimulationEngine,
    sched: SchedulingPolicy,
    cm: CacheManager,
    prefill_nodes: list[MockEngineNode],
    decode_nodes: list[MockEngineNode],
    *,
    logger_: logging.Logger,
    model_cfg: ModelConfig,
    bandwidth_cfg: BandwidthConfig,
) -> None:
    """Register simulation event handlers on *eng* (M5a split P/D).

    Event flow per request:

    1. ``REQUEST_ARRIVE`` → ``sched.schedule(req, prefill_nodes, decode_nodes, cm)``.
       Decision stored in ``_decisions``. Prefill_node admits the request
       (running or queued).
    2. ``PREFILL_START`` (on prefill_node) → enter_prefill or fast-path
       PREFILL_COMPLETE.
    3. ``DECODE_BATCH_STEP`` on prefill_node ticks chunked prefill.
    4. ``PREFILL_COMPLETE`` → prefill_node.complete() (release slot, possibly
       promote queued prefill); emit ``KV_TRANSFER_START`` (now) +
       ``KV_TRANSFER_COMPLETE`` (now + cost).
    5. ``KV_TRANSFER_COMPLETE`` → if decode_node at capacity, REJECT
       (M5a B1); else cm.admit() on decode_node + decode_node.admit() +
       start_decode + wake batch step.
    6. Decode batch steps emit ``TOKEN_GENERATED`` + ``DECODE_COMPLETE``.

    Args:
        eng: Fresh engine to register handlers on.
        sched: Scheduler implementing the M5a split signature.
        cm: CacheManager — tracks decode-pool nodes only.
        prefill_nodes: Prefill pool in stable index order.
        decode_nodes: Decode pool in stable index order.
        logger_: Module-level logger forwarded from callers.
        model_cfg: For ``kv_bytes_per_token``.
        bandwidth_cfg: For ``gpu_to_gpu`` (KV transfer cost).
    """
    prefill_nodes_by_id: dict[str, MockEngineNode] = {n.node_id: n for n in prefill_nodes}
    decode_nodes_by_id: dict[str, MockEngineNode] = {n.node_id: n for n in decode_nodes}
    all_nodes_by_id: dict[str, MockEngineNode] = {**prefill_nodes_by_id, **decode_nodes_by_id}

    # Per-request bookkeeping (single source of truth across handlers).
    _decisions: dict[str, Any] = {}   # request_id → SchedulingDecision
    _requests: dict[str, Request] = {}  # request_id → Request

    # M5a KV-transfer bookkeeping.
    _pending_transfers: dict[str, dict[str, Any]] = {}
    _transfer_counter: dict[str, int] = {}  # request_id → next counter

    def _next_transfer_id(request_id: str) -> str:
        """Deterministic, run-reproducible transfer ID (S4)."""
        n = _transfer_counter.get(request_id, 0)
        _transfer_counter[request_id] = n + 1
        return f"{request_id}-xfer-{n}"

    def _compute_n_chunks(req: Request, decode_node_id: str, chunk_size: int) -> int:
        """Return prefill chunks needed for *req*.

        Looks up cache state on the decode pool because that is where KV
        cache lives (M5a). The prefill node skips cached tokens as if it
        had the same prefix locally (Mooncake §3 simplification).
        """
        try:
            matched = cm.lookup(req, decode_node_id).matched_tokens
        except KeyError:
            matched = 0
        uncached = max(0, len(req.token_ids) - matched)
        return (uncached + chunk_size - 1) // chunk_size if uncached > 0 else 0

    def _wake_batch_step(
        node: MockEngineNode,
        node_id: str,
        engine: SimulationEngine,
        *,
        time: float | None = None,
    ) -> None:
        """Schedule a DECODE_BATCH_STEP if the node has work and no step is in flight."""
        if (node.decoding or node._prefill_remaining) and not node.is_batch_step_in_flight():
            t = engine.now() if time is None else time
            engine.schedule(Event(
                time=t,
                type=EventType.DECODE_BATCH_STEP,
                payload={"node_id": node_id},
            ))
            node.mark_batch_step_scheduled()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_arrive(event: Event, engine: SimulationEngine) -> None:
        req = event.payload["request"]
        decision = sched.schedule(req, prefill_nodes, decode_nodes, cm)
        if decision.is_rejected:
            engine.schedule(Event(
                time=engine.now(),
                type=EventType.REQUEST_REJECTED,
                payload={"request_id": req.request_id, "reason": decision.reject_reason},
            ))
            return

        # Cache lookup on decode_node (where KV cache lives).
        decode_lookup = cm.lookup(req, decision.decode_node)
        engine.schedule(Event(
            time=engine.now(),
            type=EventType.SCHEDULED,
            payload={
                "request_id": req.request_id,
                "decision": decision,
                "matched_tokens": decode_lookup.matched_tokens,
                "matched_blocks_by_tier": decode_lookup.matched_blocks_by_tier,
            },
        ))

        _decisions[req.request_id] = decision
        _requests[req.request_id] = req

        prefill_node = prefill_nodes_by_id[decision.prefill_node]
        is_running = prefill_node.admit(
            req.request_id, expected_output_len=req.expected_output_len
        )
        if is_running:
            chunk_size = prefill_node.model_config.prefill_chunk_size
            engine.schedule(Event(
                time=engine.now(),
                type=EventType.PREFILL_START,
                payload={
                    "request_id": req.request_id,
                    "node_id": decision.prefill_node,
                    "request": req,
                    "n_chunks": _compute_n_chunks(req, decision.decode_node, chunk_size),
                },
            ))
        # else: queued — PREFILL_START fires from on_prefill_complete when promoted.

    def on_prefill_start(event: Event, engine: SimulationEngine) -> None:
        req = event.payload["request"]
        node_id = event.payload["node_id"]
        request_id = event.payload["request_id"]
        node = prefill_nodes_by_id[node_id]

        decision = _decisions.get(request_id)
        decode_node_id = decision.decode_node if decision is not None else node_id
        try:
            matched = cm.lookup(req, decode_node_id).matched_tokens
        except KeyError:
            matched = 0
        uncached = max(0, len(req.token_ids) - matched)

        if uncached == 0:
            # Fully cached on decode side: skip the chunked pipeline; fire
            # PREFILL_COMPLETE immediately (still triggers KV transfer).
            engine.schedule(Event(
                time=engine.now(),
                type=EventType.PREFILL_COMPLETE,
                payload={"request_id": request_id, "node_id": node_id, "request": req},
            ))
            return

        node.enter_prefill(request_id, uncached)
        _wake_batch_step(node, node_id, engine)

    def on_prefill_complete(event: Event, engine: SimulationEngine) -> None:
        req = event.payload["request"]
        node_id = event.payload["node_id"]
        request_id = event.payload["request_id"]
        prefill_node = prefill_nodes_by_id[node_id]
        decision = _decisions.get(request_id)
        if decision is None:
            logger_.warning("on_prefill_complete: no decision for %s", request_id)
            return

        # Release prefill slot — may promote a queued prefill request.
        try:
            promoted_id = prefill_node.complete(request_id)
        except ValueError:
            promoted_id = None

        if promoted_id is not None:
            promoted_req = _requests.get(promoted_id)
            promoted_decision = _decisions.get(promoted_id)
            if promoted_req is None or promoted_decision is None:
                logger_.warning(
                    "prefill_node %s promoted %s but state missing",
                    node_id, promoted_id,
                )
            else:
                prefill_node.init_promoted(
                    promoted_id, expected_output_len=promoted_req.expected_output_len
                )
                chunk_size = prefill_node.model_config.prefill_chunk_size
                engine.schedule(Event(
                    time=engine.now(),
                    type=EventType.PREFILL_START,
                    payload={
                        "request_id": promoted_id,
                        "node_id": node_id,
                        "request": promoted_req,
                        "n_chunks": _compute_n_chunks(
                            promoted_req, promoted_decision.decode_node, chunk_size
                        ),
                    },
                ))

        # Emit KV transfer event pair (Mooncake one-shot post-prefill).
        kv_bytes = len(req.token_ids) * model_cfg.kv_bytes_per_token
        cost_ms = (kv_bytes / bandwidth_cfg.gpu_to_gpu) * 1000.0
        transfer_id = _next_transfer_id(request_id)
        _pending_transfers[transfer_id] = {
            "request_id": request_id,
            "src": decision.prefill_node,
            "dst": decision.decode_node,
            "cost_ms": cost_ms,
            "request": req,
        }
        engine.schedule(Event(
            time=engine.now(),
            type=EventType.KV_TRANSFER_START,
            payload={
                "request_id": request_id,
                "transfer_id": transfer_id,
                "src_node_id": decision.prefill_node,
                "dst_node_id": decision.decode_node,
                "cost_ms": cost_ms,
            },
        ))
        engine.schedule(Event(
            time=engine.now() + cost_ms,
            type=EventType.KV_TRANSFER_COMPLETE,
            payload={
                "request_id": request_id,
                "transfer_id": transfer_id,
                "src_node_id": decision.prefill_node,
                "dst_node_id": decision.decode_node,
                "request": req,
                "cost_ms": cost_ms,
            },
        ))

    def on_kv_transfer_start(event: Event, engine: SimulationEngine) -> None:
        """Metrics hook only — actual cost paid at KV_TRANSFER_COMPLETE."""
        logger_.debug(
            "KV_TRANSFER_START: transfer_id=%s src=%s dst=%s cost=%.3fms",
            event.payload.get("transfer_id"),
            event.payload.get("src_node_id"),
            event.payload.get("dst_node_id"),
            event.payload.get("cost_ms", 0.0),
        )

    def on_kv_transfer_complete(event: Event, engine: SimulationEngine) -> None:
        transfer_id = event.payload.get("transfer_id")
        if transfer_id is None or transfer_id not in _pending_transfers:
            return  # stale or unknown — silently drop
        info = _pending_transfers.pop(transfer_id)
        request_id = info["request_id"]
        decision = _decisions.get(request_id)
        if decision is None:
            return
        req = info["request"]
        decode_node = decode_nodes_by_id[decision.decode_node]

        # [B1] Decode-side back-pressure: reject if decode_node at capacity.
        # Don't pollute cache, don't queue on decode — the request is dropped
        # and the prefill work + KV transfer are wasted (M5a back-pressure).
        if len(decode_node.running_requests) >= decode_node.node_config.capacity:
            engine.schedule(Event(
                time=engine.now(),
                type=EventType.REQUEST_REJECTED,
                payload={
                    "request_id": request_id,
                    "reason": "decode_capacity_exhausted",
                },
            ))
            _decisions.pop(request_id, None)
            _requests.pop(request_id, None)
            return

        # Admit KV into decode_node cache.
        try:
            cm.admit(req.token_ids, decision.decode_node)
        except MemoryError:
            logger_.warning(
                "cm.admit MemoryError for %s on %s; KV not cached",
                request_id, decision.decode_node,
            )

        # Admit into decode_node. We just checked capacity → returns True.
        decode_node.admit(request_id, expected_output_len=req.expected_output_len)
        decode_node.start_decode(request_id)
        _wake_batch_step(decode_node, decode_node.node_id, engine)

    def on_decode_batch_step(event: Event, engine: SimulationEngine) -> None:
        node_id = event.payload["node_id"]
        node = all_nodes_by_id[node_id]

        node.mark_batch_step_completed()

        if not node.decoding and not node._prefill_remaining:
            return

        # Capture pre-tick batch_size for metrics — injected at execute time so
        # MetricsCollector sees the correct value regardless of attach order.
        # (M5a: the old per-tick "interleave" payload field was removed; cluster
        # dual-phase concurrency is now tracked by collector lifecycle counters.)
        bs = len(node.decoding)
        event.payload["batch_size"] = bs

        next_time, completed_ids, prefill_completed_id = node.tick_batch_step(engine.now())

        # Prefill chunk finished → PREFILL_COMPLETE on this prefill_node.
        if prefill_completed_id is not None:
            req = _requests.get(prefill_completed_id)
            if req is not None:
                engine.schedule(Event(
                    time=next_time,
                    type=EventType.PREFILL_COMPLETE,
                    payload={
                        "request_id": prefill_completed_id,
                        "node_id": node_id,
                        "request": req,
                    },
                ))
            else:
                logger_.warning(
                    "node %s prefill_completed_id %s missing request object",
                    node_id, prefill_completed_id,
                )

        # TOKEN_GENERATED for every decode stream that participated this tick.
        step_streams = set(node.decoding) | set(completed_ids)
        for req_id in step_streams:
            step_idx = node._output_tokens[req_id] - 1
            engine.schedule(Event(
                time=next_time,
                type=EventType.TOKEN_GENERATED,
                payload={"request_id": req_id, "step_index": step_idx},
            ))

        for req_id in completed_ids:
            engine.schedule(Event(
                time=next_time,
                type=EventType.DECODE_COMPLETE,
                payload={"request_id": req_id, "node_id": node_id},
            ))

        if node.decoding or node._prefill_remaining:
            _wake_batch_step(node, node_id, engine, time=next_time)

    def on_decode_complete(event: Event, engine: SimulationEngine) -> None:
        request_id = event.payload["request_id"]
        node_id = event.payload["node_id"]
        node = decode_nodes_by_id[node_id]

        try:
            promoted_id = node.complete(request_id)
        except ValueError:
            promoted_id = None

        _decisions.pop(request_id, None)
        _requests.pop(request_id, None)

        # M5a rejects on decode-full, so decode queue should stay empty.
        # Defensive: if promotion happens (shouldn't), log and ignore.
        if promoted_id is not None:
            logger_.warning(
                "decode_node %s unexpectedly promoted %s — M5a should never queue on decode",
                node_id, promoted_id,
            )

        _wake_batch_step(node, node_id, engine)

    eng.on(EventType.REQUEST_ARRIVE, on_arrive)
    eng.on(EventType.PREFILL_START, on_prefill_start)
    eng.on(EventType.PREFILL_COMPLETE, on_prefill_complete)
    eng.on(EventType.KV_TRANSFER_START, on_kv_transfer_start)
    eng.on(EventType.KV_TRANSFER_COMPLETE, on_kv_transfer_complete)
    eng.on(EventType.DECODE_BATCH_STEP, on_decode_batch_step)
    eng.on(EventType.DECODE_COMPLETE, on_decode_complete)


# ----------------- run one simulation -----------------

def _run_one(cfg: NanoKVConfig, scheduler_name: str) -> dict:
    """Run a complete simulation and return the metrics summary.

    M5a builds two MockEngineNode pools (``p{i}`` prefill / ``d{i}`` decode)
    and tracks KV cache only on the decode pool.

    Args:
        cfg: Full cluster configuration.
        scheduler_name: Name of the scheduler to instantiate.

    Returns:
        ``MetricsCollector.summary()`` dict for this run.
    """
    eng = SimulationEngine()
    prefill_nodes = [
        MockEngineNode(f"p{i}", cfg.model, cfg.node)
        for i in range(cfg.cluster.prefill_nodes)
    ]
    decode_nodes = [
        MockEngineNode(f"d{i}", cfg.model, cfg.node)
        for i in range(cfg.cluster.decode_nodes)
    ]
    cm = CacheManager(
        node_ids=[n.node_id for n in decode_nodes],
        model_config=cfg.model,
        node_config=cfg.node,
        bandwidth_config=cfg.bandwidth,
        clock=eng.now,
    )
    sched = _build_scheduler(scheduler_name, cfg.scheduler.params, cfg.model, cfg.bandwidth)
    metrics = MetricsCollector()
    if cfg.trace is not None:
        gen = TraceGenerator(cfg, cfg.trace, trace_path=Path(cfg.trace.path))
    else:
        gen = RequestGenerator(cfg)

    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=logger, model_cfg=cfg.model, bandwidth_cfg=cfg.bandwidth,
    )
    all_nodes = {n.node_id: n for n in [*prefill_nodes, *decode_nodes]}
    metrics.attach(eng, nodes=all_nodes)
    gen.attach(eng)
    eng.run()

    return metrics.summary()


# ----------------- output rendering -----------------

_TABLE_KEYS = [
    "total_arrived",
    "completed",
    "rejected",
    "rejection_rate",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "ttft_avg_ms",
    "tbt_p50_ms",
    "tbt_avg_ms",
    "e2e_p50_ms",
    "e2e_avg_ms",
    "slo_ttft_hit_rate",
    "cache_hit_ratio",
    "cache_hit_by_tier_blocks",
    "cache_hit_by_tier_ratio",
    "throughput_req_per_s",
    "avg_batch_size",
    "decode_throughput_tokens_per_s",
    "avg_chunked_prefill_steps_per_request",
    "dual_phase_tick_count",
    "kv_transfer_time_avg_ms",
]

_SWEEP_KEYS = [
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tbt_avg_ms",
    "cache_hit_ratio",
    "rejection_rate",
    "throughput_req_per_s",
    "avg_batch_size",
    "decode_throughput_tokens_per_s",
    "kv_transfer_time_avg_ms",
]


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    if isinstance(v, dict):
        return "  ".join(f"{k}:{_fmt(val)}" for k, val in v.items())
    return str(v)


def _resolve_related_config_path(config_path: str, related_path: str) -> Path:
    candidate = Path(related_path)
    if candidate.is_absolute():
        return candidate

    config_parent_candidate = Path(config_path).resolve().parent / candidate
    if config_parent_candidate.exists():
        return config_parent_candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return config_parent_candidate


def _get_nested_attr(cfg: NanoKVConfig, field_path: str) -> Any:
    current: Any = cfg
    for part in field_path.split("."):
        if not hasattr(current, part):
            raise AttributeError(f"Unknown config field path: {field_path!r}")
        current = getattr(current, part)
    return current


def _set_nested_config_value(cfg: NanoKVConfig, field_path: str, value: Any) -> NanoKVConfig:
    cloned = cfg.model_copy(deep=True)
    parts = field_path.split(".")
    current: Any = cloned
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise AttributeError(f"Unknown config field path: {field_path!r}")
        current = getattr(current, part)

    leaf = parts[-1]
    if not hasattr(current, leaf):
        raise AttributeError(f"Unknown config field path: {field_path!r}")
    setattr(current, leaf, value)
    return cloned


def _set_scheduler_name(cfg: NanoKVConfig, scheduler_name: str) -> NanoKVConfig:
    cloned = cfg.model_copy(deep=True)
    cloned.scheduler.name = scheduler_name
    return cloned


def _select_metrics(summary: dict[str, Any], metric_names: list[str]) -> dict[str, Any]:
    return {name: summary.get(name) for name in metric_names}


def _delta_abs(baseline: Any, candidate: Any) -> Any:
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        return {
            key: _delta_abs(baseline.get(key), candidate.get(key))
            for key in sorted(baseline.keys() | candidate.keys())
        }
    if isinstance(baseline, Number) and isinstance(candidate, Number):
        return candidate - baseline
    return None


def _delta_pct(baseline: Any, candidate: Any) -> Any:
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        return {
            key: _delta_pct(baseline.get(key), candidate.get(key))
            for key in sorted(baseline.keys() | candidate.keys())
        }
    if not isinstance(baseline, Number) or not isinstance(candidate, Number):
        return None
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline


def _iter_metric_leaves(metric_name: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        leaves: list[tuple[str, Any]] = []
        for key, inner_value in value.items():
            leaves.extend(_iter_metric_leaves(f"{metric_name}.{key}", inner_value))
        return leaves
    return [(metric_name, value)]


def _metric_threshold_met(metric_name: str, baseline: Any, candidate: Any) -> bool:
    if not isinstance(baseline, Number) or not isinstance(candidate, Number):
        return False

    delta_abs = abs(candidate - baseline)
    delta_pct = None if baseline == 0 else abs((candidate - baseline) / baseline)
    root_metric = metric_name.split(".", 1)[0]

    if root_metric.endswith("_ratio") or root_metric.endswith("_rate"):
        return delta_abs >= 0.005
    if root_metric.endswith("_ms"):
        return delta_abs >= 0.5 or (delta_pct is not None and delta_pct >= 0.01)
    if "throughput" in root_metric or "rejection" in root_metric:
        return delta_abs >= 0.005 or (delta_pct is not None and delta_pct >= 0.01)
    if root_metric.endswith("_count") or "steps_per_request" in root_metric or "batch_size" in root_metric:
        return delta_abs >= 0.5 or (delta_pct is not None and delta_pct >= 0.01)
    return delta_abs > 1e-9


def _changed_metric_names(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> list[str]:
    changed: list[str] = []
    for metric_name, baseline_value in baseline_metrics.items():
        candidate_value = candidate_metrics.get(metric_name)
        baseline_leaves = dict(_iter_metric_leaves(metric_name, baseline_value))
        candidate_leaves = dict(_iter_metric_leaves(metric_name, candidate_value))
        for leaf_name in baseline_leaves.keys() | candidate_leaves.keys():
            if _metric_threshold_met(
                leaf_name,
                baseline_leaves.get(leaf_name),
                candidate_leaves.get(leaf_name),
            ):
                changed.append(leaf_name)
    return sorted(set(changed))


def _run_sensitivity_experiment(
    experiment: SensitivityExperiment,
    *,
    sensitivity_config_path: str,
) -> dict[str, Any]:
    base_config_path = _resolve_related_config_path(sensitivity_config_path, experiment.base_config)
    base_cfg = _set_scheduler_name(load_config(str(base_config_path)), experiment.scheduler)
    baseline_value = _get_nested_attr(base_cfg, experiment.field)
    baseline_summary = _run_one(base_cfg, experiment.scheduler)
    baseline_metrics = _select_metrics(baseline_summary, experiment.primary_metrics)

    candidates: list[dict[str, Any]] = []
    for candidate_value in experiment.values:
        candidate_cfg = _set_nested_config_value(base_cfg, experiment.field, candidate_value)
        candidate_summary = _run_one(candidate_cfg, experiment.scheduler)
        candidate_metrics = _select_metrics(candidate_summary, experiment.primary_metrics)
        changed_metrics = _changed_metric_names(baseline_metrics, candidate_metrics)
        candidates.append(
            {
                "field": experiment.field,
                "base_config": experiment.base_config,
                "scheduler": experiment.scheduler,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "primary_metrics": list(experiment.primary_metrics),
                "note": experiment.note,
                "metrics": {
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "delta_abs": _delta_abs(baseline_metrics, candidate_metrics),
                    "delta_pct": _delta_pct(baseline_metrics, candidate_metrics),
                },
                "candidate_summary": candidate_summary,
                "changed_metrics": changed_metrics,
                "changed": bool(changed_metrics),
            }
        )

    return {
        "field": experiment.field,
        "base_config": experiment.base_config,
        "scheduler": experiment.scheduler,
        "primary_metrics": list(experiment.primary_metrics),
        "note": experiment.note,
        "baseline": {
            "value": baseline_value,
            "summary": baseline_summary,
            "metrics": baseline_metrics,
        },
        "candidates": candidates,
        "changed": any(candidate["changed"] for candidate in candidates),
    }


def _run_sensitivity_report(config_path: str) -> dict[str, Any]:
    sensitivity_cfg = load_sensitivity_config(config_path)
    experiments: list[dict[str, Any]] = []

    for experiment in sensitivity_cfg.experiments:
        console.print(
            f"[dim]running sensitivity {experiment.field} "
            f"({experiment.base_config}, {experiment.scheduler})...[/dim]"
        )
        experiments.append(
            _run_sensitivity_experiment(
                experiment,
                sensitivity_config_path=config_path,
            )
        )

    passed_fields = sum(1 for experiment in experiments if experiment["changed"])
    return {
        "config": config_path,
        "experiments": experiments,
        "passed_fields": passed_fields,
        "total_fields": len(experiments),
    }


def _print_single_table(summary: dict, scheduler_name: str) -> None:
    table = Table(title=f"nano-kvrouter — {scheduler_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for k in _TABLE_KEYS:
        table.add_row(k, _fmt(summary.get(k)))
    console.print(table)


def _print_comparison_table(results: dict[str, dict]) -> None:
    table = Table(title="nano-kvrouter — 5-scheduler comparison")
    table.add_column("Scheduler", style="cyan")
    for k in _SWEEP_KEYS:
        table.add_column(k.replace("_", " "), justify="right")
    for name in SCHEDULER_NAMES:
        if name not in results:
            continue
        row = [name] + [_fmt(results[name].get(k)) for k in _SWEEP_KEYS]
        table.add_row(*row)
    console.print(table)


def _print_sensitivity_table(report: dict[str, Any]) -> None:
    table = Table(
        title=(
            "nano-kvrouter — sensitivity acceptance "
            f"({report['passed_fields']}/{report['total_fields']} fields PASS)"
        )
    )
    table.add_column("Field", style="cyan")
    table.add_column("Scenario")
    table.add_column("Value")
    table.add_column("Changed Metrics")
    table.add_column("Candidate", justify="center")
    table.add_column("Field LIVE", justify="center")

    for experiment in report["experiments"]:
        field_live = "PASS" if experiment["changed"] else "FAIL"
        scenario = f"{experiment['base_config']} / {experiment['scheduler']}"
        for candidate in experiment["candidates"]:
            changed_metrics = ", ".join(candidate["changed_metrics"]) or "—"
            candidate_live = "PASS" if candidate["changed"] else "FAIL"
            table.add_row(
                experiment["field"],
                scenario,
                f"{candidate['baseline_value']} -> {candidate['candidate_value']}",
                changed_metrics,
                candidate_live,
                field_live,
            )

    console.print(table)


def _write_csv(results: dict[str, dict], output_path: str) -> None:
    """Long format: one row per (scheduler, metric)."""
    with Path(output_path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheduler", "metric", "value"])
        for name, summary in results.items():
            for k in _TABLE_KEYS:
                w.writerow([name, k, summary.get(k)])


def _write_json(results: dict[str, dict], output_path: str) -> None:
    Path(output_path).write_text(json.dumps(results, indent=2, default=str))


def _flatten_metric_rows(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value):
            inner_value = value[key]
            rows.extend(_flatten_metric_rows(f"{prefix}.{key}", inner_value))
        return rows
    return [(prefix, value)]


def _write_sensitivity_csv(report: dict[str, Any], output_path: str) -> None:
    with Path(output_path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "field",
                "base_config",
                "scheduler",
                "baseline_value",
                "candidate_value",
                "metric",
                "baseline_metric_value",
                "candidate_metric_value",
                "delta_abs",
                "delta_pct",
                "candidate_changed",
                "field_changed",
            ]
        )
        for experiment in report["experiments"]:
            for candidate in experiment["candidates"]:
                baseline_rows = dict(_flatten_metric_rows("", candidate["metrics"]["baseline"]))
                candidate_rows = dict(_flatten_metric_rows("", candidate["metrics"]["candidate"]))
                delta_abs_rows = dict(_flatten_metric_rows("", candidate["metrics"]["delta_abs"]))
                delta_pct_rows = dict(_flatten_metric_rows("", candidate["metrics"]["delta_pct"]))
                for metric_name in sorted(baseline_rows.keys() | candidate_rows.keys()):
                    writer.writerow(
                        [
                            experiment["field"],
                            experiment["base_config"],
                            experiment["scheduler"],
                            candidate["baseline_value"],
                            candidate["candidate_value"],
                            metric_name.lstrip("."),
                            baseline_rows.get(metric_name),
                            candidate_rows.get(metric_name),
                            delta_abs_rows.get(metric_name),
                            delta_pct_rows.get(metric_name),
                            candidate["changed"],
                            experiment["changed"],
                        ]
                    )


def _write_sensitivity_json(report: dict[str, Any], output_path: str) -> None:
    Path(output_path).write_text(json.dumps(report, indent=2, default=str))


# ----------------- argparse + entrypoints -----------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run a single scheduler and print results."""
    cfg = load_config(args.config)
    if cfg.trace is not None:
        abs_path = _resolve_related_config_path(args.config, cfg.trace.path)
        cfg.trace.path = str(abs_path)
    sched_name = args.scheduler or cfg.scheduler.name
    if sched_name not in SCHEDULER_NAMES:
        raise SystemExit(
            f"unknown scheduler: {sched_name!r}; valid: {SCHEDULER_NAMES}"
        )

    console.print(f"[dim]running {sched_name}...[/dim]")
    summary = _run_one(cfg, sched_name)

    if args.output:
        if args.output.endswith(".csv"):
            _write_csv({sched_name: summary}, args.output)
        elif args.output.endswith(".json"):
            _write_json({sched_name: summary}, args.output)
        else:
            raise SystemExit(
                f"--output must end in .csv or .json; got {args.output!r}"
            )
        console.print(f"[dim]results written to {args.output}[/dim]")

    _print_single_table(summary, sched_name)


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run all 5 schedulers sequentially and print a comparison table."""
    cfg = load_config(args.config)
    if cfg.trace is not None:
        abs_path = _resolve_related_config_path(args.config, cfg.trace.path)
        cfg.trace.path = str(abs_path)
    results: dict[str, dict] = {}
    for name in SCHEDULER_NAMES:
        console.print(f"[dim]running {name}...[/dim]")
        results[name] = _run_one(cfg, name)

    if args.output:
        if args.output.endswith(".csv"):
            _write_csv(results, args.output)
        elif args.output.endswith(".json"):
            _write_json(results, args.output)
        else:
            raise SystemExit(
                f"--output must end in .csv or .json; got {args.output!r}"
            )
        console.print(f"[dim]results written to {args.output}[/dim]")

    _print_comparison_table(results)


def cmd_sensitivity(args: argparse.Namespace) -> None:
    """Run config-driven LIVE field sensitivity acceptance experiments."""
    report = _run_sensitivity_report(args.config)

    if args.output:
        if args.output.endswith(".csv"):
            _write_sensitivity_csv(report, args.output)
        elif args.output.endswith(".json"):
            _write_sensitivity_json(report, args.output)
        else:
            raise SystemExit(
                f"--output must end in .csv or .json; got {args.output!r}"
            )
        console.print(f"[dim]results written to {args.output}[/dim]")

    _print_sensitivity_table(report)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        prog="nano-kvrouter",
        description="KV-cache-aware LLM serving simulator",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single scheduler")
    p_run.add_argument("--config", required=True, help="path to YAML config file")
    p_run.add_argument(
        "--scheduler",
        default=None,
        help=f"override config.scheduler.name; one of {SCHEDULER_NAMES}",
    )
    p_run.add_argument(
        "--output",
        default=None,
        help="write results to FILE.csv or FILE.json",
    )
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", help="Run all 5 schedulers and compare")
    p_sweep.add_argument("--config", required=True, help="path to YAML config file")
    p_sweep.add_argument(
        "--output",
        default=None,
        help="write results to FILE.csv or FILE.json",
    )
    p_sweep.set_defaults(func=cmd_sweep)

    p_sensitivity = sub.add_parser(
        "sensitivity",
        help="Run config-driven LIVE field sensitivity acceptance experiments",
    )
    p_sensitivity.add_argument(
        "--config",
        required=True,
        help="path to sensitivity YAML config file",
    )
    p_sensitivity.add_argument(
        "--output",
        default=None,
        help="write results to FILE.csv or FILE.json",
    )
    p_sensitivity.set_defaults(func=cmd_sensitivity)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

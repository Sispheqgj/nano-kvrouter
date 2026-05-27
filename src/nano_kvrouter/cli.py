"""CLI entry point — wires generator + scheduler + cache + metrics into SimEngine."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from nano_kvrouter.config import ModelConfig, NanoKVConfig, load_config
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

logger = logging.getLogger(__name__)
console = Console()

SCHEDULER_NAMES = ["round_robin", "least_loaded", "prefix_greedy", "e2_policy", "conductor"]


# ----------------- scheduler factory -----------------

def _build_scheduler(
    name: str,
    params: dict[str, Any],
    model_cfg: ModelConfig,
) -> SchedulingPolicy:
    """Construct a scheduler by name.

    Numeric params are coerced through ``float()`` so YAML string values
    (e.g. ``"2.0"`` from environment overrides) are accepted without error.

    Args:
        name: Scheduler name; must be in :data:`SCHEDULER_NAMES`.
        params: Scheduler-specific keyword params from config.
        model_cfg: Forwarded to schedulers that need latency constants.

    Returns:
        A concrete scheduler satisfying :class:`SchedulingPolicy`.

    Raises:
        ValueError: If *name* is not recognised.
    """
    if name == "round_robin":
        return RoundRobinPolicy()
    if name == "least_loaded":
        return LeastLoadedPolicy()
    if name == "prefix_greedy":
        min_hit_ratio = float(params.get("min_hit_ratio", 0.25))
        return PrefixGreedyPolicy(min_hit_ratio=min_hit_ratio)
    if name == "e2_policy":
        return E2Policy(
            w_historical=float(params.get("w_historical", 1.0)),
            w_eviction=float(params.get("w_eviction", 1.0)),
            w_run=float(params.get("w_run", 1.0)),
            model_config=model_cfg,
        )
    if name == "conductor":
        return MooncakeConductor(
            alpha=float(params.get("alpha", 1.0)),
            beta=float(params.get("beta", 1.0)),
            gamma=float(params.get("gamma", 1.0)),
            model_config=model_cfg,
        )
    raise ValueError(f"Unknown scheduler {name!r}; valid: {SCHEDULER_NAMES}")


# ----------------- event wiring -----------------

def _wire_simulator(
    eng: SimulationEngine,
    sched: SchedulingPolicy,
    cm: CacheManager,
    nodes: list[MockEngineNode],
    *,
    logger_: logging.Logger,
) -> None:
    """Register the 5 simulation event handlers on *eng*.

    All state flows through event payloads — no request-id -> node-id dict is
    maintained externally. ``cm.admit()`` MemoryErrors and ``node.complete()``
    ValueErrors are caught defensively so the simulation never crashes mid-run.

    Args:
        eng: Fresh engine to register handlers on.
        sched: Scheduler to call on REQUEST_ARRIVE.
        cm: CacheManager for prefix lookup and KV admission.
        nodes: Cluster nodes in stable index order.
        logger_: Module-level logger forwarded from callers.
    """
    nodes_by_id: dict[str, MockEngineNode] = {n.node_id: n for n in nodes}
    # Per-node registry of Request objects that are running or queued.
    # Keyed by request_id; removed at DECODE_COMPLETE. Required so
    # on_decode_complete can fire PREFILL_START for the promoted request.
    _queued_reqs: dict[str, dict[str, Request]] = {n.node_id: {} for n in nodes}

    def on_arrive(event: Event, engine: SimulationEngine) -> None:
        req = event.payload["request"]
        decision = sched.schedule(req, nodes, cm)
        if decision.is_rejected:
            engine.schedule(Event(
                time=engine.now(),
                type=EventType.REQUEST_REJECTED,
                payload={"request_id": req.request_id, "reason": decision.reject_reason},
            ))
            return
        # Lookup cache after routing so matched_tokens reflects the chosen node.
        matched = cm.lookup(req, decision.prefill_node).matched_tokens
        engine.schedule(Event(
            time=engine.now(),
            type=EventType.SCHEDULED,
            payload={
                "request_id": req.request_id,
                "decision": decision,
                "matched_tokens": matched,
            },
        ))
        node = nodes_by_id[decision.prefill_node]
        is_running = node.admit(req.request_id)
        # Store req so on_decode_complete can fire PREFILL_START if queued.
        _queued_reqs[decision.prefill_node][req.request_id] = req
        if is_running:
            engine.schedule(Event(
                time=engine.now(),
                type=EventType.PREFILL_START,
                payload={
                    "request_id": req.request_id,
                    "node_id": decision.prefill_node,
                    "request": req,
                },
            ))
        # else: queued — PREFILL_START fires from on_decode_complete when promoted

    def on_prefill_start(event: Event, engine: SimulationEngine) -> None:
        req = event.payload["request"]
        node_id = event.payload["node_id"]
        request_id = event.payload["request_id"]
        node = nodes_by_id[node_id]
        matched = cm.lookup(req, node_id).matched_tokens
        prefill_ms = node.estimate_prefill_time(len(req.token_ids), cached_tokens=matched)
        engine.schedule(Event(
            time=engine.now() + prefill_ms,
            type=EventType.PREFILL_COMPLETE,
            payload={"request_id": request_id, "node_id": node_id, "request": req},
        ))

    def on_prefill_complete(event: Event, engine: SimulationEngine) -> None:
        req = event.payload["request"]
        node_id = event.payload["node_id"]
        request_id = event.payload["request_id"]
        try:
            cm.admit(req.token_ids, node_id)
        except MemoryError:
            # Pool exhausted with all leaves pinned — KV not materialised.
            # Simulation continues; future lookups will miss, imposing a cold
            # prefill penalty without stopping the run.
            logger_.warning(
                "cm.admit MemoryError for %s on %s; KV not cached", request_id, node_id
            )
        engine.schedule(Event(
            time=engine.now(),
            type=EventType.DECODE_STEP,
            payload={
                "request_id": request_id,
                "node_id": node_id,
                "request": req,
                "step_index": 0,
            },
        ))

    def on_decode_step(event: Event, engine: SimulationEngine) -> None:
        req = event.payload["request"]
        node_id = event.payload["node_id"]
        request_id = event.payload["request_id"]
        step_index = event.payload["step_index"]
        node = nodes_by_id[node_id]
        if step_index >= req.expected_output_len - 1:
            engine.schedule(Event(
                time=engine.now(),
                type=EventType.DECODE_COMPLETE,
                payload={"request_id": request_id, "node_id": node_id},
            ))
        else:
            tbt = node.estimate_decode_time(len(node.running_requests))
            engine.schedule(Event(
                time=engine.now() + tbt,
                type=EventType.DECODE_STEP,
                payload={
                    "request_id": request_id,
                    "node_id": node_id,
                    "request": req,
                    "step_index": step_index + 1,
                },
            ))

    def on_decode_complete(event: Event, engine: SimulationEngine) -> None:
        request_id = event.payload["request_id"]
        node_id = event.payload["node_id"]
        try:
            promoted_id = nodes_by_id[node_id].complete(request_id)
        except ValueError:
            # Defensive: request already removed (duplicate DECODE_COMPLETE guard).
            promoted_id = None

        # Clean up the completed request's stored object.
        _queued_reqs[node_id].pop(request_id, None)

        # If complete() promoted a queued request, fire its PREFILL_START now.
        if promoted_id is not None:
            promoted_req = _queued_reqs[node_id].get(promoted_id)
            if promoted_req is None:
                logger_.warning(
                    "node %s promoted %s but request object missing from _queued_reqs",
                    node_id, promoted_id,
                )
            else:
                engine.schedule(Event(
                    time=engine.now(),
                    type=EventType.PREFILL_START,
                    payload={
                        "request_id": promoted_id,
                        "node_id": node_id,
                        "request": promoted_req,
                    },
                ))

    eng.on(EventType.REQUEST_ARRIVE, on_arrive)
    eng.on(EventType.PREFILL_START, on_prefill_start)
    eng.on(EventType.PREFILL_COMPLETE, on_prefill_complete)
    eng.on(EventType.DECODE_STEP, on_decode_step)
    eng.on(EventType.DECODE_COMPLETE, on_decode_complete)


# ----------------- run one simulation -----------------

def _run_one(cfg: NanoKVConfig, scheduler_name: str) -> dict:
    """Run a complete simulation and return the metrics summary.

    Every call constructs fresh engine / nodes / cache / metrics / generator
    instances so repeated calls produce independent results. The
    ``RequestGenerator`` raises RuntimeError on double-attach, making reuse
    unsafe — always pass a freshly instantiated generator.

    Args:
        cfg: Full cluster configuration.
        scheduler_name: Name of the scheduler to instantiate; must be in
            :data:`SCHEDULER_NAMES`.

    Returns:
        ``MetricsCollector.summary()`` dict for this run.
    """
    eng = SimulationEngine()
    nodes = [
        MockEngineNode(f"n{i}", cfg.model, cfg.node)
        for i in range(cfg.cluster.prefill_nodes)
    ]
    cm = CacheManager(
        node_ids=[n.node_id for n in nodes],
        model_config=cfg.model,
        node_config=cfg.node,
        bandwidth_config=cfg.bandwidth,
        clock=eng.now,
    )
    sched = _build_scheduler(scheduler_name, cfg.scheduler.params, cfg.model)
    metrics = MetricsCollector()
    gen = RequestGenerator(cfg)

    _wire_simulator(eng, sched, cm, nodes, logger_=logger)
    metrics.attach(eng)
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
    "throughput_req_per_s",
]

_SWEEP_KEYS = [
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tbt_avg_ms",
    "cache_hit_ratio",
    "rejection_rate",
    "throughput_req_per_s",
]


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


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


# ----------------- argparse + entrypoints -----------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run a single scheduler and print results."""
    cfg = load_config(args.config)
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

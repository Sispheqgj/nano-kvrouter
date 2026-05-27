"""End-to-end CLI tests — wire all modules through SimulationEngine."""
from __future__ import annotations

import pytest

from nano_kvrouter.config import (
    BandwidthConfig,
    ClusterConfig,
    GeneratorConfig,
    ModelConfig,
    NanoKVConfig,
    NodeConfig,
    SchedulerConfig,
    SLOConfig,
    WorkloadConfig,
)
from nano_kvrouter.scheduler.base import SchedulingPolicy
from nano_kvrouter.scheduler.conductor import MooncakeConductor
from nano_kvrouter.cli import SCHEDULER_NAMES, _build_scheduler, _run_one, _wire_simulator
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import Event, EventType
import logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_cfg(
    scheduler_name: str = "round_robin",
    duration: float = 1.0,
    rate: float = 20.0,
) -> NanoKVConfig:
    """Small config for fast tests — 1s × 20 req/s ≈ 20 requests."""
    return NanoKVConfig(
        cluster=ClusterConfig(prefill_nodes=2, decode_nodes=2),
        node=NodeConfig(gpu_blocks=200, cpu_blocks=400, disk_blocks=800, capacity=16),
        model=ModelConfig(block_size=16),
        bandwidth=BandwidthConfig(),
        slo=SLOConfig(),
        workload=WorkloadConfig(
            request_rate=rate,
            duration_s=duration,
            avg_prompt_len=128,
            avg_output_len=4,
            prefix_sharing_ratio=0.5,
        ),
        scheduler=SchedulerConfig(name=scheduler_name),
        generator=GeneratorConfig(num_buckets=5, seed=42),
    )


# ---------------------------------------------------------------------------
# _build_scheduler factory tests
# ---------------------------------------------------------------------------

def test_build_scheduler_each_name_returns_protocol() -> None:
    for name in SCHEDULER_NAMES:
        sched = _build_scheduler(name, {}, ModelConfig())
        assert isinstance(sched, SchedulingPolicy), f"{name} not SchedulingPolicy"


def test_build_scheduler_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler"):
        _build_scheduler("nonexistent", {}, ModelConfig())


def test_build_scheduler_coerces_string_params() -> None:
    """YAML may deliver numeric params as strings; factory must coerce to float."""
    sched = _build_scheduler("conductor", {"alpha": "2.0", "beta": "1.5"}, ModelConfig())
    assert isinstance(sched, MooncakeConductor)


def test_build_scheduler_prefix_greedy_custom_ratio() -> None:
    from nano_kvrouter.scheduler.prefix_greedy import PrefixGreedyPolicy

    sched = _build_scheduler("prefix_greedy", {"min_hit_ratio": 0.5}, ModelConfig())
    assert isinstance(sched, PrefixGreedyPolicy)


def test_build_scheduler_e2_policy_custom_weights() -> None:
    from nano_kvrouter.scheduler.e2_policy import E2Policy

    sched = _build_scheduler(
        "e2_policy", {"w_historical": 2.0, "w_eviction": 0.5, "w_run": 1.0}, ModelConfig()
    )
    assert isinstance(sched, E2Policy)


# ---------------------------------------------------------------------------
# End-to-end smoke tests
# ---------------------------------------------------------------------------

def test_single_scheduler_run_produces_summary() -> None:
    cfg = _small_cfg("round_robin")
    summary = _run_one(cfg, "round_robin")
    assert summary["total_arrived"] > 0
    assert summary["completed"] > 0
    assert summary["ttft_p50_ms"] is not None
    # round_robin never emits REQUEST_REJECTED
    assert summary["rejection_rate"] == 0.0


@pytest.mark.parametrize("scheduler_name", SCHEDULER_NAMES)
def test_each_scheduler_can_complete_a_run(scheduler_name: str) -> None:
    cfg = _small_cfg(scheduler_name)
    summary = _run_one(cfg, scheduler_name)
    assert summary["total_arrived"] > 0
    # Conductor may reject on SLO; all others should never reject
    if scheduler_name != "conductor":
        assert summary["rejection_rate"] == 0.0


def test_cache_aware_scheduler_gets_nonzero_cache_hits() -> None:
    """High prefix_sharing + longer duration → cache-aware policy shows hits."""
    cfg = _small_cfg("prefix_greedy", duration=2.0, rate=20.0)
    cfg.workload.prefix_sharing_ratio = 0.75
    cfg.workload.avg_prompt_len = 256  # ensures shared prefix >= block_size
    summary = _run_one(cfg, "prefix_greedy")
    assert summary["cache_hit_ratio"] is not None
    assert summary["cache_hit_ratio"] > 0.0


def test_cache_blind_baseline_hit_ratio_below_half() -> None:
    """Round-robin does not colocate same-bucket requests, so hit ratio stays low."""
    cfg = _small_cfg("round_robin", duration=2.0, rate=20.0)
    cfg.workload.prefix_sharing_ratio = 0.75
    summary = _run_one(cfg, "round_robin")
    # Some incidental hits may occur (same bucket lands same node by chance),
    # but the ratio should be well below a cache-aware policy.
    assert (summary["cache_hit_ratio"] or 0.0) < 0.5


def test_metrics_total_arrived_within_poisson_range() -> None:
    """Sanity: every REQUEST_ARRIVE scheduled by generator is counted."""
    cfg = _small_cfg(rate=10.0, duration=1.0)
    summary = _run_one(cfg, "round_robin")
    # Poisson(λ=10) over 1s: P(3 ≤ X ≤ 25) ≈ 1.
    assert 1 <= summary["total_arrived"] <= 30


def test_fresh_instances_per_scheduler_in_sweep() -> None:
    """Running two schedulers back-to-back must yield independent results."""
    cfg_rr = _small_cfg("round_robin")
    cfg_ll = _small_cfg("least_loaded")
    summary_rr = _run_one(cfg_rr, "round_robin")
    summary_ll = _run_one(cfg_ll, "least_loaded")
    # Both should complete all arrived requests (no rejection for either).
    assert summary_rr["rejection_rate"] == 0.0
    assert summary_ll["rejection_rate"] == 0.0
    # same seed → same arrival count
    assert summary_rr["total_arrived"] == summary_ll["total_arrived"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_dummy_request(req_id: str, n_tokens: int = 128) -> Request:
    return Request(
        request_id=req_id,
        token_ids=list(range(n_tokens)),
        prefix_hash="x",
        expected_output_len=4,
        arrival_time=0.0,
        slo_ttft=2000.0,
        slo_tbt=100.0,
    )


# ---------------------------------------------------------------------------
# Codex regression — capacity gating actually enforced in cli wire path
# ---------------------------------------------------------------------------

def test_queued_request_does_not_get_prefill_start_immediately() -> None:
    """Two requests arrive at t=0 on a capacity=1 node. Only the first
    should receive PREFILL_START at t=0; the second must wait until the
    first's DECODE_COMPLETE promotes it."""
    cfg = _small_cfg("round_robin")
    cfg.cluster.prefill_nodes = 1
    cfg.node.capacity = 1

    eng = SimulationEngine()
    nodes = [MockEngineNode("n0", cfg.model, cfg.node)]
    cm = CacheManager(["n0"], cfg.model, cfg.node, cfg.bandwidth, clock=eng.now)
    sched = RoundRobinPolicy()

    _wire_simulator(eng, sched, cm, nodes, logger_=logging.getLogger("test"))

    # Record (simulated_time, request_id) for every PREFILL_START event.
    starts: list[tuple[float, str]] = []
    eng.on(
        EventType.PREFILL_START,
        lambda ev, e: starts.append((ev.time, ev.payload["request_id"])),
    )

    req_a = _make_dummy_request("a", n_tokens=160)
    req_b = _make_dummy_request("b", n_tokens=160)
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_a}))
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_b}))

    eng.run()

    assert len(starts) == 2, f"Expected 2 PREFILL_START events, got {len(starts)}"
    a_start = next(t for t, rid in starts if rid == "a")
    b_start = next(t for t, rid in starts if rid == "b")
    assert a_start == 0.0, f"req a should start prefill at t=0, got {a_start}"
    assert b_start > a_start, (
        f"queued req b started at t={b_start}, expected > {a_start}; "
        "capacity gating appears broken"
    )


def test_capacity_full_node_throttles_throughput() -> None:
    """With capacity=1 and heavy load, throughput is bounded by service rate,
    not arrival rate. Verifies end-to-end capacity gating in _run_one."""
    cfg = _small_cfg("round_robin", duration=2.0, rate=100.0)
    cfg.cluster.prefill_nodes = 1
    cfg.node.capacity = 1

    summary = _run_one(cfg, "round_robin")

    # Cold service time ≈ 128×0.033 + 3×5.5 ≈ 20.7ms → service rate ≈ 48 req/s.
    # With proper serialisation, throughput < 80 req/s.
    # Without capacity gating all requests complete "instantly" → >> 100 req/s.
    tput = summary["throughput_req_per_s"]
    if tput is not None:
        assert tput < 80.0, (
            f"capacity=1 should serialise requests; observed throughput "
            f"{tput:.1f} req/s (expected < 80). Capacity gating may be broken."
        )

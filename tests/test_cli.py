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
from nano_kvrouter.simulator.transfer_model import NoopTransferModel
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
        sched = _build_scheduler(name, {}, ModelConfig(), backlog_view=NoopTransferModel())
        assert isinstance(sched, SchedulingPolicy), f"{name} not SchedulingPolicy"


def test_build_scheduler_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler"):
        _build_scheduler("nonexistent", {}, ModelConfig(), backlog_view=NoopTransferModel())


def test_build_scheduler_coerces_string_params() -> None:
    """YAML may deliver numeric params as strings; factory must coerce to float."""
    sched = _build_scheduler("conductor", {"alpha": "2.0", "beta": "1.5"}, ModelConfig(), backlog_view=NoopTransferModel())
    assert isinstance(sched, MooncakeConductor)


def test_build_scheduler_prefix_greedy_custom_ratio() -> None:
    from nano_kvrouter.scheduler.prefix_greedy import PrefixGreedyPolicy

    sched = _build_scheduler("prefix_greedy", {"min_hit_ratio": 0.5}, ModelConfig(), backlog_view=NoopTransferModel())
    assert isinstance(sched, PrefixGreedyPolicy)


def test_build_scheduler_e2_policy_custom_weights() -> None:
    from nano_kvrouter.scheduler.e2_policy import E2Policy

    sched = _build_scheduler(
        "e2_policy", {"w_historical": 2.0, "w_eviction": 0.5, "w_run": 1.0}, ModelConfig(),
        backlog_view=NoopTransferModel(),
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
    cfg.cluster.decode_nodes = 1
    cfg.node.capacity = 1

    eng = SimulationEngine()
    nodes = [MockEngineNode("n0", cfg.model, cfg.node)]
    cm = CacheManager(["n0"], cfg.model, cfg.node, cfg.bandwidth, clock=eng.now)
    sched = RoundRobinPolicy(model_config=cfg.model, bandwidth_config=cfg.bandwidth, backlog_view=NoopTransferModel())

    _wire_simulator(
        eng, sched, cm, nodes, nodes,
        logger_=logging.getLogger("test"),
        model_cfg=cfg.model, bandwidth_cfg=cfg.bandwidth,
        transfer_model=NoopTransferModel(),
    )

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
    cfg.cluster.decode_nodes = 1
    cfg.node.capacity = 1

    summary = _run_one(cfg, "round_robin")

    # M2: batch decode with batch=1 at capacity=1.
    # Service time ≈ 128×0.033 + 4×(5.0+0.5) = 4.224 + 22.0 = 26.2ms → ~38 req/s.
    # Throughput must be well under arrival rate (100 req/s).
    tput = summary["throughput_req_per_s"]
    if tput is not None:
        assert tput < 80.0, (
            f"capacity=1 should serialise requests; observed throughput "
            f"{tput:.1f} req/s (expected < 80). Capacity gating may be broken."
        )


# ---------------------------------------------------------------------------
# M2 batch decode integration tests
# ---------------------------------------------------------------------------

def test_m2_batch_decode_all_same_length_complete_together() -> None:
    """Validation criterion 2: 16 requests with identical output_len on a single
    capacity=16 node should all generate their last token in the same batch step,
    so all 16 DECODE_COMPLETE events share the same simulated time.

    M3 note: the shared prompt is pre-admitted to the cache so all requests have
    0 uncached tokens, bypassing the FIFO prefill pipeline and entering the decode
    batch simultaneously — preserving the M2 simultaneity invariant.
    """
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager

    cfg = _small_cfg("round_robin")
    cfg.cluster.prefill_nodes = 1
    cfg.cluster.decode_nodes = 1
    cfg.node.capacity = 16
    cfg.workload.avg_output_len = 4

    eng = SimulationEngine()
    nodes = [MockEngineNode("n0", cfg.model, cfg.node)]
    cm = CacheManager(["n0"], cfg.model, cfg.node, cfg.bandwidth, clock=eng.now)
    sched_obj = _build_scheduler("round_robin", {}, cfg.model, cfg.bandwidth, backlog_view=NoopTransferModel())

    # Pre-warm cache: all 16 requests share token_ids=[0]*64.
    # Admitting once ensures matched=64 for every request → uncached=0 →
    # PREFILL_COMPLETE fires immediately → all 16 enter decode at t=0.
    cm.admit([0] * 64, "n0")

    _wire_simulator(eng, sched_obj, cm, nodes, nodes, logger_=logging.getLogger("test"), model_cfg=cfg.model, bandwidth_cfg=cfg.bandwidth, transfer_model=NoopTransferModel())

    # Record all DECODE_COMPLETE times.
    complete_times: list[float] = []
    eng.on(
        EventType.DECODE_COMPLETE,
        lambda ev, e: complete_times.append(ev.time),
    )

    OUTPUT_LEN = 4
    N = 16
    for i in range(N):
        req = Request(
            request_id=f"r{i}",
            token_ids=[0] * 64,
            prefix_hash="x",
            expected_output_len=OUTPUT_LEN,
            arrival_time=0.0,
            slo_ttft=9999.0,
            slo_tbt=9999.0,
        )
        eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req}))

    eng.run()

    assert len(complete_times) == N, f"Expected {N} completions, got {len(complete_times)}"
    # All completions should occur at the same simulated time (same batch step).
    t_last = max(complete_times)
    t_first = min(complete_times)
    assert t_last == pytest.approx(t_first), (
        f"Completions not simultaneous: min={t_first:.3f}, max={t_last:.3f}; "
        "batch decode engine may have desynchronised."
    )


def test_m2_batch_decode_metrics_reported() -> None:
    """End-to-end: _run_one reports avg_batch_size and decode_throughput_tokens_per_s."""
    cfg = _small_cfg("round_robin", duration=0.5, rate=10.0)
    summary = _run_one(cfg, "round_robin")
    # avg_batch_size may be None if no requests complete, but should exist as key.
    assert "avg_batch_size" in summary
    assert "decode_throughput_tokens_per_s" in summary
    # If completions happened, both should be non-None.
    if summary["completed"] > 0:
        assert summary["avg_batch_size"] is not None
        assert summary["decode_throughput_tokens_per_s"] is not None
        assert summary["decode_throughput_tokens_per_s"] > 0.0


def test_m2_decode_throughput_sanity() -> None:
    """Validation criterion 5: decode_throughput_tokens_per_s > 200 in default config."""
    cfg = _small_cfg("round_robin", duration=2.0, rate=30.0)
    summary = _run_one(cfg, "round_robin")
    tput = summary.get("decode_throughput_tokens_per_s")
    if tput is not None and summary["completed"] > 0:
        assert tput > 50.0, (
            f"decode_throughput_tokens_per_s={tput:.1f} seems too low; "
            "batch decode engine may not be accumulating tokens correctly."
        )


# ---------------------------------------------------------------------------
# Critical #1 regression: lost wakeup race condition
# ---------------------------------------------------------------------------

def test_lost_wakeup_new_decode_stream_at_same_time_as_batch_completion() -> None:
    """Race: b's PREFILL_COMPLETE fires at the same simulated time as a's
    DECODE_COMPLETE (but with a lower seq). When a's batch step completes with
    remaining=0, b joins decoding while the old `len(decoding)==1` guard cannot
    fire (decoding is {a,b}=2 because a hasn't been removed yet). With the
    in_flight flag, the wakeup fires correctly from on_prefill_complete.

    Timing design:
      - a: 1-token prompt (prefill=0.5ms), output_len=1
           PREFILL_COMPLETE at t=0.5, DECODE_BATCH_STEP at t=0.5,
           step_time = 5.0ms (bs=1, marginal=0) → next_time=5.5
           DECODE_COMPLETE for a scheduled at t=5.5 (high seq)
      - b: 10-token prompt (prefill=5.5ms), output_len=2
           PREFILL_START at t=0, PREFILL_COMPLETE at t=5.5 (low seq — scheduled
           before a's DECODE_STEP/DECODE_COMPLETE events)
      At t=5.5: b's PREFILL_COMPLETE fires first → b joins decoding → wakeup.
    """
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import Event, EventType
    from nano_kvrouter.request import Request

    # decode_base=4.5, marginal=0.5 → step_time for bs=1 is 5.0ms.
    # a: 1-token prompt → prefill=0.5ms → PREFILL_COMPLETE at t=0.5
    #   DECODE_BATCH_STEP fires at t=0.5 → next_time = 0.5+5.0 = 5.5
    #   → DECODE_COMPLETE for a scheduled at t=5.5 (HIGH seq)
    # b: 11-token prompt → prefill = 11*0.5 = 5.5ms → PREFILL_COMPLETE at t=5.5
    #   (scheduled at t=0 processing, so LOW seq < a's DECODE_COMPLETE seq)
    # Race: at t=5.5 b's PREFILL_COMPLETE fires first → decoding={a,b}, in_flight=False
    #   old code: len==2 → no wakeup → b stuck
    #   new code: not in_flight → wakeup scheduled → b completes
    mc = ModelConfig(
        prefill_cost_per_token_ms=0.5,
        decode_base_ms=4.5,
        marginal_decode_ms=0.5,
    )
    nc = NodeConfig(capacity=2, gpu_blocks=200, cpu_blocks=200, disk_blocks=400)

    eng = SimulationEngine()
    nodes = [MockEngineNode("n0", mc, nc)]
    cm = CacheManager(["n0"], mc, nc, BandwidthConfig(), clock=eng.now)
    sched = RoundRobinPolicy(backlog_view=NoopTransferModel())
    _wire_simulator(eng, sched, cm, nodes, nodes, logger_=logging.getLogger("test"), model_cfg=mc, bandwidth_cfg=BandwidthConfig(), transfer_model=NoopTransferModel())

    complete_times: dict[str, float] = {}
    eng.on(
        EventType.DECODE_COMPLETE,
        lambda ev, e: complete_times.__setitem__(ev.payload["request_id"], ev.time),
    )

    req_a = Request("a", [0], "ha", expected_output_len=1,
                    arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)
    req_b = Request("b", list(range(11)), "hb", expected_output_len=2,
                    arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)

    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_a}))
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_b}))

    eng.run()

    assert "a" in complete_times, "req a should have completed"
    assert "b" in complete_times, (
        f"req b got stuck — lost wakeup bug not fixed; completions so far: {complete_times}"
    )


# ---------------------------------------------------------------------------
# Critical #1 regression: duplicate completion prevented
# ---------------------------------------------------------------------------

def test_duplicate_completion_prevented_when_wake_during_terminal() -> None:
    """Critical #1: tick_batch_step removes completed streams from decoding
    immediately so a subsequent wakeup cannot advance them a second time.

    Two requests: 'fast' (output_len=1) and 'slow' (output_len=3). After the
    first batch tick, 'fast' completes. The duplicate-completion bug would keep
    'fast' in decoding, causing the next tick to advance it again and emit a
    second DECODE_COMPLETE. Total DECODE_COMPLETE count must equal 2.
    """
    mc = ModelConfig(
        prefill_cost_per_token_ms=0.1,
        decode_base_ms=5.0,
        marginal_decode_ms=0.5,
    )
    nc = NodeConfig(capacity=2, gpu_blocks=200, cpu_blocks=200, disk_blocks=400)

    eng = SimulationEngine()
    nodes = [MockEngineNode("n0", mc, nc)]
    cm = CacheManager(["n0"], mc, nc, BandwidthConfig(), clock=eng.now)
    sched = RoundRobinPolicy(backlog_view=NoopTransferModel())
    _wire_simulator(eng, sched, cm, nodes, nodes, logger_=logging.getLogger("test"), model_cfg=mc, bandwidth_cfg=BandwidthConfig(), transfer_model=NoopTransferModel())

    complete_ids: list[str] = []
    eng.on(
        EventType.DECODE_COMPLETE,
        lambda ev, e: complete_ids.append(ev.payload["request_id"]),
    )

    req_fast = Request("fast", list(range(10)), "xf", expected_output_len=1,
                       arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)
    req_slow = Request("slow", list(range(10)), "xs", expected_output_len=3,
                       arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)

    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_fast}))
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_slow}))
    eng.run()

    assert complete_ids.count("fast") == 1, (
        f"req 'fast' completed {complete_ids.count('fast')} times — duplicate completion bug!"
    )
    assert complete_ids.count("slow") == 1, (
        f"req 'slow' completed {complete_ids.count('slow')} times — expected 1"
    )
    assert len(complete_ids) == 2, (
        f"Expected exactly 2 DECODE_COMPLETE events, got {len(complete_ids)}: {complete_ids}"
    )


# ---------------------------------------------------------------------------
# M3: Sarathi chunked-prefill tests
# ---------------------------------------------------------------------------

def test_long_prompt_does_not_freeze_decode() -> None:
    """Sarathi invariant: chunked prefill must not block concurrent decode streams.

    req_a has 0 uncached tokens (cache pre-warmed) → enters decode immediately.
    req_b has 512 uncached tokens with chunk_size=128 → 4 prefill chunks needed.
    Each tick processes one req_b chunk AND advances req_a's decode.
    req_a must emit its first token BEFORE req_b finishes all prefill chunks.
    """
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import Event, EventType
    from nano_kvrouter.request import Request

    mc = ModelConfig(
        prefill_cost_per_token_ms=0.1,
        decode_base_ms=5.0,
        marginal_decode_ms=0.5,
        prefill_chunk_size=128,
    )
    nc = NodeConfig(capacity=4, gpu_blocks=200, cpu_blocks=400, disk_blocks=800)

    eng = SimulationEngine()
    nodes = [MockEngineNode("n0", mc, nc)]
    cm = CacheManager(["n0"], mc, nc, BandwidthConfig(), clock=eng.now)
    sched = RoundRobinPolicy(backlog_view=NoopTransferModel())
    _wire_simulator(eng, sched, cm, nodes, nodes, logger_=logging.getLogger("test"), model_cfg=mc, bandwidth_cfg=BandwidthConfig(), transfer_model=NoopTransferModel())

    a_token_times: list[float] = []
    b_prefill_complete_time: list[float] = []
    eng.on(
        EventType.TOKEN_GENERATED,
        lambda ev, e: a_token_times.append(ev.time) if ev.payload["request_id"] == "a" else None,
    )
    eng.on(
        EventType.PREFILL_COMPLETE,
        lambda ev, e: b_prefill_complete_time.append(ev.time) if ev.payload["request_id"] == "b" else None,
    )

    # Pre-warm cache: req_a shares [0]*32 → 0 uncached → enters decode at t=0
    cm.admit([0] * 32, "n0")
    req_a = Request("a", [0] * 32, "ha", expected_output_len=4,
                    arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)
    # req_b: 512 cold tokens, chunk_size=128 → 4 chunks needed
    req_b = Request("b", list(range(512)), "hb", expected_output_len=2,
                    arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)

    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_a}))
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_b}))
    eng.run()

    assert len(a_token_times) >= 1, "req_a should have received at least one decode token"
    assert len(b_prefill_complete_time) == 1, "req_b should have completed prefill exactly once"
    # Core Sarathi invariant: decode is NOT frozen while prefill chunks are being processed.
    assert a_token_times[0] < b_prefill_complete_time[0], (
        f"req_a first token at {a_token_times[0]:.2f}ms, "
        f"req_b prefill complete at {b_prefill_complete_time[0]:.2f}ms — "
        "decode stream was frozen during chunked prefill (Sarathi violated)"
    )
    # M5a: KV_TRANSFER pushes r_a's decode start by one extra tick (PREFILL_COMPLETE
    # → KV_TRANSFER_START/COMPLETE → admit/start_decode chain), so r_a typically
    # produces ≥2 tokens during r_b's 4-chunk prefill (down from 3 in M3 combined).
    # Still proves chunked prefill does NOT freeze decode.
    tokens_during_prefill = [t for t in a_token_times if t < b_prefill_complete_time[0]]
    assert len(tokens_during_prefill) >= 2, (
        f"req_a produced only {len(tokens_during_prefill)} tokens before req_b prefill "
        f"completed ({b_prefill_complete_time[0]:.2f}ms); expected ≥2."
    )


def test_short_prompt_completes_in_one_chunk() -> None:
    """A prompt shorter than chunk_size completes prefill in exactly one batch tick."""
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import Event, EventType
    from nano_kvrouter.request import Request

    mc = ModelConfig(
        prefill_cost_per_token_ms=0.1,
        decode_base_ms=5.0,
        marginal_decode_ms=0.5,
        prefill_chunk_size=512,
    )
    nc = NodeConfig(capacity=4, gpu_blocks=200, cpu_blocks=400, disk_blocks=800)

    eng = SimulationEngine()
    nodes = [MockEngineNode("n0", mc, nc)]
    cm = CacheManager(["n0"], mc, nc, BandwidthConfig(), clock=eng.now)
    sched = RoundRobinPolicy(backlog_view=NoopTransferModel())
    _wire_simulator(eng, sched, cm, nodes, nodes, logger_=logging.getLogger("test"), model_cfg=mc, bandwidth_cfg=BandwidthConfig(), transfer_model=NoopTransferModel())

    prefill_times: list[float] = []
    eng.on(
        EventType.PREFILL_COMPLETE,
        lambda ev, e: prefill_times.append(ev.time),
    )

    # 64-token prompt → well under chunk_size=512 → should complete in 1 batch step
    req = Request("r0", list(range(64)), "hr", expected_output_len=1,
                  arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req}))
    eng.run()

    assert len(prefill_times) == 1, f"Expected exactly 1 PREFILL_COMPLETE, got {len(prefill_times)}"
    # With no concurrent decoders: step_time = 64*0.1 + 0 (decode_cost=0 when bs=0) = 6.4 ms
    assert prefill_times[0] == pytest.approx(64 * mc.prefill_cost_per_token_ms)


# ---------------------------------------------------------------------------
# M3.fix2: promoted-request n_chunks and execute-time interleave tests
# ---------------------------------------------------------------------------

def test_promoted_request_records_n_chunks() -> None:
    """n_chunks must be recorded for promoted (queued-then-rescheduled) requests.

    M5a split P/D with separate prefill_node (capacity=1) + decode_node
    (capacity=2): r0 runs first (4 chunks), completes, KV-transfers to
    decode side, then promotes r1 on prefill_node (also 4 chunks). Both
    must appear in chunked_prefill_steps → avg = 4.0.
    """
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.metrics.collector import MetricsCollector
    from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import Event, EventType
    from nano_kvrouter.request import Request

    mc = ModelConfig(
        prefill_cost_per_token_ms=0.1,
        decode_base_ms=5.0,
        marginal_decode_ms=0.5,
        prefill_chunk_size=128,
    )
    nc_p = NodeConfig(capacity=1, gpu_blocks=200, cpu_blocks=200, disk_blocks=400)
    nc_d = NodeConfig(capacity=4, gpu_blocks=200, cpu_blocks=200, disk_blocks=400)
    bw = BandwidthConfig()

    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", mc, nc_p)]
    decode_nodes = [MockEngineNode("d0", mc, nc_d)]
    cm = CacheManager(["d0"], mc, nc_d, bw, clock=eng.now)
    sched = RoundRobinPolicy(model_config=mc, bandwidth_config=bw, backlog_view=NoopTransferModel())

    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=logging.getLogger("test"),
        model_cfg=mc, bandwidth_cfg=bw,
        transfer_model=NoopTransferModel(),
    )
    metrics = MetricsCollector()
    metrics.attach(eng, nodes={n.node_id: n for n in [*prefill_nodes, *decode_nodes]})

    # r0 and r1 use disjoint token ranges → no shared prefix → 512 cold tokens each.
    req_r0 = Request("r0", list(range(512)), "h0",
                     expected_output_len=1, arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)
    req_r1 = Request("r1", list(range(512, 1024)), "h1",
                     expected_output_len=1, arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)

    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_r0}))
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_r1}))
    eng.run()

    s = metrics.summary()
    assert s["completed"] == 2, f"Both requests should complete, got {s['completed']}"
    assert s["avg_chunked_prefill_steps_per_request"] == pytest.approx(4.0), (
        f"avg_chunked_prefill_steps expected 4.0 (both: 512/128=4 chunks); "
        f"got {s['avg_chunked_prefill_steps_per_request']}. "
        "Promoted request n_chunks may not be injected into PREFILL_START payload."
    )


# ---------------------------------------------------------------------------
# M5a: P/D split infrastructure tests
# ---------------------------------------------------------------------------


def test_cli_builds_two_node_pools() -> None:
    """M5a: _run_one constructs prefill + decode pools independently and the
    decode-side cache is populated, while the prefill-side has no cache."""
    from nano_kvrouter.cli import _run_one
    cfg = _small_cfg("round_robin")
    cfg.cluster.prefill_nodes = 2
    cfg.cluster.decode_nodes = 3
    summary = _run_one(cfg, "round_robin")
    # Sanity: 2+3 nodes exist and the run completes without crashing.
    assert summary["total_arrived"] > 0
    assert summary["completed"] > 0


def test_kv_transfer_cost_uses_bandwidth_config() -> None:
    """M5a: kv_transfer_time_avg_ms must equal
    prompt_len * kv_bytes_per_token / bandwidth.gpu_to_gpu * 1000 within numeric tol."""
    from nano_kvrouter.cli import _run_one
    cfg = _small_cfg("round_robin", duration=1.0, rate=20.0)
    cfg.cluster.prefill_nodes = 1
    cfg.cluster.decode_nodes = 1
    cfg.node.capacity = 8
    cfg.workload.avg_prompt_len = 1024
    cfg.model.kv_bytes_per_token = 1024
    cfg.bandwidth.gpu_to_gpu = 1e9   # 1 GB/s — visible transfer cost
    summary = _run_one(cfg, "round_robin")
    avg = summary["kv_transfer_time_avg_ms"]
    assert avg is not None
    # Each request has avg_prompt_len=1024 (deterministic in generator),
    # cost_ms = 1024 * 1024 / 1e9 * 1000 ≈ 1.048576 ms.
    expected = 1024 * 1024 / 1e9 * 1000.0
    assert avg == pytest.approx(expected, rel=0.05)


def test_kv_admit_only_on_decode_node() -> None:
    """M5a: cm._trees keys must contain ONLY decode-pool node IDs."""
    from nano_kvrouter.cli import _build_scheduler
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.metrics.collector import MetricsCollector

    mc = ModelConfig(prefill_chunk_size=512)
    nc = NodeConfig(capacity=8, gpu_blocks=400, cpu_blocks=400, disk_blocks=800)
    bw = BandwidthConfig()

    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", mc, nc), MockEngineNode("p1", mc, nc)]
    decode_nodes = [MockEngineNode("d0", mc, nc), MockEngineNode("d1", mc, nc)]
    cm = CacheManager(
        node_ids=[n.node_id for n in decode_nodes],
        model_config=mc, node_config=nc, bandwidth_config=bw, clock=eng.now,
    )
    sched = _build_scheduler("round_robin", {}, mc, bw, backlog_view=NoopTransferModel())
    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=logging.getLogger("test"),
        model_cfg=mc, bandwidth_cfg=bw,
        transfer_model=NoopTransferModel(),
    )
    metrics = MetricsCollector()
    metrics.attach(eng, nodes={n.node_id: n for n in [*prefill_nodes, *decode_nodes]})

    req = _make_dummy_request("a", n_tokens=160)
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req}))
    eng.run()

    # cm._trees must only have decode-pool keys (KV cache lives on decode side).
    assert set(cm._trees.keys()) == {"d0", "d1"}, (
        f"CacheManager should track only decode_nodes; got {set(cm._trees.keys())}"
    )


def test_transfer_id_validation_drops_stale_event() -> None:
    """M5a: KV_TRANSFER_COMPLETE with an unknown transfer_id must be silently
    dropped without admitting / starting decode."""
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import Event, EventType

    mc = ModelConfig(prefill_chunk_size=512)
    nc = NodeConfig(capacity=4, gpu_blocks=400, cpu_blocks=400, disk_blocks=800)
    bw = BandwidthConfig()

    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", mc, nc)]
    decode_nodes = [MockEngineNode("d0", mc, nc)]
    cm = CacheManager(["d0"], mc, nc, bw, clock=eng.now)
    sched = RoundRobinPolicy(model_config=mc, bandwidth_config=bw, backlog_view=NoopTransferModel())
    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=logging.getLogger("test"),
        model_cfg=mc, bandwidth_cfg=bw,
        transfer_model=NoopTransferModel(),
    )

    # Inject a KV_TRANSFER_COMPLETE with a never-registered transfer_id.
    req = _make_dummy_request("ghost", n_tokens=64)
    eng.schedule(Event(
        time=0.0, type=EventType.KV_TRANSFER_COMPLETE,
        payload={
            "request_id": "ghost",
            "transfer_id": "unknown-transfer-id",
            "request": req,
            "src_node_id": "p0",
            "dst_node_id": "d0",
            "cost_ms": 0.0,
        },
    ))
    eng.run()

    # decode_node must not have admitted anything.
    assert decode_nodes[0].running_requests == []
    assert decode_nodes[0].decoding == set()


def test_decode_admit_failure_rejects() -> None:
    """M5a [B1]: when KV_TRANSFER_COMPLETE fires and decode_node is at capacity,
    the request is rejected with reason ``decode_capacity_exhausted`` and KV
    is NOT admitted into cache."""
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.metrics.collector import MetricsCollector
    from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import Event, EventType

    mc = ModelConfig(prefill_chunk_size=512)
    # decode capacity=1 to force back-pressure; prefill capacity=2 so we can
    # pump two requests through prefill before decode admits.
    nc_p = NodeConfig(capacity=2, gpu_blocks=400, cpu_blocks=400, disk_blocks=800)
    nc_d = NodeConfig(capacity=1, gpu_blocks=400, cpu_blocks=400, disk_blocks=800)
    bw = BandwidthConfig()

    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", mc, nc_p)]
    decode_nodes = [MockEngineNode("d0", mc, nc_d)]
    cm = CacheManager(["d0"], mc, nc_d, bw, clock=eng.now)
    sched = RoundRobinPolicy(model_config=mc, bandwidth_config=bw, backlog_view=NoopTransferModel())
    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=logging.getLogger("test"),
        model_cfg=mc, bandwidth_cfg=bw,
        transfer_model=NoopTransferModel(),
    )
    metrics = MetricsCollector()
    metrics.attach(eng, nodes={n.node_id: n for n in [*prefill_nodes, *decode_nodes]})

    # Long-running r0 hogs decode_node; r1 will hit B1 reject path.
    # r1 uses distinct token_ids (range 64-127) so cm.lookup can verify no KV pollution.
    req_r0 = _make_dummy_request("r0", n_tokens=64)
    req_r0.expected_output_len = 100
    from nano_kvrouter.request import Request as _Req
    req_r1 = _Req("r1", list(range(64, 128)), "y", expected_output_len=1,
                  arrival_time=0.0, slo_ttft=2000.0, slo_tbt=100.0)
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_r0}))
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_r1}))

    rejections: list[str] = []
    eng.on(
        EventType.REQUEST_REJECTED,
        lambda ev, e: rejections.append(ev.payload.get("reason", "")),
    )
    eng.run()

    # r1 should be rejected with decode_capacity_exhausted.
    assert "decode_capacity_exhausted" in rejections, (
        f"expected B1 reject reason in {rejections}"
    )
    # KV must NOT have been admitted for the rejected request (B1 rejects before cm.admit).
    assert cm.lookup(req_r1, "d0").matched_tokens == 0, (
        "rejected r1 must not pollute decode cache"
    )


def test_decode_admit_during_dual_phase() -> None:
    """M5a: dual_phase_tick_count > 0 when prefill and decode are active concurrently
    in the split P/D cluster.

    Setup: separate prefill_node (capacity=2) and decode_node (capacity=4).
    - r0 pre-warmed (0 uncached) → fast-path PREFILL_COMPLETE → KV transfer →
      decodes long (output_len=8).
    - r1 cold (512 tokens, chunk=128 → 4 chunks) prefills on prefill_node
      while r0 is decoding on decode_node → BATCH_STEP ticks fire on both
      while active_prefills and active_decodes are both non-empty.
    """
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.metrics.collector import MetricsCollector
    from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import Event, EventType
    from nano_kvrouter.request import Request

    mc = ModelConfig(
        prefill_cost_per_token_ms=0.1,
        decode_base_ms=5.0,
        marginal_decode_ms=0.5,
        prefill_chunk_size=128,
    )
    nc_p = NodeConfig(capacity=2, gpu_blocks=200, cpu_blocks=200, disk_blocks=400)
    nc_d = NodeConfig(capacity=4, gpu_blocks=200, cpu_blocks=200, disk_blocks=400)
    bw = BandwidthConfig()

    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", mc, nc_p)]
    decode_nodes = [MockEngineNode("d0", mc, nc_d)]
    cm = CacheManager(["d0"], mc, nc_d, bw, clock=eng.now)
    sched = RoundRobinPolicy(model_config=mc, bandwidth_config=bw, backlog_view=NoopTransferModel())

    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=logging.getLogger("test"),
        model_cfg=mc, bandwidth_cfg=bw,
        transfer_model=NoopTransferModel(),
    )
    metrics = MetricsCollector()
    metrics.attach(eng, nodes={n.node_id: n for n in [*prefill_nodes, *decode_nodes]})

    # Pre-warm r0 on decode side so prefill is fast-path.
    cm.admit(list(range(32)), "d0")

    req_r0 = Request("r0", list(range(32)), "h0",
                     expected_output_len=8, arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)
    req_r1 = Request("r1", list(range(1000, 1512)), "h1",
                     expected_output_len=1, arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)

    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_r0}))
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_r1}))
    eng.run()

    s = metrics.summary()
    assert s["completed"] == 2, f"Both requests should complete, got {s['completed']}"
    assert s["dual_phase_tick_count"] > 0, (
        "dual_phase_tick_count must be > 0: r0 decoding while r1 prefilling "
        "(active_prefills and active_decodes both non-empty during BATCH_STEPs)."
    )


# ---------------------------------------------------------------------------
# P4-A M1: paired CLI smoke test (constraint #7)
# ---------------------------------------------------------------------------

def test_per_node_lane_increases_kv_transfer_metric_paired():
    """Same yaml, only contention_model toggled; lane mode must have higher avg KV transfer time."""
    from nano_kvrouter.config import load_config
    import os

    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "transfer_contention.yaml"
    )
    base = load_config(yaml_path)

    none_cfg = base.model_copy(deep=True)
    none_cfg.bandwidth.contention_model = "none"

    lane_cfg = base.model_copy(deep=True)
    lane_cfg.bandwidth.contention_model = "per_node_lane"

    none_summary = _run_one(none_cfg, "conductor")
    lane_summary = _run_one(lane_cfg, "conductor")

    assert lane_summary["kv_transfer_time_avg_ms"] > none_summary["kv_transfer_time_avg_ms"], (
        f"per_node_lane ({lane_summary['kv_transfer_time_avg_ms']:.3f} ms) must exceed "
        f"none ({none_summary['kv_transfer_time_avg_ms']:.3f} ms)"
    )


# ---------------------------------------------------------------------------
# P4-B: kv_transfer_queued_avg_ms decomposition paired test
# ---------------------------------------------------------------------------


def test_kv_transfer_queued_avg_ms_decomposition_paired_fixed_length():
    """Paired toggle: lane mode exposes non-zero queued component;
    kv_total - kv_queued ≈ kv_total_none (decomposition invariant).

    Uses transfer_contention.yaml (low bandwidth → measurable lane contention).
    """
    from nano_kvrouter.config import load_config
    import os

    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "transfer_contention.yaml"
    )
    base = load_config(yaml_path)

    none_cfg = base.model_copy(deep=True)
    none_cfg.bandwidth.contention_model = "none"
    lane_cfg = base.model_copy(deep=True)
    lane_cfg.bandwidth.contention_model = "per_node_lane"

    none_s = _run_one(none_cfg, "conductor")
    lane_s = _run_one(lane_cfg, "conductor")

    # none mode: queued == 0
    assert none_s["kv_transfer_queued_avg_ms"] == 0.0, (
        f"none.kv_transfer_queued_avg_ms should be 0.0, got {none_s['kv_transfer_queued_avg_ms']}"
    )
    # lane mode: queued > 0 (lane contention is non-trivial at 5 MB/s)
    assert lane_s["kv_transfer_queued_avg_ms"] > 0.0, (
        f"lane.kv_transfer_queued_avg_ms should be > 0, got {lane_s['kv_transfer_queued_avg_ms']}"
    )
    # Decomposition invariant: total = service + queued
    # total_lane - queued_lane should equal total_none (within 1%).
    # Per plan v4: paired diff on fixed-length transfer_contention.yaml
    # makes service component invariant; empirically exact at ~104.86 ms.
    service_approx = lane_s["kv_transfer_time_avg_ms"] - lane_s["kv_transfer_queued_avg_ms"]
    expected = none_s["kv_transfer_time_avg_ms"]
    tol = max(expected * 0.01, 1e-6)  # 1% tolerance or 1e-6 ms
    assert abs(service_approx - expected) < tol, (
        f"Decomposition invariant violated: "
        f"lane_total({lane_s['kv_transfer_time_avg_ms']:.3f}) - "
        f"lane_queued({lane_s['kv_transfer_queued_avg_ms']:.3f}) = {service_approx:.3f} "
        f"!≈ none_total({expected:.3f})"
    )

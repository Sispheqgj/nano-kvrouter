"""Integration tests for Bidaw I/O-aware scheduling — event path and metrics."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nano_kvrouter.cli import (
    SCHEDULER_NAMES,
    _build_scheduler,
    _resolve_bidaw_answer_profile_path,
    _resolve_related_config_path,
    _run_one,
    _wire_simulator,
)
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
    load_config,
)
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.metrics.collector import MetricsCollector
from nano_kvrouter.request import Request
from nano_kvrouter.simulator.bidaw_controller import BidawAdmissionController
from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import Event, EventType
from nano_kvrouter.simulator.generator import RequestGenerator
from nano_kvrouter.simulator.transfer_model import NoopTransferModel

_BIDAW_YAML = str(Path(__file__).parent.parent / "configs" / "bidaw.yaml")
_BIDAW_STRESS_YAML = str(Path(__file__).parent.parent / "configs" / "bidaw-stress.yaml")
_BIDAW_M3_STRESS_YAML = str(
    Path(__file__).parent.parent / "configs" / "bidaw-m3-stress.yaml"
)
_BIDAW_INTERACTIVE_YAML = str(
    Path(__file__).parent.parent / "configs" / "bidaw-interactive.yaml"
)
_DEFAULT_YAML = str(Path(__file__).parent.parent / "configs" / "default.yaml")

_LOG = logging.getLogger("test_bidaw_cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bidaw_cfg() -> NanoKVConfig:
    """Load the canonical bidaw.yaml config."""
    return load_config(_BIDAW_YAML)


def _small_disk_cfg() -> NanoKVConfig:
    """Minimal config that quickly forces disk hits: GPU holds ~2 requests per node."""
    return NanoKVConfig(
        cluster=ClusterConfig(prefill_nodes=2, decode_nodes=4),
        # gpu_blocks=32 @ block_size=16 → 512 tokens per node; avg_prompt=256 fills it in 2 reqs.
        node=NodeConfig(gpu_blocks=32, cpu_blocks=0, disk_blocks=4000, capacity=16),
        model=ModelConfig(block_size=16, kv_bytes_per_token=4096,
                          prefill_cost_per_token_ms=0.04, decode_base_ms=5.0,
                          marginal_decode_ms=0.5, prefill_chunk_size=512),
        bandwidth=BandwidthConfig(gpu_to_gpu=3.0e11, gpu_to_cpu=3.2e10,
                                  cpu_to_disk=5.0e9),
        slo=SLOConfig(ttft_target_ms=600.0, tbt_target_ms=50.0),
        workload=WorkloadConfig(request_rate=20.0, duration_s=10.0,
                                prefix_sharing_ratio=0.95,
                                avg_prompt_len=256, avg_output_len=8),
        scheduler=SchedulerConfig(name="bidaw"),
        generator=GeneratorConfig(num_buckets=4, vocab_size=32000, seed=42),
    )


def _build_sim(cfg: NanoKVConfig):
    """Build engine + nodes + cm + generator from config. Returns (eng, prefill, decode, cm, gen)."""
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
    gen = RequestGenerator(cfg)
    return eng, prefill_nodes, decode_nodes, cm, gen


def test_bidaw_interactive_config_runs_with_answer_eviction_metrics() -> None:
    """Interactive fixture config should replay session history and export M2 metrics."""
    cfg = load_config(_BIDAW_INTERACTIVE_YAML)
    assert cfg.trace is not None
    cfg.trace.path = str(_resolve_related_config_path(_BIDAW_INTERACTIVE_YAML, cfg.trace.path))
    _resolve_bidaw_answer_profile_path(_BIDAW_INTERACTIVE_YAML, cfg)

    summary = _run_one(cfg, "bidaw")

    assert summary["total_arrived"] == 6
    assert "bidaw_answer_eviction_count" in summary
    assert "bidaw_answer_eviction_cpu_hit_rate" in summary


# ---------------------------------------------------------------------------
# test_bidaw_disk_request_waits_for_kv_load_complete
# ---------------------------------------------------------------------------


def test_bidaw_disk_request_waits_for_kv_load_complete() -> None:
    """Event ordering: for every disk-hit request, KV_LOAD_COMPLETE fires
    before (or at the same simulated time as) its PREFILL_START."""
    cfg = _small_disk_cfg()
    eng, prefill_nodes, decode_nodes, cm, gen = _build_sim(cfg)
    transfer_model = NoopTransferModel()
    sched = _build_scheduler("bidaw", {}, cfg.model, cfg.bandwidth, backlog_view=transfer_model)

    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=_LOG, model_cfg=cfg.model, bandwidth_cfg=cfg.bandwidth,
        transfer_model=transfer_model, bidaw_mode=True,
    )

    # Record KV_LOAD_COMPLETE and PREFILL_START times per request_id.
    load_complete_times: dict[str, float] = {}
    prefill_start_times: dict[str, float] = {}

    def record_load_complete(ev: Event, engine: SimulationEngine) -> None:
        req_id = ev.payload.get("request_id")
        if req_id:
            load_complete_times[req_id] = ev.time

    def record_prefill_start(ev: Event, engine: SimulationEngine) -> None:
        req_id = ev.payload.get("request_id")
        if req_id and req_id not in prefill_start_times:
            prefill_start_times[req_id] = ev.time

    eng.on(EventType.KV_LOAD_COMPLETE, record_load_complete)
    eng.on(EventType.PREFILL_START, record_prefill_start)

    metrics = MetricsCollector()
    all_nodes = {n.node_id: n for n in [*prefill_nodes, *decode_nodes]}
    metrics.attach(eng, nodes=all_nodes)
    gen.attach(eng)
    eng.run()

    # Must have at least some disk-hit requests (otherwise the test is vacuous).
    summary = metrics.summary()
    assert summary["bidaw_preparing_promotions"] > 0, (
        "No disk-hit requests found; increase prefix_sharing or reduce gpu_blocks"
    )

    # For every request that went through preparing (KV_LOAD_COMPLETE fired),
    # its PREFILL_START must not precede the load completion.
    violations = []
    for req_id, load_t in load_complete_times.items():
        if req_id in prefill_start_times:
            prefill_t = prefill_start_times[req_id]
            if prefill_t < load_t - 1e-9:
                violations.append(
                    f"req {req_id}: PREFILL_START={prefill_t:.3f}ms < "
                    f"KV_LOAD_COMPLETE={load_t:.3f}ms"
                )
    assert not violations, "Event ordering violated:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# test_bidaw_small_ready_not_blocked_by_large_preparing
# ---------------------------------------------------------------------------


def test_bidaw_small_ready_not_blocked_by_large_preparing() -> None:
    """small.PREFILL_START.time must be strictly before large.KV_LOAD_COMPLETE.time.

    Setup: 1 decode node (forces same-node routing).
    t=0:     large disk-hit request arrives → enters preparing → KV_LOAD_START.
    t=0.001: small no-disk request arrives → ready path → PREFILL_START immediately.
    Assert: small.PREFILL_START.time < large.KV_LOAD_COMPLETE.time.

    Disk bandwidth is intentionally slowed to 10 MB/s so load_service_ms (~50ms)
    is well above the ε=0.001ms gap, making the assertion robust to parameter drift.
    """
    # Slow disk bandwidth → load_service_ms >> ε.
    model_cfg = ModelConfig(block_size=16, kv_bytes_per_token=4096,
                            prefill_cost_per_token_ms=0.04, decode_base_ms=5.0,
                            marginal_decode_ms=0.5, prefill_chunk_size=512)
    bw_cfg = BandwidthConfig(gpu_to_gpu=3.0e11, gpu_to_cpu=3.2e10, cpu_to_disk=1e7)
    node_cfg = NodeConfig(gpu_blocks=32, cpu_blocks=0, disk_blocks=4000, capacity=16)

    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", model_cfg, node_cfg)]
    decode_nodes = [MockEngineNode("d0", model_cfg, node_cfg)]
    cm = CacheManager(["d0"], model_cfg, node_cfg, bw_cfg, clock=eng.now)

    # Seed large prefix (128 tokens = 8 blocks) as disk-tier blocks.
    # test-fixture-only: materialize to GPU, then manually move to disk.
    LARGE_TOKENS = list(range(128))      # 8 blocks @ block_size=16
    SMALL_TOKENS = list(range(50000, 50016))  # 1 block, unique — no cache hit

    cm.materialize_request("_seed", LARGE_TOKENS, "d0")
    cm.release_request("_seed", "d0")
    pool = cm._pools["d0"]
    tree = cm._trees["d0"]
    _, seed_bids = tree.match_prefix_path(LARGE_TOKENS)
    for bid in seed_bids:
        if pool.tier_of(bid) == "gpu":
            pool.move(bid, "gpu", "disk")

    # Confirm seeding.
    large_req_seed_check = Request(
        request_id="_check",
        token_ids=LARGE_TOKENS,
        prefix_hash="large000",
        expected_output_len=4,
        arrival_time=0.0,
        slo_ttft=9999.0,
        slo_tbt=9999.0,
    )
    seeded_lookup = cm.lookup(large_req_seed_check, "d0")
    assert seeded_lookup.matched_blocks_by_tier.get("disk", 0) > 0, (
        "Fixture setup failed: LARGE_TOKENS prefix not on disk tier"
    )

    # Build requests for the actual simulation.
    large_req = Request(
        request_id="large-req",
        token_ids=LARGE_TOKENS,
        prefix_hash="large000",
        expected_output_len=4,
        arrival_time=0.0,
        slo_ttft=9999.0,
        slo_tbt=9999.0,
    )
    small_req = Request(
        request_id="small-req",
        token_ids=SMALL_TOKENS,
        prefix_hash="small000",
        expected_output_len=4,
        arrival_time=0.001,
        slo_ttft=9999.0,
        slo_tbt=9999.0,
    )

    transfer_model = NoopTransferModel()
    sched = _build_scheduler("bidaw", {}, model_cfg, bw_cfg, backlog_view=transfer_model)
    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=_LOG, model_cfg=model_cfg, bandwidth_cfg=bw_cfg,
        transfer_model=transfer_model, bidaw_mode=True,
    )

    # Track event timestamps.
    prefill_start_times: dict[str, float] = {}
    kv_load_complete_times: dict[str, float] = {}

    def record_prefill_start(ev: Event, engine: SimulationEngine) -> None:
        req_id = ev.payload.get("request_id")
        if req_id and req_id not in prefill_start_times:
            prefill_start_times[req_id] = ev.time

    def record_kv_load_complete(ev: Event, engine: SimulationEngine) -> None:
        req_id = ev.payload.get("request_id")
        if req_id:
            kv_load_complete_times[req_id] = ev.time

    eng.on(EventType.PREFILL_START, record_prefill_start)
    eng.on(EventType.KV_LOAD_COMPLETE, record_kv_load_complete)

    # Inject both requests at controlled times (large first, then small at ε=0.001ms).
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": large_req}))
    eng.schedule(Event(time=0.001, type=EventType.REQUEST_ARRIVE, payload={"request": small_req}))

    eng.run()

    large_kv_complete = kv_load_complete_times.get("large-req")
    small_prefill_start = prefill_start_times.get("small-req")

    assert large_kv_complete is not None, "large request never completed KV load"
    assert small_prefill_start is not None, "small request never started prefill"
    assert small_prefill_start < large_kv_complete, (
        f"HoL violation: small.PREFILL_START={small_prefill_start:.6f}ms "
        f">= large.KV_LOAD_COMPLETE={large_kv_complete:.6f}ms"
    )


# ---------------------------------------------------------------------------
# test_bidaw_metrics_populated_on_bidaw_yaml
# ---------------------------------------------------------------------------


def test_bidaw_metrics_populated_on_bidaw_yaml() -> None:
    """Running bidaw on bidaw.yaml populates all Bidaw-specific metric fields."""
    cfg = _bidaw_cfg()
    summary = _run_one(cfg, "bidaw")

    # Disk loads fired — these two must always be positive on bidaw.yaml.
    assert summary["bidaw_preparing_promotions"] > 0, (
        "bidaw_preparing_promotions should be > 0 on bidaw.yaml"
    )
    assert summary["bidaw_disk_load_service_avg_ms"] > 0.0, (
        "bidaw_disk_load_service_avg_ms should be > 0 on bidaw.yaml"
    )
    # Preparing queue wait can legitimately be 0 when disk loads complete faster
    # than the inter-arrival gap (0.37ms load vs 67ms gap at 15 req/s).
    assert summary["bidaw_preparing_wait_avg_ms"] >= 0.0
    assert summary["bidaw_preparing_wait_p99_ms"] >= 0.0

    # Normal metrics must still be present and sane.
    assert summary["total_arrived"] > 0
    assert summary["completed"] > 0
    assert summary["ttft_p50_ms"] is not None


def test_bidaw_kv_load_complete_physically_promotes_disk_blocks_to_cpu() -> None:
    """KV_LOAD_COMPLETE should mutate matched disk blocks into CPU-tier blocks."""
    model_cfg = ModelConfig(block_size=16, kv_bytes_per_token=4096,
                            prefill_cost_per_token_ms=0.04, decode_base_ms=5.0,
                            marginal_decode_ms=0.5, prefill_chunk_size=512)
    bw_cfg = BandwidthConfig(gpu_to_gpu=3.0e11, gpu_to_cpu=3.2e10,
                             cpu_to_disk=1e9)
    node_cfg = NodeConfig(gpu_blocks=32, cpu_blocks=16, disk_blocks=4000, capacity=16)

    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", model_cfg, node_cfg)]
    decode_nodes = [MockEngineNode("d0", model_cfg, node_cfg)]
    cm = CacheManager(["d0"], model_cfg, node_cfg, bw_cfg, clock=eng.now)

    tokens = list(range(64))  # 4 blocks
    cm.materialize_request("_seed", tokens, "d0")
    cm.release_request("_seed", "d0")
    pool = cm._pools["d0"]
    _, seed_bids = cm._trees["d0"].match_prefix_path(tokens)
    for bid in seed_bids:
        if pool.tier_of(bid) == "gpu":
            pool.move(bid, "gpu", "disk")

    req = Request(
        request_id="disk-req",
        token_ids=tokens,
        prefix_hash="disk0000",
        expected_output_len=2,
        arrival_time=0.0,
        slo_ttft=9999.0,
        slo_tbt=9999.0,
    )
    assert cm.lookup(req, "d0").matched_blocks_by_tier == {"disk": 4}

    transfer_model = NoopTransferModel()
    sched = _build_scheduler("bidaw", {}, model_cfg, bw_cfg, backlog_view=transfer_model)
    _wire_simulator(
        eng, sched, cm, prefill_nodes, decode_nodes,
        logger_=_LOG, model_cfg=model_cfg, bandwidth_cfg=bw_cfg,
        transfer_model=transfer_model, bidaw_mode=True,
    )

    complete_payloads: list[dict] = []

    def record_kv_load_complete(ev: Event, engine: SimulationEngine) -> None:
        if ev.payload.get("request_id") == "disk-req":
            complete_payloads.append(dict(ev.payload))

    eng.on(EventType.KV_LOAD_COMPLETE, record_kv_load_complete)
    metrics = MetricsCollector()
    metrics.attach(eng, nodes={n.node_id: n for n in [*prefill_nodes, *decode_nodes]})
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req}))
    eng.run()

    assert complete_payloads
    assert complete_payloads[0]["promoted_count"] == 4
    assert complete_payloads[0]["skipped_count"] == 0
    assert cm.lookup(req, "d0").matched_blocks_by_tier == {"cpu": 4}
    summary = metrics.summary()
    assert summary["bidaw_physical_promoted_blocks"] == 4
    assert summary["bidaw_physical_skipped_blocks"] == 0


def test_bidaw_stress_yaml_exercises_preparing_queue_wait() -> None:
    """The stress config should make disk-HRRN observable beyond unit tests."""
    cfg = load_config(_BIDAW_STRESS_YAML)
    summary = _run_one(cfg, "bidaw")

    assert summary["bidaw_preparing_promotions"] > 0
    assert summary["bidaw_disk_load_service_avg_ms"] > 0.0
    assert summary["bidaw_preparing_wait_avg_ms"] > 0.0


# ---------------------------------------------------------------------------
# test_bidaw_no_disk_workload_behaves_like_ready_only
# ---------------------------------------------------------------------------


def test_bidaw_no_disk_workload_behaves_like_ready_only() -> None:
    """Bidaw on a zero-disk-hit workload (default-style config) degrades to the
    ready-only path: no KV_LOAD events fire, Bidaw metrics stay at zero."""
    # Small config: large GPU means no blocks are evicted to disk.
    cfg = NanoKVConfig(
        cluster=ClusterConfig(prefill_nodes=2, decode_nodes=2),
        node=NodeConfig(gpu_blocks=2000, cpu_blocks=0, disk_blocks=0, capacity=16),
        model=ModelConfig(block_size=16),
        bandwidth=BandwidthConfig(),
        slo=SLOConfig(),
        workload=WorkloadConfig(
            request_rate=10.0,
            duration_s=1.0,
            avg_prompt_len=64,
            avg_output_len=4,
            prefix_sharing_ratio=0.3,
        ),
        scheduler=SchedulerConfig(name="bidaw"),
        generator=GeneratorConfig(num_buckets=3, seed=42),
    )
    # Must not crash.
    summary = _run_one(cfg, "bidaw")
    assert summary["total_arrived"] > 0
    assert summary["completed"] > 0
    # No disk hits → Bidaw metrics stay at zero.
    assert summary["bidaw_preparing_promotions"] == 0
    assert summary["bidaw_preparing_wait_avg_ms"] == 0.0
    assert summary["bidaw_disk_load_service_avg_ms"] == 0.0


# ---------------------------------------------------------------------------
# test_bidaw_appears_in_sweep_table
# ---------------------------------------------------------------------------


def test_bidaw_appears_in_sweep_table() -> None:
    """'bidaw' appears in SCHEDULER_NAMES and produces a valid result in a sweep."""
    assert "bidaw" in SCHEDULER_NAMES

    cfg = _bidaw_cfg()
    # Run all schedulers (as cmd_sweep does) and confirm bidaw is present.
    results: dict[str, dict] = {}
    for name in SCHEDULER_NAMES:
        results[name] = _run_one(cfg, name)

    assert "bidaw" in results
    assert results["bidaw"]["total_arrived"] > 0
    assert results["bidaw"]["completed"] > 0

    # All 6 schedulers must complete without internal errors.
    for name in SCHEDULER_NAMES:
        assert results[name]["total_arrived"] > 0, f"{name} produced no arrivals"


# ---------------------------------------------------------------------------
# test_bidaw_kv_load_stale_guard (mirrors KV_TRANSFER stale guard pattern)
# ---------------------------------------------------------------------------


def test_bidaw_kv_load_stale_guard_drops_unknown_complete() -> None:
    """MetricsCollector silently drops KV_LOAD_COMPLETE for unknown request_ids,
    mirroring the KV_TRANSFER_COMPLETE stale guard (collector.py:255-275)."""
    from nano_kvrouter.simulator.engine import SimulationEngine

    eng = SimulationEngine()
    collector = MetricsCollector()
    collector.attach(eng)

    # Fire KV_LOAD_COMPLETE without a preceding KV_LOAD_START for this request.
    eng.schedule(Event(
        time=1.0,
        type=EventType.KV_LOAD_COMPLETE,
        payload={
            "request_id": "stale-req",
            "decode_node_id": "d0",
            "promoted_count": 0,
            "skipped_count": 0,
        },
    ))
    eng.run()

    # Stale event must be silently dropped — promotions count stays at 0.
    assert collector.summary()["bidaw_preparing_promotions"] == 0


def test_bidaw_m3_all_flags_stress_config_runs() -> None:
    """All three M3 flags on should run deterministically without crashing."""
    cfg = load_config(_BIDAW_M3_STRESS_YAML)
    summary = _run_one(cfg, "bidaw")

    assert summary["total_arrived"] > 0
    assert "bidaw_routing_score_avg" in summary
    assert "bidaw_session_affinity_hits" in summary
    assert "ttft_slo_rejections" in summary


def test_bidaw_capacity_reject_does_not_commit_affinity_in_cli_wiring() -> None:
    """Decode capacity rejection must happen before A3 affinity commit."""
    cfg = NanoKVConfig(
        cluster=ClusterConfig(prefill_nodes=1, decode_nodes=1),
        node=NodeConfig(capacity=1, gpu_blocks=100, cpu_blocks=10, disk_blocks=100),
        model=ModelConfig(block_size=16, kv_bytes_per_token=4096, prefill_chunk_size=512),
        bandwidth=BandwidthConfig(),
        slo=SLOConfig(ttft_target_ms=10_000.0, tbt_target_ms=10_000.0),
        workload=WorkloadConfig(
            request_rate=1.0,
            duration_s=1.0,
            avg_prompt_len=32,
            avg_output_len=2,
        ),
        scheduler=SchedulerConfig(name="bidaw", params={"enable_session_affinity": True}),
        generator=GeneratorConfig(seed=42),
    )
    eng, prefill_nodes, decode_nodes, cm, _gen = _build_sim(cfg)
    controller = BidawAdmissionController(
        [n.node_id for n in decode_nodes],
        model_config=cfg.model,
        bandwidth_config=cfg.bandwidth,
        affinity_enabled=True,
    )
    transfer_model = NoopTransferModel()
    sched = _build_scheduler(
        "bidaw",
        cfg.scheduler.params,
        cfg.model,
        cfg.bandwidth,
        backlog_view=transfer_model,
        bidaw_controller=controller,
    )
    _wire_simulator(
        eng,
        sched,
        cm,
        prefill_nodes,
        decode_nodes,
        logger_=_LOG,
        model_cfg=cfg.model,
        bandwidth_cfg=cfg.bandwidth,
        transfer_model=transfer_model,
        bidaw_mode=True,
        bidaw_controller=controller,
    )
    decode_nodes[0].admit("blocker", expected_output_len=4, prompt_len=32, uncached_tokens=0)
    req = Request(
        request_id="capacity-reject",
        token_ids=list(range(32)),
        prefix_hash="cap",
        expected_output_len=2,
        arrival_time=0.0,
        slo_ttft=10_000.0,
        slo_tbt=10_000.0,
        session_id="s-capacity",
    )

    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req}))
    eng.run()

    assert controller.peek_session_affinity("s-capacity") is None

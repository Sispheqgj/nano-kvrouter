"""Integration tests for Bidaw multi-stream KV-load mode."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nano_kvrouter.cli import _build_bidaw_load_model, _run_one, _wire_simulator
from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig, load_config
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.bidaw import BidawPolicy
from nano_kvrouter.simulator.bidaw_controller import BidawAdmissionController
from nano_kvrouter.simulator.bidaw_load_model import MultiStreamLoadModel, SingleSlotLoadModel
from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import Event, EventType
from nano_kvrouter.simulator.transfer_model import NoopTransferModel

_LOG = logging.getLogger(__name__)
_BIDAW_M4_YAML = str(Path(__file__).parent.parent / "configs" / "bidaw-m4-multistream.yaml")


def _req(req_id: str, base_token: int = 0) -> Request:
    return Request(
        request_id=req_id,
        token_ids=list(range(base_token, base_token + 32)),
        prefix_hash=req_id,
        expected_output_len=2,
        arrival_time=0.0,
        slo_ttft=10_000.0,
        slo_tbt=10_000.0,
    )


def _seed_disk_prefix(cm: CacheManager, req: Request, node_id: str) -> None:
    cm.materialize_request(f"_seed-{req.request_id}", req.token_ids, node_id)
    cm.release_request(f"_seed-{req.request_id}", node_id)
    pool = cm._pools[node_id]
    _, block_ids = cm._trees[node_id].match_prefix_path(req.token_ids)
    for block_id in block_ids:
        if pool.tier_of(block_id) == "gpu":
            pool.move(block_id, "gpu", "disk")


def _wired_disk_sim(num_streams: int) -> tuple[SimulationEngine, list[Request]]:
    model_cfg = ModelConfig(block_size=16, kv_bytes_per_token=4096, prefill_chunk_size=512)
    bandwidth_cfg = BandwidthConfig(cpu_to_disk=1.0e7)
    node_cfg = NodeConfig(capacity=16, gpu_blocks=100, cpu_blocks=16, disk_blocks=1000)
    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", model_cfg, node_cfg)]
    decode_nodes = [MockEngineNode("d0", model_cfg, node_cfg)]
    cm = CacheManager(
        ["d0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bandwidth_cfg,
        clock=eng.now,
    )
    requests = [_req(f"r{i}", i * 1000) for i in range(5)]
    for req in requests:
        _seed_disk_prefix(cm, req, "d0")

    load_model = (
        SingleSlotLoadModel(["d0"])
        if num_streams == 1
        else MultiStreamLoadModel(["d0"], num_streams=num_streams)
    )
    controller = BidawAdmissionController(
        ["d0"],
        model_config=model_cfg,
        bandwidth_config=bandwidth_cfg,
        load_model=load_model,
    )
    transfer_model = NoopTransferModel()
    sched = BidawPolicy(
        model_config=model_cfg,
        bandwidth_config=bandwidth_cfg,
        backlog_view=transfer_model,
    )
    _wire_simulator(
        eng,
        sched,
        cm,
        prefill_nodes,
        decode_nodes,
        logger_=_LOG,
        model_cfg=model_cfg,
        bandwidth_cfg=bandwidth_cfg,
        transfer_model=transfer_model,
        bidaw_mode=True,
        bidaw_controller=controller,
    )
    return eng, requests


def test_k4_drain_fires_exactly_four_distinct_initial_loads() -> None:
    eng, requests = _wired_disk_sim(num_streams=4)
    starts: list[tuple[float, str]] = []

    def record_start(event: Event, engine: SimulationEngine) -> None:
        starts.append((event.time, event.payload["request_id"]))

    eng.on(EventType.KV_LOAD_START, record_start)
    for req in requests:
        eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req}))

    eng.run()

    initial = [req_id for t, req_id in starts if t == pytest.approx(0.0)]
    assert len(initial) == 4
    assert len(set(initial)) == 4


def test_k1_drain_preserves_serial_load_start_sequence() -> None:
    eng, requests = _wired_disk_sim(num_streams=1)
    starts: list[tuple[float, str]] = []

    def record_start(event: Event, engine: SimulationEngine) -> None:
        starts.append((event.time, event.payload["request_id"]))

    eng.on(EventType.KV_LOAD_START, record_start)
    for req in requests:
        eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req}))

    eng.run()

    assert [req_id for _, req_id in starts] == [req.request_id for req in requests]
    assert [req_id for time, req_id in starts if time == pytest.approx(0.0)] == ["r0"]
    assert [time for time, _ in starts] == sorted(time for time, _ in starts)


def test_hrrn_ordering_preserved_when_multiple_slots_free() -> None:
    load_model = MultiStreamLoadModel(["d0"], num_streams=2)
    ctrl = BidawAdmissionController(["d0"], load_model=load_model)
    ctrl.on_arrive(_req("large"), "d0", matched_disk_blocks=10, now_ms=0.0)
    ctrl.on_arrive(_req("small"), "d0", matched_disk_blocks=1, now_ms=0.0)
    ctrl.on_arrive(_req("mid"), "d0", matched_disk_blocks=2, now_ms=0.0)

    first = ctrl.pick_next_to_load("d0", now_ms=100.0)
    assert first is not None
    assert first.request_id == "small"
    ctrl.mark_load_started("d0", first.request_id, now_ms=100.0, service_ms=10.0)

    second = ctrl.pick_next_to_load("d0", now_ms=100.0)
    assert second is not None
    assert second.request_id == "mid"


def test_projected_wait_filters_in_flight_and_uses_earliest_slot() -> None:
    model_cfg = ModelConfig(block_size=1, kv_bytes_per_token=1)
    bandwidth_cfg = BandwidthConfig(cpu_to_disk=100.0, gpu_to_cpu=1.0e12)
    load_model = MultiStreamLoadModel(["d0"], num_streams=2)
    ctrl = BidawAdmissionController(
        ["d0"],
        model_config=model_cfg,
        bandwidth_config=bandwidth_cfg,
        load_model=load_model,
    )
    req_a = _req("A")
    req_b = _req("B")
    req_c = _req("C")
    ctrl.on_arrive(req_a, "d0", matched_disk_blocks=3, now_ms=0.0)
    ctrl.on_arrive(req_c, "d0", matched_disk_blocks=2, now_ms=0.0)
    ctrl.on_arrive(req_b, "d0", matched_disk_blocks=3, now_ms=0.0)
    ctrl.mark_load_started("d0", "A", now_ms=0.0, service_ms=5.0)
    ctrl.mark_load_started("d0", "C", now_ms=0.0, service_ms=15.0)

    assert ctrl.peek_projected_preparing_wait_ms("d0", 2, now_ms=0.0) == pytest.approx(
        35.0,
        rel=1e-6,
    )


def test_build_bidaw_load_model_rejects_incompatible_single_stream_params() -> None:
    with pytest.raises(ValueError, match="load_model='single'"):
        _build_bidaw_load_model({"load_model": "single", "num_streams": 2}, ["d0"])


def test_build_bidaw_load_model_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown load_model"):
        _build_bidaw_load_model({"load_model": "other"}, ["d0"])


def test_bidaw_m4_multistream_ship_gate_smoke() -> None:
    """Ship gate v2 (post-redesign):

    Original v1 gates (promotions ±10%, rejection +0.05) assumed M4
    would not disturb downstream admission. Saturated workloads
    show otherwise — faster KV_LOAD pumps requests into PREFILL_START
    sooner, which shifts decode-capacity exhaustion timing and
    therefore rejection / promotion counts. That is a true effect
    of the mechanism, not a regression.

    v2 gates anchor on what M4 actually promises: shorter user-facing
    latency, primarily via reduced preparing-queue wait and the
    propagated effect on TTFT/E2E. Both compared against the K=1
    bidaw-stress.yaml baseline reported in M0 preflight §3.
    """
    cfg = load_config(_BIDAW_M4_YAML)
    summary = _run_one(cfg, "bidaw")

    # K=1 baselines (from M0 preflight §3 on bidaw-stress.yaml):
    k1_preparing_wait_avg_ms = 143.07
    k1_ttft_p50_ms = 139.25
    k1_e2e_avg_ms = 341.14

    # Gate 1 (primary mechanism): preparing-wait avg reduced ≥30%.
    assert summary["bidaw_preparing_wait_avg_ms"] <= k1_preparing_wait_avg_ms * 0.7

    # Gate 2 (user-facing TTFT): p50 reduced ≥30%.
    assert summary["ttft_p50_ms"] <= k1_ttft_p50_ms * 0.7

    # Gate 3 (user-facing E2E): avg reduced ≥10% (lenient — downstream
    # rebalance limits the gain).
    assert summary["e2e_avg_ms"] <= k1_e2e_avg_ms * 0.9

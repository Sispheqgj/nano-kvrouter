"""A2 tests for Bidaw TTFT SLO gating and projected preparing wait."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheLookup
from nano_kvrouter.scheduler.bidaw import BidawPolicy
from nano_kvrouter.simulator.bidaw_controller import BidawAdmissionController
from nano_kvrouter.simulator.bidaw_load_model import MultiStreamLoadModel
from nano_kvrouter.simulator.transfer_model import NoopTransferModel


@dataclass
class _Cache:
    lookup_result: CacheLookup

    def lookup(self, request: Request, node_id: str) -> CacheLookup:
        return self.lookup_result

    def lookup_all(self, request: Request) -> dict[str, CacheLookup]:
        return {node_id: self.lookup_result for node_id in ["d0"]}

    def free_blocks(self, node_id: str, tier: str) -> int:
        return 1024


@dataclass
class _ProjectedWaitView:
    projected_ms: float
    seen_args: tuple[str, int, float] | None = None

    def peek_preparing_disk_blocks(self, decode_node_id: str) -> int:
        return 0

    def peek_in_flight_disk_blocks(self, decode_node_id: str) -> int:
        return 0

    def peek_projected_preparing_wait_ms(
        self,
        decode_node_id: str,
        my_disk_blocks: int,
        now_ms: float,
    ) -> float:
        self.seen_args = (decode_node_id, my_disk_blocks, now_ms)
        return self.projected_ms

    def peek_session_affinity(self, session_id: str) -> str | None:
        return None


def _req(slo_ttft: float) -> Request:
    return Request(
        request_id="r0",
        token_ids=list(range(32)),
        prefix_hash="p",
        expected_output_len=4,
        arrival_time=0.0,
        slo_ttft=slo_ttft,
        slo_tbt=10_000.0,
    )


def _nodes() -> tuple[list[MockEngineNode], list[MockEngineNode]]:
    model = ModelConfig(block_size=16, kv_bytes_per_token=4096)
    node_cfg = NodeConfig(capacity=16)
    return [MockEngineNode("p0", model, node_cfg)], [MockEngineNode("d0", model, node_cfg)]


def _policy(view: _ProjectedWaitView) -> BidawPolicy:
    model = ModelConfig(block_size=16, kv_bytes_per_token=4096)
    bandwidth = BandwidthConfig(cpu_to_disk=1.0e7)
    return BidawPolicy(
        model_config=model,
        bandwidth_config=bandwidth,
        backlog_view=NoopTransferModel(),
        controller_view=view,
        enable_ttft_slo_gate=True,
    )


def test_projected_wait_short_circuits_when_no_disk_blocks() -> None:
    model = ModelConfig(block_size=16, kv_bytes_per_token=4096)
    bandwidth = BandwidthConfig(cpu_to_disk=1.0e7)
    ctrl = BidawAdmissionController(
        ["d0"],
        model_config=model,
        bandwidth_config=bandwidth,
    )
    ctrl.on_arrive(_req(10_000.0), "d0", matched_disk_blocks=3, now_ms=0.0)
    picked = ctrl.pick_next_to_load("d0", now_ms=0.0)
    assert picked is not None
    ctrl.mark_load_started("d0", picked.request_id, now_ms=0.0, service_ms=30.0)

    assert ctrl.peek_projected_preparing_wait_ms("d0", 0, now_ms=5.0) == 0.0


def test_projected_wait_includes_in_flight_residual_and_queued_blocks() -> None:
    model = ModelConfig(block_size=16, kv_bytes_per_token=4096)
    bandwidth = BandwidthConfig(cpu_to_disk=1.0e7)
    ctrl = BidawAdmissionController(
        ["d0"],
        model_config=model,
        bandwidth_config=bandwidth,
    )
    r0 = _req(10_000.0)
    r1 = Request(
        request_id="r1",
        token_ids=list(range(32, 64)),
        prefix_hash="q",
        expected_output_len=4,
        arrival_time=0.0,
        slo_ttft=10_000.0,
        slo_tbt=10_000.0,
    )
    ctrl.on_arrive(r0, "d0", matched_disk_blocks=4, now_ms=0.0)
    picked = ctrl.pick_next_to_load("d0", now_ms=0.0)
    assert picked is not None
    ctrl.mark_load_started("d0", picked.request_id, now_ms=0.0, service_ms=30.0)
    ctrl.on_arrive(r1, "d0", matched_disk_blocks=3, now_ms=1.0)

    block_bytes = model.block_size * model.kv_bytes_per_token
    per_block_ms = (
        block_bytes
        * (1.0 / bandwidth.cpu_to_disk + 1.0 / bandwidth.gpu_to_cpu)
        * 1000.0
    )
    expected = 25.0 + (3 + 2) * per_block_ms

    assert ctrl.peek_projected_preparing_wait_ms("d0", 2, now_ms=5.0) == pytest.approx(
        expected
    )


def test_projected_wait_k2_filters_in_flight_and_uses_earliest_slot() -> None:
    model = ModelConfig(block_size=1, kv_bytes_per_token=1)
    bandwidth = BandwidthConfig(cpu_to_disk=100.0, gpu_to_cpu=1.0e12)
    ctrl = BidawAdmissionController(
        ["d0"],
        model_config=model,
        bandwidth_config=bandwidth,
        load_model=MultiStreamLoadModel(["d0"], num_streams=2),
    )
    req_a = Request(
        request_id="A",
        token_ids=[1],
        prefix_hash="a",
        expected_output_len=1,
        arrival_time=0.0,
        slo_ttft=10_000.0,
        slo_tbt=10_000.0,
    )
    req_c = Request(
        request_id="C",
        token_ids=[2],
        prefix_hash="c",
        expected_output_len=1,
        arrival_time=0.0,
        slo_ttft=10_000.0,
        slo_tbt=10_000.0,
    )
    req_b = Request(
        request_id="B",
        token_ids=[3],
        prefix_hash="b",
        expected_output_len=1,
        arrival_time=0.0,
        slo_ttft=10_000.0,
        slo_tbt=10_000.0,
    )
    ctrl.on_arrive(req_a, "d0", matched_disk_blocks=3, now_ms=0.0)
    ctrl.on_arrive(req_c, "d0", matched_disk_blocks=2, now_ms=0.0)
    ctrl.on_arrive(req_b, "d0", matched_disk_blocks=3, now_ms=0.0)
    ctrl.mark_load_started("d0", "A", now_ms=0.0, service_ms=5.0)
    ctrl.mark_load_started("d0", "C", now_ms=0.0, service_ms=15.0)

    assert ctrl.peek_projected_preparing_wait_ms("d0", 2, now_ms=0.0) == pytest.approx(
        35.0,
        rel=1e-6,
    )


def test_ttft_slo_gate_rejects_when_projected_wait_exceeds_slo() -> None:
    view = _ProjectedWaitView(projected_ms=100.0)
    prefill, decode = _nodes()
    decision = _policy(view).schedule(
        _req(slo_ttft=50.0),
        prefill,
        decode,
        _Cache(CacheLookup(32, {"disk": 2}, 0.0)),
        now=7.0,
    )

    assert decision.is_rejected
    assert decision.reject_reason == "ttft_slo_exceeded"
    assert view.seen_args == ("d0", 2, 7.0)


def test_ttft_slo_gate_admits_when_projected_wait_fits_slo() -> None:
    view = _ProjectedWaitView(projected_ms=10.0)
    prefill, decode = _nodes()
    decision = _policy(view).schedule(
        _req(slo_ttft=500.0),
        prefill,
        decode,
        _Cache(CacheLookup(32, {"disk": 2}, 0.0)),
        now=0.0,
    )

    assert not decision.is_rejected
    assert decision.decode_node == "d0"

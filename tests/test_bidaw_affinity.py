"""A3 tests for Bidaw session affinity and commit ordering."""
from __future__ import annotations

from collections.abc import Callable

from nano_kvrouter.cli import _wire_simulator
from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler._testing import NullCacheQuery
from nano_kvrouter.scheduler.bidaw import BidawPolicy
from nano_kvrouter.simulator.bidaw_controller import BidawAdmissionController
from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import Event, EventType
from nano_kvrouter.simulator.transfer_model import NoopTransferModel


def _req(req_id: str = "r0", session_id: str | None = "s0") -> Request:
    return Request(
        request_id=req_id,
        token_ids=list(range(32)),
        prefix_hash="p",
        expected_output_len=2,
        arrival_time=0.0,
        slo_ttft=10_000.0,
        slo_tbt=10_000.0,
        session_id=session_id,
    )


def _nodes(count: int = 2) -> tuple[list[MockEngineNode], list[MockEngineNode]]:
    model = ModelConfig()
    node_cfg = NodeConfig(capacity=16)
    return [MockEngineNode("p0", model, node_cfg)], [
        MockEngineNode(f"d{i}", model, node_cfg) for i in range(count)
    ]


def _policy(
    controller: BidawAdmissionController,
    *,
    overload_factor: float = 1.5,
    overload_floor: float = 2.0,
) -> BidawPolicy:
    return BidawPolicy(
        model_config=ModelConfig(),
        bandwidth_config=BandwidthConfig(),
        backlog_view=NoopTransferModel(),
        controller_view=controller,
        enable_session_affinity=True,
        affinity_overload_factor=overload_factor,
        affinity_overload_abs_floor=overload_floor,
    )


def test_affinity_pin_routes_to_pinned_decode_node() -> None:
    controller = BidawAdmissionController(["d0", "d1"], affinity_enabled=True)
    controller.commit_session_affinity("s0", "d1")
    prefill, decode = _nodes()

    decision = _policy(controller).schedule(
        _req(session_id="s0"),
        prefill,
        decode,
        NullCacheQuery(["d0", "d1"]),
        now=0.0,
    )

    assert decision.decode_node == "d1"
    assert decision.affinity_hit is True


def test_affinity_falls_back_when_pinned_node_is_overloaded() -> None:
    controller = BidawAdmissionController(["d0", "d1"], affinity_enabled=True)
    controller.commit_session_affinity("s0", "d0")
    prefill, decode = _nodes()
    decode[0].admit("busy", expected_output_len=4, prompt_len=32, uncached_tokens=0)

    decision = _policy(
        controller,
        overload_factor=1.0,
        overload_floor=0.0,
    ).schedule(
        _req(session_id="s0"),
        prefill,
        decode,
        NullCacheQuery(["d0", "d1"]),
        now=0.0,
    )

    assert decision.decode_node == "d1"
    assert decision.affinity_hit is False


def test_affinity_ignores_requests_without_session_id() -> None:
    controller = BidawAdmissionController(["d0", "d1"], affinity_enabled=True)
    controller.commit_session_affinity("s0", "d1")
    prefill, decode = _nodes()

    decision = _policy(controller).schedule(
        _req(session_id=None),
        prefill,
        decode,
        NullCacheQuery(["d0", "d1"]),
        now=0.0,
    )

    assert decision.decode_node == "d0"
    assert decision.affinity_hit is False


def _run_one_request(
    controller: BidawAdmissionController,
    *,
    before_run: Callable[[MockEngineNode, CacheManager], None] | None = None,
) -> None:
    model = ModelConfig(block_size=16, kv_bytes_per_token=4096, prefill_chunk_size=512)
    bandwidth = BandwidthConfig()
    node_cfg = NodeConfig(capacity=1, gpu_blocks=100, cpu_blocks=10, disk_blocks=100)
    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", model, node_cfg)]
    decode_nodes = [MockEngineNode("d0", model, node_cfg)]
    cm = CacheManager(
        ["d0"],
        model_config=model,
        node_config=node_cfg,
        bandwidth_config=bandwidth,
        clock=eng.now,
    )
    if before_run is not None:
        before_run(decode_nodes[0], cm)
    transfer_model = NoopTransferModel()
    sched = BidawPolicy(
        model_config=model,
        bandwidth_config=bandwidth,
        backlog_view=transfer_model,
        controller_view=controller,
        enable_session_affinity=True,
    )
    _wire_simulator(
        eng,
        sched,
        cm,
        prefill_nodes,
        decode_nodes,
        logger_=__import__("logging").getLogger(__name__),
        model_cfg=model,
        bandwidth_cfg=bandwidth,
        transfer_model=transfer_model,
        bidaw_mode=True,
        bidaw_controller=controller,
    )
    eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": _req()}))
    eng.run()


def test_capacity_reject_does_not_commit_affinity() -> None:
    controller = BidawAdmissionController(
        ["d0"],
        model_config=ModelConfig(block_size=16, kv_bytes_per_token=4096),
        bandwidth_config=BandwidthConfig(),
        affinity_enabled=True,
    )

    def fill_decode_capacity(decode_node: MockEngineNode, cm: CacheManager) -> None:
        decode_node.admit("blocker", expected_output_len=4, prompt_len=32, uncached_tokens=0)

    _run_one_request(controller, before_run=fill_decode_capacity)

    assert controller.peek_session_affinity("s0") is None


def test_materialize_failure_does_not_commit_affinity() -> None:
    controller = BidawAdmissionController(
        ["d0"],
        model_config=ModelConfig(block_size=16, kv_bytes_per_token=4096),
        bandwidth_config=BandwidthConfig(),
        affinity_enabled=True,
    )

    def force_materialize_failure(decode_node: MockEngineNode, cm: CacheManager) -> None:
        def fail_materialize(*args: object, **kwargs: object) -> None:
            raise MemoryError("forced")

        cm.materialize_request = fail_materialize  # type: ignore[method-assign]

    _run_one_request(controller, before_run=force_materialize_failure)

    assert controller.peek_session_affinity("s0") is None

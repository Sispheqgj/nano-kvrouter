"""A1 tests for Bidaw routing-aware decode-node selection."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler._testing import NullCacheQuery
from nano_kvrouter.scheduler.base import CacheLookup
from nano_kvrouter.scheduler.bidaw import BidawPolicy
from nano_kvrouter.scheduler.conductor import MooncakeConductor
from nano_kvrouter.scheduler.e2_policy import E2Policy
from nano_kvrouter.scheduler.least_loaded import LeastLoadedPolicy
from nano_kvrouter.scheduler.prefix_greedy import PrefixGreedyPolicy
from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
from nano_kvrouter.simulator.transfer_model import NoopTransferModel


@dataclass
class _Cache:
    lookups: dict[str, CacheLookup]

    def lookup(self, request: Request, node_id: str) -> CacheLookup:
        return self.lookups.get(node_id, CacheLookup(0, {}, 0.0))

    def lookup_all(self, request: Request) -> dict[str, CacheLookup]:
        return dict(self.lookups)

    def free_blocks(self, node_id: str, tier: str) -> int:
        return 1024


@dataclass
class _View:
    preparing: dict[str, int] = field(default_factory=dict)
    in_flight: dict[str, int] = field(default_factory=dict)
    affinity: dict[str, str] = field(default_factory=dict)

    def peek_preparing_disk_blocks(self, decode_node_id: str) -> int:
        return self.preparing.get(decode_node_id, 0)

    def peek_in_flight_disk_blocks(self, decode_node_id: str) -> int:
        return self.in_flight.get(decode_node_id, 0)

    def peek_projected_preparing_wait_ms(
        self,
        decode_node_id: str,
        my_disk_blocks: int,
        now_ms: float,
    ) -> float:
        return 0.0

    def peek_session_affinity(self, session_id: str) -> str | None:
        return self.affinity.get(session_id)


def _nodes(prefix: str, count: int) -> list[MockEngineNode]:
    model = ModelConfig()
    node_cfg = NodeConfig(capacity=16)
    return [MockEngineNode(f"{prefix}{i}", model, node_cfg) for i in range(count)]


def _req(session_id: str | None = None) -> Request:
    return Request(
        request_id="r0",
        token_ids=list(range(64)),
        prefix_hash="p",
        expected_output_len=8,
        arrival_time=0.0,
        slo_ttft=10_000.0,
        slo_tbt=10_000.0,
        session_id=session_id,
    )


def _policy(view: _View, **kwargs: object) -> BidawPolicy:
    return BidawPolicy(
        model_config=ModelConfig(),
        bandwidth_config=BandwidthConfig(),
        backlog_view=NoopTransferModel(),
        controller_view=view,
        enable_routing_aware=True,
        **kwargs,
    )


def test_routing_prefers_matched_blocks_when_load_equal() -> None:
    prefill = _nodes("p", 1)
    decode = _nodes("d", 2)
    cache = _Cache(
        {
            "d0": CacheLookup(0, {}, 0.0),
            "d1": CacheLookup(64, {"gpu": 4}, 0.0),
        }
    )

    decision = _policy(
        _View(),
        routing_weight_load=0.0,
        routing_weight_preparing=0.0,
        routing_weight_in_flight=0.0,
    ).schedule(_req(), prefill, decode, cache, now=0.0)

    assert decision.decode_node == "d1"
    assert decision.routing_score == pytest.approx(-4.0)


def test_routing_penalizes_preparing_and_in_flight_blocks() -> None:
    prefill = _nodes("p", 1)
    decode = _nodes("d", 2)
    cache = _Cache(
        {
            "d0": CacheLookup(80, {"gpu": 5}, 0.0),
            "d1": CacheLookup(48, {"gpu": 3}, 0.0),
        }
    )
    view = _View(preparing={"d0": 3}, in_flight={"d0": 2})

    decision = _policy(view).schedule(_req(), prefill, decode, cache, now=0.0)

    assert decision.decode_node == "d1"
    assert decision.routing_score == pytest.approx(-3.0)


def test_routing_tie_breaks_by_node_id() -> None:
    prefill = _nodes("p", 1)
    decode = _nodes("d", 2)
    cache = _Cache(
        {
            "d0": CacheLookup(32, {"gpu": 2}, 0.0),
            "d1": CacheLookup(32, {"gpu": 2}, 0.0),
        }
    )

    decision = _policy(_View()).schedule(_req(), prefill, decode, cache, now=0.0)

    assert decision.decode_node == "d0"
    assert decision.routing_score == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "flag",
    ["enable_routing_aware", "enable_ttft_slo_gate", "enable_session_affinity"],
)
def test_bidaw_m3_flags_require_controller_view(flag: str) -> None:
    with pytest.raises(ValueError, match="controller_view"):
        BidawPolicy(
            model_config=ModelConfig(),
            bandwidth_config=BandwidthConfig(),
            backlog_view=NoopTransferModel(),
            **{flag: True},
        )


def test_existing_schedulers_keep_decision_defaults() -> None:
    prefill = _nodes("p", 1)
    decode = _nodes("d", 2)
    req = _req()
    cache = NullCacheQuery(["d0", "d1"])
    transfer_model = NoopTransferModel()
    policies = [
        RoundRobinPolicy(
            model_config=ModelConfig(),
            bandwidth_config=BandwidthConfig(),
            backlog_view=transfer_model,
        ),
        LeastLoadedPolicy(
            model_config=ModelConfig(),
            bandwidth_config=BandwidthConfig(),
            backlog_view=transfer_model,
        ),
        PrefixGreedyPolicy(
            model_config=ModelConfig(),
            bandwidth_config=BandwidthConfig(),
            backlog_view=transfer_model,
        ),
        E2Policy(
            model_config=ModelConfig(),
            bandwidth_config=BandwidthConfig(),
            backlog_view=transfer_model,
        ),
        MooncakeConductor(
            model_config=ModelConfig(),
            bandwidth_config=BandwidthConfig(),
            backlog_view=transfer_model,
        ),
    ]

    for policy in policies:
        decision = policy.schedule(req, prefill, decode, cache, now=0.0)
        assert decision.routing_score is None
        assert decision.affinity_hit is False

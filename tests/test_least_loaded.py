"""Tests for the LeastLoadedPolicy scheduler."""
from __future__ import annotations

import pytest

from nano_kvrouter.config import ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler._testing import NullCacheQuery
from nano_kvrouter.scheduler.base import SchedulingPolicy
from nano_kvrouter.scheduler.least_loaded import LeastLoadedPolicy


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

MODEL = ModelConfig(prefill_cost_per_token_ms=0.1, decode_base_ms=5.0, marginal_decode_ms=1.0)
NODE_CFG = NodeConfig(capacity=8)


def _make_node(node_id: str) -> MockEngineNode:
    return MockEngineNode(node_id=node_id, model_config=MODEL, node_config=NODE_CFG)


def _make_request(n_tokens: int, *, request_id: str = "req-0") -> Request:
    return Request(
        request_id=request_id,
        token_ids=list(range(n_tokens)),
        prefix_hash="00000000",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=2000.0,
        slo_tbt=100.0,
    )


@pytest.fixture
def three_nodes() -> list[MockEngineNode]:
    return [_make_node("n0"), _make_node("n1"), _make_node("n2")]


@pytest.fixture
def null_cache(three_nodes: list[MockEngineNode]) -> NullCacheQuery:
    return NullCacheQuery(node_ids=[n.node_id for n in three_nodes])


@pytest.fixture
def policy() -> LeastLoadedPolicy:
    return LeastLoadedPolicy()


@pytest.fixture
def req() -> Request:
    return _make_request(10)


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_scheduling_policy_protocol() -> None:
    assert isinstance(LeastLoadedPolicy(), SchedulingPolicy)


# ---------------------------------------------------------------------------
# 2. Single node — always picked
# ---------------------------------------------------------------------------


def test_single_node_always_picked(req: Request) -> None:
    single = [_make_node("only")]
    cache = NullCacheQuery(node_ids=["only"])
    p = LeastLoadedPolicy()
    for _ in range(5):
        dec = p.schedule(req, single, cache)
        assert dec.prefill_node == "only"


# ---------------------------------------------------------------------------
# 3. Picks node with lowest load
# ---------------------------------------------------------------------------


def test_picks_node_with_lowest_load(req: Request) -> None:
    n0 = _make_node("n0")
    n1 = _make_node("n1")
    n2 = _make_node("n2")
    # n0: 5 running, n1: 2 running, n2: 1 running → n2 has lowest load
    for i in range(5):
        n0.admit(f"r{i}")
    for i in range(2):
        n1.admit(f"s{i}")
    n2.admit("t0")

    cache = NullCacheQuery(node_ids=["n0", "n1", "n2"])
    dec = LeastLoadedPolicy().schedule(req, [n0, n1, n2], cache)
    assert dec.prefill_node == "n2"


# ---------------------------------------------------------------------------
# 4. Ties broken by index order (first in sequence wins)
# ---------------------------------------------------------------------------


def test_ties_broken_by_index_order(null_cache: NullCacheQuery, req: Request) -> None:
    # All three nodes start with 0 running → all load = 0.0 → n0 wins
    nodes = [_make_node("n0"), _make_node("n1"), _make_node("n2")]
    dec = LeastLoadedPolicy().schedule(req, nodes, null_cache)
    assert dec.prefill_node == "n0"


def test_ties_broken_by_index_order_reordered(req: Request) -> None:
    # Same load but sequence is [n2, n1, n0] → n2 should win
    nodes = [_make_node("n2"), _make_node("n1"), _make_node("n0")]
    cache = NullCacheQuery(node_ids=["n2", "n1", "n0"])
    dec = LeastLoadedPolicy().schedule(req, nodes, cache)
    assert dec.prefill_node == "n2"


# ---------------------------------------------------------------------------
# 5. Cache methods are never called
# ---------------------------------------------------------------------------


class _ExplodingCache:
    """Any call to lookup/lookup_all/free_blocks raises RuntimeError."""

    def lookup(self, *a, **kw):  # type: ignore[override]
        raise RuntimeError("LeastLoaded must not call cache.lookup")

    def lookup_all(self, *a, **kw):  # type: ignore[override]
        raise RuntimeError("LeastLoaded must not call cache.lookup_all")

    def free_blocks(self, *a, **kw):  # type: ignore[override]
        raise RuntimeError("LeastLoaded must not call cache.free_blocks")


def test_does_not_consult_cache_methods(req: Request) -> None:
    nodes = [_make_node("n0")]
    dec = LeastLoadedPolicy().schedule(req, nodes, _ExplodingCache())  # type: ignore[arg-type]
    assert not dec.is_rejected
    assert dec.prefill_node == "n0"


# ---------------------------------------------------------------------------
# 6. Empty nodes → rejection
# ---------------------------------------------------------------------------


def test_empty_nodes_returns_rejection(policy: LeastLoadedPolicy, req: Request) -> None:
    cache = NullCacheQuery(node_ids=[])
    dec = policy.schedule(req, [], cache)
    assert dec.is_rejected
    assert dec.reject_reason == "no_nodes_available"
    assert dec.prefill_node is None
    assert dec.decode_node is None


# ---------------------------------------------------------------------------
# 7. TTFT = cold prefill + queue wait
# ---------------------------------------------------------------------------


def test_ttft_equals_cold_prefill_plus_queue_wait(policy: LeastLoadedPolicy) -> None:
    node = _make_node("n0")
    # Fill to capacity so the next admit goes to queue (queue_depth=1)
    for i in range(NODE_CFG.capacity):
        node.admit(f"r{i}")
    node.admit("queued")
    assert len(node.queue) == 1

    req = _make_request(20)
    cache = NullCacheQuery(node_ids=["n0"])
    dec = policy.schedule(req, [node], cache)

    # 8 running == capacity, 0 decoding → bs = max(0,8) = 8, step_time = 13.0
    # n_blockers=2; per_req = 20*0.1+32*13.0 = 418.0; queue_wait = 836.0
    # cold prefill = 2.0; first_tick = 5.0+(8+1)*1.0 = 14.0
    # expected_ttft = 2.0 + 836.0 + 14.0 = 852.0
    expected_ttft = (
        20 * MODEL.prefill_cost_per_token_ms
        + 2 * (20 * MODEL.prefill_cost_per_token_ms + 32 * (MODEL.decode_base_ms + NODE_CFG.capacity * MODEL.marginal_decode_ms))
        + (MODEL.decode_base_ms + (NODE_CFG.capacity + 1) * MODEL.marginal_decode_ms)
    )
    assert dec.estimated_ttft_ms == pytest.approx(expected_ttft)


def test_ttft_zero_queue_cold_prefill(policy: LeastLoadedPolicy) -> None:
    req = _make_request(30)
    node = _make_node("n0")
    cache = NullCacheQuery(node_ids=["n0"])
    dec = policy.schedule(req, [node], cache)
    # idle node: queue_wait=0; cold prefill + first_tick (bs=1) = 3.0 + 6.0 = 9.0
    assert dec.estimated_ttft_ms == pytest.approx(
        30 * MODEL.prefill_cost_per_token_ms + MODEL.decode_base_ms + MODEL.marginal_decode_ms
    )


# ---------------------------------------------------------------------------
# 8. TBT reflects (running + 1) batch size
# ---------------------------------------------------------------------------


def test_tbt_reflects_running_plus_one(policy: LeastLoadedPolicy, req: Request) -> None:
    node = _make_node("n0")
    node.admit("r0")
    node.admit("r1")
    assert len(node.running_requests) == 2

    cache = NullCacheQuery(node_ids=["n0"])
    dec = policy.schedule(req, [node], cache)
    # batch_size = 2 running + 1 new = 3
    expected_tbt = MODEL.decode_base_ms + 3 * MODEL.marginal_decode_ms
    assert dec.estimated_tbt_ms == pytest.approx(expected_tbt)


# ---------------------------------------------------------------------------
# 9. prefill_node == decode_node (combined-deployment invariant)
# ---------------------------------------------------------------------------


def test_prefill_node_equals_decode_node(
    policy: LeastLoadedPolicy,
    three_nodes: list[MockEngineNode],
    null_cache: NullCacheQuery,
    req: Request,
) -> None:
    for _ in range(6):
        dec = policy.schedule(req, three_nodes, null_cache)
        assert dec.prefill_node == dec.decode_node

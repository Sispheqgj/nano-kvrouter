"""Tests for the RoundRobinPolicy scheduler."""
from __future__ import annotations

import pytest

from nano_kvrouter.config import ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler._testing import NullCacheQuery
from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy


# ---------------------------------------------------------------------------
# Fixtures
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
def policy() -> RoundRobinPolicy:
    return RoundRobinPolicy()


@pytest.fixture
def req() -> Request:
    return _make_request(10)


# ---------------------------------------------------------------------------
# Round-robin rotation
# ---------------------------------------------------------------------------


def test_rotates_across_three_nodes(
    policy: RoundRobinPolicy,
    three_nodes: list[MockEngineNode],
    null_cache: NullCacheQuery,
    req: Request,
) -> None:
    picks = [policy.schedule(req, three_nodes, null_cache).prefill_node for _ in range(3)]
    assert picks == ["n0", "n1", "n2"]


def test_wraps_around_after_full_cycle(
    policy: RoundRobinPolicy,
    three_nodes: list[MockEngineNode],
    null_cache: NullCacheQuery,
    req: Request,
) -> None:
    for _ in range(3):
        policy.schedule(req, three_nodes, null_cache)
    fourth = policy.schedule(req, three_nodes, null_cache)
    assert fourth.prefill_node == "n0"


def test_single_node_always_picked(null_cache: NullCacheQuery, req: Request) -> None:
    single = [_make_node("only")]
    cache = NullCacheQuery(node_ids=["only"])
    p = RoundRobinPolicy()
    for _ in range(5):
        dec = p.schedule(req, single, cache)
        assert dec.prefill_node == "only"


# ---------------------------------------------------------------------------
# Empty nodes — early rejection
# ---------------------------------------------------------------------------


def test_empty_nodes_returns_rejection(policy: RoundRobinPolicy, req: Request) -> None:
    cache = NullCacheQuery(node_ids=[])
    dec = policy.schedule(req, [], cache)
    assert dec.is_rejected
    assert dec.reject_reason == "no_nodes_available"
    assert dec.prefill_node is None
    assert dec.decode_node is None


# ---------------------------------------------------------------------------
# TTFT estimation
# ---------------------------------------------------------------------------


def test_ttft_equals_cold_prefill_with_empty_queue(
    policy: RoundRobinPolicy,
    three_nodes: list[MockEngineNode],
    null_cache: NullCacheQuery,
) -> None:
    req = _make_request(20)
    dec = policy.schedule(req, three_nodes, null_cache)
    # M3 chunked-prefill: 20 tokens → 1 chunk (20 < chunk_size=512)
    # bs_hint = running+1 = 0+1 = 1
    # step_per_chunk = 512*ppt + base + 1*marginal = 51.2+5.0+1.0 = 57.2
    # first_tick = base + 1*marginal = 6.0
    # expected_ttft = 57.2 + 0 (queue_wait) + 6.0 = 63.2
    expected_ttft = (
        512 * MODEL.prefill_cost_per_token_ms + MODEL.decode_base_ms + MODEL.marginal_decode_ms
        + MODEL.decode_base_ms + MODEL.marginal_decode_ms
    )
    assert dec.estimated_ttft_ms == pytest.approx(expected_ttft)


def test_ttft_includes_queue_wait(
    policy: RoundRobinPolicy,
    null_cache: NullCacheQuery,
) -> None:
    node = _make_node("n0")
    # Fill the node to capacity so the next admit goes to the queue.
    for i in range(NODE_CFG.capacity):
        node.admit(f"r{i}")
    node.admit("extra")  # lands in queue
    assert len(node.queue) == 1

    cache = NullCacheQuery(node_ids=["n0"])
    req = _make_request(10)
    dec = policy.schedule(req, [node], cache)
    # M3: bs_hint = running+1 = 8+1 = 9; step_per_chunk = 512*0.1+5+9*1 = 65.2
    # queue_wait: bs=max(0,8)=8, step=13.0, n_blockers=2, per_req=10*0.1+32*13=417.0
    # queue_wait = 2*417.0 = 834.0; first_tick = 5+9*1 = 14.0
    # expected_ttft = 65.2 + 834.0 + 14.0 = 913.2
    assert dec.estimated_ttft_ms == pytest.approx(
        (512 * 0.1 + 5.0 + 9 * 1.0) + 2 * (10 * 0.1 + 32 * (5.0 + 8 * 1.0)) + (5.0 + 9 * 1.0)
    )


# ---------------------------------------------------------------------------
# TBT estimation
# ---------------------------------------------------------------------------


def test_tbt_reflects_empty_running_batch(
    policy: RoundRobinPolicy,
    three_nodes: list[MockEngineNode],
    null_cache: NullCacheQuery,
    req: Request,
) -> None:
    dec = policy.schedule(req, three_nodes, null_cache)
    # 0 running + 1 for new request → decode_base + 1 * marginal
    expected_tbt = MODEL.decode_base_ms + 1 * MODEL.marginal_decode_ms
    assert dec.estimated_tbt_ms == pytest.approx(expected_tbt)


def test_tbt_grows_with_running_requests(
    policy: RoundRobinPolicy,
    null_cache: NullCacheQuery,
    req: Request,
) -> None:
    node = _make_node("n0")
    node.admit("already-0")
    node.admit("already-1")
    assert len(node.running_requests) == 2

    cache = NullCacheQuery(node_ids=["n0"])
    dec = policy.schedule(req, [node], cache)
    # 2 running + 1 = batch_size 3
    expected_tbt = MODEL.decode_base_ms + 3 * MODEL.marginal_decode_ms
    assert dec.estimated_tbt_ms == pytest.approx(expected_tbt)


# ---------------------------------------------------------------------------
# Combined-deployment contract
# ---------------------------------------------------------------------------


def test_prefill_node_equals_decode_node(
    policy: RoundRobinPolicy,
    three_nodes: list[MockEngineNode],
    null_cache: NullCacheQuery,
    req: Request,
) -> None:
    for _ in range(6):
        dec = policy.schedule(req, three_nodes, null_cache)
        assert dec.prefill_node == dec.decode_node


# ---------------------------------------------------------------------------
# Instance isolation
# ---------------------------------------------------------------------------


def test_two_instances_have_independent_cursors(
    three_nodes: list[MockEngineNode],
    null_cache: NullCacheQuery,
    req: Request,
) -> None:
    p1 = RoundRobinPolicy()
    p2 = RoundRobinPolicy()
    # Advance p1 by 2
    p1.schedule(req, three_nodes, null_cache)
    p1.schedule(req, three_nodes, null_cache)
    # p2 should still start at index 0
    assert p2.schedule(req, three_nodes, null_cache).prefill_node == "n0"
    assert p1.schedule(req, three_nodes, null_cache).prefill_node == "n2"


# ---------------------------------------------------------------------------
# Cache is not consulted (cold-prefill assumption)
# ---------------------------------------------------------------------------


def test_ttft_assumes_cold_prefill_regardless_of_cache(
    policy: RoundRobinPolicy,
    three_nodes: list[MockEngineNode],
) -> None:
    """Even when cache reports all tokens matched, TTFT is still cold-prefill cost."""

    class FullHitCache:
        """Stub where every token is cached."""

        def lookup(self, request: Request, node_id: str) -> object:  # type: ignore[override]
            from nano_kvrouter.scheduler.base import CacheLookup

            return CacheLookup(
                matched_tokens=len(request.token_ids),
                matched_blocks_by_tier={"gpu": len(request.token_ids) // 16},
                transfer_cost_ms=0.0,
            )

        def lookup_all(self, request: Request) -> dict:  # type: ignore[override]
            return {}

        def free_blocks(self, node_id: str, tier: str) -> int:
            return 1024

    req = _make_request(20)
    dec = policy.schedule(req, three_nodes, FullHitCache())  # type: ignore[arg-type]
    # RoundRobin never calls lookup/lookup_all — cached_tokens=0 always.
    # M3: 1 chunk (20 < 512), bs_hint=1; step_per_chunk=57.2; first_tick=6.0 → 63.2
    assert dec.estimated_ttft_ms == pytest.approx(
        512 * MODEL.prefill_cost_per_token_ms + MODEL.decode_base_ms + MODEL.marginal_decode_ms
        + MODEL.decode_base_ms + MODEL.marginal_decode_ms
    )

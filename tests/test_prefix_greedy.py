"""Tests for PrefixGreedyPolicy — cache-affinity-first scheduler."""
from __future__ import annotations

import uuid

import pytest

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import SchedulingPolicy
from nano_kvrouter.scheduler.prefix_greedy import PrefixGreedyPolicy


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16


@pytest.fixture
def cm() -> CacheManager:
    """Real CacheManager — needed because tests verify true cache hits."""
    return CacheManager(
        node_ids=["n0", "n1", "n2"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=100, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )


@pytest.fixture
def nodes() -> list[MockEngineNode]:
    mc = ModelConfig(
        block_size=BLOCK_SIZE,
        prefill_cost_per_token_ms=0.1,
        decode_base_ms=5.0,
        marginal_decode_ms=0.5,
    )
    nc = NodeConfig(capacity=8)
    return [MockEngineNode(f"n{i}", mc, nc) for i in range(3)]


def _make_request(token_ids: list[int]) -> Request:
    return Request(
        request_id=str(uuid.uuid4()),
        token_ids=list(token_ids),
        prefix_hash="x",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=2000.0,
        slo_tbt=100.0,
    )


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_scheduling_policy_protocol() -> None:
    assert isinstance(PrefixGreedyPolicy(), SchedulingPolicy)


# ---------------------------------------------------------------------------
# 2. Invalid min_hit_ratio raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_min_hit_ratio_raises() -> None:
    with pytest.raises(ValueError):
        PrefixGreedyPolicy(min_hit_ratio=-0.1)
    with pytest.raises(ValueError):
        PrefixGreedyPolicy(min_hit_ratio=1.5)


# ---------------------------------------------------------------------------
# 3. Cold-start fallback to least-loaded
# ---------------------------------------------------------------------------


def test_cold_start_falls_back_to_least_loaded(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """With no cache anywhere, fallback selects the least-loaded node."""
    # n0: 2 running, n1: 5 running → n0 is least loaded
    for i in range(2):
        nodes[0].admit(f"r{i}")
    for i in range(5):
        nodes[1].admit(f"r{i}")

    req = _make_request(list(range(128)))
    dec = PrefixGreedyPolicy().schedule(req, nodes[:2], cm)
    assert dec.prefill_node == "n0"


# ---------------------------------------------------------------------------
# 4. Picks node with highest matched_tokens
# ---------------------------------------------------------------------------


def test_picks_node_with_highest_matched_tokens(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """Node with matching cache wins even against a cold peer."""
    cm.admit(list(range(64)), "n1")  # 4 blocks on n1
    req = _make_request(list(range(64)))
    dec = PrefixGreedyPolicy().schedule(req, nodes[:2], cm)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 5. Hit below threshold → fallback to load
# ---------------------------------------------------------------------------


def test_hit_below_threshold_falls_back_to_load(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """If best hit ratio < min_hit_ratio, load decides (not cache)."""
    # 1 block (16 tokens) cached on n1; prompt = 128 tokens → ratio 12.5% < 25%
    cm.admit(list(range(16)), "n1")
    # n1 is heavily loaded; n0 is idle
    for i in range(5):
        nodes[1].admit(f"r{i}")

    req = _make_request(list(range(128)))
    dec = PrefixGreedyPolicy(min_hit_ratio=0.25).schedule(req, nodes[:2], cm)
    # hit ratio 16/128 = 12.5% < 25% → fallback → pick least loaded = n0
    assert dec.prefill_node == "n0"


# ---------------------------------------------------------------------------
# 6. Hit above threshold → cache wins over load
# ---------------------------------------------------------------------------


def test_hit_above_threshold_picks_cache_node(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """50% hit ratio exceeds 25% threshold; cache node wins despite higher load."""
    # n1 has 4 blocks (64 tokens) of the 8-block (128-token) prompt
    cm.admit(list(range(64)), "n1")
    # n1 is heavily loaded (7 running out of 8 capacity)
    for i in range(7):
        nodes[1].admit(f"r{i}")
    # n0 is idle

    req = _make_request(list(range(128)))
    dec = PrefixGreedyPolicy(min_hit_ratio=0.25).schedule(req, nodes[:2], cm)
    # hit ratio 64/128 = 50% > 25% → cache wins → pick n1
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 7. Ties in hit broken by load
# ---------------------------------------------------------------------------


def test_ties_in_hit_broken_by_load(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """When two nodes have the same matched_tokens, lower load wins."""
    prefix = list(range(64))
    cm.admit(prefix, "n0")
    cm.admit(prefix, "n1")
    # n0 is more loaded than n1
    for i in range(3):
        nodes[0].admit(f"r{i}")
    # n1 has 0 running

    req = _make_request(prefix)
    dec = PrefixGreedyPolicy().schedule(req, nodes[:2], cm)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 8. Ties in hit and load → first by index
# ---------------------------------------------------------------------------


def test_ties_in_hit_and_load_broken_by_index(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """When hit and load are equal, the node that appears first wins."""
    prefix = list(range(64))
    cm.admit(prefix, "n0")
    cm.admit(prefix, "n1")
    # Both n0 and n1 have 0 running requests (equal load)

    req = _make_request(prefix)
    dec = PrefixGreedyPolicy().schedule(req, nodes[:2], cm)
    assert dec.prefill_node == "n0"


# ---------------------------------------------------------------------------
# 9. TTFT reflects cache hit
# ---------------------------------------------------------------------------


def test_ttft_reflects_cache_hit(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """TTFT must use cached_tokens in estimate_prefill_time, not 0."""
    # 4 blocks (64 tokens) cached on n0; request is 8 blocks (128 tokens)
    cm.admit(list(range(64)), "n0")
    req = _make_request(list(range(128)))

    dec = PrefixGreedyPolicy().schedule(req, nodes[:1], cm)

    # M3: uncached=64, bs_hint=running+1=1; n_chunks=1 (64<512)
    # step_per_chunk = 512*0.1+5.0+1*0.5 = 56.7; first_tick = 5.0+0.5 = 5.5
    # expected_ttft = 56.7 + 0 + 5.5 = 62.2
    assert dec.estimated_ttft_ms == pytest.approx(56.7 + 5.5)


# ---------------------------------------------------------------------------
# 10. Empty nodes → rejection
# ---------------------------------------------------------------------------


def test_empty_nodes_returns_rejection(cm: CacheManager) -> None:
    req = _make_request(list(range(64)))
    dec = PrefixGreedyPolicy().schedule(req, [], cm)
    assert dec.is_rejected
    assert dec.reject_reason == "no_nodes_available"


# ---------------------------------------------------------------------------
# 11. prefill_node == decode_node (combined deployment invariant)
# ---------------------------------------------------------------------------


def test_prefill_node_equals_decode_node(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    req = _make_request(list(range(64)))
    dec = PrefixGreedyPolicy().schedule(req, nodes, cm)
    assert dec.prefill_node == dec.decode_node
    assert dec.prefill_node is not None

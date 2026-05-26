"""Tests for E2Policy — Preble ICLR'25 exploit-explore scheduler."""
from __future__ import annotations

import uuid

import pytest

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import SchedulingPolicy
from nano_kvrouter.scheduler.e2_policy import E2Policy


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
MODEL = ModelConfig(
    block_size=BLOCK_SIZE,
    prefill_cost_per_token_ms=0.1,
    decode_base_ms=5.0,
    marginal_decode_ms=1.0,
)
NODE_CFG = NodeConfig(capacity=8)


@pytest.fixture
def cm() -> CacheManager:
    """Real CacheManager with 4-block GPU capacity per node."""
    return CacheManager(
        node_ids=["n0", "n1", "n2"],
        model_config=MODEL,
        node_config=NodeConfig(gpu_blocks=4, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )


@pytest.fixture
def cm_large() -> CacheManager:
    """Real CacheManager with generous GPU capacity (100 blocks per node)."""
    return CacheManager(
        node_ids=["n0", "n1", "n2"],
        model_config=MODEL,
        node_config=NodeConfig(gpu_blocks=100, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )


@pytest.fixture
def nodes() -> list[MockEngineNode]:
    return [MockEngineNode(f"n{i}", MODEL, NODE_CFG) for i in range(3)]


def _make_request(token_ids: list[int], *, slo_ttft: float = 2000.0) -> Request:
    return Request(
        request_id=str(uuid.uuid4()),
        token_ids=list(token_ids),
        prefix_hash="x",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=slo_ttft,
        slo_tbt=100.0,
    )


def _policy() -> E2Policy:
    return E2Policy(model_config=MODEL)


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_scheduling_policy_protocol() -> None:
    assert isinstance(E2Policy(), SchedulingPolicy)


# ---------------------------------------------------------------------------
# 2. Negative weight raises ValueError
# ---------------------------------------------------------------------------


def test_negative_weight_raises() -> None:
    with pytest.raises(ValueError):
        E2Policy(w_run=-0.1)
    with pytest.raises(ValueError):
        E2Policy(w_historical=-1.0)
    with pytest.raises(ValueError):
        E2Policy(w_eviction=-0.001)


# ---------------------------------------------------------------------------
# 3. Cold-start: all scores equal → first node by index wins
# ---------------------------------------------------------------------------


def test_cold_start_picks_first_node_by_index(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """Empty cache everywhere + equal load → min() stable → n0."""
    req = _make_request(list(range(64)))
    dec = _policy().schedule(req, nodes, cm_large)
    assert dec.prefill_node == "n0"


# ---------------------------------------------------------------------------
# 4. Load dominates when cache equal (w_h high enough)
# ---------------------------------------------------------------------------


def test_load_dominates_when_cache_equal(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """No cache anywhere; n0=5 running, n1=2 running, n2=4 → n1 wins."""
    for i in range(5):
        nodes[0].admit(f"a{i}")
    for i in range(2):
        nodes[1].admit(f"b{i}")
    for i in range(4):
        nodes[2].admit(f"c{i}")

    req = _make_request(list(range(64)))
    dec = _policy().schedule(req, nodes, cm_large)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 5. Cache hit reduces run_cost → cache node wins
# ---------------------------------------------------------------------------


def test_cache_hit_reduces_run_cost(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """n1 has 64 tokens cached; n0 and n2 are cold → n1 selected."""
    cm_large.admit(list(range(64)), "n1")  # 4 blocks fully cached on n1
    req = _make_request(list(range(64)))
    dec = _policy().schedule(req, nodes, cm_large)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 6. Eviction cost penalises full node
# ---------------------------------------------------------------------------


def test_eviction_cost_penalises_full_node(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """n0 is at capacity (4 blocks); n1 is empty → n1 wins to avoid eviction."""
    # Fill n0's 4 GPU blocks with an unrelated prefix (no hit for new request)
    cm.admit(list(range(1600, 1664)), "n0")   # 4 blocks of different tokens
    assert cm.free_blocks("n0", "gpu") == 0

    req = _make_request(list(range(2000, 2064)))   # 4 blocks, no overlap with n0
    dec = _policy().schedule(req, nodes[:2], cm)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 7. w_run=0 makes eviction term dominant
# ---------------------------------------------------------------------------


def test_weight_w_run_zero_eviction_dominates(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """w_run=0, w_eviction=10: full n0 → penalised; empty n1 → wins."""
    cm.admit(list(range(1600, 1664)), "n0")
    assert cm.free_blocks("n0", "gpu") == 0

    policy = E2Policy(w_historical=1.0, w_eviction=10.0, w_run=0.0, model_config=MODEL)
    req = _make_request(list(range(2000, 2064)))
    dec = policy.schedule(req, nodes[:2], cm)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 8. w_historical very high → load is the deciding factor
# ---------------------------------------------------------------------------


def test_weight_w_historical_high_makes_load_dominant(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """w_h=1000 → historical proxy dominates; least-loaded node wins."""
    for i in range(5):
        nodes[0].admit(f"a{i}")   # n0: 5 running (load=0.625)
    for i in range(1):
        nodes[1].admit(f"b{i}")   # n1: 1 running (load=0.125) ← winner
    for i in range(4):
        nodes[2].admit(f"c{i}")   # n2: 4 running

    policy = E2Policy(w_historical=1000.0, w_eviction=1.0, w_run=1.0, model_config=MODEL)
    req = _make_request(list(range(64)))
    dec = policy.schedule(req, nodes, cm_large)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 9. SLO violation does NOT cause rejection (Conductor's responsibility)
# ---------------------------------------------------------------------------


def test_does_not_early_reject_even_when_slo_exceeded(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """E2Policy never rejects based on SLO; that belongs to MooncakeConductor."""
    req = _make_request(list(range(64)), slo_ttft=0.001)   # effectively 0 ms SLO
    dec = _policy().schedule(req, nodes, cm_large)
    assert not dec.is_rejected
    assert dec.prefill_node is not None


# ---------------------------------------------------------------------------
# 10. Empty nodes → rejection
# ---------------------------------------------------------------------------


def test_empty_nodes_returns_rejection(cm_large: CacheManager) -> None:
    req = _make_request(list(range(64)))
    dec = _policy().schedule(req, [], cm_large)
    assert dec.is_rejected
    assert dec.reject_reason == "no_nodes_available"
    assert dec.prefill_node is None
    assert dec.decode_node is None


# ---------------------------------------------------------------------------
# 11. prefill_node == decode_node (combined-deployment invariant)
# ---------------------------------------------------------------------------


def test_prefill_node_equals_decode_node(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    req = _make_request(list(range(64)))
    dec = _policy().schedule(req, nodes, cm_large)
    assert dec.prefill_node == dec.decode_node
    assert dec.prefill_node is not None


# ---------------------------------------------------------------------------
# 12. TTFT uses actual cached_tokens (not cold-prefill assumption)
# ---------------------------------------------------------------------------


def test_ttft_reflects_cache_hit(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """When n0 has half the prompt cached, TTFT should reflect only uncached cost."""
    cm_large.admit(list(range(64)), "n0")   # 64 tokens cached on n0
    req = _make_request(list(range(128)))   # 128-token request → 64 uncached

    # Force routing to n0 by making n1 and n2 very loaded
    for i in range(7):
        nodes[1].admit(f"x{i}")
        nodes[2].admit(f"y{i}")

    dec = _policy().schedule(req, nodes, cm_large)
    assert dec.prefill_node == "n0"
    # uncached = 128 - 64 = 64 tokens × 0.1 ms = 6.4 ms; queue = 0
    assert dec.estimated_ttft_ms == pytest.approx(64 * 0.1)

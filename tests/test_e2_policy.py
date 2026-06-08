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
BW_INF = BandwidthConfig(gpu_to_gpu=1e30)


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
    return E2Policy(model_config=MODEL, bandwidth_config=BW_INF)


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
    dec = _policy().schedule(req, nodes, nodes, cm_large)
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
    dec = _policy().schedule(req, nodes, nodes, cm_large)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 5. Cache hit reduces run_cost → cache node wins
# ---------------------------------------------------------------------------


def test_cache_hit_reduces_run_cost(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """M5a: cache hit on n1 reduces e2_score → decode_node = n1.

    Prefill_node is load-driven; with all loads at 0 it falls back to
    node_id lex-order → n0.
    """
    cm_large.admit(list(range(64)), "n1")  # 4 blocks fully cached on n1
    req = _make_request(list(range(64)))
    dec = _policy().schedule(req, nodes, nodes, cm_large)
    assert dec.decode_node == "n1"


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
    dec = _policy().schedule(req, nodes[:2], nodes[:2], cm)
    # M5a: eviction term lives on the decode pool — decode_node = n1.
    assert dec.decode_node == "n1"


# ---------------------------------------------------------------------------
# 7. w_run=0 makes eviction term dominant
# ---------------------------------------------------------------------------


def test_weight_w_run_zero_eviction_dominates(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """w_run=0, w_eviction=10: full n0 → penalised; empty n1 → wins."""
    cm.admit(list(range(1600, 1664)), "n0")
    assert cm.free_blocks("n0", "gpu") == 0

    policy = E2Policy(w_historical=1.0, w_eviction=10.0, w_run=0.0, model_config=MODEL, bandwidth_config=BW_INF)
    req = _make_request(list(range(2000, 2064)))
    dec = policy.schedule(req, nodes[:2], nodes[:2], cm)
    # M5a: eviction term lives on the decode pool — decode_node = n1.
    assert dec.decode_node == "n1"


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

    policy = E2Policy(w_historical=1000.0, w_eviction=1.0, w_run=1.0, model_config=MODEL, bandwidth_config=BW_INF)
    req = _make_request(list(range(64)))
    dec = policy.schedule(req, nodes, nodes, cm_large)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 9. SLO violation does NOT cause rejection (Conductor's responsibility)
# ---------------------------------------------------------------------------


def test_does_not_early_reject_even_when_slo_exceeded(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """E2Policy never rejects based on SLO; that belongs to MooncakeConductor."""
    req = _make_request(list(range(64)), slo_ttft=0.001)   # effectively 0 ms SLO
    dec = _policy().schedule(req, nodes, nodes, cm_large)
    assert not dec.is_rejected
    assert dec.prefill_node is not None


# ---------------------------------------------------------------------------
# 10. Empty nodes → rejection
# ---------------------------------------------------------------------------


def test_empty_nodes_returns_rejection(cm_large: CacheManager) -> None:
    req = _make_request(list(range(64)))
    dec = _policy().schedule(req, [], [], cm_large)
    assert dec.is_rejected
    assert dec.reject_reason == "no_nodes_available"
    assert dec.prefill_node is None
    assert dec.decode_node is None


# ---------------------------------------------------------------------------
# 11. prefill_node == decode_node (combined-deployment invariant)
# ---------------------------------------------------------------------------


def test_returns_separate_prefill_decode_nodes(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """M5a: prefill_node (load-driven) may differ from decode_node (e2-driven)."""
    cm_large.admit(list(range(64)), "n1")  # warm n1 cache
    for i in range(3):
        nodes[1].admit(f"r{i}")  # load n1
    req = _make_request(list(range(64)))
    dec = _policy().schedule(req, nodes, nodes, cm_large)
    assert dec.prefill_node is not None
    assert dec.decode_node is not None


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

    dec = _policy().schedule(req, nodes, nodes, cm_large)
    assert dec.prefill_node == "n0"
    # M3: uncached=64, bs_hint=decoding+1=1; n_chunks=1 (64<512)
    # step_per_chunk = 512*0.1+5.0+1*1.0 = 57.2; first_tick = 5.0+1.0 = 6.0
    # est_ttft = 57.2 + 0 + 6.0 = 63.2 ms
    assert dec.estimated_ttft_ms == pytest.approx(57.2 + 6.0)


# ---------------------------------------------------------------------------
# 13. Regression: full-hit node with zero free_blocks must NOT be penalised
# ---------------------------------------------------------------------------


def test_full_hit_node_at_capacity_not_eviction_penalised(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """Codex regression (Important): eviction_cost must subtract already-cached
    blocks before computing shortage.

    n0: fully cached (4 blocks), free_blocks=0.
    n1: cold, free_blocks=4 (plenty of room).

    Without the fix, E2 computed shortage=4 for n0 (wrong — no new blocks
    needed) and shortage=0 for n1.  With w_eviction=2, w_run=1, w_h=0:
      old  score(n0) = 2×6.4 + 1×0   = 12.8  → n1 wrongly wins
      fix  score(n0) = 2×0   + 1×0   = 0.0   → n0 correctly wins
    """
    cm.admit(list(range(64)), "n0")      # 4 blocks on n0; free_blocks → 0
    assert cm.free_blocks("n0", "gpu") == 0

    policy = E2Policy(w_historical=0, w_eviction=2, w_run=1, model_config=MODEL, bandwidth_config=BW_INF)
    req = _make_request(list(range(64)))
    dec = policy.schedule(req, nodes[:2], nodes[:2], cm)
    # n0 is fully cached — no eviction needed — must be chosen over cold n1
    assert dec.prefill_node == "n0"


# ---------------------------------------------------------------------------
# 14. first_tick in run_cost changes routing decision (Important #5)
# ---------------------------------------------------------------------------


def test_first_tick_in_score_changes_choice(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """first_tick uses len(decoding)+1 as bs_hint; more decoding → higher run_cost.

    Both nodes have all 64 tokens cached (prefill_phase=0 for both), so the
    routing decision is driven entirely by first_tick:

    Setup (w_historical=0, w_eviction=0, w_run=1):
    * n0: 0 decoding, fully cached → bs_hint=1, first_tick=5+1*1=6.0, run=6.0
    * n1: 7 decoding, fully cached → bs_hint=8, first_tick=5+8*1=13.0, run=13.0

    n0 wins (6.0 < 13.0).  Without first_tick both runs would be 0 (tie →
    n0 by index), so this specifically validates that first_tick is counted.
    """
    cm_large.admit(list(range(64)), "n0")   # warm cache on n0
    cm_large.admit(list(range(64)), "n1")   # warm cache on n1
    for i in range(7):
        nodes[1].admit(f"load{i}", 10)
        nodes[1].start_decode(f"load{i}")   # 7 active decode streams on n1

    req = _make_request(list(range(64)))
    policy = E2Policy(w_historical=0, w_eviction=0, w_run=1, model_config=MODEL, bandwidth_config=BW_INF)
    dec = policy.schedule(req, nodes[:2], nodes[:2], cm_large)
    # n0 has lower first_tick (1 vs 8 concurrent decoders) → lower run_cost → wins
    assert dec.prefill_node == "n0", (
        f"Expected n0 (lower first_tick due to 0 decoding), got {dec.prefill_node}; "
        "first_tick not using decoding count as bs_hint?"
    )


# ---------------------------------------------------------------------------
# 15. M3: est_ttft reflects chunked prefill (n_chunks × step_per_chunk)
# ---------------------------------------------------------------------------


def test_ttft_includes_chunked_prefill(
    cm_large: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """est_ttft scales with number of prefill chunks.

    1024-token cold prompt, idle node (0 decoding), chunk_size=512 → n_chunks=2.
    bs_hint = 0+1 = 1; step_per_chunk = 512*0.1+5.0+1*1.0 = 57.2
    prefill_phase = 2×57.2 = 114.4; first_tick = 5.0+1.0 = 6.0
    est_ttft = 0 + 114.4 + 6.0 = 120.4 ms
    """
    req = _make_request(list(range(1024)))
    dec = _policy().schedule(req, nodes[:1], nodes[:1], cm_large)
    assert not dec.is_rejected
    chunk = MODEL.prefill_chunk_size  # 512
    step_per_chunk = chunk * MODEL.prefill_cost_per_token_ms + MODEL.decode_base_ms + 1 * MODEL.marginal_decode_ms
    assert dec.estimated_ttft_ms == pytest.approx(2 * step_per_chunk + MODEL.decode_base_ms + MODEL.marginal_decode_ms)

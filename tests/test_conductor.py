"""Tests for MooncakeConductor — three-objective scoring + SLO early rejection."""
from __future__ import annotations

import uuid

import pytest

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import SchedulingPolicy
from nano_kvrouter.scheduler.conductor import MooncakeConductor


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
# prefill_cost_per_token_ms=0.1 makes arithmetic easy: 64 tokens → 6.4 ms
_MC = ModelConfig(
    block_size=BLOCK_SIZE,
    prefill_cost_per_token_ms=0.1,
    decode_base_ms=5.0,
    marginal_decode_ms=0.5,
)
_NC = NodeConfig(capacity=8)
BW_INF = BandwidthConfig(gpu_to_gpu=1e30)


def _conductor(alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> MooncakeConductor:
    return MooncakeConductor(
        alpha=alpha, beta=beta, gamma=gamma,
        model_config=_MC, bandwidth_config=BW_INF,
    )


@pytest.fixture
def cm() -> CacheManager:
    """Real CacheManager — needed to test true cache-benefit term."""
    return CacheManager(
        node_ids=["n0", "n1", "n2"],
        model_config=_MC,
        node_config=NodeConfig(gpu_blocks=100, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )


@pytest.fixture
def nodes() -> list[MockEngineNode]:
    return [MockEngineNode(f"n{i}", _MC, _NC) for i in range(3)]


def _req(
    token_ids: list[int],
    *,
    slo_ttft: float = 2000.0,
    slo_tbt: float = 100.0,
) -> Request:
    return Request(
        request_id=str(uuid.uuid4()),
        token_ids=list(token_ids),
        prefix_hash="x",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=slo_ttft,
        slo_tbt=slo_tbt,
    )


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_scheduling_policy_protocol() -> None:
    assert isinstance(MooncakeConductor(), SchedulingPolicy)


# ---------------------------------------------------------------------------
# 2. Negative weight raises ValueError
# ---------------------------------------------------------------------------


def test_negative_weight_raises() -> None:
    with pytest.raises(ValueError):
        MooncakeConductor(gamma=-0.1)
    with pytest.raises(ValueError):
        MooncakeConductor(alpha=-1.0)
    with pytest.raises(ValueError):
        MooncakeConductor(beta=-0.5)


# ---------------------------------------------------------------------------
# 3. Cold start: all scores equal → first node by index wins
# ---------------------------------------------------------------------------


def test_cold_start_picks_first_by_index(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """With no cache and no load anywhere, all scores are equal → n0 wins."""
    req = _req(list(range(64)))
    dec = _conductor().schedule(req, nodes, nodes, cm)
    assert dec.prefill_node == "n0"
    assert not dec.is_rejected


# ---------------------------------------------------------------------------
# 4. Picks node with highest cache_benefit
# ---------------------------------------------------------------------------


def test_picks_node_with_highest_cache_benefit(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """M5a: cache benefit drives decode_node selection (KV lives on decode)."""
    cm.admit(list(range(64)), "n1")  # 4 blocks on n1
    req = _req(list(range(64)))
    dec = _conductor().schedule(req, nodes[:2], nodes[:2], cm)
    assert dec.decode_node == "n1"


# ---------------------------------------------------------------------------
# 5. Picks node with lowest load_penalty when cache is equal
# ---------------------------------------------------------------------------


def test_picks_node_with_lowest_load_when_cache_equal(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """With no cache anywhere, lowest-loaded decode_node wins (load_penalty term)."""
    # n0: 3 running, n1: 5 running, n2: 0 running → n2 has lowest load_penalty
    for i in range(3):
        nodes[0].admit(f"a{i}")
    for i in range(5):
        nodes[1].admit(f"b{i}")

    req = _req(list(range(64)))
    dec = _conductor().schedule(req, nodes, nodes, cm)
    assert dec.decode_node == "n2"


# ---------------------------------------------------------------------------
# 6. Reject when TTFT exceeds SLO
# ---------------------------------------------------------------------------


def test_reject_when_ttft_exceeds_slo(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """A nearly-zero slo_ttft forces a rejection on any cold prefill."""
    req = _req(list(range(64)), slo_ttft=0.001)
    dec = _conductor().schedule(req, nodes[:1], nodes[:1], cm)
    assert dec.is_rejected
    assert dec.reject_reason == "ttft_slo_exceeded"


# ---------------------------------------------------------------------------
# 7. Reject when TBT exceeds SLO
# ---------------------------------------------------------------------------


def test_reject_when_tbt_exceeds_slo(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """A nearly-zero slo_tbt forces a TBT rejection even when TTFT passes."""
    # slo_ttft large so TTFT check passes; slo_tbt tiny so TBT check fails.
    # est_tbt = decode_base_ms + 1*marginal = 5.0 + 0.5 = 5.5 > 0.001
    req = _req(list(range(64)), slo_ttft=2000.0, slo_tbt=0.001)
    dec = _conductor().schedule(req, nodes[:1], nodes[:1], cm)
    assert dec.is_rejected
    assert dec.reject_reason == "tbt_slo_exceeded"


# ---------------------------------------------------------------------------
# 8. Rejection preserves estimated TTFT/TBT for diagnostics
# ---------------------------------------------------------------------------


def test_rejection_preserves_estimated_values(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """Rejected SchedulingDecision must carry non-zero estimated_ttft/tbt
    so MetricsCollector can record the violation magnitude."""
    req = _req(list(range(64)), slo_ttft=0.001)
    dec = _conductor().schedule(req, nodes[:1], nodes[:1], cm)
    assert dec.is_rejected
    # M5a: 64 cold tokens, bs_hint=decoding+1=1; n_chunks=1 (64<512)
    # step_per_chunk = 512*0.1+5.0+1*0.5 = 56.7; queue_wait=0; first_tick=5.5
    # kv_transfer ~0 (BW_INF); est_ttft = 56.7 + 0 + 0 + 5.5 = 62.2 ms
    assert dec.estimated_ttft_ms == pytest.approx(62.2)
    # decode with batch_size=1: 5.0 + 0.5 = 5.5 ms
    assert dec.estimated_tbt_ms == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# 9. alpha=0 → cache term ignored; load decides
# ---------------------------------------------------------------------------


def test_alpha_zero_ignores_cache(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """With α=0, cache_benefit is zeroed out; only load_penalty matters."""
    # n1 has warm cache but is loaded; n0 is cold but idle
    cm.admit(list(range(64)), "n1")
    for i in range(3):
        nodes[1].admit(f"r{i}")

    req = _req(list(range(64)))
    dec = _conductor(alpha=0, beta=1, gamma=0).schedule(
        req, nodes[:2], nodes[:2], cm
    )
    # n0 has lower load_penalty → higher score (cache doesn't count) → decode = n0
    assert dec.decode_node == "n0"


# ---------------------------------------------------------------------------
# 10. beta=0 → load term ignored; cache decides
# ---------------------------------------------------------------------------


def test_beta_zero_ignores_load(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """With β=0, load_penalty is zeroed out; only cache_benefit matters."""
    # n1 has warm cache AND heavy load (would normally penalise n1)
    cm.admit(list(range(64)), "n1")
    for i in range(8):          # fill running to capacity
        nodes[1].admit(f"r{i}")
    for i in range(10):         # add queue pressure too
        nodes[1].admit(f"q{i}")
    # n0 is completely idle

    # With Important #6 fix (bs=0): queue_wait = 11×(64*0.1+32*5.0) = 11×166.4 = 1830.4 ms
    # Plus Critical #2 first_tick = 5+9*0.5=9.5 → est_ttft ~1839.9 < 5000 → accepted
    # With Important #1 fix: bs = max(decoding, running) = max(0, 8) = 8
    # step_time = 5.0 + 8*0.5 = 9.0
    # queue_wait = 11×(64*0.1 + 32*9.0) = 11×294.4 = 3238.4 ms
    # first_tick = 5.0 + 9*0.5 = 9.5; est_ttft ≈ 6.4 + 3238.4 + 9.5 = 3254.3 < 5000 → accepted
    req = _req(list(range(64)), slo_ttft=5000.0)
    dec = _conductor(alpha=1, beta=0, gamma=0).schedule(
        req, nodes[:2], nodes[:2], cm
    )
    # n1 has positive cache_benefit; load doesn't matter → decode_node = n1.
    assert dec.decode_node == "n1"


# ---------------------------------------------------------------------------
# 11. transfer_penalty is always 0 in v1
# ---------------------------------------------------------------------------


def test_transfer_penalty_is_zero_in_v1(cm: CacheManager) -> None:
    """CacheLookup.transfer_cost_ms must be 0 for all nodes in v1 GPU-only."""
    cm.admit(list(range(64)), "n0")
    req = _req(list(range(64)))
    lookups = cm.lookup_all(req)
    for nid, lk in lookups.items():
        assert lk.transfer_cost_ms == 0.0, f"node {nid} has non-zero transfer_cost_ms"


# ---------------------------------------------------------------------------
# 12. Empty nodes → "no_nodes_available" (distinct from SLO rejection)
# ---------------------------------------------------------------------------


def test_empty_nodes_returns_no_nodes_available(cm: CacheManager) -> None:
    req = _req(list(range(64)))
    dec = _conductor().schedule(req, [], [], cm)
    assert dec.is_rejected
    assert dec.reject_reason == "no_nodes_available"
    # Distinguishable from SLO rejection: no estimated time recorded
    assert dec.estimated_ttft_ms == 0.0
    assert dec.estimated_tbt_ms == 0.0


# ---------------------------------------------------------------------------
# 14. Critical #2: est_ttft includes first batch tick time
# ---------------------------------------------------------------------------


def test_ttft_includes_first_batch_tick(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """est_ttft must include the first decode step time, not just prefill+queue.

    On a loaded node (8 running), the first batch tick adds
    decode_base + (8+1)*marginal = 5.0 + 9*0.5 = 9.5 ms.
    """
    for i in range(8):
        nodes[0].admit(f"run{i}")   # fill to capacity (capacity=8)

    req = _req(list(range(32)), slo_ttft=999999.0)
    dec = _conductor().schedule(req, nodes[:1], nodes[:1], cm)
    assert not dec.is_rejected

    # M5a: decode_node bs_hint = len(decoding)+1 = 1 (no decoding).
    first_tick_ms = _MC.decode_base_ms + 1 * _MC.marginal_decode_ms  # = 5.5
    assert dec.estimated_ttft_ms >= first_tick_ms, (
        f"est_ttft={dec.estimated_ttft_ms:.3f} < first_tick={first_tick_ms:.3f}; "
        "TTFT prediction omits first batch step time"
    )


# ---------------------------------------------------------------------------
# 13. SLO check based on chosen node, not all nodes (design choice 6A)
# ---------------------------------------------------------------------------


def test_slo_check_based_on_chosen_node_not_all_nodes(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """M5a design 6A: Conductor rejects when the chosen (prefill, decode) pair
    fails SLO, even if another pair could serve the request.

    Setup: fill the only prefill_node to capacity so queue_wait dominates
    TTFT; the resulting est_ttft exceeds slo and the request is rejected.
    """
    # Fill nodes[0] to capacity → queue_wait will be heavy.
    for i in range(_NC.capacity):
        nodes[0].admit(f"run{i}")

    req = _req(list(range(128)), slo_ttft=100.0, slo_tbt=100.0)
    # Only one node available — both pools = [nodes[0]] → prefill_node and
    # decode_node both forced to n0.
    dec = _conductor(alpha=30, beta=1, gamma=0).schedule(
        req, nodes[:1], nodes[:1], cm,
    )
    assert dec.is_rejected
    assert dec.reject_reason == "ttft_slo_exceeded"


# ---------------------------------------------------------------------------
# 15. M3: est_ttft reflects chunked prefill (n_chunks × step_per_chunk)
# ---------------------------------------------------------------------------


def test_ttft_includes_chunked_prefill(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """est_ttft must scale with the number of prefill chunks.

    For a 1024-token cold prompt with chunk_size=512, n_chunks=2.
    Idle node: bs_hint = decoding+1 = 1.
    step_per_chunk = 512×0.1 + 5.0 + 1×0.5 = 56.7 ms
    prefill_phase  = 2 × 56.7 = 113.4 ms
    first_tick     = 5.0 + 1×0.5 = 5.5 ms
    est_ttft       = 0 + 113.4 + 5.5 = 118.9 ms
    """
    req = _req(list(range(1024)), slo_ttft=2000.0)
    dec = _conductor().schedule(req, nodes[:1], nodes[:1], cm)
    assert not dec.is_rejected
    chunk = _MC.prefill_chunk_size  # 512
    step_per_chunk = chunk * _MC.prefill_cost_per_token_ms + _MC.decode_base_ms + 1 * _MC.marginal_decode_ms
    assert dec.estimated_ttft_ms == pytest.approx(2 * step_per_chunk + _MC.decode_base_ms + _MC.marginal_decode_ms)

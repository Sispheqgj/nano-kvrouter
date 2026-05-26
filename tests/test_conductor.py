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
    dec = MooncakeConductor(model_config=_MC).schedule(req, nodes, cm)
    assert dec.prefill_node == "n0"
    assert not dec.is_rejected


# ---------------------------------------------------------------------------
# 4. Picks node with highest cache_benefit
# ---------------------------------------------------------------------------


def test_picks_node_with_highest_cache_benefit(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """Node with warm cache has higher score and wins over cold peers."""
    cm.admit(list(range(64)), "n1")  # 4 blocks on n1
    req = _req(list(range(64)))
    dec = MooncakeConductor(model_config=_MC).schedule(req, nodes[:2], cm)
    assert dec.prefill_node == "n1"


# ---------------------------------------------------------------------------
# 5. Picks node with lowest load_penalty when cache is equal
# ---------------------------------------------------------------------------


def test_picks_node_with_lowest_load_when_cache_equal(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """With no cache on any node, the least-loaded node wins."""
    # n0: 3 running, n1: 5 running, n2: 0 running → n2 has lowest load_penalty
    for i in range(3):
        nodes[0].admit(f"a{i}")
    for i in range(5):
        nodes[1].admit(f"b{i}")

    req = _req(list(range(64)))
    dec = MooncakeConductor(model_config=_MC).schedule(req, nodes, cm)
    assert dec.prefill_node == "n2"


# ---------------------------------------------------------------------------
# 6. Reject when TTFT exceeds SLO
# ---------------------------------------------------------------------------


def test_reject_when_ttft_exceeds_slo(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """A nearly-zero slo_ttft forces a rejection on any cold prefill."""
    req = _req(list(range(64)), slo_ttft=0.001)
    dec = MooncakeConductor(model_config=_MC).schedule(req, nodes[:1], cm)
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
    dec = MooncakeConductor(model_config=_MC).schedule(req, nodes[:1], cm)
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
    dec = MooncakeConductor(model_config=_MC).schedule(req, nodes[:1], cm)
    assert dec.is_rejected
    # prefill 64 cold tokens at 0.1 ms/token = 6.4 ms
    assert dec.estimated_ttft_ms == pytest.approx(6.4)
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
    dec = MooncakeConductor(alpha=0, beta=1, gamma=0, model_config=_MC).schedule(
        req, nodes[:2], cm
    )
    # n0 has lower load_penalty → higher score (cache doesn't count)
    assert dec.prefill_node == "n0"


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

    req = _req(list(range(64)))
    dec = MooncakeConductor(alpha=1, beta=0, gamma=0, model_config=_MC).schedule(
        req, nodes[:2], cm
    )
    # n1 has positive cache_benefit; load doesn't matter → n1 wins
    assert dec.prefill_node == "n1"


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
    dec = MooncakeConductor(model_config=_MC).schedule(req, [], cm)
    assert dec.is_rejected
    assert dec.reject_reason == "no_nodes_available"
    # Distinguishable from SLO rejection: no estimated time recorded
    assert dec.estimated_ttft_ms == 0.0
    assert dec.estimated_tbt_ms == 0.0


# ---------------------------------------------------------------------------
# 13. SLO check based on chosen node, not all nodes (design choice 6A)
# ---------------------------------------------------------------------------


def test_slo_check_based_on_chosen_node_not_all_nodes(
    cm: CacheManager, nodes: list[MockEngineNode]
) -> None:
    """Design 6A verification: Conductor rejects when the max-score node
    fails SLO — it does not try alternative nodes that could pass.

    Setup:
    * n0: no cache, idle → est_ttft = 6.4 ms < slo_ttft=10 → would pass
    * n1: full cache hit, 8 running + 10 queued
         → est_ttft = 0 + 10×5.0 = 50 ms > slo_ttft=10 → would fail
         → but cache_benefit is huge so score(n1) >> score(n0)

    With α=20 β=1:
      score(n0) = 0
      score(n1) = 20×6.4 − (1.0×64×0.1 + 50.0) = 128 − 56.4 = 71.6 > 0

    Conductor picks n1, finds its est_ttft=50 > slo=10, rejects.
    n0 could serve the request but is never tried (design 6A, not 6B).
    """
    cm.admit(list(range(64)), "n1")           # 4 blocks cached on n1
    for i in range(8):
        nodes[1].admit(f"run{i}")             # fill running slots
    for i in range(10):
        nodes[1].admit(f"que{i}")             # add 10 queued requests

    req = _req(list(range(64)), slo_ttft=10.0, slo_tbt=100.0)
    dec = MooncakeConductor(
        alpha=20, beta=1, gamma=0, model_config=_MC
    ).schedule(req, nodes[:2], cm)

    assert dec.is_rejected
    assert dec.reject_reason == "ttft_slo_exceeded"
    # Estimated TTFT reflects n1's queue (50 ms), not n0's cold prefill
    assert dec.estimated_ttft_ms == pytest.approx(50.0)

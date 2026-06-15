"""Tests for scheduler/base.py types and the NullCacheQuery test stub."""
from __future__ import annotations

import pytest

from nano_kvrouter.request import Request
from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.scheduler.base import (
    CacheLookup,
    CacheQuery,
    SchedulingDecision,
    SchedulingPolicy,
    TransferBacklogView,
    compute_est_ttft,
)
from nano_kvrouter.scheduler._testing import NullCacheQuery
from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_request() -> Request:
    return Request(
        request_id="r0",
        token_ids=[1, 2, 3, 4],
        prefix_hash="aabbccdd",
        expected_output_len=64,
        arrival_time=0.0,
        slo_ttft=2000.0,
        slo_tbt=100.0,
    )


@pytest.fixture
def null_cache() -> NullCacheQuery:
    return NullCacheQuery(node_ids=["n0", "n1", "n2"])


# ---------------------------------------------------------------------------
# CacheLookup
# ---------------------------------------------------------------------------


def test_cache_lookup_fields() -> None:
    cl = CacheLookup(matched_tokens=32, matched_blocks_by_tier={"gpu": 2}, transfer_cost_ms=1.5)
    assert cl.matched_tokens == 32
    assert cl.matched_blocks_by_tier == {"gpu": 2}
    assert cl.transfer_cost_ms == 1.5


def test_cache_lookup_is_frozen() -> None:
    cl = CacheLookup(matched_tokens=0, matched_blocks_by_tier={}, transfer_cost_ms=0.0)
    with pytest.raises((AttributeError, TypeError)):
        cl.matched_tokens = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SchedulingDecision
# ---------------------------------------------------------------------------


def test_scheduling_decision_accepted_not_rejected() -> None:
    dec = SchedulingDecision(
        prefill_node="n0",
        decode_node="n0",
        estimated_ttft_ms=50.0,
        estimated_tbt_ms=10.0,
    )
    assert not dec.is_rejected
    assert dec.reject_reason is None


def test_scheduling_decision_rejected() -> None:
    dec = SchedulingDecision(
        prefill_node=None,
        decode_node=None,
        estimated_ttft_ms=0.0,
        estimated_tbt_ms=0.0,
        reject_reason="no_nodes_available",
    )
    assert dec.is_rejected
    assert dec.reject_reason == "no_nodes_available"


def test_scheduling_decision_reject_reason_none_by_default() -> None:
    dec = SchedulingDecision(
        prefill_node="n0", decode_node="n0", estimated_ttft_ms=1.0, estimated_tbt_ms=1.0
    )
    assert dec.reject_reason is None


# ---------------------------------------------------------------------------
# NullCacheQuery — lookup
# ---------------------------------------------------------------------------


def test_null_lookup_returns_zero_hits(stub_request: Request, null_cache: NullCacheQuery) -> None:
    result = null_cache.lookup(stub_request, "n0")
    assert result.matched_tokens == 0
    assert result.matched_blocks_by_tier == {}
    assert result.transfer_cost_ms == 0.0


def test_null_lookup_unknown_node_raises(
    stub_request: Request, null_cache: NullCacheQuery
) -> None:
    with pytest.raises(KeyError):
        null_cache.lookup(stub_request, "unknown")


# ---------------------------------------------------------------------------
# NullCacheQuery — lookup_all
# ---------------------------------------------------------------------------


def test_null_lookup_all_covers_all_nodes(
    stub_request: Request, null_cache: NullCacheQuery
) -> None:
    result = null_cache.lookup_all(stub_request)
    assert set(result.keys()) == {"n0", "n1", "n2"}
    for cl in result.values():
        assert cl.matched_tokens == 0
        assert cl.transfer_cost_ms == 0.0


# ---------------------------------------------------------------------------
# NullCacheQuery — free_blocks
# ---------------------------------------------------------------------------


def test_null_free_blocks_returns_configured_count(null_cache: NullCacheQuery) -> None:
    assert null_cache.free_blocks("n0", "gpu") == 1024
    assert null_cache.free_blocks("n1", "cpu") == 1024


def test_null_free_blocks_unknown_node_raises(null_cache: NullCacheQuery) -> None:
    with pytest.raises(KeyError):
        null_cache.free_blocks("nX", "gpu")


def test_null_free_blocks_custom_count() -> None:
    cache = NullCacheQuery(node_ids=["a"], free_blocks_count=42)
    assert cache.free_blocks("a", "disk") == 42


# ---------------------------------------------------------------------------
# Protocol conformance (runtime_checkable)
# ---------------------------------------------------------------------------


def test_null_cache_query_satisfies_cache_query_protocol(
    null_cache: NullCacheQuery,
) -> None:
    assert isinstance(null_cache, CacheQuery)


def test_round_robin_satisfies_scheduling_policy_protocol() -> None:
    from nano_kvrouter.simulator.transfer_model import NoopTransferModel
    policy = RoundRobinPolicy(backlog_view=NoopTransferModel())
    assert isinstance(policy, SchedulingPolicy)


# ---------------------------------------------------------------------------
# compute_est_ttft: split P/D uses prefill_node.decoding, not decode_node.decoding
# ---------------------------------------------------------------------------


def test_compute_est_ttft_uses_prefill_node_bs() -> None:
    """compute_est_ttft must use prefill_node.decoding as batch_size_hint for
    the prefill phase, not decode_node.decoding.

    In split P/D the prefill_node has an empty decoding set (prefill_bs=1),
    while the decode_node may have many active streams.  Using decode_node's
    decoding count (old behaviour) would over-estimate prefill_phase.

    Setup:
        prefill_node: 0 decoding → prefill_bs = 1
        decode_node:  7 decoding → decoding_bs = 8

    Old (wrong): prefill_phase = 1 × (512×0.1 + 5.0 + 8×0.5) = 60.2 ms
    New (right): prefill_phase = 1 × (512×0.1 + 5.0 + 1×0.5) = 56.7 ms
    first_decode_tick = 5.0 + 8×0.5 = 9.0 ms (unchanged — uses decode_node.decoding)
    est_ttft (new) = 56.7 + 0 + 0 + 9.0 = 65.7 ms
    est_ttft (old) = 60.2 + 0 + 0 + 9.0 = 69.2 ms
    """
    mc = ModelConfig(
        block_size=16,
        prefill_cost_per_token_ms=0.1,
        decode_base_ms=5.0,
        marginal_decode_ms=0.5,
    )
    nc = NodeConfig(capacity=16)
    bw = BandwidthConfig(gpu_to_gpu=1e30)  # negligible transfer cost

    prefill_node = MockEngineNode("p0", mc, nc)
    decode_node = MockEngineNode("d0", mc, nc)

    # populate decode_node with 7 active decode streams
    for i in range(7):
        decode_node.admit(f"r{i}")
        decode_node.start_decode(f"r{i}")
    assert len(decode_node.decoding) == 7
    assert len(prefill_node.decoding) == 0  # prefill_node has no decode streams

    req = Request(
        request_id="test",
        token_ids=list(range(64)),
        prefix_hash="x",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=9999.0,
        slo_tbt=9999.0,
    )
    no_cache = CacheLookup(matched_tokens=0, matched_blocks_by_tier={}, transfer_cost_ms=0.0)

    from nano_kvrouter.simulator.transfer_model import NoopTransferModel
    est = compute_est_ttft(
        prefill_node,
        decode_node,
        req,
        no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw.gpu_to_gpu,
        backlog_view=NoopTransferModel(),
        now=0.0,
    )

    import pytest as _pytest
    # New correct value: prefill_phase uses prefill_bs=1 → 56.7 + 9.0 = 65.7
    assert est == _pytest.approx(65.7), (
        f"est_ttft={est:.3f} != 65.7; "
        "compute_est_ttft may be using decode_node.decoding instead of prefill_node.decoding"
    )
    # Also verify it is NOT the old wrong value (60.2 + 9.0 = 69.2)
    assert abs(est - 69.2) > 0.5, (
        "est_ttft matches the OLD wrong value 69.2; fix did not take effect"
    )



# ---------------------------------------------------------------------------
# P4-B: compute_est_ttft backlog integration tests
# ---------------------------------------------------------------------------


def _make_p4b_base() -> tuple:
    """Helper: minimal nodes + request with real ModelConfig defaults.

    Uses default ModelConfig (prefill_cost=0.033, decode_base=5.0, etc.)
    and a high bandwidth (1 GB/s with 100 tokens×1024 bytes = 0.1024 ms service).
    Tests compare compute_est_ttft WITH backlog against WITHOUT backlog to
    isolate the lane_queue_wait term without needing zero-cost model fields.
    """
    from nano_kvrouter.simulator.transfer_model import NoopTransferModel, PerNodeLaneTransferModel
    mc = ModelConfig(kv_bytes_per_token=1024)  # use defaults for cost fields
    nc = NodeConfig(capacity=16)
    pnode = MockEngineNode("p0", mc, nc)
    dnode = MockEngineNode("d0", mc, nc)
    req = Request(
        request_id="r_p4b",
        token_ids=list(range(100)),
        prefix_hash="hash_p4b",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=9999.0,
        slo_tbt=9999.0,
    )
    no_cache = CacheLookup(matched_tokens=0, matched_blocks_by_tier={}, transfer_cost_ms=0.0)
    bw_bps = 1e9  # 1 GB/s
    return mc, nc, pnode, dnode, req, no_cache, bw_bps


def _baseline_est(mc, pnode, dnode, req, no_cache, bw_bps) -> float:
    """compute_est_ttft with NoopTransferModel (zero backlog)."""
    from nano_kvrouter.simulator.transfer_model import NoopTransferModel
    return compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=NoopTransferModel(),
        now=0.0,
    )


def test_compute_est_ttft_ingress_only_backlog() -> None:
    """Only dst ingress has backlog → kv_transfer inflated by backlog - now."""
    from nano_kvrouter.simulator.transfer_model import PerNodeLaneTransferModel
    mc, nc, pnode, dnode, req, no_cache, bw_bps = _make_p4b_base()
    model = PerNodeLaneTransferModel()
    model._ingress_available_at["d0"] = 50.0  # dst ingress backlog = 50ms
    est_with = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=model,
        now=0.0,
    )
    baseline = _baseline_est(mc, pnode, dnode, req, no_cache, bw_bps)
    import pytest as _p
    # Only ingress backlog → queue_wait = 50ms increase
    assert est_with - baseline == _p.approx(50.0, abs=1e-6), (
        f"delta={est_with - baseline}, expected 50.0 from ingress backlog"
    )


def test_compute_est_ttft_egress_only_backlog() -> None:
    """Only src egress has backlog → kv_transfer inflated."""
    from nano_kvrouter.simulator.transfer_model import PerNodeLaneTransferModel
    mc, nc, pnode, dnode, req, no_cache, bw_bps = _make_p4b_base()
    model = PerNodeLaneTransferModel()
    model._egress_available_at["p0"] = 30.0  # src egress backlog = 30ms
    est_with = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=model,
        now=0.0,
    )
    baseline = _baseline_est(mc, pnode, dnode, req, no_cache, bw_bps)
    import pytest as _p
    assert est_with - baseline == _p.approx(30.0, abs=1e-6), (
        f"delta={est_with - baseline}, expected 30.0 from egress backlog"
    )


def test_compute_est_ttft_uses_max_not_sum() -> None:
    """Both lanes have backlog → queue_wait = max(src_egress, dst_ingress), not sum.

    Per plan v4: this test constructs the asymmetric backlog state via
    the public ``request_transfer`` API (no private field access), so
    the formula and the runtime model agree on how state is produced.
    """
    from nano_kvrouter.simulator.transfer_model import PerNodeLaneTransferModel
    mc, nc, pnode, dnode, req, no_cache, bw_bps = _make_p4b_base()
    model = PerNodeLaneTransferModel()
    # p0→d1 cost=20 ⇒ p0.egress.available_at = 20, d1.ingress = 20.
    # p1→d0 cost=35 ⇒ p1.egress = 35, d0.ingress.available_at = 35.
    # Resulting state under test: p0.egress = 20, d0.ingress = 35.
    model.request_transfer("p0", "d1", now=0.0, cost_ms=20.0)
    model.request_transfer("p1", "d0", now=0.0, cost_ms=35.0)
    # queue_wait = max(20, 35) = 35 (NOT 20+35=55)
    est_with = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=model,
        now=0.0,
    )
    baseline = _baseline_est(mc, pnode, dnode, req, no_cache, bw_bps)
    delta = est_with - baseline
    import pytest as _p
    assert delta == _p.approx(35.0, abs=1e-6), (
        f"delta={delta} should be max(20,35)=35, not sum=55"
    )
    assert abs(delta - 55.0) > 1.0, "Got sum(20+35)=55 instead of max=35 — formula is wrong"


def test_compute_est_ttft_clamps_past_backlog_to_zero() -> None:
    """Backlog timestamp in the past (< now) → wait = 0, not negative."""
    from nano_kvrouter.simulator.transfer_model import PerNodeLaneTransferModel
    mc, nc, pnode, dnode, req, no_cache, bw_bps = _make_p4b_base()
    model = PerNodeLaneTransferModel()
    # Inject backlog timestamps that are EARLIER than now=1000.0
    model._egress_available_at["p0"] = 10.0   # backlog is in past
    model._ingress_available_at["d0"] = 5.0   # backlog is in past
    est_with = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=model,
        now=1000.0,
    )
    # With past backlog, result should equal NoopTransferModel at now=1000.0
    from nano_kvrouter.simulator.transfer_model import NoopTransferModel
    baseline_1000 = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=NoopTransferModel(),
        now=1000.0,
    )
    import pytest as _p
    assert est_with == _p.approx(baseline_1000, abs=1e-6), (
        f"est_with={est_with} should equal baseline={baseline_1000} (past backlog clamped to 0)"
    )


def test_compute_est_ttft_noop_is_now_invariant() -> None:
    """NoopTransferModel: kv_transfer term does not change with now value."""
    from nano_kvrouter.simulator.transfer_model import NoopTransferModel
    mc, nc, pnode, dnode, req, no_cache, bw_bps = _make_p4b_base()
    noop = NoopTransferModel()
    est0 = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=noop,
        now=0.0,
    )
    est100 = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=noop,
        now=100.0,
    )
    est1e6 = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw_bps,
        backlog_view=noop,
        now=1_000_000.0,
    )
    import pytest as _p
    assert est0 == _p.approx(est100, abs=1e-9)
    assert est0 == _p.approx(est1e6, abs=1e-9)


def test_compute_est_ttft_noop_kv_transfer_term_matches_pre_p4b_formula() -> None:
    """Under NoopTransferModel the kv_transfer = (tokens * kv_bpt / bw_bps) * 1000 exactly.

    Verified by comparing two runs with different bandwidths: the delta between
    the two estimates equals (100*1024/bw1 - 100*1024/bw2)*1000, confirming
    the kv_transfer term is the only thing that changes.
    """
    from nano_kvrouter.simulator.transfer_model import NoopTransferModel
    mc, nc, pnode, dnode, req, no_cache, _ = _make_p4b_base()
    noop = NoopTransferModel()
    bw1 = 1e9
    bw2 = 2e9
    est1 = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw1,
        backlog_view=noop,
        now=0.0,
    )
    est2 = compute_est_ttft(
        pnode, dnode, req, no_cache,
        kv_bytes_per_token=mc.kv_bytes_per_token,
        bandwidth_bytes_per_s=bw2,
        backlog_view=noop,
        now=0.0,
    )
    n_tokens = len(req.token_ids)
    kv_bpt = mc.kv_bytes_per_token
    expected_delta = (n_tokens * kv_bpt / bw1 - n_tokens * kv_bpt / bw2) * 1000.0
    import pytest as _p
    assert est1 - est2 == _p.approx(expected_delta, abs=1e-9), (
        f"delta={est1-est2} != expected={expected_delta}; "
        "kv_transfer term may not match pre-P4-B formula"
    )

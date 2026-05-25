"""Tests for scheduler/base.py types and the NullCacheQuery test stub."""
from __future__ import annotations

import pytest

from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import (
    CacheLookup,
    CacheQuery,
    SchedulingDecision,
    SchedulingPolicy,
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
    policy = RoundRobinPolicy()
    assert isinstance(policy, SchedulingPolicy)

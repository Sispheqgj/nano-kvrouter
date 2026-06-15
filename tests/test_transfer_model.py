"""Unit tests for TransferModel implementations (P4-A M1).

Hard gate: test_per_node_lane_serializes_4_simultaneous_transfers
directly calls PerNodeLaneTransferModel.request_transfer(); does NOT
go through SimulationEngine or CLI so prefill serialization cannot
mask the contention result.
"""
from __future__ import annotations

from nano_kvrouter.simulator.transfer_model import (
    NoopTransferModel,
    PerNodeLaneTransferModel,
)


# ---------------------------------------------------------------------------
# Hard gate
# ---------------------------------------------------------------------------

def test_per_node_lane_serializes_4_simultaneous_transfers():
    """Same src+dst+now=0 → p0.egress and d0.ingress both serialize."""
    model = PerNodeLaneTransferModel()
    C = 10.0
    results = [
        model.request_transfer("p0", "d0", now=0.0, cost_ms=C)
        for _ in range(4)
    ]
    assert results[0] == (0.0, C)
    assert results[1] == (C, 2 * C)
    assert results[2] == (2 * C, 3 * C)
    assert results[3] == (3 * C, 4 * C)


# ---------------------------------------------------------------------------
# Mandatory unit tests
# ---------------------------------------------------------------------------

def test_disjoint_pairs_run_in_parallel():
    """p0→d0 and p1→d1 started at same now must both return (0.0, 10.0)."""
    model = PerNodeLaneTransferModel()
    r1 = model.request_transfer("p0", "d0", now=0.0, cost_ms=10.0)
    r2 = model.request_transfer("p1", "d1", now=0.0, cost_ms=10.0)
    assert r1 == (0.0, 10.0)
    assert r2 == (0.0, 10.0)


def test_shared_egress_serializes():
    """p0→d0 and p0→d1 contend on p0.egress; second must wait."""
    model = PerNodeLaneTransferModel()
    r1 = model.request_transfer("p0", "d0", now=0.0, cost_ms=10.0)
    r2 = model.request_transfer("p0", "d1", now=0.0, cost_ms=10.0)
    assert r1 == (0.0, 10.0)
    assert r2 == (10.0, 20.0)


def test_shared_ingress_serializes():
    """p0→d0 and p1→d0 contend on d0.ingress; second must wait."""
    model = PerNodeLaneTransferModel()
    r1 = model.request_transfer("p0", "d0", now=0.0, cost_ms=10.0)
    r2 = model.request_transfer("p1", "d0", now=0.0, cost_ms=10.0)
    assert r1 == (0.0, 10.0)
    assert r2 == (10.0, 20.0)


def test_peek_backlog_is_side_effect_free():
    """Multiple peek_backlog calls must not mutate _egress/_ingress_available_at."""
    model = PerNodeLaneTransferModel()
    model.request_transfer("p0", "d0", now=0.0, cost_ms=10.0)
    snap1 = model.peek_backlog("p0")
    snap2 = model.peek_backlog("p0")
    snap3 = model.peek_backlog("p0")
    assert snap1 == snap2 == snap3
    # Internal state unchanged after three peeks.
    assert model._egress_available_at["p0"] == 10.0
    assert model._ingress_available_at["d0"] == 10.0


def test_noop_transfer_model_is_constant_cost():
    """NoopTransferModel returns (now, now+cost) regardless of args."""
    model = NoopTransferModel()
    assert model.request_transfer("p0", "d0", now=0.0, cost_ms=10.0) == (0.0, 10.0)
    assert model.request_transfer("p1", "d1", now=5.0, cost_ms=3.7) == (5.0, 8.7)
    assert model.request_transfer("pX", "dY", now=100.0, cost_ms=0.001) == (100.0, 100.001)
    # peek_backlog always returns zeros.
    assert model.peek_backlog("p0") == {"egress": 0.0, "ingress": 0.0}
    assert model.peek_backlog("d99") == {"egress": 0.0, "ingress": 0.0}


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_per_node_lane_now_after_lane_finish_resets_wait():
    """If now > lane.available_at, no extra wait — start == now."""
    model = PerNodeLaneTransferModel()
    model.request_transfer("p0", "d0", now=0.0, cost_ms=5.0)
    # Lane finishes at 5.0; now=10.0 > 5.0 → no delay.
    r2 = model.request_transfer("p0", "d0", now=10.0, cost_ms=5.0)
    assert r2 == (10.0, 15.0)


def test_per_node_lane_ingress_advances_independently():
    """Egress state on p0 doesn't affect ingress state on d1."""
    model = PerNodeLaneTransferModel()
    model.request_transfer("p0", "d0", now=0.0, cost_ms=20.0)
    # d1 ingress is free; p1 egress is free → (0.0, 5.0)
    r2 = model.request_transfer("p1", "d1", now=0.0, cost_ms=5.0)
    assert r2 == (0.0, 5.0)

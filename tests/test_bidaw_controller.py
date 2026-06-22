"""Tests for simulator/bidaw_controller.py — BidawAdmissionController state machine."""
from __future__ import annotations

import pytest

from nano_kvrouter.request import Request
from nano_kvrouter.simulator.bidaw_controller import BidawAdmissionController, PreparingEntry
from nano_kvrouter.simulator.bidaw_load_model import MultiStreamLoadModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(req_id: str = "r0", token_count: int = 32) -> Request:
    return Request(
        request_id=req_id,
        token_ids=list(range(token_count)),
        prefix_hash="deadbeef",
        expected_output_len=16,
        arrival_time=0.0,
        slo_ttft=600.0,
        slo_tbt=50.0,
    )


def _controller(node_ids: list[str] | None = None) -> BidawAdmissionController:
    return BidawAdmissionController(node_ids or ["d0", "d1", "d2", "d3"])


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


def test_controller_ready_classification() -> None:
    """on_arrive with 0 matched disk blocks returns 'ready'."""
    ctrl = _controller(["d0"])
    verdict = ctrl.on_arrive(_req("r0"), "d0", matched_disk_blocks=0, now_ms=0.0)
    assert verdict == "ready"
    # Ready request must NOT appear in the preparing queue.
    assert ctrl._preparing["d0"] == []


def test_controller_preparing_classification_disk_hit() -> None:
    """on_arrive with ≥1 matched disk block returns 'preparing' and enqueues."""
    ctrl = _controller(["d0"])
    verdict = ctrl.on_arrive(_req("r0"), "d0", matched_disk_blocks=5, now_ms=10.0)
    assert verdict == "preparing"
    assert len(ctrl._preparing["d0"]) == 1
    entry: PreparingEntry = ctrl._preparing["d0"][0]
    assert entry.request_id == "r0"
    assert entry.disk_blocks == 5
    assert entry.arrival_ms == 10.0


def test_controller_single_slot_serializes() -> None:
    """pick_next_to_load returns None when a load is already in-flight (single slot)."""
    ctrl = _controller(["d0"])
    ctrl.on_arrive(_req("r0"), "d0", matched_disk_blocks=2, now_ms=0.0)
    ctrl.on_arrive(_req("r1"), "d0", matched_disk_blocks=3, now_ms=1.0)

    # Pick and start first load.
    req_first = ctrl.pick_next_to_load("d0", now_ms=5.0)
    assert req_first is not None
    ctrl.mark_load_started("d0", req_first.request_id, now_ms=5.0, service_ms=10.0)

    # Slot is now busy — pick must return None.
    assert ctrl.pick_next_to_load("d0", now_ms=5.0) is None

    # After completing first load, pick returns the second.
    ctrl.mark_load_completed("d0", req_first.request_id, now_ms=20.0)
    req_second = ctrl.pick_next_to_load("d0", now_ms=20.0)
    assert req_second is not None
    assert req_second.request_id != req_first.request_id


def test_controller_pick_next_uses_hrrn() -> None:
    """pick_next_to_load picks the entry with the highest HRRN response ratio."""
    ctrl = _controller(["d0"])
    # r_large: disk_blocks=10, arrived at t=0 (waits 200ms at t=200)
    ctrl.on_arrive(_req("r_large"), "d0", matched_disk_blocks=10, now_ms=0.0)
    # r_small: disk_blocks=1, arrived at t=200 (waits 0ms at t=200)
    ctrl.on_arrive(_req("r_small"), "d0", matched_disk_blocks=1, now_ms=200.0)

    # At t=200:
    # r_large: hrrn(200, 10) = 1 + 200/10 = 21
    # r_small: hrrn(0, 1)   = 1 + 0/1   = 1
    # r_large should be picked.
    req = ctrl.pick_next_to_load("d0", now_ms=200.0)
    assert req is not None
    assert req.request_id == "r_large"


def test_controller_pick_next_uses_hrrn_small_wins_equal_wait() -> None:
    """With equal wait, smaller kv_size has higher HRRN ratio → picked first."""
    ctrl = _controller(["d0"])
    ctrl.on_arrive(_req("r_small"), "d0", matched_disk_blocks=1, now_ms=0.0)
    ctrl.on_arrive(_req("r_large"), "d0", matched_disk_blocks=10, now_ms=0.0)

    # At t=100: hrrn(100,1)=101 vs hrrn(100,10)=11. r_small wins.
    req = ctrl.pick_next_to_load("d0", now_ms=100.0)
    assert req is not None
    assert req.request_id == "r_small"


def test_controller_promote_to_ready_on_load_complete() -> None:
    """mark_load_completed returns (request, preparing_wait_ms) and frees the slot."""
    ctrl = _controller(["d0"])
    req0 = _req("r0")
    ctrl.on_arrive(req0, "d0", matched_disk_blocks=3, now_ms=5.0)
    picked = ctrl.pick_next_to_load("d0", now_ms=10.0)
    assert picked is not None
    ctrl.mark_load_started("d0", picked.request_id, now_ms=10.0, service_ms=10.0)

    # Slot is busy.
    assert not ctrl.has_idle_load_slot("d0")

    # Complete the load at t=50.
    returned_req, wait_ms = ctrl.mark_load_completed("d0", picked.request_id, now_ms=50.0)
    assert returned_req.request_id == "r0"
    assert wait_ms == pytest.approx(50.0 - 5.0)  # arrival=5, complete=50

    # Slot is free again.
    assert ctrl.has_idle_load_slot("d0")

    # Preparing queue is empty.
    assert ctrl._preparing["d0"] == []


def test_controller_double_load_start_raises() -> None:
    """mark_load_started raises RuntimeError when slot is already occupied."""
    ctrl = _controller(["d0"])
    ctrl.on_arrive(_req("r0"), "d0", matched_disk_blocks=2, now_ms=0.0)
    ctrl.on_arrive(_req("r1"), "d0", matched_disk_blocks=3, now_ms=0.0)

    r0 = ctrl.pick_next_to_load("d0", now_ms=5.0)
    assert r0 is not None
    ctrl.mark_load_started("d0", r0.request_id, now_ms=5.0, service_ms=10.0)

    # Slot is busy — second call must raise.
    with pytest.raises(RuntimeError, match="already has in-flight"):
        ctrl.mark_load_started("d0", "r1", now_ms=5.0, service_ms=10.0)


def test_controller_get_waiting_ms() -> None:
    """get_waiting_ms returns time since on_arrive."""
    ctrl = _controller(["d0"])
    ctrl.on_arrive(_req("r0"), "d0", matched_disk_blocks=2, now_ms=100.0)
    wait = ctrl.get_waiting_ms("r0", now_ms=350.0)
    assert wait == pytest.approx(250.0)


def test_controller_per_node_isolation() -> None:
    """Single load slot is per-node; d0 slot busy does not block d1."""
    ctrl = _controller(["d0", "d1"])
    ctrl.on_arrive(_req("r0"), "d0", matched_disk_blocks=2, now_ms=0.0)
    ctrl.on_arrive(_req("r1"), "d1", matched_disk_blocks=2, now_ms=0.0)

    r0 = ctrl.pick_next_to_load("d0", now_ms=5.0)
    ctrl.mark_load_started("d0", r0.request_id, now_ms=5.0, service_ms=10.0)

    # d0 slot busy, but d1 slot is still idle.
    assert not ctrl.has_idle_load_slot("d0")
    assert ctrl.has_idle_load_slot("d1")

    r1 = ctrl.pick_next_to_load("d1", now_ms=5.0)
    assert r1 is not None
    assert r1.request_id == "r1"


def test_controller_k2_pick_next_filters_in_flight_entries() -> None:
    """K>1 keeps in-flight entries in _preparing but never re-picks them."""
    ctrl = BidawAdmissionController(
        ["d0"],
        load_model=MultiStreamLoadModel(["d0"], num_streams=2),
    )
    ctrl.on_arrive(_req("r0"), "d0", matched_disk_blocks=1, now_ms=0.0)
    ctrl.on_arrive(_req("r1"), "d0", matched_disk_blocks=2, now_ms=0.0)

    first = ctrl.pick_next_to_load("d0", now_ms=100.0)
    assert first is not None
    ctrl.mark_load_started("d0", first.request_id, now_ms=100.0, service_ms=10.0)

    second = ctrl.pick_next_to_load("d0", now_ms=100.0)
    assert second is not None
    assert second.request_id != first.request_id


def test_controller_k4_peek_preparing_filters_all_in_flight_entries() -> None:
    """peek_preparing_disk_blocks excludes every in-flight request, not just one."""
    ctrl = BidawAdmissionController(
        ["d0"],
        load_model=MultiStreamLoadModel(["d0"], num_streams=4),
    )
    for req_id, disk_blocks in [
        ("a", 2),
        ("b", 3),
        ("c", 5),
        ("queued0", 7),
        ("queued1", 11),
    ]:
        ctrl.on_arrive(_req(req_id), "d0", matched_disk_blocks=disk_blocks, now_ms=0.0)
    for req_id in ["a", "b", "c"]:
        ctrl.mark_load_started("d0", req_id, now_ms=0.0, service_ms=100.0)

    assert ctrl.peek_preparing_disk_blocks("d0") == 18
    assert ctrl.peek_in_flight_disk_blocks("d0") == 10

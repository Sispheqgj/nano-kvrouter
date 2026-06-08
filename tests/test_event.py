"""Tests for simulator/event.py — the Event dataclass and EventType enum."""
from __future__ import annotations

import heapq

import pytest

from nano_kvrouter.simulator.event import Event, EventType


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------


def test_event_type_has_m5a_set() -> None:
    """M5a event set: 10 types — adds KV_TRANSFER_START/COMPLETE for split P/D."""
    assert {e.name for e in EventType} == {
        "REQUEST_ARRIVE",
        "SCHEDULED",
        "PREFILL_START",
        "PREFILL_COMPLETE",
        "KV_TRANSFER_START",
        "KV_TRANSFER_COMPLETE",
        "TOKEN_GENERATED",
        "DECODE_BATCH_STEP",
        "DECODE_COMPLETE",
        "REQUEST_REJECTED",
    }


def test_event_type_values_are_snake_case_strings() -> None:
    """Values are stable string identifiers — used in logs and metrics keys."""
    for member in EventType:
        assert isinstance(member.value, str)
        assert member.value == member.name.lower()


# ---------------------------------------------------------------------------
# Event ordering on heapq
# ---------------------------------------------------------------------------


def test_heap_pops_in_time_order() -> None:
    q: list[Event] = []
    heapq.heappush(q, Event(time=30, seq=0, type=EventType.PREFILL_COMPLETE))
    heapq.heappush(q, Event(time=10, seq=1, type=EventType.REQUEST_ARRIVE))
    heapq.heappush(q, Event(time=20, seq=2, type=EventType.TOKEN_GENERATED))

    popped = [heapq.heappop(q).time for _ in range(3)]
    assert popped == [10, 20, 30]


def test_heap_breaks_ties_by_seq_fifo() -> None:
    """Two events at the same simulated time fire in insertion order."""
    q: list[Event] = []
    heapq.heappush(q, Event(time=5, seq=2, type=EventType.PREFILL_COMPLETE, payload={"id": "third"}))
    heapq.heappush(q, Event(time=5, seq=0, type=EventType.REQUEST_ARRIVE, payload={"id": "first"}))
    heapq.heappush(q, Event(time=5, seq=1, type=EventType.TOKEN_GENERATED, payload={"id": "second"}))

    order = [heapq.heappop(q).payload["id"] for _ in range(3)]
    assert order == ["first", "second", "third"]


def test_payload_does_not_participate_in_comparison() -> None:
    """Identical (time, seq) with different payload dicts must NOT raise.

    If `compare=False` were missing on `payload`, heapq would try to compare
    two dicts and crash with `TypeError: '<' not supported between instances
    of 'dict' and 'dict'`. This test is the canary.
    """
    q: list[Event] = []
    heapq.heappush(q, Event(time=5, seq=0, type=EventType.REQUEST_ARRIVE, payload={"a": 1}))
    # Same time + seq, different payload → must not raise.
    heapq.heappush(q, Event(time=5, seq=0, type=EventType.PREFILL_START, payload={"b": 2}))
    assert len(q) == 2


def test_type_does_not_participate_in_comparison() -> None:
    """Same (time, seq) with different EventType must not raise either."""
    q: list[Event] = []
    heapq.heappush(q, Event(time=5, seq=0, type=EventType.REQUEST_ARRIVE))
    heapq.heappush(q, Event(time=5, seq=0, type=EventType.DECODE_COMPLETE))
    assert len(q) == 2


# ---------------------------------------------------------------------------
# Event defaults and field shape
# ---------------------------------------------------------------------------


def test_event_default_payload_is_independent() -> None:
    """Two events created without explicit payload must not share the dict."""
    a = Event(time=1.0)
    b = Event(time=2.0)
    a.payload["x"] = 1
    assert "x" not in b.payload


def test_event_default_seq_is_zero() -> None:
    ev = Event(time=1.0, type=EventType.REQUEST_ARRIVE)
    assert ev.seq == 0


def test_event_payload_accepts_arbitrary_keys() -> None:
    ev = Event(
        time=10.0,
        seq=1,
        type=EventType.SCHEDULED,
        payload={"request_id": "r-7", "node_id": "n-0", "decision": object()},
    )
    assert ev.payload["request_id"] == "r-7"
    assert ev.payload["node_id"] == "n-0"

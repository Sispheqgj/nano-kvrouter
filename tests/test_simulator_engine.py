"""Tests for SimulationEngine — discrete-event core."""
from __future__ import annotations

import pytest

from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import Event, EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arrive(time: float, **payload) -> Event:
    """Shorthand: REQUEST_ARRIVE event at *time* with optional payload."""
    return Event(time=time, type=EventType.REQUEST_ARRIVE, payload=dict(payload))


def _complete(time: float, **payload) -> Event:
    """Shorthand: PREFILL_COMPLETE event at *time* with optional payload."""
    return Event(time=time, type=EventType.PREFILL_COMPLETE, payload=dict(payload))


# ---------------------------------------------------------------------------
# 1. now() before run
# ---------------------------------------------------------------------------


def test_now_before_run_is_zero() -> None:
    assert SimulationEngine().now() == 0.0


# ---------------------------------------------------------------------------
# 2. Empty queue
# ---------------------------------------------------------------------------


def test_run_empty_queue_returns_immediately() -> None:
    SimulationEngine().run()  # must not raise


# ---------------------------------------------------------------------------
# 3. Single event triggers handler exactly once
# ---------------------------------------------------------------------------


def test_single_event_triggers_handler() -> None:
    eng = SimulationEngine()
    calls: list[tuple[Event, SimulationEngine]] = []

    eng.on(EventType.REQUEST_ARRIVE, lambda ev, e: calls.append((ev, e)))
    ev = _arrive(10.0, req="r0")
    eng.schedule(ev)
    eng.run()

    assert len(calls) == 1
    got_ev, got_eng = calls[0]
    assert got_ev is ev
    assert got_eng is eng


# ---------------------------------------------------------------------------
# 4. now() inside handler == event.time
# ---------------------------------------------------------------------------


def test_now_inside_handler_returns_event_time() -> None:
    eng = SimulationEngine()
    captured: list[float] = []

    eng.on(EventType.REQUEST_ARRIVE, lambda ev, e: captured.append(e.now()))
    eng.schedule(_arrive(42.5))
    eng.run()

    assert captured == [42.5]


# ---------------------------------------------------------------------------
# 5. seq assigned automatically; same-time events process in FIFO order
# ---------------------------------------------------------------------------


def test_seq_assigned_automatically_breaks_ties() -> None:
    eng = SimulationEngine()
    events = []
    for i in range(3):
        # Caller intentionally sets seq=999 — engine must overwrite it.
        ev = Event(time=5.0, seq=999, type=EventType.REQUEST_ARRIVE, payload={"idx": i})
        eng.schedule(ev)
        events.append(ev)

    # Verify seq was overwritten immediately on schedule()
    assert [ev.seq for ev in events] == [0, 1, 2]

    order: list[int] = []
    eng.on(EventType.REQUEST_ARRIVE, lambda ev, e: order.append(ev.payload["idx"]))
    eng.run()

    # FIFO — events processed in the order they were scheduled
    assert order == [0, 1, 2]


# ---------------------------------------------------------------------------
# 6. Multiple handlers called in registration order
# ---------------------------------------------------------------------------


def test_multiple_handlers_called_in_registration_order() -> None:
    eng = SimulationEngine()
    log: list[str] = []

    eng.on(EventType.REQUEST_ARRIVE, lambda ev, e: log.append("first"))
    eng.on(EventType.REQUEST_ARRIVE, lambda ev, e: log.append("second"))
    eng.schedule(_arrive(1.0))
    eng.run()

    assert log == ["first", "second"]


# ---------------------------------------------------------------------------
# 7. Handler can schedule new events (chaining)
# ---------------------------------------------------------------------------


def test_handler_can_schedule_new_event() -> None:
    eng = SimulationEngine()
    seen: list[float] = []

    def arrive_handler(ev: Event, engine: SimulationEngine) -> None:
        seen.append(ev.time)
        # Chain: arrive at t=1 → schedule complete at t=5
        engine.schedule(_complete(5.0))

    def complete_handler(ev: Event, engine: SimulationEngine) -> None:
        seen.append(ev.time)

    eng.on(EventType.REQUEST_ARRIVE, arrive_handler)
    eng.on(EventType.PREFILL_COMPLETE, complete_handler)
    eng.schedule(_arrive(1.0))
    eng.run()

    assert seen == [1.0, 5.0]


# ---------------------------------------------------------------------------
# 8. schedule() with past time raises ValueError
# ---------------------------------------------------------------------------


def test_schedule_past_time_raises() -> None:
    eng = SimulationEngine()

    def bad_handler(ev: Event, engine: SimulationEngine) -> None:
        # now == 100.0; trying to schedule at t=50 should raise
        engine.schedule(_arrive(50.0))

    eng.on(EventType.REQUEST_ARRIVE, bad_handler)
    eng.schedule(_arrive(100.0))

    with pytest.raises(ValueError, match="cannot schedule event at past time"):
        eng.run()


# ---------------------------------------------------------------------------
# 9. Handler exception propagates out of run()
# ---------------------------------------------------------------------------


def test_handler_exception_propagates() -> None:
    eng = SimulationEngine()

    def broken_handler(ev: Event, engine: SimulationEngine) -> None:
        raise RuntimeError("boom")

    eng.on(EventType.REQUEST_ARRIVE, broken_handler)
    eng.schedule(_arrive(1.0))

    with pytest.raises(RuntimeError, match="boom"):
        eng.run()


# ---------------------------------------------------------------------------
# 10. stop() inside handler exits the loop after current event
# ---------------------------------------------------------------------------


def test_stop_inside_handler_exits_loop() -> None:
    eng = SimulationEngine()
    processed: list[float] = []

    def handler(ev: Event, engine: SimulationEngine) -> None:
        processed.append(ev.time)
        engine.stop()  # signal stop after first event

    eng.on(EventType.REQUEST_ARRIVE, handler)
    for t in [1.0, 2.0, 3.0]:
        eng.schedule(_arrive(t))

    eng.run()

    # Only the first event must be processed; stop() fires before the loop
    # checks the queue again.
    assert processed == [1.0]


# ---------------------------------------------------------------------------
# 11. stop() preserves the remaining queue
# ---------------------------------------------------------------------------


def test_stop_preserves_queue() -> None:
    eng = SimulationEngine()

    eng.on(EventType.REQUEST_ARRIVE, lambda ev, e: e.stop())
    for t in [1.0, 2.0, 3.0]:
        eng.schedule(_arrive(t))

    eng.run()

    # Events at t=2.0 and t=3.0 remain in the queue after stop.
    assert len(eng._queue) == 2


# ---------------------------------------------------------------------------
# 12. Handler for unrelated type is not called
# ---------------------------------------------------------------------------


def test_handler_registered_for_unrelated_type_not_called() -> None:
    eng = SimulationEngine()
    called: list[bool] = []

    # Register for PREFILL_COMPLETE but only schedule REQUEST_ARRIVE
    eng.on(EventType.PREFILL_COMPLETE, lambda ev, e: called.append(True))
    eng.schedule(_arrive(1.0))
    eng.run()

    assert called == []

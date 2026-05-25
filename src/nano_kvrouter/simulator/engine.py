"""Single-threaded discrete-event simulation engine.

This module is the execution core of nano-kvrouter: a priority-queue-based
event loop that drives all simulated cluster activity without threads or
asyncio.

Design decisions
----------------
1. **Ordering by (time, seq).** :class:`Event` compares on ``(time, seq)``
   only — ``type`` and ``payload`` are excluded from comparison. ``seq`` is a
   monotonic counter owned by the engine, which guarantees FIFO tie-breaking
   for same-time events and prevents ``heapq`` from ever trying to compare
   ``EventType`` enums or ``dict`` payloads.
2. **Caller seq is always overwritten.** Any ``seq`` value a caller sets in an
   :class:`Event` before passing to :meth:`SimulationEngine.schedule` is
   replaced by the engine's internal counter. Do not rely on caller-supplied
   seq; set ``seq=0`` (the default) and let the engine manage ordering.
3. **Handler exceptions propagate unmodified.** There is no try/except around
   handler calls. A crashing handler crashes the simulation — failing loudly
   is safer than silently skipping events and producing corrupt metrics.
4. **stop() does not clear the queue.** Remaining events are preserved so a
   simulation can be interrupted and inspected mid-run. :meth:`run` can be
   called again after resetting ``_stopped`` externally, though the intended
   usage is a single :meth:`run` per engine instance.
"""
from __future__ import annotations

import heapq
import logging
from collections.abc import Callable

from nano_kvrouter.simulator.event import Event, EventType

logger = logging.getLogger(__name__)

__all__ = ["SimulationEngine"]


class SimulationEngine:
    """Single-threaded discrete-event simulation engine.

    Maintains a min-heap of :class:`~nano_kvrouter.simulator.event.Event`
    objects ordered by ``(time, seq)``. Callers register handlers for specific
    :class:`~nano_kvrouter.simulator.event.EventType` values via :meth:`on`;
    handlers may call :meth:`schedule` to chain further events (e.g.
    ``PREFILL_COMPLETE`` → ``DECODE_STEP``).

    Usage::

        eng = SimulationEngine()
        eng.on(EventType.REQUEST_ARRIVE, my_handler)
        eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={}))
        eng.run()

    The engine is not thread-safe and must be driven from a single thread.
    """

    def __init__(self) -> None:
        self._queue: list[Event] = []
        self._current_time: float = 0.0
        self._seq_counter: int = 0
        self._handlers: dict[EventType, list[Callable[[Event, SimulationEngine], None]]] = {}
        self._stopped: bool = False

    def now(self) -> float:
        """Return the current simulated time in milliseconds.

        Returns:
            ``0.0`` before :meth:`run` starts. Inside a handler, returns the
            ``time`` of the event currently being processed.
        """
        return self._current_time

    def schedule(self, event: Event) -> None:
        """Add *event* to the simulation queue.

        The caller's ``event.seq`` value is **always overwritten** with the
        engine's internal monotonic counter before the event is pushed onto the
        heap. Do not rely on any ``seq`` you set in the :class:`Event`
        constructor — the engine is the sole authority on insertion order.

        Args:
            event: Event to enqueue. ``event.time`` must be >= :meth:`now`.

        Raises:
            ValueError: If ``event.time < self.now()`` (scheduling into the
                past is a simulation invariant violation).
        """
        if event.time < self._current_time:
            raise ValueError(
                f"cannot schedule event at past time {event.time} < now {self._current_time}"
            )
        event.seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._queue, event)

    def on(
        self,
        event_type: EventType,
        handler: Callable[[Event, SimulationEngine], None],
    ) -> None:
        """Subscribe *handler* to *event_type*.

        Multiple handlers may be registered for the same type; they are invoked
        in registration order. Registering the same callable twice causes it to
        be called twice per matching event — no deduplication is applied.

        Args:
            event_type: The :class:`EventType` to subscribe to.
            handler: ``(event, engine) -> None``. May call :meth:`schedule` to
                chain new events. Must not catch and swallow exceptions.
        """
        self._handlers.setdefault(event_type, []).append(handler)

    def stop(self) -> None:
        """Signal the engine to exit after the current event's handlers finish.

        Sets ``_stopped = True``. The remaining events in the queue are **not**
        cleared — they survive for post-run inspection or a future :meth:`run`
        call. The current event's remaining handlers still execute to
        completion before the loop checks the flag.
        """
        self._stopped = True

    def run(self) -> None:
        """Process events until the queue is empty or :meth:`stop` is called.

        For each event popped from the heap:

        1. Advances :meth:`now` to ``event.time``.
        2. Calls every registered handler for the event's type in registration
           order.

        If a handler raises, the exception propagates immediately out of
        :meth:`run` — remaining handlers for the same event are skipped, but
        the queue itself is unchanged.

        :meth:`run` does **not** reset :attr:`_stopped`. If you need to resume
        after a :meth:`stop`, either create a new :class:`SimulationEngine` or
        reset ``_stopped`` manually before calling :meth:`run` again.
        """
        while self._queue and not self._stopped:
            ev = heapq.heappop(self._queue)
            self._current_time = ev.time
            logger.debug(
                "t=%.3fms seq=%d %s payload=%s",
                ev.time,
                ev.seq,
                ev.type.name,
                ev.payload,
            )
            for handler in self._handlers.get(ev.type, []):
                handler(ev, self)

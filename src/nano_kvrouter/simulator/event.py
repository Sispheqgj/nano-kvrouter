"""Simulation event types and the Event dataclass.

The simulator is a single-threaded, discrete-event system. Every state
change in the cluster — a request arriving, a prefill finishing, a
decode step happening — is represented as an :class:`Event` placed on
the global heap. The :class:`SimulationEngine` pops events in
time-then-insertion order, invokes the registered handlers, and lets
the handlers schedule further events (which is how time advances).

This module defines the event vocabulary; the engine itself lives in
``simulator.engine`` and handlers are registered by schedulers,
nodes, and metrics collectors at startup.

Design notes
------------
* **Event set (8 types).** Cross-tier / cross-node transfers
  and rebalance ticks will be added when the corresponding features
  land (Llumnix migration, Mooncake transfer-aware scoring).
* **dict payload, not per-event dataclasses.** Handlers reach into
  ``ev.payload["request_id"]`` etc. We trade type-safety for the
  ability to evolve a payload shape without breaking the Event class.
  If payloads grow complex we can revisit per-event dataclasses.
* **Sequence number for stable ordering.** ``heapq`` would otherwise
  fall back to comparing ``EventType`` (an enum, would work) and then
  ``payload`` (a dict, would crash) when two events share the same
  ``time``. The ``seq`` field is assigned by the engine on insertion
  to break ties in FIFO order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["EventType", "Event"]


class EventType(Enum):
    """All event kinds the simulator currently understands.

    Lifecycle of a normal request (M2 continuous-batching model):

    ``REQUEST_ARRIVE`` → ``SCHEDULED`` → ``PREFILL_START`` →
    ``PREFILL_COMPLETE`` → [joins node decode batch] →
    ``TOKEN_GENERATED`` × N (per-request metrics signal, one per batch tick) →
    ``DECODE_COMPLETE``

    Node-level decode pipeline (one per active node):
    ``DECODE_BATCH_STEP`` → ``DECODE_BATCH_STEP`` → … (until node idle)

    Rejected requests skip everything after ``SCHEDULED`` and emit
    ``REQUEST_REJECTED`` instead.

    Future additions (not in scope for the current milestone):
    ``KV_TRANSFER_COMPLETE`` (Llumnix migration), ``REBALANCE_TICK``
    (periodic rebalance trigger).
    """

    REQUEST_ARRIVE = "request_arrive"
    SCHEDULED = "scheduled"
    PREFILL_START = "prefill_start"
    PREFILL_COMPLETE = "prefill_complete"
    TOKEN_GENERATED = "token_generated"  # per-request metrics signal; renamed from DECODE_STEP
    DECODE_BATCH_STEP = "decode_batch_step"
    DECODE_COMPLETE = "decode_complete"
    REQUEST_REJECTED = "request_rejected"


@dataclass(slots=True, order=True)
class Event:
    """A single time-stamped event on the simulation queue.

    Events are ordered first by ``time`` and then by ``seq`` so that
    two events scheduled for the exact same simulated time fire in
    the order they were inserted. ``type`` and ``payload`` are
    excluded from comparison — otherwise ``heapq`` would attempt to
    compare ``EventType`` enums and ``dict`` payloads, which is
    fragile (dict comparison raises ``TypeError``).

    Attributes:
        time: Simulated time at which the event fires, in milliseconds
            since the start of the run. The engine pops events in
            non-decreasing ``time`` order.
        seq: Monotonic insertion counter assigned by
            ``SimulationEngine.schedule``. Callers should leave this
            as the default; the engine will overwrite it before
            pushing onto the heap.
        type: One of :class:`EventType`.
        payload: Event-specific data. Conventional keys (not enforced):

            * ``request_id`` — present on every request-scoped event
            * ``node_id`` — present on ``SCHEDULED`` and lifecycle events
            * ``decision`` — :class:`SchedulingDecision` on ``SCHEDULED``
            * ``reason`` — rejection reason string on ``REQUEST_REJECTED``
            * ``step_index`` — decode step number on ``TOKEN_GENERATED``
    """

    time: float
    seq: int = 0
    type: EventType = field(compare=False, default=EventType.REQUEST_ARRIVE)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)

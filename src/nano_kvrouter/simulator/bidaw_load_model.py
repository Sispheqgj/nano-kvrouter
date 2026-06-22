"""Bidaw KV-load slot models."""
from __future__ import annotations

from typing import Protocol

SlotEntry = tuple[str, float, int]

__all__ = [
    "BidawLoadModel",
    "MultiStreamLoadModel",
    "SingleSlotLoadModel",
]


class BidawLoadModel(Protocol):
    """Pluggable model for per-node Bidaw KV-load slots."""

    def has_capacity(self, decode_node_id: str) -> bool:
        """Return True when *decode_node_id* has at least one idle load slot."""
        ...

    def start_load(
        self,
        decode_node_id: str,
        request_id: str,
        now_ms: float,
        service_ms: float,
        disk_blocks: int,
    ) -> None:
        """Claim a slot for *request_id*.

        Raises RuntimeError if no slot is idle or if *request_id* is already
        in-flight on the same node.
        """
        ...

    def complete_load(
        self,
        decode_node_id: str,
        request_id: str,
        now_ms: float,
    ) -> None:
        """Free the slot occupied by *request_id*."""
        ...

    def slot_residuals_ms(self, decode_node_id: str, now_ms: float) -> list[float]:
        """Return one residual per slot, with 0.0 for idle slots."""
        ...

    def in_flight_request_ids(self, decode_node_id: str) -> frozenset[str]:
        """Return request IDs currently occupying load slots on this node."""
        ...

    def in_flight_disk_blocks(self, decode_node_id: str) -> int:
        """Return the sum of disk blocks across all active slots on this node."""
        ...

    @property
    def num_streams(self) -> int:
        """Number of load slots per decode node."""
        ...


class SingleSlotLoadModel:
    """Single-slot load model matching the M3 controller's observable behavior."""

    def __init__(self, decode_node_ids: list[str]) -> None:
        self._in_flight: dict[str, SlotEntry | None] = {
            node_id: None for node_id in decode_node_ids
        }

    @property
    def num_streams(self) -> int:
        return 1

    def has_capacity(self, decode_node_id: str) -> bool:
        return self._in_flight.get(decode_node_id) is None

    def start_load(
        self,
        decode_node_id: str,
        request_id: str,
        now_ms: float,
        service_ms: float,
        disk_blocks: int,
    ) -> None:
        if request_id in self.in_flight_request_ids(decode_node_id):
            raise RuntimeError(
                f"BidawLoadModel: request {request_id!r} already in-flight "
                f"on node {decode_node_id!r}; cannot start a second load"
            )
        current = self._in_flight.get(decode_node_id)
        if current is not None:
            raise RuntimeError(
                f"BidawLoadModel: decode_node {decode_node_id!r} already has "
                f"in-flight load for request {current[0]!r}; cannot start load "
                f"for {request_id!r}"
            )
        self._in_flight[decode_node_id] = (request_id, now_ms + service_ms, disk_blocks)

    def complete_load(
        self,
        decode_node_id: str,
        request_id: str,
        now_ms: float,
    ) -> None:
        entry = self._in_flight.get(decode_node_id)
        if entry is None or entry[0] != request_id:
            raise RuntimeError(
                f"BidawLoadModel: request {request_id!r} is not in-flight "
                f"on node {decode_node_id!r}"
            )
        self._in_flight[decode_node_id] = None

    def slot_residuals_ms(self, decode_node_id: str, now_ms: float) -> list[float]:
        entry = self._in_flight.get(decode_node_id)
        if entry is None:
            return [0.0]
        return [max(0.0, entry[1] - now_ms)]

    def in_flight_request_ids(self, decode_node_id: str) -> frozenset[str]:
        entry = self._in_flight.get(decode_node_id)
        return frozenset() if entry is None else frozenset({entry[0]})

    def in_flight_disk_blocks(self, decode_node_id: str) -> int:
        entry = self._in_flight.get(decode_node_id)
        return 0 if entry is None else entry[2]


class MultiStreamLoadModel:
    """Fixed-width multi-stream load model with deterministic slot assignment."""

    def __init__(self, decode_node_ids: list[str], *, num_streams: int) -> None:
        if num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {num_streams}")
        self._num_streams = num_streams
        self._slots: dict[str, list[SlotEntry | None]] = {
            node_id: [None] * num_streams for node_id in decode_node_ids
        }

    @property
    def num_streams(self) -> int:
        return self._num_streams

    def has_capacity(self, decode_node_id: str) -> bool:
        return any(entry is None for entry in self._slots[decode_node_id])

    def start_load(
        self,
        decode_node_id: str,
        request_id: str,
        now_ms: float,
        service_ms: float,
        disk_blocks: int,
    ) -> None:
        if request_id in self.in_flight_request_ids(decode_node_id):
            raise RuntimeError(
                f"BidawLoadModel: request {request_id!r} already in-flight "
                f"on node {decode_node_id!r}; cannot start a second load"
            )
        slots = self._slots[decode_node_id]
        for idx, entry in enumerate(slots):
            if entry is None:
                slots[idx] = (request_id, now_ms + service_ms, disk_blocks)
                return
        raise RuntimeError(
            f"BidawLoadModel: decode_node {decode_node_id!r} has no idle load slot"
        )

    def complete_load(
        self,
        decode_node_id: str,
        request_id: str,
        now_ms: float,
    ) -> None:
        slots = self._slots[decode_node_id]
        for idx, entry in enumerate(slots):
            if entry is not None and entry[0] == request_id:
                slots[idx] = None
                return
        raise RuntimeError(
            f"BidawLoadModel: request {request_id!r} is not in-flight "
            f"on node {decode_node_id!r}"
        )

    def slot_residuals_ms(self, decode_node_id: str, now_ms: float) -> list[float]:
        return [
            0.0 if entry is None else max(0.0, entry[1] - now_ms)
            for entry in self._slots[decode_node_id]
        ]

    def in_flight_request_ids(self, decode_node_id: str) -> frozenset[str]:
        return frozenset(
            entry[0]
            for entry in self._slots[decode_node_id]
            if entry is not None
        )

    def in_flight_disk_blocks(self, decode_node_id: str) -> int:
        return sum(
            entry[2]
            for entry in self._slots[decode_node_id]
            if entry is not None
        )

"""BidawAdmissionController — dual-queue state machine for I/O-aware request gating.

This module implements the admission controller that holds a ready queue
and a preparing queue per decode node. It is a pure state machine — no
event scheduling. The cli wiring (_wire_bidaw_branch in cli.py) calls into
this controller and schedules KV_LOAD_START / KV_LOAD_COMPLETE events
based on the controller's decisions.

Metadata boundary:
KV_LOAD_COMPLETE marks the request "ready" in the controller. The cli wiring
then attempts physical metadata promotion (disk→CPU) through CacheManager.
This controller remains a pure state machine and never mutates BlockPool
directly.

Single load slot invariant:
At most one in-flight KV_LOAD per decode node at a time. This serializes
disk loads per-node. Cross-node loads run in parallel (each node has its
own slot).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

from nano_kvrouter.config import BandwidthConfig, ModelConfig
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.bidaw import hrrn_priority

logger = logging.getLogger(__name__)

__all__ = ["BidawAdmissionController", "PreparingEntry"]


@dataclass(slots=True)
class PreparingEntry:
    """One request waiting in a decode node's preparing queue.

    Attributes:
        request_id: Matches Request.request_id.
        decode_node_id: The decode node this request is assigned to.
        disk_blocks: Matched disk block count (HRRN denominator).
        arrival_ms: Simulated time when on_arrive was called.
        request: Original Request object (retained for PREFILL_START wiring).
    """

    request_id: str
    decode_node_id: str
    disk_blocks: int
    arrival_ms: float
    request: Request


class BidawAdmissionController:
    """Dual-queue admission controller: classifies requests as ready vs preparing.

    One instance is owned by _wire_simulator when scheduler="bidaw". The
    5 existing schedulers never interact with this controller.

    Public methods:
        on_arrive — classify and enqueue (ready or preparing).
        pick_next_to_load — HRRN pick from preparing queue (slot must be idle).
        mark_load_started — claim the single in-flight load slot.
        mark_load_completed — free the slot and return (request, wait_ms).
        has_idle_load_slot — predicate for the single-slot invariant.
        get_waiting_ms — helper for KV_LOAD_START payload construction.
    """

    def __init__(
        self,
        decode_node_ids: Sequence[str],
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
        affinity_enabled: bool = False,
    ) -> None:
        self._model_cfg = model_config
        self._bw_cfg = bandwidth_config
        self.affinity_enabled = affinity_enabled
        # Per-node preparing queue.
        self._preparing: dict[str, list[PreparingEntry]] = {
            nid: [] for nid in decode_node_ids
        }
        # Per-node in-flight load slot: request_id or None (idle).
        self._in_flight_load: dict[str, str | None] = {
            nid: None for nid in decode_node_ids
        }
        # Per-node in-flight load finish time, used for deterministic A2
        # projected-wait estimates. None means the load slot is idle.
        self._in_flight_finish_ms: dict[str, float | None] = {
            nid: None for nid in decode_node_ids
        }
        # Arrival time registry for HRRN waiting_ms calculation.
        self._arrival_ms_by_req: dict[str, float] = {}
        # Session affinity is bounded by the distinct session IDs in the workload.
        self._session_to_decode: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_arrive(
        self,
        request: Request,
        decode_node_id: str,
        matched_disk_blocks: int,
        now_ms: float,
    ) -> Literal["ready", "preparing"]:
        """Classify request, push onto the right queue, return verdict.

        Args:
            request: The incoming request.
            decode_node_id: Decode node assigned by BidawPolicy.
            matched_disk_blocks: Matched disk block count from cache lookup.
            now_ms: Current simulated time (ms) for HRRN accounting.

        Returns:
            "ready" if no disk blocks matched; "preparing" otherwise.
        """
        self._arrival_ms_by_req[request.request_id] = now_ms
        if matched_disk_blocks == 0:
            return "ready"
        entry = PreparingEntry(
            request_id=request.request_id,
            decode_node_id=decode_node_id,
            disk_blocks=matched_disk_blocks,
            arrival_ms=now_ms,
            request=request,
        )
        self._preparing[decode_node_id].append(entry)
        return "preparing"

    def pick_next_to_load(self, decode_node_id: str, now_ms: float) -> Request | None:
        """HRRN pick from the preparing queue; None if empty or slot busy.

        Uses response_ratio = 1 + waiting_ms / max(1, disk_blocks).
        Picks the entry with the highest ratio (longest-waited relative
        to its KV size). Does not modify state — call mark_load_started
        to claim the slot.

        Args:
            decode_node_id: Decode node whose preparing queue to scan.
            now_ms: Current simulated time for waiting_ms calculation.

        Returns:
            The best-priority Request, or None.
        """
        if not self.has_idle_load_slot(decode_node_id):
            return None
        entries = self._preparing.get(decode_node_id, [])
        if not entries:
            return None
        best = max(
            entries,
            key=lambda e: hrrn_priority(now_ms - e.arrival_ms, e.disk_blocks),
        )
        return best.request

    def mark_load_started(
        self,
        decode_node_id: str,
        req_id: str,
        now_ms: float,
        service_ms: float,
    ) -> None:
        """Claim the in-flight load slot for req_id on decode_node_id.

        Args:
            decode_node_id: Decode node whose slot to claim.
            req_id: Request starting the load.
            now_ms: Current simulated time.
            service_ms: Load service time in milliseconds.

        Raises:
            RuntimeError: If the slot is already occupied (single-slot invariant
                violation). The caller must check has_idle_load_slot first.
        """
        current = self._in_flight_load.get(decode_node_id)
        if current is not None:
            raise RuntimeError(
                f"BidawAdmissionController: decode_node {decode_node_id!r} "
                f"already has in-flight load for request {current!r}; "
                f"cannot start load for {req_id!r}"
            )
        self._in_flight_load[decode_node_id] = req_id
        self._in_flight_finish_ms[decode_node_id] = now_ms + service_ms

    def mark_load_completed(
        self, decode_node_id: str, req_id: str, now_ms: float
    ) -> tuple[Request, float]:
        """Free the load slot and return (request, preparing_wait_ms).

        Removes the entry from the preparing queue and clears the in-flight
        slot so pick_next_to_load can pick the next candidate.

        Args:
            decode_node_id: Decode node whose slot to free.
            req_id: Request completing the load.
            now_ms: Current simulated time for wait measurement.

        Returns:
            (request, preparing_wait_ms) — preparing_wait_ms is the total
            time from on_arrive to this call.

        Raises:
            StopIteration: If req_id is not found in the preparing queue.
        """
        entries = self._preparing[decode_node_id]
        entry = next(e for e in entries if e.request_id == req_id)
        entries.remove(entry)
        self._in_flight_load[decode_node_id] = None
        self._in_flight_finish_ms[decode_node_id] = None
        preparing_wait_ms = now_ms - entry.arrival_ms
        self._arrival_ms_by_req.pop(req_id, None)
        return entry.request, preparing_wait_ms

    def has_idle_load_slot(self, decode_node_id: str) -> bool:
        """Return True if decode_node_id has no in-flight disk load.

        Args:
            decode_node_id: Decode node to check.

        Returns:
            True when the single load slot is free.
        """
        return self._in_flight_load.get(decode_node_id) is None

    def get_waiting_ms(self, req_id: str, now_ms: float) -> float:
        """Return how long req_id has waited since on_arrive.

        Used by cli wiring to populate KV_LOAD_START payload's
        preparing_wait_ms field.

        Args:
            req_id: Request ID to look up.
            now_ms: Current simulated time.

        Returns:
            Time since on_arrive was called, in ms.
        """
        arrival = self._arrival_ms_by_req.get(req_id, now_ms)
        return now_ms - arrival

    def peek_preparing_disk_blocks(self, decode_node_id: str) -> int:
        """Return queued preparing disk blocks, excluding the active load."""
        in_flight_req = self._in_flight_load.get(decode_node_id)
        return sum(
            entry.disk_blocks
            for entry in self._preparing.get(decode_node_id, [])
            if entry.request_id != in_flight_req
        )

    def peek_in_flight_disk_blocks(self, decode_node_id: str) -> int:
        """Return disk block count for the request currently loading."""
        req_id = self._in_flight_load.get(decode_node_id)
        if req_id is None:
            return 0
        for entry in self._preparing.get(decode_node_id, []):
            if entry.request_id == req_id:
                return entry.disk_blocks
        return 0

    def peek_projected_preparing_wait_ms(
        self,
        decode_node_id: str,
        my_disk_blocks: int,
        now_ms: float,
    ) -> float:
        """Estimate single-slot wait plus own load service for a disk-hit request.

        The formula mirrors the cli.py KV_LOAD service path exactly:
        ``disk_blocks * block_bytes * (1/cpu_to_disk + 1/gpu_to_cpu) * 1000``
        — both hops Disk→CPU and CPU→GPU, matching CacheManager's
        transfer_cost_ms estimate.
        """
        if my_disk_blocks == 0:
            return 0.0
        if self._model_cfg is None or self._bw_cfg is None:
            raise ValueError(
                "BidawAdmissionController needs model_config and bandwidth_config "
                "to project preparing wait"
            )

        finish_ms = self._in_flight_finish_ms.get(decode_node_id)
        residual_ms = 0.0 if finish_ms is None else max(0.0, finish_ms - now_ms)
        queued_disk_blocks = self.peek_preparing_disk_blocks(decode_node_id)
        block_bytes = self._model_cfg.block_size * self._model_cfg.kv_bytes_per_token
        service_ms_per_block = (
            block_bytes
            * (1.0 / self._bw_cfg.cpu_to_disk + 1.0 / self._bw_cfg.gpu_to_cpu)
            * 1000.0
        )
        return residual_ms + (queued_disk_blocks + my_disk_blocks) * service_ms_per_block

    def peek_session_affinity(self, session_id: str) -> str | None:
        """Return pinned decode node for *session_id*, if any."""
        return self._session_to_decode.get(session_id)

    def commit_session_affinity(self, session_id: str, decode_node_id: str) -> None:
        """Pin *session_id* to *decode_node_id* for later routing decisions."""
        if not self.affinity_enabled:
            return
        self._session_to_decode[session_id] = decode_node_id

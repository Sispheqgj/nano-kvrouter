"""KV transfer cost / contention models (P4-A M1).

Two implementations:

* NoopTransferModel — constant-cost passthrough, byte-identical to
  the pre-P4-A baseline. Selected when bandwidth.contention_model == "none".
* PerNodeLaneTransferModel — per-node egress + ingress lane queue.
  Selected when bandwidth.contention_model == "per_node_lane".

The Protocol defines the interface both implementations satisfy; cli.py
constructs one implementation at _run_one time and passes it through
_wire_simulator so the rest of the code never branches on
contention_model.
"""
from __future__ import annotations

from typing import Protocol


class TransferModel(Protocol):
    """Pluggable cost / contention model for KV transfer.

    Two implementations live in this module:

    * NoopTransferModel — constant cost, byte-identical to pre-P4-A
      behavior. Always selected when bandwidth.contention_model == "none".
    * PerNodeLaneTransferModel — per-node egress + ingress lane queue.
      Selected when bandwidth.contention_model == "per_node_lane".

    Implementations must be deterministic given the call sequence
    (request_transfer is the only state-mutating method).
    """

    def request_transfer(
        self,
        src_node_id: str,
        dst_node_id: str,
        now: float,
        cost_ms: float,
    ) -> tuple[float, float]:
        """Reserve lanes and return (start_time, finish_time).

        cost_ms is the *service* cost computed from
        kv_bytes / bandwidth.gpu_to_gpu. Implementations may delay
        start_time past `now` if upstream resources are still busy.
        """

    def peek_backlog(self, node_id: str) -> dict[str, float]:
        """Return {"egress": float, "ingress": float} available_at.

        MUST be side-effect-free: repeated calls before any
        request_transfer between them return identical results.

        Consumed by ``scheduler.base.compute_est_ttft`` (called by all
        5 schedulers since P4-B M1) to include current lane queue wait
        in the KV-transfer cost estimate. Conductor uses the estimate
        for its SLO gate; E2 uses it as its ``run_cost`` term in the
        3-objective score; the other three schedulers embed it in
        ``SchedulingDecision.estimated_ttft_ms`` but do not gate
        routing on it.
        """


class NoopTransferModel:
    """Constant-cost passthrough.

    Returns (now, now + cost_ms) every call. peek_backlog always
    returns {"egress": 0.0, "ingress": 0.0}. With this model,
    KV_TRANSFER_COMPLETE event timing is byte-identical to the
    pre-P4-A baseline.
    """

    def request_transfer(self, src_node_id: str, dst_node_id: str, now: float, cost_ms: float) -> tuple[float, float]:
        return (now, now + cost_ms)

    def peek_backlog(self, node_id: str) -> dict[str, float]:
        return {"egress": 0.0, "ingress": 0.0}


class PerNodeLaneTransferModel:
    """Per-node egress + ingress lane queue.

    Each node has one egress lane (used when it's the src of a
    transfer) and one ingress lane (used when it's the dst). A
    transfer occupies BOTH lanes simultaneously for [start, finish):

        start = max(now, egress.available_at[src], ingress.available_at[dst])
        finish = start + cost_ms

    Both lanes' available_at are then advanced to finish. Disjoint
    (src, dst) pairs run in parallel; transfers sharing a src or dst
    serialize. This matches Mooncake's "per-node KV transfer
    throughput is the bottleneck" semantics without modeling RDMA
    link topology.
    """

    def __init__(self) -> None:
        self._egress_available_at: dict[str, float] = {}
        self._ingress_available_at: dict[str, float] = {}

    def request_transfer(self, src_node_id: str, dst_node_id: str, now: float, cost_ms: float) -> tuple[float, float]:
        start = max(
            now,
            self._egress_available_at.get(src_node_id, 0.0),
            self._ingress_available_at.get(dst_node_id, 0.0),
        )
        finish = start + cost_ms
        self._egress_available_at[src_node_id] = finish
        self._ingress_available_at[dst_node_id] = finish
        return (start, finish)

    def peek_backlog(self, node_id: str) -> dict[str, float]:
        return {
            "egress": self._egress_available_at.get(node_id, 0.0),
            "ingress": self._ingress_available_at.get(node_id, 0.0),
        }

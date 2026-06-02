"""Round-robin scheduling policy — baseline scheduler.

Rotates requests across nodes in stable index order without consulting
cache state or current load. Every node receives an equal share of
traffic in the long run, making this the natural baseline against which
cache-aware and load-aware policies are benchmarked.
"""
from __future__ import annotations

import logging
from typing import Sequence

from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheQuery, SchedulingDecision

logger = logging.getLogger(__name__)

__all__ = ["RoundRobinPolicy"]


class RoundRobinPolicy:
    """Round-robin scheduler: rotate across nodes in sequential order.

    The cursor advances by one on every :meth:`schedule` call and wraps
    modulo the current cluster size, so the distribution is uniform even
    across bursty arrival patterns.

    Design notes:
        * Cache state is intentionally ignored — the policy assumes a cold
          prefill on the chosen node, which overestimates TTFT but is
          correct for a baseline that has no cache information.
        * ``prefill_node`` and ``decode_node`` are always the same (combined
          deployment). P/D separation is out of scope for this baseline.
        * Cursor state is instance-bound; two ``RoundRobinPolicy`` instances
          maintain independent counters.
    """

    def __init__(self) -> None:
        self._cursor: int = 0

    def schedule(
        self,
        request: Request,
        nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> SchedulingDecision:
        """Pick the next node in round-robin order.

        Args:
            request: The incoming request. Only ``len(token_ids)`` is used
                for TTFT estimation; ``prefix_hash`` is ignored.
            nodes: Live cluster nodes in stable order. An empty sequence
                causes an early rejection.
            cache: Not consulted. RoundRobin always assumes zero cached
                tokens on the chosen node, so smarter policies will
                generally produce lower TTFT estimates.

        Returns:
            :class:`~nano_kvrouter.scheduler.base.SchedulingDecision` with
            ``prefill_node == decode_node``. Rejected with
            ``reject_reason="no_nodes_available"`` when *nodes* is empty.
        """
        if not nodes:
            return SchedulingDecision(
                prefill_node=None,
                decode_node=None,
                estimated_ttft_ms=0.0,
                estimated_tbt_ms=0.0,
                reject_reason="no_nodes_available",
            )

        idx = self._cursor % len(nodes)
        self._cursor += 1
        node = nodes[idx]

        prompt_len = len(request.token_ids)
        # Assume cold prefill: cached_tokens=0 because we never query cache.
        prefill_ms = node.estimate_prefill_time(prompt_len, cached_tokens=0)
        # Queue wait reflects real back-pressure even without cache-awareness.
        queue_ms = node.queue_wait_time(prompt_len, request.expected_output_len)
        # first_tick: time of the first decode step once the request is admitted
        # (batch grows by 1). Included in TTFT so all schedulers are comparable.
        first_tick_ms = node.estimate_decode_time(len(node.running_requests) + 1)
        ttft_ms = prefill_ms + queue_ms + first_tick_ms
        tbt_ms = first_tick_ms

        logger.debug(
            "RoundRobin: request %s → node %s (cursor_idx=%d)",
            request.request_id,
            node.node_id,
            idx,
        )

        return SchedulingDecision(
            prefill_node=node.node_id,
            decode_node=node.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
        )

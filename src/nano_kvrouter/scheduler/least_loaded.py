"""Least-loaded scheduling policy — baseline load-aware scheduler.

Picks the node with the lowest current load fraction, breaking ties
by stable index order in the ``nodes`` sequence. Like RoundRobin, it
never consults cache state and always assumes a cold prefill.
"""
from __future__ import annotations

import logging
from typing import Sequence

from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheQuery, SchedulingDecision

logger = logging.getLogger(__name__)

__all__ = ["LeastLoadedPolicy"]


class LeastLoadedPolicy:
    """Least-loaded scheduler: pick the node with the lowest current_load().

    This is a baseline load-aware policy — it makes routing decisions
    purely from ``node.current_load()`` without examining cache state.

    Design notes:
        * **Cache-blind**: the ``cache`` argument is accepted to satisfy
          the :class:`~nano_kvrouter.scheduler.base.SchedulingPolicy`
          Protocol but **no cache methods are ever called**. TTFT is
          estimated assuming a cold prefill (``cached_tokens=0``), which
          overestimates cost but is correct for a policy that has no
          cache information.
        * **Tie-breaking**: when multiple nodes share the minimum load,
          the one with the lowest index in the ``nodes`` sequence is
          chosen. This guarantees determinism without sorting by node_id,
          which could break if IDs are not lexicographically ordered.
        * **Combined deployment**: ``prefill_node == decode_node`` always.
          P/D separation is out of scope for this baseline.
        * **No internal state**: unlike RoundRobin, no cursor is needed.
          Each call is stateless, making the policy safe to share between
          simulations.
    """

    def __init__(self) -> None:
        pass

    def schedule(
        self,
        request: Request,
        nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> SchedulingDecision:
        """Pick the least-loaded node for *request*.

        Args:
            request: The incoming request. Only ``len(token_ids)`` is used
                for TTFT estimation.
            nodes: Live cluster nodes in stable order. Ties in
                ``current_load()`` are broken by position in this sequence
                (first wins). An empty sequence causes an early rejection.
            cache: Not consulted. LeastLoaded always assumes zero cached
                tokens (cold prefill). No methods on this object are called.

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

        # min() with a key is stable: ties preserve the original index order.
        node = min(nodes, key=lambda n: n.current_load())

        prompt_len = len(request.token_ids)
        # M3 chunked-prefill TTFT: assume cold prefill (cached_tokens=0).
        bs_hint = len(node.running_requests) + 1
        prefill_ms = node.estimate_prefill_time(
            prompt_len, cached_tokens=0, batch_size_hint=bs_hint
        )
        queue_ms = node.queue_wait_time(prompt_len, request.expected_output_len)
        first_tick_ms = node.estimate_decode_time(bs_hint)
        ttft_ms = prefill_ms + queue_ms + first_tick_ms
        tbt_ms = first_tick_ms

        logger.debug(
            "LeastLoaded: request %s → node %s (load=%.3f)",
            request.request_id,
            node.node_id,
            node.current_load(),
        )

        return SchedulingDecision(
            prefill_node=node.node_id,
            decode_node=node.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
        )

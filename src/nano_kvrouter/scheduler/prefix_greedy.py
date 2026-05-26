"""Prefix-greedy scheduling policy — SGLang RadixAttention-style cache-aware scheduler.

Routes each request to the node with the most cached prefix tokens, falling
back to least-loaded selection when the best match falls below
``min_hit_ratio`` (cold-start / low-hit scenario).
"""
from __future__ import annotations

import logging
from typing import Sequence

from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheQuery, SchedulingDecision

logger = logging.getLogger(__name__)

__all__ = ["PrefixGreedyPolicy"]


class PrefixGreedyPolicy:
    """SGLang RadixAttention-style cache-aware scheduler.

    Routes each request to the node that maximises prefix cache reuse.
    Falls back to least-loaded selection when no node has a meaningful
    hit (best ``matched_tokens / prompt_length < min_hit_ratio``).

    Selection priority (highest to lowest):

    1. Highest ``matched_tokens`` from ``cache.lookup_all()``.
    2. Cold-start fallback: if best hit ratio < ``min_hit_ratio``, select
       by ``current_load()`` (LeastLoaded semantics) instead.
    3. Ties in cache hit broken by ``current_load()`` (lower wins).
    4. Ties in both cache hit and load broken by stable index order
       (first node in the ``nodes`` sequence wins).

    Design notes:
        * No SLO early-rejection — this is a baseline cache-aware policy,
          not a Conductor-style admission controller.
        * ``prefill_node == decode_node`` (combined deployment).
        * The fallback LeastLoaded logic is implemented inline to avoid
          depending on a sibling module that may not exist yet.
    """

    def __init__(self, min_hit_ratio: float = 0.25) -> None:
        """Create a PrefixGreedyPolicy.

        Args:
            min_hit_ratio: When the best-hitting node's matched_tokens /
                prompt length is below this threshold, the policy falls
                back to LeastLoaded behavior (pick lowest current_load).
                Default 0.25 — chosen as a conservative "any meaningful
                cache benefit" cutoff; tune via experiments.

        Raises:
            ValueError: If ``min_hit_ratio`` is outside [0.0, 1.0].
        """
        if not 0.0 <= min_hit_ratio <= 1.0:
            raise ValueError(f"min_hit_ratio must be in [0, 1], got {min_hit_ratio}")
        self._min_hit_ratio = min_hit_ratio

    def schedule(
        self,
        request: Request,
        nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> SchedulingDecision:
        """Pick the node that maximises cache reuse, or least-loaded on cold start.

        Args:
            request: The incoming request.
            nodes: Live cluster nodes in stable order. An empty sequence
                causes an early rejection.
            cache: Read-only handle for per-node prefix-hit details.

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

        lookups = cache.lookup_all(request)
        prompt_len = len(request.token_ids)

        # Collect (node, matched_tokens) in stable index order.
        hits = [(n, lookups[n.node_id].matched_tokens) for n in nodes]
        max_hit = max(matched for _, matched in hits)

        # Fall back to LeastLoaded when prompt is empty or best hit is too weak.
        use_fallback = (prompt_len == 0) or (max_hit / prompt_len < self._min_hit_ratio)

        if use_fallback:
            # Inline LeastLoaded: stable min by current_load (ties → first by index).
            chosen = min(nodes, key=lambda n: n.current_load())
        else:
            # Cache-affinity: highest hit first, then min load, ties by index.
            best = [n for n, matched in hits if matched == max_hit]
            chosen = min(best, key=lambda n: n.current_load())

        chosen_matched = lookups[chosen.node_id].matched_tokens
        prefill_ms = chosen.estimate_prefill_time(prompt_len, cached_tokens=chosen_matched)
        queue_ms = chosen.queue_wait_time()
        ttft_ms = prefill_ms + queue_ms
        tbt_ms = chosen.estimate_decode_time(len(chosen.running_requests) + 1)

        logger.debug(
            "PrefixGreedy: request %s → node %s (matched=%d/%d, fallback=%s)",
            request.request_id,
            chosen.node_id,
            chosen_matched,
            prompt_len,
            use_fallback,
        )

        return SchedulingDecision(
            prefill_node=chosen.node_id,
            decode_node=chosen.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
        )

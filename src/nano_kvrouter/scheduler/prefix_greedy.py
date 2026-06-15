"""Prefix-greedy scheduling policy — SGLang RadixAttention-style cache-aware scheduler.

M5a split P/D: cache lives on the decode pool, so cache-affinity drives
decode_node selection. Prefill_node selection is purely load-based
(min current_load), tie-broken by node_id. Cold-start fallback applies
to decode_node when the best hit ratio is below ``min_hit_ratio``.
"""
from __future__ import annotations

import logging
from typing import Sequence

from nano_kvrouter.config import BandwidthConfig, ModelConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import (
    CacheLookup,
    CacheQuery,
    SchedulingDecision,
    TransferBacklogView,
    compute_est_tbt,
    compute_est_ttft,
)

logger = logging.getLogger(__name__)

__all__ = ["PrefixGreedyPolicy"]


class PrefixGreedyPolicy:
    """SGLang RadixAttention-style cache-aware scheduler for split P/D.

    Selection (M5a split):

    * **decode_node**: highest ``matched_tokens`` from ``cache.lookup_all()``;
      falls back to least-loaded when the best hit ratio is below
      ``min_hit_ratio``. Cache lives on the decode pool only.
    * **prefill_node**: lowest ``current_load()`` — the prefill side
      has no cache knowledge in M5a. (M5b will introduce cache-aware
      prefill routing.)

    Tie-breaking is deterministic via ``(metric, node_id)``.

    Design notes:
        * No SLO early-rejection — this is a baseline cache-aware policy,
          not a Conductor-style admission controller.
        * In combined deployments (``prefill_nodes is decode_nodes``)
          ``prefill_node`` may still differ from ``decode_node`` because
          the two selectors use different keys.
    """

    def __init__(
        self,
        min_hit_ratio: float = 0.25,
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
        backlog_view: TransferBacklogView,
    ) -> None:
        """Create a PrefixGreedyPolicy.

        Args:
            min_hit_ratio: When the best-hitting decode node's
                ``matched_tokens / prompt_length`` is below this
                threshold the policy falls back to LeastLoaded behavior
                on the decode pool. Default 0.25.
            model_config: Forwarded to ``compute_est_ttft``.
            bandwidth_config: Forwarded to ``compute_est_ttft``.
            backlog_view: Read-only KV transfer lane backlog view (P4-B).

        Raises:
            ValueError: If ``min_hit_ratio`` is outside [0.0, 1.0].
        """
        if not 0.0 <= min_hit_ratio <= 1.0:
            raise ValueError(f"min_hit_ratio must be in [0, 1], got {min_hit_ratio}")
        self._min_hit_ratio = min_hit_ratio
        self._model_cfg = model_config if model_config is not None else ModelConfig()
        self._bw_cfg = bandwidth_config if bandwidth_config is not None else BandwidthConfig()
        self._backlog_view = backlog_view

    def schedule(
        self,
        request: Request,
        prefill_nodes: Sequence[MockEngineNode],
        decode_nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
        *,
        now: float,
    ) -> SchedulingDecision:
        """Pick decode_node by cache affinity and prefill_node by load.

        Returns:
            :class:`SchedulingDecision`. Rejected with
            ``reject_reason="no_nodes_available"`` when either pool is
            empty.
        """
        if not prefill_nodes or not decode_nodes:
            return SchedulingDecision(
                prefill_node=None,
                decode_node=None,
                estimated_ttft_ms=0.0,
                estimated_tbt_ms=0.0,
                reject_reason="no_nodes_available",
            )

        lookups = cache.lookup_all(request)
        prompt_len = len(request.token_ids)

        # decode_node: cache-affinity-first.
        hits = [(n, lookups.get(n.node_id, CacheLookup(0, {}, 0.0)).matched_tokens)
                for n in decode_nodes]
        max_hit = max(matched for _, matched in hits)
        use_fallback = (prompt_len == 0) or (max_hit / prompt_len < self._min_hit_ratio)

        if use_fallback:
            decode = min(decode_nodes, key=lambda n: (n.current_load(), n.node_id))
        else:
            best = [n for n, matched in hits if matched == max_hit]
            decode = min(best, key=lambda n: (n.current_load(), n.node_id))

        # prefill_node: load-only (M5a; cache lives on decode).
        prefill = min(prefill_nodes, key=lambda n: (n.current_load(), n.node_id))

        decode_match = lookups.get(
            decode.node_id, CacheLookup(0, {}, 0.0)
        )
        ttft_ms = compute_est_ttft(
            prefill,
            decode,
            request,
            decode_match,
            kv_bytes_per_token=self._model_cfg.kv_bytes_per_token,
            bandwidth_bytes_per_s=self._bw_cfg.gpu_to_gpu,
            backlog_view=self._backlog_view,
            now=now,
        )
        tbt_ms = compute_est_tbt(decode)

        logger.debug(
            "PrefixGreedy: request %s → prefill=%s decode=%s (matched=%d/%d, fallback=%s)",
            request.request_id,
            prefill.node_id,
            decode.node_id,
            decode_match.matched_tokens,
            prompt_len,
            use_fallback,
        )

        return SchedulingDecision(
            prefill_node=prefill.node_id,
            decode_node=decode.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
        )

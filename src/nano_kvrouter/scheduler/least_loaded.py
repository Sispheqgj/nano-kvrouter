"""Least-loaded scheduling policy — baseline load-aware scheduler.

Picks the node with the lowest current load fraction in each pool,
breaking ties by ``(load, node_id)`` so cross-run reproducibility holds.
Like RoundRobin, it never consults cache state and assumes the prefill
side is cold.
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
    compute_est_tbt,
    compute_est_ttft,
)

logger = logging.getLogger(__name__)

__all__ = ["LeastLoadedPolicy"]


class LeastLoadedPolicy:
    """Least-loaded scheduler: pick the lowest-load node in each pool.

    M5a split P/D: prefill and decode pools are independently scored
    by ``current_load()``; the same algorithm runs twice.

    Design notes:
        * **Cache-blind**: the ``cache`` argument is accepted to satisfy
          the :class:`~nano_kvrouter.scheduler.base.SchedulingPolicy`
          Protocol; ``compute_est_ttft`` uses the decode-side cache hit
          (if any) but the selection itself never reads cache.
        * **Tie-breaking**: ``(current_load(), node_id)`` so ties are
          broken deterministically by node_id lexicographic order. This
          stays reproducible across runs regardless of
          ``PYTHONHASHSEED``.
        * **Combined deployment**: when ``prefill_nodes is decode_nodes``
          the same load function selects the same node for both, so
          ``prefill_node == decode_node``.
        * **No internal state**: stateless; safe to share between
          simulations.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
    ) -> None:
        self._model_cfg = model_config if model_config is not None else ModelConfig()
        self._bw_cfg = bandwidth_config if bandwidth_config is not None else BandwidthConfig()

    def schedule(
        self,
        request: Request,
        prefill_nodes: Sequence[MockEngineNode],
        decode_nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> SchedulingDecision:
        """Pick the least-loaded node in each pool.

        Args:
            request: The incoming request.
            prefill_nodes: Live prefill-pool nodes in stable order.
                Ties broken by ``(load, node_id)``.
            decode_nodes: Live decode-pool nodes in stable order.
            cache: Decode-pool cache view (only used by
                ``compute_est_ttft``).

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

        prefill = min(prefill_nodes, key=lambda n: (n.current_load(), n.node_id))
        decode = min(decode_nodes, key=lambda n: (n.current_load(), n.node_id))

        try:
            decode_match = cache.lookup(request, decode.node_id)
        except KeyError:
            decode_match = CacheLookup(
                matched_tokens=0, matched_blocks_by_tier={}, transfer_cost_ms=0.0
            )

        ttft_ms = compute_est_ttft(
            prefill,
            decode,
            request,
            decode_match,
            kv_bytes_per_token=self._model_cfg.kv_bytes_per_token,
            bandwidth_bytes_per_s=self._bw_cfg.gpu_to_gpu,
        )
        tbt_ms = compute_est_tbt(decode)

        logger.debug(
            "LeastLoaded: request %s → prefill=%s(load=%.3f) decode=%s(load=%.3f)",
            request.request_id,
            prefill.node_id,
            prefill.current_load(),
            decode.node_id,
            decode.current_load(),
        )

        return SchedulingDecision(
            prefill_node=prefill.node_id,
            decode_node=decode.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
        )

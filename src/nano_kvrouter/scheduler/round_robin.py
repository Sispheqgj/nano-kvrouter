"""Round-robin scheduling policy — baseline scheduler.

Rotates requests across nodes in stable index order without consulting
cache state or current load. Every node receives an equal share of
traffic in the long run, making this the natural baseline against which
cache-aware and load-aware policies are benchmarked.
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

__all__ = ["RoundRobinPolicy"]


class RoundRobinPolicy:
    """Round-robin scheduler: rotate across nodes in sequential order.

    M5a split P/D: keeps two independent cursors so that prefill and
    decode pools advance separately. The two cursors always advance by
    one per :meth:`schedule` call so the long-run distribution is
    uniform across each pool independently. In combined deployments
    (``prefill_nodes is decode_nodes``) both cursors stay in sync
    yielding ``prefill_node == decode_node``.

    Design notes:
        * Cache state is intentionally ignored — the policy assumes a
          cold prefill on the chosen prefill_node, which overestimates
          TTFT but is correct for a baseline that has no cache
          information. ``compute_est_ttft`` still consults the
          decode-side CacheLookup so it gets a cheap cache benefit if
          the same prompt previously landed on the same decode_node.
        * Cursor state is instance-bound; two ``RoundRobinPolicy``
          instances maintain independent counters.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
    ) -> None:
        self._prefill_cursor: int = 0
        self._decode_cursor: int = 0
        self._model_cfg = model_config if model_config is not None else ModelConfig()
        self._bw_cfg = bandwidth_config if bandwidth_config is not None else BandwidthConfig()

    def schedule(
        self,
        request: Request,
        prefill_nodes: Sequence[MockEngineNode],
        decode_nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> SchedulingDecision:
        """Pick the next node in round-robin order from each pool.

        Args:
            request: The incoming request.
            prefill_nodes: Live prefill-pool nodes in stable order.
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

        p_idx = self._prefill_cursor % len(prefill_nodes)
        d_idx = self._decode_cursor % len(decode_nodes)
        self._prefill_cursor += 1
        self._decode_cursor += 1
        prefill = prefill_nodes[p_idx]
        decode = decode_nodes[d_idx]

        # Cache lookup is on the decode node (where KV cache lives).
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
            "RoundRobin: request %s → prefill=%s decode=%s",
            request.request_id,
            prefill.node_id,
            decode.node_id,
        )

        return SchedulingDecision(
            prefill_node=prefill.node_id,
            decode_node=decode.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
        )

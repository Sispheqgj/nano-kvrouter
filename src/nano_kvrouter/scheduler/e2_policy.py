"""E2 (exploit-explore) scheduling policy — Preble ICLR'25 §4.

Routes requests by minimising a three-term joint cost over decode-pool
candidates:

    e2_score(decode_node, request) =
        w_historical × historical_load_proxy(decode_node)
      + w_eviction   × eviction_cost(decode_node, request)
      + w_run        × run_cost(prefill_node, decode_node, request)

where every term is expressed in milliseconds so the weights are
dimensionally consistent and directly comparable.

M5a split P/D: cache lives on the decode pool, so e2_score is computed
over decode_nodes. Prefill_node is selected independently by lowest
``current_load()`` because the prefill side has no cache state in M5a
(M5b will introduce transfer-aware prefill selection).
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

__all__ = ["E2Policy"]


class E2Policy:
    """Preble-style exploit-explore scheduler (ICLR'25 §4).

    Selects the decode_node that minimises a three-term weighted cost:

    * **historical_load** (w_historical):
      ``current_load() × prompt_len × prefill_cost_per_token_ms``
    * **eviction_cost** (w_eviction): equivalent re-prefill time for
      blocks that would need to be evicted on the decode_node.
    * **run_cost** (w_run): ``compute_est_ttft(...)`` — total predicted
      time from arrival to first token (Mooncake style, includes KV
      transfer cost and lane queue wait).

    Ties broken by ``(score, node_id)`` so cross-run reproducibility holds.

    Design notes:
        * Cold-start behaviour: with cold caches and equal load the
          minimum is tie-broken by node_id → first node lexicographic.
        * Prefill_node is chosen separately by lowest load.
        * No SLO rejection — admission control belongs in
          MooncakeConductor.
    """

    def __init__(
        self,
        w_historical: float = 1.0,
        w_eviction: float = 1.0,
        w_run: float = 1.0,
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
        backlog_view: TransferBacklogView,
    ) -> None:
        for name, val in [
            ("w_historical", w_historical),
            ("w_eviction", w_eviction),
            ("w_run", w_run),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

        self._w_h = w_historical
        self._w_e = w_eviction
        self._w_r = w_run
        self._model_cfg: ModelConfig = model_config if model_config is not None else ModelConfig()
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
        bs = self._model_cfg.block_size
        ppt = self._model_cfg.prefill_cost_per_token_ms
        total_blocks = prompt_len // bs

        # prefill_node: load-only (M5a; cache lives on decode).
        prefill = min(prefill_nodes, key=lambda n: (n.current_load(), n.node_id))

        def _score(decode: MockEngineNode) -> tuple[float, str]:
            match = lookups.get(decode.node_id, CacheLookup(0, {}, 0.0))
            matched = match.matched_tokens
            new_blocks = max(0, total_blocks - matched // bs)
            hist = decode.current_load() * prompt_len * ppt
            shortage = max(0, new_blocks - cache.free_blocks(decode.node_id, "gpu"))
            evict = shortage * bs * ppt
            run = compute_est_ttft(
                prefill,
                decode,
                request,
                match,
                kv_bytes_per_token=self._model_cfg.kv_bytes_per_token,
                bandwidth_bytes_per_s=self._bw_cfg.gpu_to_gpu,
                backlog_view=self._backlog_view,
                now=now,
            )
            score = self._w_h * hist + self._w_e * evict + self._w_r * run
            return (score, decode.node_id)

        decode = min(decode_nodes, key=_score)
        decode_match = lookups.get(decode.node_id, CacheLookup(0, {}, 0.0))

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
            "E2Policy: request %s → prefill=%s decode=%s (matched=%d/%d)",
            request.request_id,
            prefill.node_id,
            decode.node_id,
            decode_match.matched_tokens,
            prompt_len,
        )

        return SchedulingDecision(
            prefill_node=prefill.node_id,
            decode_node=decode.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
        )

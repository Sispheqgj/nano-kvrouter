"""Mooncake Conductor scheduler (FAST'25 Best Paper §4).

M5a split P/D: three-objective scoring over the decode pool +
SLO early rejection. Prefill_node is selected independently by lowest
``current_load()`` because the prefill side has no cache state in M5a.

    score(decode_node) = α × cache_benefit(decode_node)
                       − β × load_penalty(decode_node)
                       − γ × transfer_penalty(decode_node)

In M5a ``transfer_penalty`` is still always ``0.0`` (the CacheLookup
field is unused on decode side because KV is already there). M5b will
attach a KV-transfer penalty term keyed on ``bandwidth.gpu_to_gpu``.
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

__all__ = ["MooncakeConductor"]


class MooncakeConductor:
    """Mooncake Conductor scheduler (FAST'25 Best Paper §4) — split P/D.

    Selects the decode_node that maximises a three-objective score and
    applies SLO early rejection based on the chosen-node prediction.

    Design notes:
        * **SLO 6A**: rejection is based on the single max-score
          decode_node; no fallback to alternates.
        * **Prefill_node**: lowest ``current_load()`` — independent of
          decode scoring in M5a.
        * **Ties**: ``(−score, node_id)`` so ties resolve by node_id
          deterministically.
        * **transfer_penalty** (roadmap):

          transfer_penalty:
              Currently ``0.0`` in M5a — ``CacheManager.lookup`` returns
              ``transfer_cost_ms=0.0`` because all matched blocks are on GPU
              HBM (GPU-only tier, no cross-node hit in M5a).

          ``transfer_penalty`` in the score formula is always ``0.0`` in
          M5a — ``CacheManager.lookup`` returns ``transfer_cost_ms=0.0``
          because all matched blocks are on GPU HBM (GPU-only tier, no
          cross-node hit possible in M5a).

          The field is reserved for **M6 multi-tier HiCache**: when a
          prefix hit resides on a CPU or Disk tier instead of GPU HBM,
          ``CacheManager.lookup`` will return a non-zero
          ``transfer_cost_ms`` reflecting the bandwidth-bound cost to
          load those blocks back into GPU before prefill starts.  The
          conductor can then trade off "use this distant cache hit" vs
          "recompute from scratch on a lightly-loaded node".

          **Important**: ``transfer_penalty`` here is the *within-node
          tier-migration cost* (CPU → GPU or Disk → CPU → GPU).  It is
          **not** the prefill_node → decode_node KV transfer cost —
          that one is already baked into ``est_ttft`` via
          ``compute_est_ttft`` (the ``kv_transfer`` term) and feeds the
          SLO gate.  Do not add it to the score as well (double count).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
    ) -> None:
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError(
                f"Weights must be non-negative; got alpha={alpha}, beta={beta}, gamma={gamma}"
            )
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._model_cfg = model_config if model_config is not None else ModelConfig()
        self._bw_cfg = bandwidth_config if bandwidth_config is not None else BandwidthConfig()

    def schedule(
        self,
        request: Request,
        prefill_nodes: Sequence[MockEngineNode],
        decode_nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
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
        ppt = self._model_cfg.prefill_cost_per_token_ms

        # prefill_node: load-only (M5a; cache lives on decode).
        prefill = min(prefill_nodes, key=lambda n: (n.current_load(), n.node_id))

        def score_key(decode: MockEngineNode) -> tuple[float, str]:
            match = lookups.get(decode.node_id, CacheLookup(0, {}, 0.0))
            matched = match.matched_tokens
            cache_benefit = matched * ppt
            load_penalty = (
                decode.current_load() * prompt_len * ppt
                + decode.queue_wait_time(prompt_len, request.expected_output_len)
            )
            transfer_penalty = match.transfer_cost_ms  # 0 in M5a
            score = (
                self._alpha * cache_benefit
                - self._beta * load_penalty
                - self._gamma * transfer_penalty
            )
            # min on (-score, node_id) so higher score wins; ties resolve by node_id.
            return (-score, decode.node_id)

        decode = min(decode_nodes, key=score_key)
        decode_match = lookups.get(decode.node_id, CacheLookup(0, {}, 0.0))

        est_ttft = compute_est_ttft(
            prefill,
            decode,
            request,
            decode_match,
            kv_bytes_per_token=self._model_cfg.kv_bytes_per_token,
            bandwidth_bytes_per_s=self._bw_cfg.gpu_to_gpu,
        )
        est_tbt = compute_est_tbt(decode)

        # SLO early rejection (design 6A) — based on the chosen decode pair.
        if est_ttft > request.slo_ttft:
            logger.debug(
                "Conductor: reject %s ttft=%.1f > slo=%.1f",
                request.request_id, est_ttft, request.slo_ttft,
            )
            return SchedulingDecision(
                prefill_node=None,
                decode_node=None,
                estimated_ttft_ms=est_ttft,
                estimated_tbt_ms=est_tbt,
                reject_reason="ttft_slo_exceeded",
            )
        if est_tbt > request.slo_tbt:
            logger.debug(
                "Conductor: reject %s tbt=%.1f > slo=%.1f",
                request.request_id, est_tbt, request.slo_tbt,
            )
            return SchedulingDecision(
                prefill_node=None,
                decode_node=None,
                estimated_ttft_ms=est_ttft,
                estimated_tbt_ms=est_tbt,
                reject_reason="tbt_slo_exceeded",
            )

        logger.debug(
            "Conductor: request %s → prefill=%s decode=%s (matched=%d/%d)",
            request.request_id,
            prefill.node_id,
            decode.node_id,
            decode_match.matched_tokens,
            prompt_len,
        )

        return SchedulingDecision(
            prefill_node=prefill.node_id,
            decode_node=decode.node_id,
            estimated_ttft_ms=est_ttft,
            estimated_tbt_ms=est_tbt,
        )

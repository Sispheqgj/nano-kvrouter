"""Mooncake Conductor scheduler (FAST'25 Best Paper §4).

Three-objective scoring + SLO early rejection.  Routes each request to the
node that maximises a composite score, then immediately rejects the request
if even the best node cannot satisfy the SLO targets.
"""
from __future__ import annotations

import logging
from typing import Sequence

from nano_kvrouter.config import ModelConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheQuery, SchedulingDecision

logger = logging.getLogger(__name__)

__all__ = ["MooncakeConductor"]


class MooncakeConductor:
    """Mooncake Conductor scheduler (Mooncake FAST'25 Best Paper, §4).

    Selects the node that maximises a three-objective score, then applies
    SLO early rejection based on the selected node's predicted TTFT / TBT:

        score(node) = α × cache_benefit(node)
                    − β × load_penalty(node)
                    − γ × transfer_penalty(node)

    where:

    * ``cache_benefit(n) = matched_tokens × prefill_cost_per_token_ms``
      — reusable computation; larger is better.
    * ``load_penalty(n)  = current_load() × prompt_len × prefill_cost_per_token_ms
                           + queue_wait_time(prompt_len, expected_output_len)``
      — scheduling pressure from existing traffic; smaller is better.
    * ``transfer_penalty(n) = CacheLookup.transfer_cost_ms``
      — cost of migrating KV blocks across tiers / nodes to this node.
      **Always 0.0 in v1 (GPU-only).**  The γ parameter is retained for
      forward compatibility when multi-tier v2 is introduced.

    **SLO early rejection (design choice 6A)**:

    1. Choose ``chosen = max(nodes, key=score)`` — the globally best node.
    2. Compute ``est_ttft`` and ``est_tbt`` for *that* node.
    3. If either prediction exceeds the request's SLO, reject immediately.
       No fallback to another node is attempted (design choice 6A) — the
       request is rejected even if an alternative node could serve it.

    Rejection carries ``estimated_ttft_ms`` / ``estimated_tbt_ms`` set to
    the actual predicted values (not 0) so ``MetricsCollector`` can record
    the *reason* for rejection and track SLO violation magnitudes.

    Design notes:
        * ``prefill_node == decode_node`` (combined deployment, no P/D split).
        * ``transfer_penalty`` is always 0 in v1; γ does not affect routing.
        * Ties in score broken by position in the *nodes* sequence (Python's
          ``max()`` returns the first element with the maximum value).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        *,
        model_config: ModelConfig | None = None,
    ) -> None:
        """Create a MooncakeConductor.

        Args:
            alpha: Weight on the cache_benefit term. Higher values give
                stronger preference to nodes with warm prefix cache.
            beta: Weight on the load_penalty term. Higher values give
                stronger preference to lightly loaded nodes.
            gamma: Weight on the transfer_penalty term. In v1 GPU-only
                mode this term is always 0 because
                ``CacheLookup.transfer_cost_ms`` is 0; the parameter is
                kept for forward compatibility with multi-tier v2.
            model_config: Source of ``prefill_cost_per_token_ms`` for the
                cache_benefit and load_penalty terms.  Defaults to
                ``ModelConfig()`` (default constants) when ``None``.

        Raises:
            ValueError: If any weight (alpha, beta, gamma) is negative.
        """
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError(
                f"Weights must be non-negative; got alpha={alpha}, beta={beta}, gamma={gamma}"
            )
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._model_cfg = model_config if model_config is not None else ModelConfig()

    def schedule(
        self,
        request: Request,
        nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> SchedulingDecision:
        """Select the highest-scoring node, then apply SLO early rejection.

        Args:
            request: The incoming request. ``slo_ttft`` and ``slo_tbt`` are
                used for the post-selection rejection check.
            nodes: Live cluster nodes in stable order. Empty sequence →
                immediate rejection with ``"no_nodes_available"``.
            cache: Read-only handle for per-node prefix-hit details and
                transfer costs.

        Returns:
            :class:`~nano_kvrouter.scheduler.base.SchedulingDecision` with
            ``prefill_node == decode_node`` on acceptance.  On rejection,
            ``prefill_node`` and ``decode_node`` are ``None`` but
            ``estimated_ttft_ms`` / ``estimated_tbt_ms`` carry the
            predicted values for diagnostic accounting.
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
        ppt = self._model_cfg.prefill_cost_per_token_ms

        def score(n: MockEngineNode) -> float:
            matched = lookups[n.node_id].matched_tokens
            cache_benefit = matched * ppt
            load_penalty = n.current_load() * prompt_len * ppt + n.queue_wait_time(prompt_len, request.expected_output_len)
            transfer_penalty = lookups[n.node_id].transfer_cost_ms
            return (
                self._alpha * cache_benefit
                - self._beta * load_penalty
                - self._gamma * transfer_penalty
            )

        # max() is stable: ties → first node in the sequence (by index).
        chosen = max(nodes, key=score)

        chosen_matched = lookups[chosen.node_id].matched_tokens
        est_ttft = (
            chosen.estimate_prefill_time(prompt_len, cached_tokens=chosen_matched)
            + chosen.queue_wait_time(prompt_len, request.expected_output_len)
        )
        est_tbt = chosen.estimate_decode_time(len(chosen.running_requests) + 1)

        # SLO early rejection (design 6A): based on the best node only —
        # no fallback to other nodes even if they could serve the request.
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
            "Conductor: request %s → node %s (score=%.3f, matched=%d/%d)",
            request.request_id,
            chosen.node_id,
            score(chosen),
            chosen_matched,
            prompt_len,
        )

        return SchedulingDecision(
            prefill_node=chosen.node_id,
            decode_node=chosen.node_id,
            estimated_ttft_ms=est_ttft,
            estimated_tbt_ms=est_tbt,
        )

"""E2 (exploit-explore) scheduling policy — Preble ICLR'25 §4.

Routes requests by minimising a three-term joint cost over all nodes:

    e2_score(node, request) =
        w_historical × historical_load_proxy(node)
      + w_eviction   × eviction_cost(node, request)
      + w_run        × run_cost(node, request)

where every term is expressed in milliseconds so the weights are
dimensionally consistent and directly comparable.

v1 simplifications (documented in DESIGN.md §3.4):
    * historical_load is proxied by current_load() — no rolling window.
    * eviction_cost is expressed as "equivalent re-prefill time" for the
      blocks that would need to be evicted.
    * No SLO early-rejection — that is the Conductor's responsibility.
"""
from __future__ import annotations

import logging
from typing import Sequence

from nano_kvrouter.config import ModelConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheQuery, SchedulingDecision

logger = logging.getLogger(__name__)

__all__ = ["E2Policy"]


class E2Policy:
    """Preble-style exploit-explore scheduler (ICLR'25 §4).

    Selects the node that minimises a three-term weighted cost:

    * **historical_load** (w_historical): proxy for past utilisation.
      ``current_load() × prompt_len × prefill_cost_per_token_ms`` — a
      load-proportional penalty that discourages routing to busy nodes.

    * **eviction_cost** (w_eviction): penalty when the node's GPU cache
      is too full to absorb the new prompt without eviction. Expressed
      as equivalent re-prefill time for the evicted tokens so it is
      dimensionally compatible with run_cost.

    * **run_cost** (w_run): actual cost of running the request.
      ``estimate_prefill_time(prompt_len, cached_tokens) +
      queue_wait_time()`` — lower when there is a cache hit and a short
      queue.

    Ties (equal score) are broken by stable index order in the ``nodes``
    sequence (Python's ``min()`` is stable).

    Design notes:
        * Cold-start behaviour: when the cache is empty everywhere,
          historical and eviction terms are 0, and run_cost is identical
          across nodes, so the first node in the sequence is picked. No
          explicit cold-start fallback is needed; this also avoids
          masking the policy's true scoring logic.
        * ``prefill_node == decode_node`` always (combined deployment).
        * No SLO rejection — E2Policy is a pure routing policy; admission
          control belongs in MooncakeConductor.
    """

    def __init__(
        self,
        w_historical: float = 1.0,
        w_eviction: float = 1.0,
        w_run: float = 1.0,
        *,
        model_config: ModelConfig | None = None,
    ) -> None:
        """Initialise E2Policy with optional weight tuning.

        Args:
            w_historical: Weight on the historical load proxy term.
                Higher values bias routing away from busy nodes.
            w_eviction: Weight on the eviction cost term.
                Higher values bias routing toward nodes with free cache.
            w_run: Weight on the run cost (prefill + queue) term.
                Higher values bias toward cache-warm, short-queue nodes.
            model_config: Source of ``block_size`` and
                ``prefill_cost_per_token_ms`` for historical and eviction
                calculations. Defaults to ``ModelConfig()`` (16-token
                blocks, default prefill cost) when ``None``.

        Raises:
            ValueError: If any weight is negative.
        """
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

    def schedule(
        self,
        request: Request,
        nodes: Sequence[MockEngineNode],
        cache: CacheQuery,
    ) -> SchedulingDecision:
        """Pick the node with the lowest E2 cost for *request*.

        Args:
            request: The incoming request.
            nodes: Live cluster nodes in stable order. Ties in E2 score
                are broken by position in this sequence (first wins).
                An empty sequence causes an early rejection.
            cache: Read-only handle used for prefix-hit details
                (``lookup_all``) and free-block counts (``free_blocks``).

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
        bs = self._model_cfg.block_size
        ppt = self._model_cfg.prefill_cost_per_token_ms
        needed_blocks = prompt_len // bs

        def _score(n: MockEngineNode) -> float:
            matched = lookups[n.node_id].matched_tokens
            hist = n.current_load() * prompt_len * ppt
            shortage = max(0, needed_blocks - cache.free_blocks(n.node_id, "gpu"))
            evict = shortage * bs * ppt
            run = n.estimate_prefill_time(prompt_len, cached_tokens=matched) + n.queue_wait_time()
            return self._w_h * hist + self._w_e * evict + self._w_r * run

        chosen = min(nodes, key=_score)

        chosen_matched = lookups[chosen.node_id].matched_tokens
        ttft_ms = (
            chosen.estimate_prefill_time(prompt_len, cached_tokens=chosen_matched)
            + chosen.queue_wait_time()
        )
        tbt_ms = chosen.estimate_decode_time(len(chosen.running_requests) + 1)

        logger.debug(
            "E2Policy: request %s → node %s (score=%.3f, matched=%d/%d)",
            request.request_id,
            chosen.node_id,
            _score(chosen),
            chosen_matched,
            prompt_len,
        )

        return SchedulingDecision(
            prefill_node=chosen.node_id,
            decode_node=chosen.node_id,
            estimated_ttft_ms=ttft_ms,
            estimated_tbt_ms=tbt_ms,
        )

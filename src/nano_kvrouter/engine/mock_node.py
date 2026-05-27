from __future__ import annotations

import logging

from nano_kvrouter.config import ModelConfig, NodeConfig

logger = logging.getLogger(__name__)


class MockEngineNode:
    """Latency-model-only stand-in for a real GPU inference engine.

    Holds no tensors. The node tracks two queues — `running_requests`
    (admitted, bounded by `node_config.capacity`) and `queue` (waiting) —
    and answers latency questions the scheduler asks before placing a
    request: prefill time, decode-step time, current load, queue wait.

    Design notes:
        * All time values are milliseconds and *simulated* — never wall
          clock. The event loop converts them into future events.
        * Capacity-driven admission lives here rather than in the
          scheduler because every policy needs the same back-pressure
          semantics; the scheduler only chooses which node to call
          `admit()` on.
        * `model_config` and `node_config` are kept as separate refs
          (rather than merged into one) so a heterogeneous cluster can
          give different nodes different capacity without duplicating
          model constants.
    """

    def __init__(self, node_id: str, model_config: ModelConfig, node_config: NodeConfig) -> None:
        """Initialize a node with empty queues.

        Args:
            node_id: Stable identifier used in metrics and logs.
            model_config: Source of prefill / decode latency constants.
            node_config: Source of the concurrent-request capacity.
        """
        self.node_id = node_id
        self.model_config = model_config
        self.node_config = node_config
        self.running_requests: list[str] = []
        self.queue: list[str] = []

    # ------------------------------------------------------------------
    # Latency estimation
    # ------------------------------------------------------------------

    def estimate_prefill_time(self, prompt_len: int, cached_tokens: int) -> float:
        """Estimate prefill latency for a prompt with partial cache hit.

        Args:
            prompt_len: Total prompt token count.
            cached_tokens: Tokens already covered by this node's KV
                cache — these are free.

        Returns:
            Estimated prefill time in milliseconds. Clamped at 0 so a
            cache hit longer than the prompt (which can happen during
            speculative prefix matching) does not produce negative time.
        """
        uncached = max(0, prompt_len - cached_tokens)
        return uncached * self.model_config.prefill_cost_per_token_ms

    def estimate_decode_time(self, batch_size: int) -> float:
        """Estimate per-step decode latency at a given batch size.

        Args:
            batch_size: Number of concurrent decode streams the node
                would run.

        Returns:
            Decode-step time in milliseconds: `decode_base_ms +
            batch_size * marginal_decode_ms`. Linear model is a
            deliberate simplification — sufficient for ranking
            schedulers, not for absolute throughput claims.
        """
        return self.model_config.decode_base_ms + batch_size * self.model_config.marginal_decode_ms

    # ------------------------------------------------------------------
    # Load metrics
    # ------------------------------------------------------------------

    def current_load(self) -> float:
        """Current load as a fraction of node capacity.

        Returns:
            `len(running_requests) / capacity` in [0, 1]. Schedulers
            like `LeastLoaded` and the load-penalty term of
            `MooncakeConductor` consume this value directly.
        """
        return len(self.running_requests) / self.node_config.capacity

    def queue_wait_time(
        self,
        prompt_len: int | None = None,
        expected_output_len: int | None = None,
    ) -> float:
        """Conservative upper-bound estimate of how long a new admittee waits.

        Models capacity-aware admission:
        * If ``running_requests`` < ``capacity``, a slot is open for immediate
          admission — wait time is 0.
        * Otherwise the new request must wait for ``queue_length + 1``
          currently-blocking requests to complete (each modelled as a full
          prefill + decode lifecycle, the conservative upper bound).

        Args:
            prompt_len: New request's prompt length in tokens. When supplied
                with expected_output_len, computes full lifecycle estimate.
            expected_output_len: New request's expected decode steps. Required
                together with prompt_len for the accurate estimate.

        Returns:
            Wait time in milliseconds.
            - 0.0 when there is a free slot at admission time.
            - With both args: ``n_blockers × (prompt_len × ppt + output_len ×
              (decode_base + marginal))``
            - Without args (legacy): ``n_blockers × decode_base_ms``. Coarse
              fallback; new callers should always supply args.

        Known limitation (v1):
            Uses *the new request's* ``prompt_len`` / ``expected_output_len``
            to estimate the lifecycle of every blocker — because the node
            stores only request IDs, not the prompt/output lengths of
            already-admitted requests. Accurate for the current fixed-length
            :class:`RequestGenerator`. For trace replay / bursty workloads
            with heterogeneous lengths, short blockers ahead of a long new
            request will underestimate wait, and a long blocker ahead of a
            short new request will overestimate. This is acceptable for v1
            because the Conductor's SLO check is intentionally a conservative
            *gate* (false accepts are worse than false rejects); to make it
            length-aware in v2, ``MockEngineNode`` would need to track the
            ``(prompt_len, expected_output_len)`` of each running/queued
            request.
        """
        if len(self.running_requests) < self.node_config.capacity:
            return 0.0

        n_blockers = len(self.queue) + 1  # 1 running must finish + N queued ahead

        if prompt_len is None or expected_output_len is None:
            # Legacy fallback; new callers should always supply args.
            return n_blockers * self.model_config.decode_base_ms

        per_req_lifecycle_ms = (
            prompt_len * self.model_config.prefill_cost_per_token_ms
            + expected_output_len * (
                self.model_config.decode_base_ms + self.model_config.marginal_decode_ms
            )
        )
        return n_blockers * per_req_lifecycle_ms

    # ------------------------------------------------------------------
    # Request lifecycle (used by the simulation engine)
    # ------------------------------------------------------------------

    def admit(self, request_id: str) -> bool:
        """Admit a request — into `running_requests` if capacity allows, else `queue`.

        Args:
            request_id: Opaque ID owned by the scheduler/simulator.

        Returns:
            True if the request entered `running_requests` immediately and
            can start processing. False if it was placed in `queue` because
            capacity is exhausted; callers must defer downstream events until
            :meth:`complete` promotes this request.
        """
        if len(self.running_requests) < self.node_config.capacity:
            self.running_requests.append(request_id)
            logger.debug("node %s admitted %s (running)", self.node_id, request_id)
            return True
        self.queue.append(request_id)
        logger.debug("node %s queued %s", self.node_id, request_id)
        return False

    def complete(self, request_id: str) -> str | None:
        """Mark a request as finished and promote the next queued one if room frees up.

        Args:
            request_id: ID previously passed to `admit()`. May live in
                either `running_requests` or `queue` — both are checked
                because a request can be cancelled while still queued.

        Returns:
            The request_id of the promoted request if one was moved from
            `queue` into `running_requests`; `None` if the queue was
            empty or the just-completed request was itself still queued.

        Raises:
            ValueError: If *request_id* is in neither `running_requests`
                nor `queue`.
        """
        try:
            self.running_requests.remove(request_id)
        except ValueError:
            # Not running — must be queued (or the caller has a bug).
            self.queue.remove(request_id)
        logger.debug("node %s completed %s", self.node_id, request_id)
        if self.queue and len(self.running_requests) < self.node_config.capacity:
            promoted = self.queue.pop(0)
            self.running_requests.append(promoted)
            logger.debug("node %s promoted queued %s", self.node_id, promoted)
            return promoted
        return None

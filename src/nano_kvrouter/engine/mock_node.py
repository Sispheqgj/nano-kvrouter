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

    def queue_wait_time(self) -> float:
        """Rough wait-time estimate for everything currently queued.

        Returns:
            `queue_depth * decode_base_ms`. This is intentionally
            optimistic — it assumes each queued request blocks for one
            decode step worth of time — and is used only as a relative
            penalty when comparing nodes, not as an absolute SLO check.
        """
        return len(self.queue) * self.model_config.decode_base_ms

    # ------------------------------------------------------------------
    # Request lifecycle (used by the simulation engine)
    # ------------------------------------------------------------------

    def admit(self, request_id: str) -> None:
        """Admit a request — into `running_requests` if capacity allows, else `queue`.

        Args:
            request_id: Opaque ID owned by the scheduler/simulator.

        Returns:
            None. The simulator inspects `running_requests` / `queue`
            after the call to decide which event to emit next.
        """
        if len(self.running_requests) < self.node_config.capacity:
            self.running_requests.append(request_id)
            logger.debug("node %s admitted %s", self.node_id, request_id)
        else:
            self.queue.append(request_id)
            logger.debug("node %s queued %s", self.node_id, request_id)

    def complete(self, request_id: str) -> None:
        """Mark a request as finished and promote the next queued one if room frees up.

        Args:
            request_id: ID previously passed to `admit()`. May live in
                either `running_requests` or `queue` — both are checked
                because a request can be cancelled while still queued.

        Returns:
            None. If a queued request gets promoted, it is appended to
            `running_requests` so the caller can tell the simulator to
            schedule it.
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

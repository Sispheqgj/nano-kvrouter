from __future__ import annotations

import logging

from nano_kvrouter.config import ModelConfig, NodeConfig

logger = logging.getLogger(__name__)


class MockEngineNode:
    """Latency-model-only stand-in for a real GPU inference engine.

    Holds no tensors. The node tracks two queues — `running_requests`
    (admitted, bounded by `node_config.capacity`) and `queue` (waiting) —
    plus a `decoding` set of requests that have finished prefill and are
    actively participating in batch decode steps.

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
        * Decode batch state (`decoding`, `_output_tokens`,
          `_expected_output`) is populated at admit() / start_decode()
          time and cleaned up by complete(). tick_batch_step() advances
          only the `decoding` set — prefilling requests in
          `running_requests` are untouched.
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
        # Requests that have finished prefill and are in the decode batch.
        self.decoding: set[str] = set()
        # Decode progress tracking (only for requests in `decoding`).
        self._output_tokens: dict[str, int] = {}
        self._expected_output: dict[str, int] = {}
        # True between mark_batch_step_scheduled() and mark_batch_step_completed().
        # Prevents double-scheduling a DECODE_BATCH_STEP event when a new request
        # joins decoding while a step is already in flight.
        self._batch_step_in_flight: bool = False

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
            request will **overestimate** wait (we apply the long lifecycle
            to short blockers), and a long blocker ahead of a short new
            request will **underestimate** (we apply the short lifecycle to
            long blockers). This is acceptable for v1 because the Conductor's
            SLO check is intentionally a conservative *gate* (false accepts
            are worse than false rejects); to make it length-aware in v2,
            ``MockEngineNode`` would need to track the
            ``(prompt_len, expected_output_len)`` of each running/queued
            request.

        Formula:
            ``n_blockers × (prompt_len × ppt + output_len × step_time)``
            where ``step_time = decode_base + bs × marginal``
            and ``bs = max(len(decoding), len(running_requests))``.

            Using ``running_requests`` as the lower bound on ``bs`` handles
            the common case where all slots are occupied but none have yet
            entered decode (all still prefilling): once prefill finishes,
            every running request will join the decode batch, so using
            ``running_requests`` is a more accurate upper bound than 0.
        """
        if len(self.running_requests) < self.node_config.capacity:
            return 0.0

        n_blockers = len(self.queue) + 1  # 1 running must finish + N queued ahead

        if prompt_len is None or expected_output_len is None:
            # Legacy fallback; new callers should always supply args.
            return n_blockers * self.model_config.decode_base_ms

        # Use max(decoding, running) as batch-size estimate. When all requests
        # are still prefilling (decoding=0), running_requests is the upper
        # bound because they will all join the decode batch once prefill ends.
        bs = max(len(self.decoding), len(self.running_requests))
        step_time = self.model_config.decode_base_ms + bs * self.model_config.marginal_decode_ms
        per_req_lifecycle_ms = (
            prompt_len * self.model_config.prefill_cost_per_token_ms
            + expected_output_len * step_time
        )
        return n_blockers * per_req_lifecycle_ms

    # ------------------------------------------------------------------
    # Request lifecycle (used by the simulation engine)
    # ------------------------------------------------------------------

    def admit(self, request_id: str, expected_output_len: int = 0) -> bool:
        """Admit a request — into `running_requests` if capacity allows, else `queue`.

        Also initialises output-tracking state (`_output_tokens`,
        `_expected_output`) when the request enters `running_requests` so
        that :meth:`tick_batch_step` can advance it once :meth:`start_decode`
        is called.

        Args:
            request_id: Opaque ID owned by the scheduler/simulator.
            expected_output_len: Total number of decode tokens this request
                is expected to produce. Used by :meth:`tick_batch_step` to
                detect completion. Defaults to 0 for backward compatibility
                with tests that do not exercise the decode path.

        Returns:
            True if the request entered `running_requests` immediately and
            can start processing. False if it was placed in `queue` because
            capacity is exhausted; callers must defer downstream events until
            :meth:`complete` promotes this request.
        """
        if len(self.running_requests) < self.node_config.capacity:
            self.running_requests.append(request_id)
            self._output_tokens[request_id] = 0
            self._expected_output[request_id] = expected_output_len
            logger.debug("node %s admitted %s (running)", self.node_id, request_id)
            return True
        self.queue.append(request_id)
        logger.debug("node %s queued %s", self.node_id, request_id)
        return False

    # ------------------------------------------------------------------
    # Batch-step scheduling guards (Critical #1 lost-wakeup fix)
    # ------------------------------------------------------------------

    def mark_batch_step_scheduled(self) -> None:
        """Record that a DECODE_BATCH_STEP event has been enqueued for this node.

        Must be called exactly once per in-flight step. Raises RuntimeError
        on double-scheduling so double-schedule bugs surface immediately rather
        than silently running two overlapping batch pipelines.

        Raises:
            RuntimeError: If a batch step is already in flight.
        """
        if self._batch_step_in_flight:
            raise RuntimeError(
                f"node {self.node_id}: double-scheduled batch step"
            )
        self._batch_step_in_flight = True

    def mark_batch_step_completed(self) -> None:
        """Record that the DECODE_BATCH_STEP handler has started executing.

        Called at the top of the DECODE_BATCH_STEP event handler so that
        any new decode stream added during the same tick can immediately
        reschedule the pipeline.
        """
        self._batch_step_in_flight = False

    def is_batch_step_in_flight(self) -> bool:
        """Return True if a DECODE_BATCH_STEP is already scheduled for this node."""
        return self._batch_step_in_flight

    # ------------------------------------------------------------------
    # Request lifecycle (used by the simulation engine)
    # ------------------------------------------------------------------

    def start_decode(self, request_id: str) -> None:
        """Mark a request as finished with prefill and ready for batch decoding.

        After this call, :meth:`tick_batch_step` will advance `request_id`
        on every batch tick. The request must already be in `running_requests`
        (i.e., :meth:`admit` must have returned True for it).

        Args:
            request_id: ID previously admitted via :meth:`admit`.

        Raises:
            RuntimeError: If *request_id* is not currently in
                ``running_requests`` — indicates a caller bug (e.g. calling
                start_decode before admit, or after complete).
        """
        if request_id not in self.running_requests:
            raise RuntimeError(
                f"node {self.node_id}: start_decode called for {request_id!r} "
                "which is not in running_requests"
            )
        self.decoding.add(request_id)
        logger.debug("node %s started decode for %s", self.node_id, request_id)

    def init_promoted(self, request_id: str, expected_output_len: int) -> None:
        """Initialise output-tracking for a request promoted from `queue` to `running_requests`.

        Requests that were originally queued (admit returned False) bypass
        the normal admit() output-tracking initialisation. Call this from
        the simulator after :meth:`complete` returns a promoted ID and
        before scheduling PREFILL_START for the promoted request.

        Args:
            request_id: ID returned by :meth:`complete` as the promoted request.
            expected_output_len: Expected decode output tokens for this request.

        Raises:
            RuntimeError: If *request_id* is not in ``running_requests`` —
                indicates init_promoted was called before complete() promoted
                the request, which is a caller bug.
        """
        if request_id not in self.running_requests:
            raise RuntimeError(
                f"node {self.node_id}: init_promoted called for {request_id!r} "
                "which is not in running_requests"
            )
        self._output_tokens[request_id] = 0
        self._expected_output[request_id] = expected_output_len

    def tick_batch_step(self, now: float) -> tuple[float, list[str]]:
        """Advance one decode batch step for all active decode streams.

        Increments `_output_tokens` by 1 for every request in `decoding`.
        The batch step time is computed from the current `decoding` size:
        ``step_time = decode_base_ms + len(decoding) * marginal_decode_ms``.

        Engine invariant: each token is an integer event — no fractional
        tokens. This preserves the contract for future P3+ spec-decoding
        integration where draft+verify rounds are integer-length chains.

        Args:
            now: Current simulated time in milliseconds.

        Returns:
            ``(next_time, completed_ids)`` where:

            - ``next_time`` = now + step_time: the simulated time at which
              this batch step completes (and the next one may start).
            - ``completed_ids``: request IDs whose ``_output_tokens`` have
              reached or exceeded ``_expected_output`` after this tick.
              These requests should have ``DECODE_COMPLETE`` scheduled at
              ``next_time`` by the caller.

            **Critical invariant**: completed streams are removed from
            ``decoding`` before this method returns, so a re-wakeup that
            fires between the tick and the ``DECODE_COMPLETE`` handler
            cannot advance the same stream a second time. ``running_requests``
            is NOT modified here — ``complete()`` is responsible for
            releasing the capacity slot.

        Raises:
            RuntimeError: If ``decoding`` is empty. Callers must only invoke
                this method when at least one decode stream is active.
        """
        if not self.decoding:
            raise RuntimeError(
                f"tick_batch_step called on idle node {self.node_id} (no active decode streams)"
            )
        bs = len(self.decoding)
        step_time = self.model_config.decode_base_ms + bs * self.model_config.marginal_decode_ms

        # Iterate in sorted order so completed_ids is deterministic regardless
        # of set iteration order — important for reproducible test assertions.
        completed: list[str] = []
        for req_id in sorted(self.decoding):
            self._output_tokens[req_id] += 1
            if self._output_tokens[req_id] >= self._expected_output[req_id]:
                completed.append(req_id)

        # Critical #1: immediately remove terminal streams from decoding.
        # Prevents a subsequent wake from advancing them again before
        # DECODE_COMPLETE fires and calls complete().
        for req_id in completed:
            self.decoding.discard(req_id)

        return (now + step_time, completed)

    def complete(self, request_id: str) -> str | None:
        """Mark a request as finished and promote the next queued one if room frees up.

        Also cleans up decode-tracking state for `request_id`.

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

        # Clean up decode tracking.
        self._output_tokens.pop(request_id, None)
        self._expected_output.pop(request_id, None)
        self.decoding.discard(request_id)

        logger.debug("node %s completed %s", self.node_id, request_id)
        if self.queue and len(self.running_requests) < self.node_config.capacity:
            promoted = self.queue.pop(0)
            self.running_requests.append(promoted)
            logger.debug("node %s promoted queued %s", self.node_id, promoted)
            return promoted
        return None

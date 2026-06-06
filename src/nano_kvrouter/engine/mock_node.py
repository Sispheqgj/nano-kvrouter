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
        # Chunked prefill pipeline (M3). Maps request_id → remaining uncached tokens.
        # Insertion order is preserved (Python 3.7+ dict) for FIFO scheduling.
        self._prefill_remaining: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Latency estimation
    # ------------------------------------------------------------------

    def estimate_prefill_time(
        self,
        prompt_len: int,
        cached_tokens: int,
        batch_size_hint: int | None = None,
    ) -> float:
        """Estimate prefill latency for a prompt with partial cache hit.

        Args:
            prompt_len: Total prompt token count.
            cached_tokens: Tokens already covered by this node's KV
                cache — these are free.
            batch_size_hint: If None (legacy), returns the P1 formula:
                ``uncached * prefill_cost_per_token_ms``.
                If supplied, returns the M3 chunked-piggyback formula:
                ``n_chunks * (chunk_size * ppt + decode_base + bs * marginal)``,
                which is a conservative upper-bound because the last chunk may
                have fewer than ``chunk_size`` tokens.

        Returns:
            Estimated prefill time in milliseconds.

        Note — M3 conservative upper bound:
            The chunked-aware formula treats every chunk as a full
            ``chunk_size``, including the final (possibly partial) chunk,
            and adds ``decode_base + bs * marginal`` even when the node is
            idle (``bs = 1`` rather than ``0``).  Both choices are
            deliberately pessimistic so that Conductor's SLO gate is a
            *guard* (false rejects preferable to false accepts).  The actual
            per-tick cost in ``cli.tick_batch_step`` uses
            ``min(remaining, chunk_size)`` for the last chunk and incurs
            ``0`` decode cost when ``decoding`` is empty.
        """
        uncached = max(0, prompt_len - cached_tokens)
        if batch_size_hint is None:
            return uncached * self.model_config.prefill_cost_per_token_ms
        chunk = self.model_config.prefill_chunk_size
        n_chunks = (uncached + chunk - 1) // chunk if uncached > 0 else 0
        step_per_chunk = (
            chunk * self.model_config.prefill_cost_per_token_ms
            + self.model_config.decode_base_ms
            + batch_size_hint * self.model_config.marginal_decode_ms
        )
        return n_chunks * step_per_chunk

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

    def enter_prefill(self, request_id: str, uncached_tokens: int) -> None:
        """Mark a request as entering the chunked prefill pipeline.

        The request is placed in the FIFO ``_prefill_remaining`` queue.
        Each subsequent :meth:`tick_batch_step` will consume up to
        ``prefill_chunk_size`` tokens per step until the queue is drained,
        at which point the step returns ``prefill_completed_id``.

        Args:
            request_id: ID that must already be in ``running_requests``.
            uncached_tokens: Number of tokens that still require prefill
                computation (i.e. ``prompt_len - matched_tokens``).

        Raises:
            RuntimeError: If *request_id* is not in ``running_requests``.
        """
        if request_id not in self.running_requests:
            raise RuntimeError(
                f"node {self.node_id}: enter_prefill called for {request_id!r} "
                "which is not in running_requests"
            )
        self._prefill_remaining[request_id] = uncached_tokens
        logger.debug(
            "node %s enter_prefill %s (%d uncached tokens)",
            self.node_id, request_id, uncached_tokens,
        )

    def tick_batch_step(self, now: float) -> tuple[float, list[str], str | None]:
        """Advance one batch step: optional prefill chunk piggybacked with all decode streams.

        Each call processes at most one prefill chunk (FIFO from ``_prefill_remaining``)
        plus all active decode streams simultaneously (Sarathi-Serve piggyback model).

        Step time formula::

            step_time = chunk_cost + decode_cost
            where:
              chunk_cost  = min(remaining, chunk_size) * prefill_cost_per_token_ms
              decode_cost = decode_base_ms + bs * marginal_decode_ms  (0 when bs == 0)

        Args:
            now: Current simulated time in milliseconds.

        Returns:
            ``(next_time, completed_decode_ids, prefill_completed_id)`` where:

            - ``next_time`` = now + step_time.
            - ``completed_decode_ids``: request IDs whose decode finished this tick
              (removed from ``decoding`` immediately — Critical #1 invariant).
            - ``prefill_completed_id``: the request_id whose prefill finished
              this tick (removed from ``_prefill_remaining``), or ``None`` if no
              prefill chunk was processed or the prefill still has remaining tokens.

        Raises:
            RuntimeError: If both ``decoding`` and ``_prefill_remaining`` are empty.
        """
        if not self.decoding and not self._prefill_remaining:
            raise RuntimeError(f"node {self.node_id}: tick on idle node")

        bs = len(self.decoding)

        # FIFO prefill chunk: process the oldest entry in _prefill_remaining.
        prefill_id: str | None = None
        chunk_cost = 0.0
        if self._prefill_remaining:
            prefill_id = next(iter(self._prefill_remaining))
            remaining = self._prefill_remaining[prefill_id]
            chunk_this_step = min(remaining, self.model_config.prefill_chunk_size)
            chunk_cost = chunk_this_step * self.model_config.prefill_cost_per_token_ms
            self._prefill_remaining[prefill_id] -= chunk_this_step

        # Decode step cost (0 when no active decode streams).
        decode_cost = (
            self.model_config.decode_base_ms + bs * self.model_config.marginal_decode_ms
        ) if bs > 0 else 0.0

        step_time = chunk_cost + decode_cost

        # Advance all decode streams by one token.
        completed: list[str] = []
        for req_id in sorted(self.decoding):
            self._output_tokens[req_id] += 1
            if self._output_tokens[req_id] >= self._expected_output[req_id]:
                completed.append(req_id)

        # Critical #1: remove completed streams immediately.
        for req_id in completed:
            self.decoding.discard(req_id)

        # Check whether the prefill chunk just completed.
        prefill_completed_id: str | None = None
        if prefill_id is not None and self._prefill_remaining[prefill_id] == 0:
            del self._prefill_remaining[prefill_id]
            prefill_completed_id = prefill_id

        return (now + step_time, completed, prefill_completed_id)

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

        # Clean up all per-request state.
        self._output_tokens.pop(request_id, None)
        self._expected_output.pop(request_id, None)
        self.decoding.discard(request_id)
        self._prefill_remaining.pop(request_id, None)

        logger.debug("node %s completed %s", self.node_id, request_id)
        if self.queue and len(self.running_requests) < self.node_config.capacity:
            promoted = self.queue.pop(0)
            self.running_requests.append(promoted)
            logger.debug("node %s promoted queued %s", self.node_id, promoted)
            return promoted
        return None

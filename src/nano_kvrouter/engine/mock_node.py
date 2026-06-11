from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal

from nano_kvrouter.config import ModelConfig, NodeConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NodeRequestState:
    """Runtime metadata for one request currently owned by a mock node."""

    request_id: str
    prompt_len: int
    expected_output_len: int
    phase: Literal["queued", "admitted", "prefilling", "prefill_done", "decoding"]
    uncached_total: int | None = None
    uncached_remaining: int = 0
    generated_tokens: int = 0
    has_prompt_len: bool = False
    blocks_prefill_slot: bool = True


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
        # Per-request runtime metadata used for length-aware queue wait estimates.
        self._states: dict[str, NodeRequestState] = {}

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

        Models capacity-aware admission with a deterministic multi-slot FIFO
        release estimate:
        * If ``running_requests`` < ``capacity``, a slot is open for immediate
          admission, so wait time is 0.
        * Otherwise each running blocker contributes its own remaining
          lifecycle, queued-ahead requests occupy released slots in FIFO order,
          and the new request waits for the first slot available after them.

        Args:
            prompt_len: Legacy fallback prompt length for blockers that were
                admitted by older tests without per-request metadata.
            expected_output_len: Legacy fallback decode length for blockers
                admitted without output metadata.

        Returns:
            Wait time in milliseconds.
            - 0.0 when there is a free slot at admission time.
            - For runtime-aware requests: FIFO slot wait computed from each
              blocker request's own phase and remaining uncached/decode work.
              ``prefill_done`` requests that no longer block a prefill slot
              contribute 0; ``decoding`` requests still contribute remaining
              decode work on decode nodes.
            - For legacy tests that admitted blockers without prompt/output
              metadata: falls back to the pre-v2 coarse estimate.
        """
        if len(self.running_requests) < self.node_config.capacity:
            return 0.0

        blocker_ids = [*self.running_requests, *self.queue]
        if self._uses_legacy_wait_fallback(blocker_ids, prompt_len, expected_output_len):
            legacy_wait_units = len(self.queue) + 1
            return legacy_wait_units * self.model_config.decode_base_ms

        running_remaining = [
            self._remaining_lifecycle_ms(
                self._state_for_wait(request_id),
                fallback_prompt_len=prompt_len,
                fallback_output_len=expected_output_len,
            )
            for request_id in self.running_requests
        ]
        if not running_remaining:
            return 0.0

        # FIFO multi-server estimate: queued-ahead requests occupy the earliest
        # slot releases before the new request can enter.
        slot_release_times = sorted(running_remaining)
        for request_id in self.queue:
            state = self._state_for_wait(request_id)
            earliest = slot_release_times.pop(0)
            finish_time = earliest + self._remaining_lifecycle_ms(
                state,
                fallback_prompt_len=prompt_len,
                fallback_output_len=expected_output_len,
            )
            slot_release_times.append(finish_time)
            slot_release_times.sort()
        return slot_release_times[0]

    def _uses_legacy_wait_fallback(
        self,
        blocker_ids: list[str],
        prompt_len: int | None,
        expected_output_len: int | None,
    ) -> bool:
        """Preserve old no-arg tests when no blocker metadata exists."""
        if prompt_len is not None or expected_output_len is not None:
            return False
        for request_id in blocker_ids:
            state = self._states.get(request_id)
            if state is None:
                return True
            if state.has_prompt_len or state.expected_output_len > 0:
                return False
        return True

    def _state_for_wait(self, request_id: str) -> NodeRequestState:
        state = self._states.get(request_id)
        if state is not None:
            return state
        phase: Literal["queued", "admitted", "prefilling", "prefill_done", "decoding"] = (
            "queued" if request_id in self.queue else "admitted"
        )
        return NodeRequestState(
            request_id=request_id,
            prompt_len=0,
            expected_output_len=0,
            phase=phase,
        )

    def _remaining_lifecycle_ms(
        self,
        state: NodeRequestState,
        *,
        fallback_prompt_len: int | None = None,
        fallback_output_len: int | None = None,
    ) -> float:
        expected_output = state.expected_output_len
        if expected_output == 0 and not state.has_prompt_len and fallback_output_len is not None:
            expected_output = fallback_output_len

        remaining_decode_tokens = max(0, expected_output - state.generated_tokens)
        decode_step = self._wait_decode_step_ms()

        if state.phase == "prefill_done" and not state.blocks_prefill_slot:
            return 0.0

        if state.phase in {"prefill_done", "decoding"}:
            return remaining_decode_tokens * decode_step

        if state.phase == "prefilling":
            prefill_tokens = max(0, state.uncached_remaining)
        elif state.uncached_total is not None:
            prefill_tokens = max(0, state.uncached_total)
        else:
            prompt_tokens = state.prompt_len if state.has_prompt_len else (fallback_prompt_len or 0)
            prefill_tokens = max(0, prompt_tokens)

        return (
            prefill_tokens * self.model_config.prefill_cost_per_token_ms
            + remaining_decode_tokens * decode_step
        )

    def _wait_decode_step_ms(self) -> float:
        bs = max(len(self.decoding), len(self.running_requests), 1)
        return self.model_config.decode_base_ms + bs * self.model_config.marginal_decode_ms

    # ------------------------------------------------------------------
    # Request lifecycle (used by the simulation engine)
    # ------------------------------------------------------------------

    def admit(
        self,
        request_id: str,
        expected_output_len: int = 0,
        prompt_len: int | None = None,
        uncached_tokens: int | None = None,
    ) -> bool:
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
            prompt_len: Optional prompt length for length-aware wait
                estimation. Omitted legacy calls remain supported.
            uncached_tokens: Optional number of prompt tokens that still need
                prefill after cache hit. When known, queued/admitted wait
                estimates use this value instead of prompt-length fallback.

        Returns:
            True if the request entered `running_requests` immediately and
            can start processing. False if it was placed in `queue` because
            capacity is exhausted; callers must defer downstream events until
            :meth:`complete` promotes this request.
        """
        uncached_total = max(0, uncached_tokens) if uncached_tokens is not None else None
        state = NodeRequestState(
            request_id=request_id,
            prompt_len=prompt_len if prompt_len is not None else 0,
            expected_output_len=expected_output_len,
            phase="admitted" if len(self.running_requests) < self.node_config.capacity else "queued",
            uncached_total=uncached_total,
            uncached_remaining=uncached_total if uncached_total is not None else 0,
            has_prompt_len=prompt_len is not None,
        )
        self._states[request_id] = state

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
        state = self._states.get(request_id)
        if state is not None:
            state.phase = "decoding"
            state.uncached_total = 0 if state.uncached_total is None else state.uncached_total
            state.uncached_remaining = 0
            state.generated_tokens = self._output_tokens.get(request_id, 0)
            state.blocks_prefill_slot = True
        self.decoding.add(request_id)
        logger.debug("node %s started decode for %s", self.node_id, request_id)

    def init_promoted(
        self,
        request_id: str,
        expected_output_len: int,
        prompt_len: int | None = None,
        uncached_tokens: int | None = None,
    ) -> None:
        """Initialise output-tracking for a request promoted from `queue` to `running_requests`.

        Requests that were originally queued (admit returned False) bypass
        the normal admit() output-tracking initialisation. Call this from
        the simulator after :meth:`complete` returns a promoted ID and
        before scheduling PREFILL_START for the promoted request.

        Args:
            request_id: ID returned by :meth:`complete` as the promoted request.
            expected_output_len: Expected decode output tokens for this request.
            prompt_len: Optional prompt length. When omitted, preserves the
                value stored while the request was queued.
            uncached_tokens: Optional remaining prefill tokens after cache hit.
                When omitted, preserves the value stored while queued.

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
        uncached_total = max(0, uncached_tokens) if uncached_tokens is not None else None
        state = self._states.get(request_id)
        if state is None:
            state = NodeRequestState(
                request_id=request_id,
                prompt_len=prompt_len if prompt_len is not None else 0,
                expected_output_len=expected_output_len,
                phase="admitted",
                uncached_total=uncached_total,
                uncached_remaining=uncached_total if uncached_total is not None else 0,
                has_prompt_len=prompt_len is not None,
            )
            self._states[request_id] = state
        else:
            state.phase = "admitted"
            state.expected_output_len = expected_output_len
            state.generated_tokens = 0
            if prompt_len is not None:
                state.prompt_len = prompt_len
                state.has_prompt_len = True
            if uncached_total is not None:
                state.uncached_total = uncached_total
                state.uncached_remaining = uncached_total

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
        state = self._states.get(request_id)
        if state is not None:
            state.phase = "prefilling"
            state.uncached_total = max(0, uncached_tokens)
            state.uncached_remaining = uncached_tokens
            state.blocks_prefill_slot = True
        logger.debug(
            "node %s enter_prefill %s (%d uncached tokens)",
            self.node_id, request_id, uncached_tokens,
        )

    def mark_prefill_done(self, request_id: str) -> None:
        """Record that a running request has no remaining prefill work.

        This covers both fully-cached fast paths and requests whose chunked
        prefill just drained. It keeps later wait estimates from charging full
        prompt prefill to a blocker that is waiting only for transfer/decode.
        """
        if request_id not in self.running_requests:
            raise RuntimeError(
                f"node {self.node_id}: mark_prefill_done called for {request_id!r} "
                "which is not in running_requests"
            )
        state = self._states.get(request_id)
        if state is not None:
            state.phase = "prefill_done"
            state.uncached_total = 0
            state.uncached_remaining = 0
            state.blocks_prefill_slot = False

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
            state = self._states.get(prefill_id)
            if state is not None:
                state.uncached_remaining = self._prefill_remaining[prefill_id]

        # Decode step cost (0 when no active decode streams).
        decode_cost = (
            self.model_config.decode_base_ms + bs * self.model_config.marginal_decode_ms
        ) if bs > 0 else 0.0

        step_time = chunk_cost + decode_cost

        # Advance all decode streams by one token.
        completed: list[str] = []
        for req_id in sorted(self.decoding):
            self._output_tokens[req_id] += 1
            state = self._states.get(req_id)
            if state is not None:
                state.generated_tokens = self._output_tokens[req_id]
            if self._output_tokens[req_id] >= self._expected_output[req_id]:
                completed.append(req_id)

        # Critical #1: remove completed streams immediately.
        for req_id in completed:
            self.decoding.discard(req_id)

        # Check whether the prefill chunk just completed.
        prefill_completed_id: str | None = None
        if prefill_id is not None and self._prefill_remaining[prefill_id] == 0:
            del self._prefill_remaining[prefill_id]
            state = self._states.get(prefill_id)
            if state is not None:
                state.phase = "prefill_done"
                state.uncached_total = 0
                state.uncached_remaining = 0
                state.blocks_prefill_slot = False
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
        self._states.pop(request_id, None)

        logger.debug("node %s completed %s", self.node_id, request_id)
        if self.queue and len(self.running_requests) < self.node_config.capacity:
            promoted = self.queue.pop(0)
            self.running_requests.append(promoted)
            state = self._states.get(promoted)
            if state is not None:
                state.phase = "admitted"
                state.blocks_prefill_slot = True
            logger.debug("node %s promoted queued %s", self.node_id, promoted)
            return promoted
        return None

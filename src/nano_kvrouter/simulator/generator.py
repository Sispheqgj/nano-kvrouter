"""Streaming Poisson request generator with K conversation buckets.

Models a chatbot/code-assistant workload where:
* Inter-arrival times follow an exponential distribution (Poisson
  process) with mean ``1000 / request_rate`` ms.
* Each request belongs to one of K "conversation buckets" — buckets
  share a common (block-aligned) prefix, with per-request random
  suffix tokens. This generates measurable prefix cache hits for
  cache-aware schedulers like PrefixGreedy / E2Policy / Conductor.
* The prefix length is ``workload.prefix_sharing_ratio × prompt_len``,
  floored to a ``block_size`` multiple to align with vLLM PagedAttention
  semantics (avoids the sub-block-node accounting hazard documented
  in ``doc/code-review/kv_cache/cache_manager.md``).
* Streaming: the generator schedules only the first arrival eagerly;
  each subsequent arrival is chained inside the ``REQUEST_ARRIVE``
  handler. This mirrors a real online system where the scheduler does
  not know future request times.
"""
from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from nano_kvrouter.config import NanoKVConfig
from nano_kvrouter.request import Request, make_request
from nano_kvrouter.simulator.event import Event, EventType

if TYPE_CHECKING:
    from nano_kvrouter.simulator.engine import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["RequestGenerator"]


class RequestGenerator:
    """Streaming Poisson generator with K-bucket prefix sharing.

    Attach to a :class:`~nano_kvrouter.simulator.engine.SimulationEngine`
    once via :meth:`attach` before calling ``engine.run()``. The generator
    then self-sustains by chaining each ``REQUEST_ARRIVE`` event into the
    next, stopping when the next arrival would exceed
    ``workload.duration_s * 1000`` ms.

    Design notes:
        * ``_prefix_len`` is forced to a ``block_size`` multiple so every
          bucket prefix occupies an integer number of KV blocks. This
          prevents the sub-block-node accounting issue described in
          ``doc/code-review/kv_cache/cache_manager.md`` (known limitation #2).
        * When ``prefix_sharing_ratio × avg_prompt_len < block_size``,
          ``_prefix_len`` collapses to 0 — requests have no shared prefix
          and every node is a cold-start for cache-aware schedulers.
        * ``_suffix_len + _prefix_len == avg_prompt_len`` always.
        * The PRNG is instance-local (``random.Random(seed)``) for full
          reproducibility without affecting the global ``random`` module state.
    """

    def __init__(self, config: NanoKVConfig) -> None:
        """Initialise generator from a complete NanoKVConfig.

        Args:
            config: Root config. ``config.workload``, ``config.model``,
                ``config.slo``, and ``config.generator`` sub-sections are
                all consumed. The full config object is retained so
                :func:`~nano_kvrouter.request.make_request` can stamp SLO
                and output-length fields consistently.
        """
        self._config = config
        self._workload = config.workload
        self._model = config.model
        self._gen = config.generator

        self._rng = random.Random(config.generator.seed)
        self._mean_interarrival_ms: float = 1000.0 / self._workload.request_rate
        self._end_time_ms: float = self._workload.duration_s * 1000.0

        bs = self._model.block_size
        raw_prefix = int(self._workload.prefix_sharing_ratio * self._workload.avg_prompt_len)
        self._prefix_len: int = (raw_prefix // bs) * bs
        self._suffix_len: int = self._workload.avg_prompt_len - self._prefix_len

        # Pre-generate K bucket prefixes using the seeded RNG so the set
        # is stable across runs with the same seed.
        self._bucket_prefixes: list[list[int]] = [
            [
                self._rng.randint(0, self._gen.vocab_size - 1)
                for _ in range(self._prefix_len)
            ]
            for _ in range(self._gen.num_buckets)
        ]

        self._next_request_id: int = 0
        self._attached: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self, engine: SimulationEngine) -> None:
        """Bind to *engine*: register chain handler and schedule first arrival.

        Must be called **exactly once** per generator instance. The second
        call raises ``RuntimeError`` — duplicate chain handlers would cause
        each ARRIVE event to spawn multiple successor chains, producing
        exponential rather than linear growth in arrival count. (Empirically:
        at rate=5/s, duration=1s, attaching 2× yields ~400 arrivals, 3× yields
        ~32000.)

        For a fresh simulation, instantiate a new ``RequestGenerator``.

        Args:
            engine: The simulation engine that will drive the generator.
                The generator registers itself on
                ``EventType.REQUEST_ARRIVE`` and does not touch any other
                event type.

        Raises:
            RuntimeError: If ``attach()`` has already been called on this
                instance.
        """
        if self._attached:
            raise RuntimeError(
                "RequestGenerator.attach() called more than once; arrival "
                "events would spawn explosively (K^N growth where K = chain "
                "fanout per ARRIVE, N = simulated steps). Construct a fresh "
                "generator for a fresh simulation."
            )
        self._attached = True

        engine.on(EventType.REQUEST_ARRIVE, self._on_arrive_chain)
        first_t = self._sample_interarrival()
        if first_t < self._end_time_ms:
            self._enqueue_arrival(engine, first_t)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_interarrival(self) -> float:
        """Sample one exponential inter-arrival time in milliseconds."""
        return self._rng.expovariate(1.0 / self._mean_interarrival_ms)

    def _build_request(self, arrival_time: float) -> Request:
        """Build a new Request with a bucket prefix + random suffix."""
        bucket_idx = self._rng.randint(0, self._gen.num_buckets - 1)
        suffix = [
            self._rng.randint(0, self._gen.vocab_size - 1)
            for _ in range(self._suffix_len)
        ]
        token_ids = self._bucket_prefixes[bucket_idx] + suffix
        # make_request computes prefix_hash and stamps SLO / output_len from config.
        req = make_request(token_ids, arrival_time, self._config)
        # Overwrite UUID with a sequential ID for deterministic test assertions.
        req.request_id = f"req-{self._next_request_id}"
        self._next_request_id += 1
        return req

    def _enqueue_arrival(self, engine: SimulationEngine, t: float) -> None:
        """Schedule a REQUEST_ARRIVE event at time *t*."""
        req = self._build_request(t)
        engine.schedule(Event(
            time=t,
            type=EventType.REQUEST_ARRIVE,
            payload={"request": req},
        ))

    def _on_arrive_chain(self, event: Event, engine: SimulationEngine) -> None:
        """Chain handler: schedule the next arrival after the current one.

        Does not consume the event itself — ``MetricsCollector`` and other
        subscribers handle the request payload. This handler's sole job is
        to keep the arrival stream alive until ``_end_time_ms`` is reached.
        """
        next_t = engine.now() + self._sample_interarrival()
        if next_t < self._end_time_ms:
            self._enqueue_arrival(engine, next_t)

"""Trace-driven request generator (Mooncake / BurstGPT-style JSONL).

Replays a pre-recorded JSONL trace through the simulation engine, scheduling
REQUEST_ARRIVE events in arrival-time order.  The generator is streaming —
it reads one record at a time and schedules the next ARRIVE inside the handler
for the current one, so the full trace never needs to be loaded into memory.

Supports three prefix_mode values:

- ``"hash_ids"``: requires ``hash_ids`` field per record (Mooncake FAST'25
  trace format, block_size=512).  Token IDs are derived from hash IDs to
  preserve real prefix-sharing structure.
- ``"synthesis"``: requires no prefix field; pairs with PrefixSynthesisModel
  to synthesize prefix sharing on top of (arrival_ms, input_length).  Used
  for BurstGPT / Azure-style length-only traces.
- ``"none"``: pure random tokens, no prefix sharing (conservative baseline).

Internal JSONL schema per line::

    {
      "arrival_ms": float,       # ms from trace start (0-based); or "timestamp" for Mooncake
      "input_length": int,       # prompt token count
      "output_length": int,      # decode token count
      "hash_ids": list[int],     # optional; required when prefix_mode="hash_ids"
      "session_id": str | null,  # optional; preserved for future P3-D use
    }
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import IO

from nano_kvrouter.config import NanoKVConfig, TraceConfig
from nano_kvrouter.simulator.prefix_synthesis import PrefixSynthesisModel
from nano_kvrouter.request import make_request
from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import Event, EventType

logger = logging.getLogger(__name__)

__all__ = ["TraceGenerator"]


def _build_token_ids(
    hash_ids: list[int],
    input_length: int,
    block_size: int,
    rng: random.Random,
    vocab_size: int,
) -> list[int]:
    """Synthesise token_ids from Mooncake hash_ids, enforcing len == input_length.

    Mooncake hash_ids are block-level identifiers: each hash_id represents
    block_size consecutive tokens. Two requests sharing a leading hash_ids
    prefix produce identical leading token sequences, so the RadixTree can
    cache-hit on that shared prefix.

    Expansion rules:
    - The first min(len(hash_ids), full_blocks) hash_ids are each expanded
      into block_size tokens via ``hid * block_size + offset`` (deterministic,
      collision-free, reproducible).
    - Remaining full blocks not covered by hash_ids are filled with random
      tokens (no prefix sharing for those blocks).
    - The tail (input_length % block_size tokens) is filled with random
      tokens and does NOT contribute to prefix sharing.

    NOTE: ``hid * block_size + offset`` can exceed vocab_size, but the
    RadixTree compares token IDs only for equality. Safe for M1; requires
    revisiting if a real tokenizer is wired in.

    Args:
        hash_ids: Block-level prefix identifiers from the trace record.
        input_length: Target token count; enforced by final assert.
        block_size: Tokens per KV-cache block (must match trace block_size).
        rng: Instance-local PRNG (does not touch global state).
        vocab_size: Upper bound for random token sampling.

    Returns:
        List of exactly ``input_length`` integers.
    """
    full_blocks = input_length // block_size
    tail_len = input_length - full_blocks * block_size

    n_to_expand = min(len(hash_ids), full_blocks)
    tokens: list[int] = []

    for hid in hash_ids[:n_to_expand]:
        tokens.extend(hid * block_size + i for i in range(block_size))

    for _ in range(full_blocks - n_to_expand):
        tokens.extend(rng.randint(0, vocab_size - 1) for _ in range(block_size))

    tokens.extend(rng.randint(0, vocab_size - 1) for _ in range(tail_len))

    assert len(tokens) == input_length, (
        f"len(tokens)={len(tokens)} != input_length={input_length} "
        f"(hash_ids={len(hash_ids)}, block_size={block_size})"
    )
    return tokens


class TraceGenerator:
    """Schedule REQUEST_ARRIVE events from a JSONL trace file.

    Duck-type compatible with RequestGenerator.attach — both expose a
    single ``attach(engine)`` method and use an event-driven chain to
    avoid pre-loading all events.

    Args:
        config: Full NanoKVConfig (block_size, vocab_size, seed, SLO).
        trace_config: TraceConfig (path resolved to absolute by CLI).
        trace_path: Absolute path to the JSONL trace file.
    """

    def __init__(
        self,
        config: NanoKVConfig,
        trace_config: TraceConfig,
        trace_path: Path,
        synthesis_model: PrefixSynthesisModel | None = None,
    ) -> None:
        if not trace_path.is_absolute():
            raise ValueError(
                f"trace_path must be absolute (CLI resolves it via "
                f"_resolve_related_config_path); got {trace_path!r}"
            )
        if trace_config.prefix_mode == "synthesis" and synthesis_model is None:
            raise ValueError(
                "prefix_mode='synthesis' requires synthesis_model; "
                "construct TraceGenerator with synthesis_model=PrefixSynthesisModel(...)"
            )
        self._config = config
        self._trace_cfg = trace_config
        self._trace_path = trace_path
        self._synthesis_model = synthesis_model
        self._rng = random.Random(config.generator.seed)
        self._idx = 0
        self._file: IO[str] | None = None
        self._attached: bool = False

    def attach(self, engine: SimulationEngine) -> None:
        """Open the trace file, schedule first ARRIVE, register chain handler."""
        if self._attached:
            raise RuntimeError(
                "TraceGenerator.attach() called more than once; arrival "
                "events would interleave from two parallel chains. "
                "Construct a fresh TraceGenerator for a fresh simulation."
            )
        self._attached = True
        self._file = open(self._trace_path, encoding="utf-8")
        engine.on(EventType.REQUEST_ARRIVE, self._on_arrive_chain)
        self._schedule_next(engine)

    def _close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _schedule_next(self, engine: SimulationEngine) -> None:
        """Read one record from the file and schedule its ARRIVE event."""
        if self._file is None:
            return
        if (
            self._trace_cfg.max_requests is not None
            and self._idx >= self._trace_cfg.max_requests
        ):
            self._close()
            return

        line = self._file.readline()
        if not line:
            self._close()
            return

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("trace: skipping malformed JSON at line %d", self._idx)
            self._schedule_next(engine)
            return

        try:
            # BurstGPT uses "arrival_ms" (already zero-based ms);
            # Mooncake uses "timestamp" (also ms). Prefer arrival_ms if present.
            raw_arrival_ms = record.get("arrival_ms", record.get("timestamp", 0))
            arrival_ms = raw_arrival_ms / self._trace_cfg.speedup
            token_ids = self._build_token_ids_from_record(record)
            req = make_request(
                token_ids,
                arrival_time=arrival_ms,
                config=self._config,
                expected_output_len=record["output_length"],
            )
        except Exception:
            self._close()
            raise
        # Override UUID request_id with stable trace index.
        req = replace(req, request_id=f"trace-{self._idx}")
        self._idx += 1

        engine.schedule(Event(
            time=arrival_ms,
            type=EventType.REQUEST_ARRIVE,
            payload={"request": req},
        ))

    def _on_arrive_chain(self, event: Event, engine: SimulationEngine) -> None:
        """Called on each REQUEST_ARRIVE; schedules the next record."""
        self._schedule_next(engine)

    def _build_token_ids_from_record(self, record: dict) -> list[int]:
        """Dispatch token synthesis based on prefix_mode."""
        mode = self._trace_cfg.prefix_mode
        input_length = record["input_length"]
        block_size = self._config.model.block_size
        vocab_size = self._config.generator.vocab_size

        if mode == "none":
            tokens = [self._rng.randint(0, vocab_size - 1) for _ in range(input_length)]
            assert len(tokens) == input_length
            return tokens
        elif mode == "hash_ids":
            if "hash_ids" not in record:
                raise ValueError(
                    f"prefix_mode='hash_ids' requires 'hash_ids' field; "
                    f"got record without it (request idx={self._idx}). "
                    f"Use a cleaner trace or set prefix_mode='none'."
                )
            hash_ids = record["hash_ids"]
            return _build_token_ids(hash_ids, input_length, block_size, self._rng, vocab_size)
        elif mode == "synthesis":
            raw_arrival_ms = record.get("arrival_ms", record.get("timestamp", 0))
            prefix = self._synthesis_model.assign_prefix_tokens(
                arrival_ms=raw_arrival_ms,
                prompt_len=input_length,
            )
            tail_len = input_length - len(prefix)
            suffix = [self._rng.randint(0, vocab_size - 1) for _ in range(tail_len)]
            tokens = prefix + suffix
            assert len(tokens) == input_length
            return tokens
        else:
            raise ValueError(f"Unknown prefix_mode: {mode!r}")

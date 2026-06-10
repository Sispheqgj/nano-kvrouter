"""Synthetic prefix-sharing model for length-only traces (e.g. BurstGPT).

BurstGPT records contain arrival timestamps and request/response lengths but
NO prefix structure.  This module synthesises a plausible prefix-reuse
pattern on top of those (timestamp, length) pairs so that cache-aware
schedulers can be evaluated on the trace.

Design:
  - Instance-local PRNG (random.Random(seed)) — never touches global state.
  - Bucket prefixes start at initial_prompt_len and grow lazily as longer
    prompts arrive; no hard upper bound.
  - Sliding-window deque for time-locality with O(1) amortised purge.
  - Layer probabilities are validated at construction time (sum ≈ 1.0,
    each ratio in [0, 1], each prob ≥ 0).

IMPORTANT: Results obtained with this model reflect the prefix-sharing
HYPOTHESIS layered on top of real trace arrival/length, NOT observed
BurstGPT prefix reuse.  Report cache_hit numbers as
"trace arrival/length real + prefix-sharing synthetic", never as
ground truth.
"""
from __future__ import annotations

import collections
import logging
import random

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

__all__ = ["PrefixSynthesisConfig", "PrefixSynthesisModel"]


class PrefixSynthesisConfig(BaseModel):
    """Configure synthetic prefix-sharing structure for length-only traces.

    This synthesis is a HYPOTHESIS layered on top of trace
    arrival+length, NOT an observation of actual prefix reuse.  Report
    cache_hit numbers obtained with this model as "trace arrival/length
    real + prefix-sharing synthetic", never as ground truth.
    """

    model_config = ConfigDict(extra="forbid")

    num_buckets: int = Field(default=64, ge=1)
    zipf_alpha: float = Field(default=1.0, gt=0)
    p_local: float = Field(default=0.6, ge=0.0, le=1.0)
    local_window_s: float = Field(default=60.0, gt=0)
    sharing_layers: list[tuple[float, float]] = Field(
        default_factory=lambda: [
            (0.2, 0.5),
            (0.5, 0.25),
            (0.3, 0.0),
        ],
        min_length=1,
    )
    seed: int = Field(default=42)

    @field_validator("sharing_layers")
    @classmethod
    def _validate_sharing_layers(
        cls, v: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        if not v:
            raise ValueError("sharing_layers must be non-empty")
        for i, (p, r) in enumerate(v):
            if p < 0.0:
                raise ValueError(
                    f"sharing_layers[{i}].prob must be >= 0, got {p}"
                )
            if not 0.0 <= r <= 1.0:
                raise ValueError(
                    f"sharing_layers[{i}].ratio must be in [0, 1], got {r}"
                )
        total = sum(p for p, _ in v)
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"sharing_layers probabilities must sum to ~1.0 (tolerance ±0.01), "
                f"got {total:.4f}"
            )
        return v


class PrefixSynthesisModel:
    """Three-stage synthesis: sharing layer -> bucket selection -> block-aligned prefix tokens.

    Stage 1: sample one sharing layer; that layer ratio determines raw
        prefix_len = ratio * prompt_len.  Align down to block_size.
    Stage 2: bucket selection
        - With prob p_local, sample uniformly from buckets active in last
          local_window_s seconds (sliding window).  If no recent buckets,
          fall through to global Zipf.
        - Else, sample bucket_idx ~ Zipf over [0, num_buckets) with given alpha.
    Stage 3: emit prefix tokens
        - Return bucket_prefixes[bucket_idx][:aligned_prefix_len].
        - Bucket prefixes grow lazily if aligned_prefix_len exceeds current length.
        - Record (bucket_idx, arrival_ms) in recent-active deque for stage 2.

    All randomness from an instance-local random.Random; never touches global
    random module state.
    """

    def __init__(
        self,
        config: PrefixSynthesisConfig,
        block_size: int,
        vocab_size: int,
        initial_prompt_len: int = 1024,   # performance hint only; buckets grow lazily
    ) -> None:
        self._cfg = config
        self._block_size = block_size
        self._vocab_size = vocab_size
        self._rng = random.Random(config.seed)

        # Pre-compute Zipf weights for fast random.choices()
        # weight[i] = 1 / (i+1)**alpha
        self._zipf_weights: list[float] = [
            1.0 / ((i + 1) ** config.zipf_alpha)
            for i in range(config.num_buckets)
        ]
        self._all_buckets: list[int] = list(range(config.num_buckets))

        # Pre-generate K bucket prefixes at initial_prompt_len length.
        # Buckets extend lazily when longer prompts arrive (no hard cap).
        initial_blocks = max(1, (initial_prompt_len + block_size - 1) // block_size)
        self._bucket_prefixes: list[list[int]] = [
            [self._rng.randint(0, vocab_size - 1) for _ in range(initial_blocks * block_size)]
            for _ in range(config.num_buckets)
        ]

        # Sliding window of recently active buckets: deque of (bucket_idx, arrival_ms)
        self._recent: collections.deque[tuple[int, float]] = collections.deque()
        self._local_window_ms = config.local_window_s * 1000.0

    def assign_prefix_tokens(
        self,
        arrival_ms: float,
        prompt_len: int,
    ) -> list[int]:
        """Return prefix tokens of length aligned to block_size.

        Caller pads with random suffix tokens to reach total prompt_len.
        Caller owns the len(total) == input_length invariant.
        """
        # Stage 1: pick sharing layer
        layer_p = [p for p, _ in self._cfg.sharing_layers]
        layer_r = [r for _, r in self._cfg.sharing_layers]
        layer_idx = self._rng.choices(range(len(layer_r)), weights=layer_p, k=1)[0]
        layer_ratio = layer_r[layer_idx]

        raw_prefix_len = int(layer_ratio * prompt_len)
        aligned_prefix_len = (raw_prefix_len // self._block_size) * self._block_size
        if aligned_prefix_len == 0:
            return []

        # Stage 2: bucket selection
        self._purge_expired(arrival_ms)
        use_local = (
            self._rng.random() < self._cfg.p_local and len(self._recent) > 0
        )
        if use_local:
            bucket_idx = self._rng.choice([b for b, _ in self._recent])
        else:
            bucket_idx = self._rng.choices(
                self._all_buckets, weights=self._zipf_weights, k=1
            )[0]

        # Stage 3: lazy extend if needed, then emit + record
        bucket = self._bucket_prefixes[bucket_idx]
        if len(bucket) < aligned_prefix_len:
            n_to_add = aligned_prefix_len - len(bucket)
            bucket.extend(
                self._rng.randint(0, self._vocab_size - 1)
                for _ in range(n_to_add)
            )

        self._recent.append((bucket_idx, arrival_ms))
        return bucket[:aligned_prefix_len]

    def _purge_expired(self, now_ms: float) -> None:
        """Drop entries older than local_window_ms from the front of the deque."""
        cutoff = now_ms - self._local_window_ms
        while self._recent and self._recent[0][1] < cutoff:
            self._recent.popleft()

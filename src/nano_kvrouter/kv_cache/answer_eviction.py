"""Bidaw previous-answer-based eviction helpers.

This module is metadata-only. It does not know about tensors or storage
engines; it maps a request's previous answer length to an estimated CPU-tier
hit potential, then CacheManager uses that potential when selecting CPU blocks
to demote under capacity pressure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AnswerEvictionStats:
    """Counters exported by CacheManager after answer-aware evictions."""

    eviction_count: int = 0
    evicted_blocks: int = 0
    cpu_saved_blocks: int = 0
    evicted_potential_sum: float = 0.0
    cpu_hit_blocks: int = 0
    cpu_lookup_blocks: int = 0

    def summary(self) -> dict[str, float | int]:
        avg_potential = (
            self.evicted_potential_sum / self.eviction_count
            if self.eviction_count > 0 else 0.0
        )
        cpu_hit_rate = (
            self.cpu_hit_blocks / self.cpu_lookup_blocks
            if self.cpu_lookup_blocks > 0 else 0.0
        )
        return {
            "bidaw_answer_eviction_count": self.eviction_count,
            "bidaw_answer_evicted_blocks": self.evicted_blocks,
            "bidaw_answer_eviction_cpu_saved_blocks": self.cpu_saved_blocks,
            "bidaw_answer_eviction_hit_potential_avg": avg_potential,
            "bidaw_answer_eviction_cpu_hit_rate": cpu_hit_rate,
        }


@dataclass(frozen=True, slots=True)
class AnswerEvictionPolicy:
    """Answer-length bucket policy used by Bidaw CPU-tier eviction."""

    short_answer_max: int = 128
    medium_answer_max: int = 512
    potential_short: float = 1.0
    potential_medium: float = 0.5
    potential_long: float = 0.0
    default_potential: float = 0.5

    @classmethod
    def default(cls) -> "AnswerEvictionPolicy":
        return cls()

    @classmethod
    def from_profile_path(cls, path: str | Path) -> "AnswerEvictionPolicy":
        data = json.loads(Path(path).read_text())
        return cls.from_profile(data)

    @classmethod
    def from_profile(cls, data: dict[str, Any]) -> "AnswerEvictionPolicy":
        boundaries = data.get("bucket_boundaries", {})
        potentials = data.get("bucket_potential", {})
        return cls(
            short_answer_max=int(boundaries.get("short_answer_max", 128)),
            medium_answer_max=int(boundaries.get("medium_answer_max", 512)),
            potential_short=float(potentials.get("short", 1.0)),
            potential_medium=float(potentials.get("medium", 0.5)),
            potential_long=float(potentials.get("long", 0.0)),
            default_potential=float(data.get("default_potential", 0.5)),
        )

    def bucket_name(self, previous_answer_len: int | None) -> str:
        if previous_answer_len is None:
            return "unknown"
        if previous_answer_len <= self.short_answer_max:
            return "short"
        if previous_answer_len <= self.medium_answer_max:
            return "medium"
        return "long"

    def potential(self, previous_answer_len: int | None) -> float:
        bucket = self.bucket_name(previous_answer_len)
        if bucket == "short":
            return self.potential_short
        if bucket == "medium":
            return self.potential_medium
        if bucket == "long":
            return self.potential_long
        return self.default_potential


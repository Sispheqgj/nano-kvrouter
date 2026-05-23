from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TIER_DEFAULTS: dict[str, int] = {
    "gpu": 2_000,
    "cpu": 10_000,
    "disk": 100_000,
}


@dataclass(slots=True)
class _TierState:
    capacity: int
    allocated: set[str] = field(default_factory=set)

    @property
    def used(self) -> int:
        return len(self.allocated)

    @property
    def free(self) -> int:
        return self.capacity - self.used


class BlockPool:
    """Metadata-only block pool across GPU / CPU / Disk tiers."""

    def __init__(self, capacities: dict[str, int] | None = None) -> None:
        caps = {**TIER_DEFAULTS, **(capacities or {})}
        self._tiers: dict[str, _TierState] = {
            tier: _TierState(capacity=cap) for tier, cap in caps.items()
        }
        self._block_tier: dict[str, str] = {}  # block_id → tier

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_tier(self, tier: str) -> _TierState:
        try:
            return self._tiers[tier]
        except KeyError:
            raise ValueError(f"Unknown tier {tier!r}. Valid: {set(self._tiers)}")

    def _move(self, block_id: str, from_tier: str, to_tier: str) -> None:
        src = self._require_tier(from_tier)
        dst = self._require_tier(to_tier)

        current = self._block_tier.get(block_id)
        if current != from_tier:
            raise KeyError(
                f"Block {block_id!r} is on tier {current!r}, not {from_tier!r}"
            )
        if dst.free < 1:
            raise MemoryError(
                f"Tier {to_tier!r} is full (capacity {dst.capacity})"
            )

        src.allocated.discard(block_id)
        dst.allocated.add(block_id)
        self._block_tier[block_id] = to_tier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, num_blocks: int, tier: str) -> list[str]:
        """Allocate *num_blocks* blocks on *tier*; return their block_ids."""
        state = self._require_tier(tier)
        if num_blocks > state.free:
            raise MemoryError(
                f"Tier {tier!r}: requested {num_blocks} blocks, "
                f"only {state.free} free (capacity {state.capacity})"
            )
        ids = [str(uuid.uuid4()) for _ in range(num_blocks)]
        state.allocated.update(ids)
        for bid in ids:
            self._block_tier[bid] = tier
        logger.debug("Allocated %d block(s) on %s", num_blocks, tier)
        return ids

    def free(self, block_ids: list[str]) -> None:
        """Release blocks back to their tier."""
        for bid in block_ids:
            tier = self._block_tier.pop(bid, None)
            if tier is None:
                raise KeyError(f"Block {bid!r} is not allocated")
            self._tiers[tier].allocated.discard(bid)
        logger.debug("Freed %d block(s)", len(block_ids))

    def promote(self, block_id: str, from_tier: str, to_tier: str) -> None:
        """Move a block toward a faster tier (e.g. cpu → gpu)."""
        self._move(block_id, from_tier, to_tier)
        logger.debug("Promoted %s: %s → %s", block_id, from_tier, to_tier)

    def demote(self, block_id: str, from_tier: str, to_tier: str) -> None:
        """Move a block toward a slower tier (e.g. gpu → cpu)."""
        self._move(block_id, from_tier, to_tier)
        logger.debug("Demoted %s: %s → %s", block_id, from_tier, to_tier)

    def stats(self) -> dict[str, dict[str, int]]:
        """Return per-tier usage: capacity / used / free."""
        return {
            tier: {"capacity": s.capacity, "used": s.used, "free": s.free}
            for tier, s in self._tiers.items()
        }

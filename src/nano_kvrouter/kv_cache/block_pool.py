from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from nano_kvrouter.config import NodeConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TierState:
    """Internal bookkeeping for one storage tier.

    Tracks the block-id set rather than counters so that `free()` and
    `_move()` can validate membership in O(1) and fail loudly on stale
    block ids instead of silently double-counting.
    """

    capacity: int
    allocated: set[str] = field(default_factory=set)

    @property
    def used(self) -> int:
        return len(self.allocated)

    @property
    def free(self) -> int:
        return self.capacity - self.used


class BlockPool:
    """Metadata-only block pool across GPU / CPU / Disk tiers.

    Blocks are KV-cache slots; this pool only tracks *which tier a
    block lives on* and *how many slots each tier has free*. No real
    KV tensors exist — the simulator works entirely on block ids.

    Promotion (slower → faster) and demotion (faster → slower) are both
    expressed as `_move`; the public methods exist mainly to make
    intent — and log lines — readable when a scheduler decides to pull
    a block toward the GPU vs push it out to disk.

    Design notes:
        * Tier capacities come from `NodeConfig` so the YAML config is
          the single source of truth — same principle that drives
          `MockEngineNode`'s capacity / latency parameters.
        * Only the three named tiers ``"gpu"``, ``"cpu"``, ``"disk"``
          exist; extending the hierarchy would require touching this
          class *and* `NodeConfig` together, which is intentional.
        * Errors are raised, not swallowed: an unknown tier or an
          over-allocation is a programmer bug, not a runtime decision
          point. Callers (schedulers, cache managers) are expected to
          check `stats()` before allocating.
    """

    def __init__(self, node_config: NodeConfig) -> None:
        """Initialize an empty pool with per-tier capacities from config.

        Args:
            node_config: Source of `gpu_blocks` / `cpu_blocks` /
                `disk_blocks`. Each becomes the capacity of the
                correspondingly-named tier; the three tiers are fixed.
        """
        self._tiers: dict[str, _TierState] = {
            "gpu": _TierState(capacity=node_config.gpu_blocks),
            "cpu": _TierState(capacity=node_config.cpu_blocks),
            "disk": _TierState(capacity=node_config.disk_blocks),
        }
        self._block_tier: dict[str, str] = {}  # block_id → tier
        self._ref_count: dict[str, int] = {}
        self._pin_count: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_tier(self, tier: str) -> _TierState:
        """Return the `_TierState` for `tier` or raise `ValueError`."""
        try:
            return self._tiers[tier]
        except KeyError:
            raise ValueError(f"Unknown tier {tier!r}. Valid: {set(self._tiers)}")

    def _move(self, block_id: str, from_tier: str, to_tier: str) -> None:
        """Move one block between tiers, validating source and capacity.

        Raises `KeyError` if the block isn't on `from_tier` and
        `MemoryError` if the destination tier is full. Callers that may
        legitimately encounter a full destination must check `stats()`
        first or catch the error explicitly.
        """
        src = self._require_tier(from_tier)
        dst = self._require_tier(to_tier)

        current = self._block_tier.get(block_id)
        if current != from_tier:
            raise KeyError(
                f"Block {block_id!r} is on tier {current!r}, not {from_tier!r}"
            )
        if self.pin_count(block_id) > 0:
            raise RuntimeError(f"Block {block_id!r} is pinned and cannot be moved")
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
        """Allocate fresh block ids on a tier.

        Args:
            num_blocks: Number of blocks to allocate.
            tier: One of ``"gpu"``, ``"cpu"``, ``"disk"``.

        Returns:
            Newly minted block ids (UUID4 strings). The pool records
            each one's tier so subsequent `free` / `promote` / `demote`
            calls can find them.

        Raises:
            ValueError: `tier` is not one of the three known tiers.
            MemoryError: Not enough free slots on `tier`.
        """
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
            self._ref_count[bid] = 0
            self._pin_count[bid] = 0
        logger.debug("Allocated %d block(s) on %s", num_blocks, tier)
        return ids

    def free(self, block_ids: list[str]) -> None:
        """Release blocks back to their owning tier.

        Args:
            block_ids: Block ids to free. Each must currently be
                allocated; unknown ids raise `KeyError` so that
                accidental double-free surfaces immediately rather than
                silently corrupting capacity accounting.

        Raises:
            ValueError: Duplicate block ids in one call.
            KeyError: Any block id is not currently allocated.
            RuntimeError: Any block is pinned by an active request.
        """
        tiers: dict[str, str] = {}
        for bid in block_ids:
            if bid in tiers:
                raise ValueError(f"Duplicate block id {bid!r} in free()")
            tier = self._block_tier.get(bid)
            if tier is None:
                raise KeyError(f"Block {bid!r} is not allocated")
            if self.pin_count(bid) > 0:
                raise RuntimeError(f"Block {bid!r} is pinned and cannot be freed")
            tiers[bid] = tier

        for bid, tier in tiers.items():
            self._block_tier.pop(bid)
            self._tiers[tier].allocated.discard(bid)
            self._ref_count.pop(bid, None)
            self._pin_count.pop(bid, None)
        logger.debug("Freed %d block(s)", len(block_ids))

    def pin(self, block_ids: list[str]) -> None:
        """Increment ref/pin counters for active request ownership.

        Args:
            block_ids: Allocated block IDs to retain for an active request.

        Raises:
            KeyError: If any block is not currently allocated.
        """
        for bid in block_ids:
            if bid not in self._block_tier:
                raise KeyError(f"Block {bid!r} is not allocated")
        for bid in block_ids:
            self._ref_count[bid] = self._ref_count.get(bid, 0) + 1
            self._pin_count[bid] = self._pin_count.get(bid, 0) + 1

    def unpin(self, block_ids: list[str]) -> None:
        """Decrement ref/pin counters for active request release.

        Args:
            block_ids: Previously pinned block IDs.

        Raises:
            KeyError: If a block is unknown.
            RuntimeError: If a block is not currently pinned.
        """
        for bid in block_ids:
            if bid not in self._block_tier:
                raise KeyError(f"Block {bid!r} is not allocated")
            if self._pin_count.get(bid, 0) <= 0 or self._ref_count.get(bid, 0) <= 0:
                raise RuntimeError(f"Block {bid!r} is not pinned")
        for bid in block_ids:
            self._ref_count[bid] -= 1
            self._pin_count[bid] -= 1

    def ref_count(self, block_id: str) -> int:
        """Return active request reference count for *block_id*."""
        return self._ref_count.get(block_id, 0)

    def pin_count(self, block_id: str) -> int:
        """Return active request pin count for *block_id*."""
        return self._pin_count.get(block_id, 0)

    def promote(self, block_id: str, from_tier: str, to_tier: str) -> None:
        """Move a block toward a faster tier (e.g. ``cpu`` → ``gpu``).

        Args:
            block_id: Block to move.
            from_tier: Current tier (validated).
            to_tier: Destination tier; must have free capacity.

        Returns:
            None. Same semantics as `demote` — the directional split is
            for log clarity, not behavior.
        """
        self._move(block_id, from_tier, to_tier)
        logger.debug("Promoted %s: %s → %s", block_id, from_tier, to_tier)

    def demote(self, block_id: str, from_tier: str, to_tier: str) -> None:
        """Move a block toward a slower tier (e.g. ``gpu`` → ``cpu``).

        Args:
            block_id: Block to move.
            from_tier: Current tier (validated).
            to_tier: Destination tier; must have free capacity.

        Returns:
            None.
        """
        self._move(block_id, from_tier, to_tier)
        logger.debug("Demoted %s: %s → %s", block_id, from_tier, to_tier)

    def used(self, tier: str) -> int:
        """Number of allocated blocks on *tier*.

        Args:
            tier: One of ``"gpu"``, ``"cpu"``, ``"disk"``.

        Returns:
            Count of currently allocated block IDs on that tier.

        Raises:
            ValueError: *tier* is not one of the three known tiers.
        """
        return self._require_tier(tier).used

    def tier_breakdown(self) -> dict[str, dict[str, int]]:
        """Per-tier capacity / used / free snapshot.

        Returns:
            ``{tier: {"used": int, "free": int, "capacity": int}}``.
        """
        return {
            tier: {"used": s.used, "free": s.free, "capacity": s.capacity}
            for tier, s in self._tiers.items()
        }

    def tier_of(self, block_id: str) -> str | None:
        """Return the tier of *block_id*, or ``None`` if not allocated."""
        return self._block_tier.get(block_id)

    def tiers_of(self, block_ids: list[str]) -> dict[str, str]:
        """Map each *block_id* to its current tier, omitting freed blocks."""
        return {
            bid: tier
            for bid in block_ids
            if (tier := self._block_tier.get(bid)) is not None
        }

    def move(self, block_id: str, from_tier: str, to_tier: str) -> None:
        """Move one block between tiers (public wrapper around ``_move``).

        Args:
            block_id: Block to move.
            from_tier: Current tier (validated).
            to_tier: Destination tier; must have free capacity.

        Raises:
            KeyError: Block not on ``from_tier``.
            MemoryError: ``to_tier`` has no free slots.
        """
        self._move(block_id, from_tier, to_tier)
        logger.debug("Moved %s: %s -> %s", block_id, from_tier, to_tier)

    def stats(self) -> dict[str, dict[str, int]]:
        """Snapshot of per-tier capacity / used / free counts.

        Returns:
            ``{tier: {"capacity": int, "used": int, "free": int}}``.
            Schedulers consult this before requesting allocations or
            promotions so they can predict whether the call would
            raise `MemoryError`.
        """
        return {
            tier: {"capacity": s.capacity, "used": s.used, "free": s.free}
            for tier, s in self._tiers.items()
        }

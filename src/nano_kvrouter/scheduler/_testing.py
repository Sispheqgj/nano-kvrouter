"""Test stubs for the scheduler package.

Not part of the production API. Import only from ``tests/``.
"""
from __future__ import annotations

from dataclasses import dataclass

from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheLookup

__all__ = ["NullCacheQuery"]


@dataclass
class NullCacheQuery:
    """CacheQuery stub that always returns zero cache hits.

    Satisfies the :class:`~nano_kvrouter.scheduler.base.CacheQuery` Protocol
    via structural subtyping — no explicit inheritance. Intended for unit
    tests that need a scheduler exercised but don't care about real cache
    state.

    Attributes:
        node_ids: The set of node IDs this stub recognises. Queries for
            unknown IDs raise ``KeyError``, mirroring the real
            ``CacheManager`` contract.
        free_blocks_count: Constant returned by :meth:`free_blocks` for
            every ``(node_id, tier)`` pair. Default is large enough that
            tests never hit capacity limits unintentionally.
    """

    node_ids: list[str]
    free_blocks_count: int = 1024

    def lookup(self, request: Request, node_id: str) -> CacheLookup:
        """Return a zero-hit CacheLookup for *node_id*.

        Args:
            request: Ignored.
            node_id: Must be in ``self.node_ids``.

        Returns:
            :class:`CacheLookup` with ``matched_tokens=0``, empty tier
            breakdown, and zero transfer cost.

        Raises:
            KeyError: If *node_id* is not in ``self.node_ids``.
        """
        if node_id not in self.node_ids:
            raise KeyError(node_id)
        return CacheLookup(matched_tokens=0, matched_blocks_by_tier={}, transfer_cost_ms=0.0)

    def lookup_all(self, request: Request) -> dict[str, CacheLookup]:
        """Return zero-hit CacheLookup for every known node.

        Args:
            request: Ignored.

        Returns:
            Mapping ``node_id → CacheLookup`` with zero cache hits for
            each node in ``self.node_ids``.
        """
        return {
            nid: CacheLookup(matched_tokens=0, matched_blocks_by_tier={}, transfer_cost_ms=0.0)
            for nid in self.node_ids
        }

    def free_blocks(self, node_id: str, tier: str) -> int:
        """Return ``self.free_blocks_count`` for any valid node and tier.

        Args:
            node_id: Must be in ``self.node_ids``.
            tier: Ignored — all tiers return the same count.

        Returns:
            ``self.free_blocks_count``.

        Raises:
            KeyError: If *node_id* is not in ``self.node_ids``.
        """
        if node_id not in self.node_ids:
            raise KeyError(node_id)
        return self.free_blocks_count

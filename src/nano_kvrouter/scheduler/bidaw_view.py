"""Read-only Bidaw controller view consumed by the scheduler."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["BidawControllerView"]


@runtime_checkable
class BidawControllerView(Protocol):
    """Narrow read-only view over Bidaw admission-controller state.

    BidawPolicy uses this protocol for optional M3 routing features without
    gaining access to controller mutators or simulator event scheduling.
    """

    def peek_preparing_disk_blocks(self, decode_node_id: str) -> int:
        """Return queued preparing disk blocks for *decode_node_id*."""
        ...

    def peek_in_flight_disk_blocks(self, decode_node_id: str) -> int:
        """Return disk blocks currently occupying the node's load slot."""
        ...

    def peek_projected_preparing_wait_ms(
        self,
        decode_node_id: str,
        my_disk_blocks: int,
        now_ms: float,
    ) -> float:
        """Return projected preparing wait/load time for a candidate request."""
        ...

    def peek_session_affinity(self, session_id: str) -> str | None:
        """Return pinned decode node for *session_id*, if any."""
        ...

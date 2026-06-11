from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RequestBlockTable:
    """Logical-to-physical KV block table for one active request."""

    request_id: str
    node_id: str
    block_ids: list[str]
    matched_blocks: int
    new_blocks: int

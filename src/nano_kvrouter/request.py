from __future__ import annotations

import hashlib
import struct
import uuid
from dataclasses import dataclass, field

from nano_kvrouter.config import NanoKVConfig


def _hash_prefix(token_ids: list[int], prefix_len: int) -> str:
    """Hash the first `prefix_len` tokens for cache-key matching.

    Args:
        token_ids: Full token sequence of the request.
        prefix_len: Number of leading tokens to hash — typically the
            model's `block_size`, so the hash identifies the first KV
            block of a prompt.

    Returns:
        First 8 hex chars of SHA-256 over the packed prefix bytes. 32-bit
        collision space is enough for simulation-scale workloads (~k
        requests per run); using a fixed-width prefix keeps RadixTree
        keys compact and `Request` debug logs readable.
    """
    prefix = token_ids[:prefix_len]
    # Pack as big-endian unsigned ints so the byte representation is
    # stable across platforms and independent of Python's int repr.
    data = struct.pack(f">{len(prefix)}I", *prefix) if prefix else b""
    return hashlib.sha256(data).hexdigest()[:8]


@dataclass(slots=True)
class Request:
    """A single inference request flowing through the simulator.

    Carries everything a scheduler needs to make a routing decision:
    full token sequence (for RadixTree prefix lookup), the prefix hash
    (for fast first-block matching), the SLO targets it must meet, and
    arrival/priority metadata for queue ordering.

    Notes:
        `token_ids` are random integers — no real tokenizer is involved.
        `prefix_hash` is a derived field but stored to avoid recomputing
        it on every scheduling pass.
    """

    request_id: str
    token_ids: list[int]
    prefix_hash: str
    expected_output_len: int
    arrival_time: float
    slo_ttft: float
    slo_tbt: float
    priority: int = field(default=0)


def make_request(
    token_ids: list[int],
    arrival_time: float,
    config: NanoKVConfig,
    *,
    priority: int = 0,
    expected_output_len: int | None = None,
) -> Request:
    """Construct a `Request` with SLO and output-length fields filled from config.

    Args:
        token_ids: Prompt token sequence (random ints in this simulator).
        arrival_time: Simulated arrival timestamp (ms), produced by the
            event loop, not wall-clock.
        config: Full `NanoKVConfig` — model.block_size is consulted for
            the prefix hash, slo.* for the per-request SLO stamp,
            workload.avg_output_len as the expected output length fallback.
        priority: Optional scheduling priority. Higher = earlier; equal
            priorities preserve arrival order.
        expected_output_len: Override the expected output length for this
            request. When None (default), falls back to
            config.workload.avg_output_len so existing Poisson callers
            require no change.

    Returns:
        A freshly-initialized `Request` with a new UUID `request_id` and
        a `token_ids` copy (defensive — caller's list is not retained).
    """
    if expected_output_len is None:
        expected_output_len = config.workload.avg_output_len
    prefix_hash = _hash_prefix(token_ids, config.model.block_size)
    return Request(
        request_id=str(uuid.uuid4()),
        token_ids=list(token_ids),
        prefix_hash=prefix_hash,
        expected_output_len=expected_output_len,
        arrival_time=arrival_time,
        slo_ttft=config.slo.ttft_target_ms,
        slo_tbt=config.slo.tbt_target_ms,
        priority=priority,
    )

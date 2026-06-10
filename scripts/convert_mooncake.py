#!/usr/bin/env python3
"""Convert Mooncake upstream JSONL to nano-kvrouter internal JSONL.

Upstream Mooncake JSONL already uses the same field names as the internal
schema (timestamp/input_length/output_length/hash_ids), so this script
mainly does:
  - Field existence validation (skip + warn on missing fields)
  - hash_ids length consistency check: len(hash_ids) == ceil(input_length/512)
  - Timestamp zero-normalization (Mooncake is already 0-based, but for safety)
  - Write valid records to output path

Usage:
    python scripts/convert_mooncake.py <input.jsonl> <output.jsonl>
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"timestamp", "input_length", "output_length", "hash_ids"}
MOONCAKE_BLOCK_SIZE = 512


def convert(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Convert Mooncake JSONL to internal format.

    Args:
        input_path: Path to upstream Mooncake JSONL.
        output_path: Path to write internal JSONL.

    Returns:
        (total_read, total_written) line counts.
    """
    first_timestamp: int | None = None
    total_read = 0
    total_written = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for lineno, line in enumerate(fin, start=1):
            total_read += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("line %d: JSON parse error (%s), skipping", lineno, exc)
                continue

            missing = REQUIRED_FIELDS - set(record)
            if missing:
                logger.warning("line %d: missing fields %s, skipping", lineno, missing)
                continue

            input_length = record["input_length"]
            hash_ids = record["hash_ids"]
            expected_n_blocks = math.ceil(input_length / MOONCAKE_BLOCK_SIZE)
            if len(hash_ids) != expected_n_blocks:
                logger.warning(
                    "line %d: len(hash_ids)=%d != ceil(%d/512)=%d, skipping",
                    lineno, len(hash_ids), input_length, expected_n_blocks,
                )
                continue

            # Zero-normalize timestamp (Mooncake already 0-based, but defensive)
            ts = record["timestamp"]
            if first_timestamp is None:
                first_timestamp = ts
            ts_normalized = ts - first_timestamp

            out = {
                "timestamp": ts_normalized,
                "input_length": input_length,
                "output_length": record["output_length"],
                "hash_ids": hash_ids,
            }
            fout.write(json.dumps(out) + "\n")
            total_written += 1

    return total_read, total_written


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_read, total_written = convert(input_path, output_path)
    logger.info("Processed %d lines, wrote %d valid records to %s",
                total_read, total_written, output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert BurstGPT CSV trace to nano-kvrouter internal JSONL format.

Usage:
    python scripts/convert_burstgpt.py <input.csv> <output.jsonl> [--limit N]

BurstGPT CSV schema (verified 2026-06-09):
    Timestamp,Session ID,Elapsed time,Model,Request tokens,Response tokens,...
    - Timestamp: seconds, NOT zero-based (epoch-like)
    - Session ID: UUID string
    - Request tokens: int (input_length)
    - Response tokens: int (output_length)

Output JSONL schema (one object per line):
    {"request_id": "burstgpt-<N>", "arrival_ms": <float>, "input_length": <int>,
     "output_length": <int>, "session_id": "<uuid>"}

Note: no hash_ids field — BurstGPT has no prefix structure.
session_id is preserved for future P3-D use; M2 does not use it for synthesis.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"Timestamp", "Session ID", "Request tokens", "Response tokens"}


def convert(input_path: Path, output_path: Path, limit: int | None = None, require_session_id: bool = False) -> int:
    """Stream-convert BurstGPT CSV to JSONL.

    Returns:
        Number of rows written.
    """
    written = 0
    skipped = 0
    first_timestamp: float | None = None

    with input_path.open(encoding="utf-8", newline="") as csv_file, \
         output_path.open("w", encoding="utf-8") as out_file:

        reader = csv.DictReader(csv_file)

        for row_idx, row in enumerate(reader):
            if limit is not None and written >= limit:
                break

            # Check required fields
            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                logger.warning("row %d: skipping — missing fields %s", row_idx, missing)
                skipped += 1
                continue

            # Parse timestamp
            try:
                ts = float(row["Timestamp"])
            except (ValueError, TypeError):
                logger.warning("row %d: skipping — non-numeric Timestamp %r", row_idx, row.get("Timestamp"))
                skipped += 1
                continue

            # Parse lengths
            try:
                input_length = int(row["Request tokens"])
                output_length = int(row["Response tokens"])
            except (ValueError, TypeError):
                logger.warning("row %d: skipping — non-int token counts", row_idx)
                skipped += 1
                continue

            if input_length <= 0:
                logger.warning("row %d: skipping — input_length=%d <= 0", row_idx, input_length)
                skipped += 1
                continue
            if output_length <= 0:
                logger.warning("row %d: skipping — output_length=%d <= 0", row_idx, output_length)
                skipped += 1
                continue

            if require_session_id and not row.get("Session ID", "").strip():
                logger.warning("row %d: skipping — blank Session ID (API log)", row_idx)
                skipped += 1
                continue

            # Zero-base timestamps on first valid row
            if first_timestamp is None:
                first_timestamp = ts
            arrival_ms = (ts - first_timestamp) * 1000.0

            record = {
                "request_id": f"burstgpt-{written}",
                "arrival_ms": arrival_ms,
                "input_length": input_length,
                "output_length": output_length,
                "session_id": row.get("Session ID", ""),
            }
            out_file.write(json.dumps(record) + "\n")
            written += 1

    logger.info("converted: %d rows written, %d skipped", written, skipped)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BurstGPT CSV to nano-kvrouter JSONL"
    )
    parser.add_argument("input_csv", help="Path to BurstGPT CSV file")
    parser.add_argument("output_jsonl", help="Path to output JSONL file")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of output rows (for generating sample files)"
    )
    parser.add_argument(
        "--require-session-id",
        action="store_true",
        help="Skip rows with blank Session ID (filters out API-log entries)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_jsonl)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = convert(input_path, output_path, limit=args.limit, require_session_id=args.require_session_id)
    if n == 0:
        logger.error("No rows written — check input file format")
        sys.exit(1)


if __name__ == "__main__":
    main()

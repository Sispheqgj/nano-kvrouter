#!/usr/bin/env python3
"""Convert Bidaw interactive conversation workload CSV to nano-kvrouter JSONL.

Expected upstream columns:
    User_id, Timestamp(seconds), Query_length, Response_length, Round_index

The output JSONL is consumed by TraceGenerator with
``trace.prefix_mode: session_history``. A second optional JSON profile captures
answer-length buckets and weighted reuse-distance-derived hit potential for
Bidaw previous-answer-based eviction.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "User_id",
    "Timestamp(seconds)",
    "Query_length",
    "Response_length",
    "Round_index",
}


@dataclass(slots=True)
class _RawTurn:
    user_id: str
    timestamp_s: float
    query_length: int
    response_length: int
    round_index: int


@dataclass(slots=True)
class _Access:
    user_id: str
    kv_size_after: int


def _parse_rows(input_path: Path, limit: int | None) -> list[_RawTurn]:
    rows: list[_RawTurn] = []
    skipped = 0
    with input_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader):
            if limit is not None and len(rows) >= limit:
                break
            try:
                user_id = str(row["User_id"]).strip()
                timestamp_s = float(row["Timestamp(seconds)"])
                query_length = int(row["Query_length"])
                response_length = int(row["Response_length"])
                round_index = int(row["Round_index"])
            except (TypeError, ValueError) as exc:
                logger.warning("row %d: skipping unparsable row (%s)", row_idx, exc)
                skipped += 1
                continue
            if not user_id or query_length <= 0 or response_length <= 0:
                logger.warning("row %d: skipping invalid values", row_idx)
                skipped += 1
                continue
            rows.append(_RawTurn(
                user_id=user_id,
                timestamp_s=timestamp_s,
                query_length=query_length,
                response_length=response_length,
                round_index=round_index,
            ))
    rows.sort(key=lambda r: (r.timestamp_s, r.user_id, r.round_index))
    logger.info("parsed: %d valid turns, %d skipped", len(rows), skipped)
    return rows


def _quantile(values: list[int], q: float, default: int) -> int:
    if not values:
        return default
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _profile_potential(wrd: int, performance_layer_tokens: int, ghost_layer_tokens: int) -> float:
    if wrd <= performance_layer_tokens:
        return 1.0
    if wrd <= ghost_layer_tokens:
        return 0.5
    return 0.0


def _build_profile(
    observations: list[tuple[int, int]],
    performance_layer_tokens: int,
    ghost_layer_tokens: int,
) -> dict[str, Any]:
    answer_lengths = [prev_answer for prev_answer, _wrd in observations]
    short_max = _quantile(answer_lengths, 1 / 3, 128)
    medium_max = max(short_max, _quantile(answer_lengths, 2 / 3, 512))

    buckets: dict[str, list[tuple[int, int]]] = {"short": [], "medium": [], "long": []}
    for prev_answer, wrd in observations:
        if prev_answer <= short_max:
            buckets["short"].append((prev_answer, wrd))
        elif prev_answer <= medium_max:
            buckets["medium"].append((prev_answer, wrd))
        else:
            buckets["long"].append((prev_answer, wrd))

    bucket_potential: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    bucket_avg_wrd: dict[str, float] = {}
    for name, vals in buckets.items():
        bucket_counts[name] = len(vals)
        if not vals:
            bucket_potential[name] = 0.5
            bucket_avg_wrd[name] = 0.0
            continue
        potentials = [
            _profile_potential(wrd, performance_layer_tokens, ghost_layer_tokens)
            for _answer, wrd in vals
        ]
        bucket_potential[name] = sum(potentials) / len(potentials)
        bucket_avg_wrd[name] = sum(wrd for _answer, wrd in vals) / len(vals)

    return {
        "source": "ShipengHu-777/Interactive-conversation-workload",
        "performance_layer_tokens": performance_layer_tokens,
        "ghost_layer_tokens": ghost_layer_tokens,
        "observations": len(observations),
        "bucket_boundaries": {
            "short_answer_max": short_max,
            "medium_answer_max": medium_max,
        },
        "bucket_potential": bucket_potential,
        "bucket_counts": bucket_counts,
        "bucket_avg_wrd": bucket_avg_wrd,
        "default_potential": 0.5,
    }


def convert(
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path | None = None,
    limit: int | None = None,
    performance_layer_tokens: int = 8192,
    ghost_layer_tokens: int | None = None,
) -> tuple[int, int]:
    """Convert CSV turns to internal JSONL and optional eviction profile.

    Returns:
        ``(total_valid_rows, written_rows)``.
    """
    if ghost_layer_tokens is None:
        ghost_layer_tokens = performance_layer_tokens * 4
    rows = _parse_rows(input_path, limit)
    if not rows:
        output_path.write_text("")
        if profile_path is not None:
            profile = _build_profile([], performance_layer_tokens, ghost_layer_tokens)
            profile_path.write_text(json.dumps(profile, indent=2))
        return 0, 0

    first_ts = rows[0].timestamp_s
    session_tokens: dict[str, int] = {}
    previous_answer: dict[str, int] = {}
    last_access_idx: dict[str, int] = {}
    accesses: list[_Access] = []
    observations: list[tuple[int, int]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as out_file:
        for turn in rows:
            prev_answer_len = previous_answer.get(turn.user_id, 0)
            history_len = session_tokens.get(turn.user_id, 0)
            input_length = history_len + turn.query_length

            if turn.user_id in last_access_idx:
                start = last_access_idx[turn.user_id] + 1
                unique_other_sizes: dict[str, int] = {}
                for access in accesses[start:]:
                    if access.user_id == turn.user_id:
                        continue
                    unique_other_sizes.setdefault(access.user_id, access.kv_size_after)
                observations.append((prev_answer_len, sum(unique_other_sizes.values())))

            record = {
                "request_id": f"interactive-{written}",
                "arrival_ms": (turn.timestamp_s - first_ts) * 1000.0,
                "input_length": input_length,
                "output_length": turn.response_length,
                "session_id": turn.user_id,
                "round_index": turn.round_index,
                "query_length": turn.query_length,
                "previous_answer_length": prev_answer_len,
            }
            out_file.write(json.dumps(record) + "\n")
            written += 1

            session_tokens[turn.user_id] = (
                history_len + turn.query_length + turn.response_length
            )
            previous_answer[turn.user_id] = turn.response_length
            accesses.append(_Access(turn.user_id, session_tokens[turn.user_id]))
            last_access_idx[turn.user_id] = len(accesses) - 1

    if profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile = _build_profile(observations, performance_layer_tokens, ghost_layer_tokens)
        profile_path.write_text(json.dumps(profile, indent=2) + "\n")

    logger.info("converted: %d rows written to %s", written, output_path)
    return len(rows), written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Bidaw interactive conversation workload CSV"
    )
    parser.add_argument("input_csv", help="Path to upstream CSV")
    parser.add_argument("output_jsonl", help="Path to output internal JSONL")
    parser.add_argument("--profile-json", help="Optional eviction profile JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum output rows")
    parser.add_argument(
        "--performance-layer-tokens",
        type=int,
        default=8192,
        help="CPU/performance layer size used for WRD bucket potential",
    )
    parser.add_argument(
        "--ghost-layer-tokens",
        type=int,
        default=None,
        help="Ghost-cache window size; defaults to 4x performance-layer-tokens",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    total, written = convert(
        input_path,
        Path(args.output_jsonl),
        profile_path=Path(args.profile_json) if args.profile_json else None,
        limit=args.limit,
        performance_layer_tokens=args.performance_layer_tokens,
        ghost_layer_tokens=args.ghost_layer_tokens,
    )
    if total == 0 or written == 0:
        logger.error("No rows written; check input format")
        sys.exit(1)


if __name__ == "__main__":
    main()

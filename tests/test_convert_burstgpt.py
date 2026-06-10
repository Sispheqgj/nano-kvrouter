"""Tests for convert_burstgpt.py converter."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# Import the converter module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import convert_burstgpt


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a CSV file with BurstGPT headers."""
    fieldnames = ["Timestamp", "Session ID", "Elapsed time", "Model",
                  "Request tokens", "Response tokens", "Total tokens", "Log Type"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            full_row = {k: "" for k in fieldnames}
            full_row.update(row)
            w.writerow(full_row)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: basic conversion
# ─────────────────────────────────────────────────────────────────────────────

def test_basic_conversion(tmp_path: Path):
    """3-row inline CSV → 3 JSONL rows with correct field mapping."""
    csv_path = tmp_path / "input.csv"
    out_path = tmp_path / "output.jsonl"

    _write_csv(csv_path, [
        {"Timestamp": "100.0", "Session ID": "uuid-1", "Request tokens": "906", "Response tokens": "446"},
        {"Timestamp": "161.0", "Session ID": "uuid-2", "Request tokens": "36",  "Response tokens": "29"},
        {"Timestamp": "192.0", "Session ID": "uuid-3", "Request tokens": "1779","Response tokens": "123"},
    ])

    n = convert_burstgpt.convert(csv_path, out_path)
    assert n == 3

    rows = _read_jsonl(out_path)
    assert len(rows) == 3

    # Check field names
    for row in rows:
        assert "request_id" in row
        assert "arrival_ms" in row
        assert "input_length" in row
        assert "output_length" in row
        assert "session_id" in row

    # Check values for first row
    assert rows[0]["input_length"] == 906
    assert rows[0]["output_length"] == 446
    assert rows[0]["session_id"] == "uuid-1"
    assert "hash_ids" not in rows[0], "BurstGPT output must not have hash_ids"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: timestamp zero offset
# ─────────────────────────────────────────────────────────────────────────────

def test_timestamp_zero_offset(tmp_path: Path):
    """First row Timestamp=100, second=200 → arrival_ms 0 and 100000."""
    csv_path = tmp_path / "input.csv"
    out_path = tmp_path / "output.jsonl"

    _write_csv(csv_path, [
        {"Timestamp": "100.0", "Session ID": "s1", "Request tokens": "10", "Response tokens": "5"},
        {"Timestamp": "200.0", "Session ID": "s2", "Request tokens": "20", "Response tokens": "8"},
    ])

    convert_burstgpt.convert(csv_path, out_path)
    rows = _read_jsonl(out_path)

    assert rows[0]["arrival_ms"] == pytest.approx(0.0)
    assert rows[1]["arrival_ms"] == pytest.approx(100_000.0)  # (200-100)*1000


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: skip invalid rows
# ─────────────────────────────────────────────────────────────────────────────

def test_skip_invalid_rows(tmp_path: Path):
    """Rows with missing fields or input_length=0 are skipped; rest output."""
    csv_path = tmp_path / "input.csv"
    out_path = tmp_path / "output.jsonl"

    _write_csv(csv_path, [
        {"Timestamp": "100.0", "Session ID": "s1", "Request tokens": "50",  "Response tokens": "10"},
        {"Timestamp": "200.0", "Session ID": "s2", "Request tokens": "0",   "Response tokens": "10"},   # skip
        {"Timestamp": "300.0", "Session ID": "s3", "Request tokens": "100", "Response tokens": "0"},   # skip
        {"Timestamp": "400.0", "Session ID": "s4", "Request tokens": "200", "Response tokens": "30"},
    ])

    n = convert_burstgpt.convert(csv_path, out_path)
    assert n == 2, f"Expected 2 valid rows, got {n}"

    rows = _read_jsonl(out_path)
    assert len(rows) == 2
    assert rows[0]["input_length"] == 50
    assert rows[1]["input_length"] == 200


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: --limit arg
# ─────────────────────────────────────────────────────────────────────────────

def test_limit_arg(tmp_path: Path):
    """CSV with 5 rows + --limit 3 → exactly 3 rows output."""
    csv_path = tmp_path / "input.csv"
    out_path = tmp_path / "output.jsonl"

    _write_csv(csv_path, [
        {"Timestamp": str(100.0 + i * 10), "Session ID": f"s{i}",
         "Request tokens": str(50 + i), "Response tokens": "10"}
        for i in range(5)
    ])

    n = convert_burstgpt.convert(csv_path, out_path, limit=3)
    assert n == 3, f"Expected 3 rows with limit=3, got {n}"

    rows = _read_jsonl(out_path)
    assert len(rows) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: --require-session-id filters blank session_id rows
# ─────────────────────────────────────────────────────────────────────────────

def test_require_session_id_filters_blank(tmp_path: Path):
    """important 3: require_session_id=True skips rows with blank session_id."""
    csv_path = tmp_path / "input.csv"
    out_path_with = tmp_path / "with_filter.jsonl"
    out_path_without = tmp_path / "without_filter.jsonl"

    _write_csv(csv_path, [
        {"Timestamp": "100.0", "Session ID": "uuid-1", "Request tokens": "50", "Response tokens": "10"},
        {"Timestamp": "200.0", "Session ID": "",       "Request tokens": "60", "Response tokens": "15"},  # blank
        {"Timestamp": "300.0", "Session ID": "uuid-3", "Request tokens": "70", "Response tokens": "20"},
    ])

    # Without flag: all 3 rows emitted
    n_without = convert_burstgpt.convert(csv_path, out_path_without, require_session_id=False)
    assert n_without == 3, f"Expected 3 without filter, got {n_without}"

    # With flag: blank session_id row skipped → 2 rows
    n_with = convert_burstgpt.convert(csv_path, out_path_with, require_session_id=True)
    assert n_with == 2, f"Expected 2 with filter, got {n_with}"

    rows = _read_jsonl(out_path_with)
    assert all(r["session_id"] != "" for r in rows), "All rows should have non-blank session_id"

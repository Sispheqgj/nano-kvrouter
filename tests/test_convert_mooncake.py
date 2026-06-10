from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.convert_mooncake import convert


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_missing_field_row_skipped(tmp_path: Path) -> None:
    """3 inline rows, 1 missing required field → 2 valid output rows."""
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    block = 512
    rows = [
        # valid
        {"timestamp": 0, "input_length": 512, "output_length": 128,
         "hash_ids": [1]},
        # missing hash_ids → should be skipped
        {"timestamp": 1000, "input_length": 512, "output_length": 64},
        # valid
        {"timestamp": 2000, "input_length": 1024, "output_length": 256,
         "hash_ids": [2, 3]},
    ]
    _write_jsonl(inp, rows)
    total_read, total_written = convert(inp, out)

    assert total_read == 3
    assert total_written == 2
    written = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(written) == 2
    assert written[0]["input_length"] == 512
    assert written[1]["input_length"] == 1024


def test_inconsistent_hash_ids_length_skipped(tmp_path: Path) -> None:
    """Rows where len(hash_ids) != ceil(input_length/512) are skipped."""
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        # valid: ceil(512/512)=1, len(hash_ids)=1
        {"timestamp": 0, "input_length": 512, "output_length": 64,
         "hash_ids": [10]},
        # invalid: ceil(1024/512)=2, but len=3
        {"timestamp": 500, "input_length": 1024, "output_length": 64,
         "hash_ids": [1, 2, 3]},
        # invalid: ceil(768/512)=2, but len=1
        {"timestamp": 1000, "input_length": 768, "output_length": 64,
         "hash_ids": [99]},
    ]
    _write_jsonl(inp, rows)
    total_read, total_written = convert(inp, out)

    assert total_read == 3
    assert total_written == 1
    written = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(written) == 1
    assert written[0]["input_length"] == 512

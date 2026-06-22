from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_converter() -> ModuleType:
    script_path = Path(__file__).parent.parent / "scripts" / "convert_interactive_workload.py"
    spec = importlib.util.spec_from_file_location("convert_interactive_workload", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "User_id",
                "Timestamp(seconds)",
                "Query_length",
                "Response_length",
                "Round_index",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_convert_interactive_workload_builds_session_history_and_profile(
    tmp_path: Path,
) -> None:
    converter = _load_converter()
    input_csv = tmp_path / "interactive.csv"
    output_jsonl = tmp_path / "interactive.jsonl"
    profile_json = tmp_path / "profile.json"
    _write_csv(
        input_csv,
        [
            {
                "User_id": "u1",
                "Timestamp(seconds)": 10.0,
                "Query_length": 16,
                "Response_length": 4,
                "Round_index": 0,
            },
            {
                "User_id": "u2",
                "Timestamp(seconds)": 10.5,
                "Query_length": 16,
                "Response_length": 100,
                "Round_index": 0,
            },
            {
                "User_id": "u1",
                "Timestamp(seconds)": 11.0,
                "Query_length": 8,
                "Response_length": 5,
                "Round_index": 1,
            },
        ],
    )

    total, written = converter.convert(
        input_csv,
        output_jsonl,
        profile_path=profile_json,
        performance_layer_tokens=32,
        ghost_layer_tokens=128,
    )

    assert (total, written) == (3, 3)
    records = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert records[0]["arrival_ms"] == 0.0
    assert records[1]["arrival_ms"] == 500.0
    assert records[2]["session_id"] == "u1"
    assert records[2]["input_length"] == 16 + 4 + 8
    assert records[2]["previous_answer_length"] == 4

    profile = json.loads(profile_json.read_text())
    assert profile["observations"] == 1
    assert set(profile["bucket_potential"]) == {"short", "medium", "long"}

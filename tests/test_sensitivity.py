from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from nano_kvrouter.cli import (
    _resolve_related_config_path,
    _run_sensitivity_experiment,
    _run_sensitivity_report,
    _set_nested_config_value,
    main,
)
from nano_kvrouter.config import NanoKVConfig, SensitivityExperiment


def _write_base_config(path: Path) -> None:
    path.write_text(
        """
cluster:
  prefill_nodes: 1
  decode_nodes: 1

node:
  gpu_blocks: 64
  cpu_blocks: 0
  disk_blocks: 0
  capacity: 4

model:
  block_size: 16
  kv_bytes_per_token: 512
  prefill_cost_per_token_ms: 0.1
  decode_base_ms: 5.0
  marginal_decode_ms: 0.5
  prefill_chunk_size: 128

bandwidth:
  gpu_to_gpu: 300000000000.0
  gpu_to_cpu: 32000000000.0
  cpu_to_disk: 5000000000.0

slo:
  ttft_target_ms: 9999.0
  tbt_target_ms: 9999.0

workload:
  request_rate: 8.0
  duration_s: 1.0
  prefix_sharing_ratio: 0.6
  avg_prompt_len: 128
  avg_output_len: 8

generator:
  num_buckets: 4
  vocab_size: 32000
  seed: 42

scheduler:
  name: conductor
  params:
    alpha: 1.0
    beta: 1.0
    gamma: 1.0
""".strip()
    )


def _write_sensitivity_config(path: Path, base_name: str = "base.yaml") -> None:
    path.write_text(
        f"""
experiments:
  - field: model.decode_base_ms
    base_config: {base_name}
    scheduler: conductor
    values: [10.0]
    primary_metrics: [tbt_avg_ms, ttft_p50_ms]

  - field: model.prefill_chunk_size
    base_config: {base_name}
    scheduler: conductor
    values: [64]
    primary_metrics: [avg_chunked_prefill_steps_per_request]
""".strip()
    )


def test_set_nested_config_value_returns_deep_copy() -> None:
    cfg = NanoKVConfig()

    updated = _set_nested_config_value(cfg, "model.decode_base_ms", 9.0)

    assert updated.model.decode_base_ms == pytest.approx(9.0)
    assert cfg.model.decode_base_ms == pytest.approx(5.0)


def test_set_nested_config_value_unknown_path_raises() -> None:
    with pytest.raises(AttributeError, match="Unknown config field path"):
        _set_nested_config_value(NanoKVConfig(), "model.no_such_field", 1)


def test_resolve_related_config_path_prefers_sensitivity_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    base_cfg = suite_dir / "default.yaml"
    sensitivity_cfg = suite_dir / "sensitivity.yaml"
    _write_base_config(base_cfg)
    _write_sensitivity_config(sensitivity_cfg, base_name="default.yaml")

    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    resolved = _resolve_related_config_path(str(sensitivity_cfg.resolve()), "default.yaml")

    assert resolved == base_cfg.resolve()


def test_sensitivity_cli_smoke_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base_cfg = tmp_path / "base.yaml"
    sensitivity_cfg = tmp_path / "sensitivity.yaml"
    output = tmp_path / "sensitivity.json"
    _write_base_config(base_cfg)
    _write_sensitivity_config(sensitivity_cfg)

    old_argv = sys.argv
    sys.argv = [
        "nano-kvrouter",
        "sensitivity",
        "--config",
        str(sensitivity_cfg),
        "--output",
        str(output),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    assert output.exists()
    captured = capsys.readouterr()
    assert "model.decode_base_ms" in captured.out
    assert "PASS" in captured.out


def test_sensitivity_report_json_schema(tmp_path: Path) -> None:
    base_cfg = tmp_path / "base.yaml"
    sensitivity_cfg = tmp_path / "sensitivity.yaml"
    output = tmp_path / "report.json"
    _write_base_config(base_cfg)
    _write_sensitivity_config(sensitivity_cfg)

    report = _run_sensitivity_report(str(sensitivity_cfg))
    output.write_text(json.dumps(report, indent=2, default=str))
    payload = json.loads(output.read_text())

    assert "experiments" in payload
    first = payload["experiments"][0]
    assert first["field"] == "model.decode_base_ms"
    assert "baseline" in first
    assert "candidates" in first
    assert "changed" in first
    assert first["candidates"][0]["metrics"]["baseline"]["tbt_avg_ms"] is not None


def test_sensitivity_report_uses_absolute_config_path_from_other_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    base_cfg = suite_dir / "default.yaml"
    sensitivity_cfg = suite_dir / "sensitivity.yaml"
    _write_base_config(base_cfg)
    _write_sensitivity_config(sensitivity_cfg, base_name="default.yaml")

    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    report = _run_sensitivity_report(str(sensitivity_cfg.resolve()))

    assert report["experiments"][0]["baseline"]["value"] == pytest.approx(5.0)
    assert report["experiments"][0]["changed"] is True


def test_decode_base_ms_experiment_changes_tbt_avg_ms(tmp_path: Path) -> None:
    base_cfg = tmp_path / "base.yaml"
    sensitivity_cfg = tmp_path / "sensitivity.yaml"
    _write_base_config(base_cfg)
    _write_sensitivity_config(sensitivity_cfg)

    experiment = SensitivityExperiment(
        field="model.decode_base_ms",
        base_config="base.yaml",
        scheduler="conductor",
        values=[10.0],
        primary_metrics=["tbt_avg_ms", "ttft_p50_ms"],
    )
    result = _run_sensitivity_experiment(
        experiment,
        sensitivity_config_path=str(sensitivity_cfg),
    )

    candidate = result["candidates"][0]
    assert result["changed"] is True
    assert candidate["changed"] is True
    assert "tbt_avg_ms" in candidate["changed_metrics"]
    assert candidate["metrics"]["candidate"]["tbt_avg_ms"] > candidate["metrics"]["baseline"]["tbt_avg_ms"]

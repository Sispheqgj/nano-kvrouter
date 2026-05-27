from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nano_kvrouter.config import (
    NanoKVConfig,
    load_config,
)

DEFAULT_YAML = Path(__file__).parent.parent / "configs" / "default.yaml"


# ------------------------------------------------------------------
# 加载 default.yaml
# ------------------------------------------------------------------

def test_load_default_yaml_succeeds():
    cfg = load_config(str(DEFAULT_YAML))
    assert isinstance(cfg, NanoKVConfig)


def test_default_yaml_cluster_values():
    cfg = load_config(str(DEFAULT_YAML))
    assert cfg.cluster.prefill_nodes == 4
    assert cfg.cluster.decode_nodes == 4


def test_default_yaml_node_values():
    cfg = load_config(str(DEFAULT_YAML))
    assert cfg.node.gpu_blocks == 2_000
    assert cfg.node.cpu_blocks == 10_000
    assert cfg.node.disk_blocks == 100_000


def test_default_yaml_model_values():
    cfg = load_config(str(DEFAULT_YAML))
    assert cfg.model.block_size == 16
    assert cfg.model.kv_bytes_per_token == 512
    assert cfg.model.prefill_cost_per_token_ms == pytest.approx(0.033)
    assert cfg.model.decode_base_ms == pytest.approx(5.0)


def test_default_yaml_bandwidth_values():
    cfg = load_config(str(DEFAULT_YAML))
    assert cfg.bandwidth.gpu_to_gpu == pytest.approx(300e9)
    assert cfg.bandwidth.gpu_to_cpu == pytest.approx(32e9)
    assert cfg.bandwidth.cpu_to_disk == pytest.approx(5e9)


def test_default_yaml_slo_values():
    cfg = load_config(str(DEFAULT_YAML))
    assert cfg.slo.ttft_target_ms == pytest.approx(2000.0)
    assert cfg.slo.tbt_target_ms == pytest.approx(100.0)


def test_default_yaml_workload_values():
    cfg = load_config(str(DEFAULT_YAML))
    assert cfg.workload.request_rate == pytest.approx(50.0)
    assert cfg.workload.duration_s == pytest.approx(10.0)
    assert cfg.workload.prefix_sharing_ratio == pytest.approx(0.6)
    assert cfg.workload.avg_prompt_len == 1024
    assert cfg.workload.avg_output_len == 64


def test_empty_yaml_uses_all_defaults(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("{}\n")
    cfg = load_config(str(f))
    assert cfg.cluster.prefill_nodes == 4
    assert cfg.workload.avg_output_len == 256


def test_partial_yaml_overrides_only_specified_fields(tmp_path):
    f = tmp_path / "partial.yaml"
    f.write_text("cluster:\n  prefill_nodes: 8\n")
    cfg = load_config(str(f))
    assert cfg.cluster.prefill_nodes == 8
    assert cfg.cluster.decode_nodes == 4   # default preserved


# ------------------------------------------------------------------
# 字段缺失 / 约束违反时报错
# ------------------------------------------------------------------

def test_missing_required_field_none_raises():
    # Passing None for a non-Optional int field triggers ValidationError
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"cluster": {"prefill_nodes": None}})


def test_constraint_violation_zero_nodes_raises():
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"cluster": {"prefill_nodes": 0}})


def test_constraint_violation_negative_blocks_raises():
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"node": {"gpu_blocks": -1}})


def test_constraint_violation_prefix_ratio_out_of_range_raises():
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"workload": {"prefix_sharing_ratio": 1.5}})


def test_extra_field_raises():
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"unknown_section": {}})


def test_extra_field_in_sub_model_raises():
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"cluster": {"nonexistent_key": 99}})


# ------------------------------------------------------------------
# 类型错误时报错
# ------------------------------------------------------------------

def test_type_error_string_for_int_raises():
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"cluster": {"prefill_nodes": "not_an_int"}})


def test_type_error_string_for_float_raises():
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"slo": {"ttft_target_ms": "fast"}})


def test_type_error_list_for_int_raises():
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"model": {"block_size": [16, 32]}})


def test_type_error_from_yaml_file(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("cluster:\n  prefill_nodes: not_a_number\n")
    with pytest.raises(ValidationError):
        load_config(str(f))


# ------------------------------------------------------------------
# 文件不存在时报错
# ------------------------------------------------------------------

def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


# ------------------------------------------------------------------
# SchedulerConfig 新字段
# ------------------------------------------------------------------

def test_default_scheduler_config():
    cfg = NanoKVConfig()
    assert cfg.scheduler.name == "round_robin"
    assert cfg.scheduler.params == {}


def test_scheduler_config_accepts_arbitrary_params():
    cfg = NanoKVConfig.model_validate({
        "scheduler": {"name": "conductor", "params": {"alpha": 2.0, "beta": 0.5}}
    })
    assert cfg.scheduler.name == "conductor"
    assert cfg.scheduler.params["alpha"] == 2.0


def test_scheduler_extra_field_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        NanoKVConfig.model_validate({"scheduler": {"name": "x", "unknown_key": 1}})


# ------------------------------------------------------------------
# GeneratorConfig 新字段
# ------------------------------------------------------------------

def test_default_generator_config():
    cfg = NanoKVConfig()
    assert cfg.generator.num_buckets == 10
    assert cfg.generator.vocab_size == 32_000
    assert cfg.generator.seed == 42


def test_generator_config_custom_values():
    cfg = NanoKVConfig.model_validate({
        "generator": {"num_buckets": 5, "vocab_size": 1000, "seed": 0}
    })
    assert cfg.generator.num_buckets == 5
    assert cfg.generator.vocab_size == 1000
    assert cfg.generator.seed == 0


def test_generator_num_buckets_zero_raises():
    with pytest.raises(Exception):
        NanoKVConfig.model_validate({"generator": {"num_buckets": 0}})


def test_load_config_with_scheduler_and_generator(tmp_path):
    f = tmp_path / "full.yaml"
    f.write_text(
        "scheduler:\n"
        "  name: e2_policy\n"
        "  params:\n"
        "    w_historical: 2.0\n"
        "generator:\n"
        "  num_buckets: 20\n"
        "  seed: 99\n"
    )
    cfg = load_config(str(f))
    assert cfg.scheduler.name == "e2_policy"
    assert cfg.scheduler.params["w_historical"] == 2.0
    assert cfg.generator.num_buckets == 20
    assert cfg.generator.seed == 99

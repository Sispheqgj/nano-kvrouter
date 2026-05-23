from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ClusterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefill_nodes: int = Field(default=4, ge=1)
    decode_nodes: int = Field(default=4, ge=1)


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_blocks: int = Field(default=2_000, ge=0)
    cpu_blocks: int = Field(default=10_000, ge=0)
    disk_blocks: int = Field(default=100_000, ge=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_size: int = Field(default=16, ge=1)
    kv_bytes_per_token: int = Field(default=512, ge=1)
    prefill_cost_per_token_ms: float = Field(default=0.033, gt=0)
    decode_base_ms: float = Field(default=5.0, gt=0)


class BandwidthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_to_gpu: float = Field(default=300e9, gt=0)
    gpu_to_cpu: float = Field(default=32e9, gt=0)
    cpu_to_disk: float = Field(default=5e9, gt=0)


class SLOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttft_target_ms: float = Field(default=2000.0, gt=0)
    tbt_target_ms: float = Field(default=100.0, gt=0)


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_rate: float = Field(default=50.0, gt=0)
    duration_s: float = Field(default=60.0, gt=0)
    prefix_sharing_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    avg_prompt_len: int = Field(default=1024, ge=1)
    avg_output_len: int = Field(default=256, ge=1)


class NanoKVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    node: NodeConfig = Field(default_factory=NodeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    bandwidth: BandwidthConfig = Field(default_factory=BandwidthConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)


def load_config(path: str) -> NanoKVConfig:
    """Load a NanoKVConfig from a YAML file."""
    raw = Path(path).read_text()
    data = yaml.safe_load(raw) or {}
    cfg = NanoKVConfig.model_validate(data)
    logger.debug("Loaded config from %s", path)
    return cfg

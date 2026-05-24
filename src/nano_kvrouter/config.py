from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ClusterConfig(BaseModel):
    """Cluster topology: how many prefill / decode mock nodes to spin up.

    Mirrors Mooncake's P/D separation — prefill and decode counts are
    independent so experiments can vary the ratio.
    """

    model_config = ConfigDict(extra="forbid")

    prefill_nodes: int = Field(default=4, ge=1)
    decode_nodes: int = Field(default=4, ge=1)


class NodeConfig(BaseModel):
    """Per-node resource limits.

    `gpu_blocks` / `cpu_blocks` / `disk_blocks` size the three-tier
    `BlockPool`. `capacity` caps the number of concurrent running requests
    on a node — exceeding it pushes new admits into the wait queue.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_blocks: int = Field(default=2_000, ge=0)
    cpu_blocks: int = Field(default=10_000, ge=0)
    disk_blocks: int = Field(default=100_000, ge=0)
    capacity: int = Field(default=32, ge=1)


class ModelConfig(BaseModel):
    """Model-level constants that drive the latency model.

    Prefill cost is linear in uncached tokens; decode cost is
    `decode_base_ms + batch_size * marginal_decode_ms` per step. All
    values are tunable per-experiment so the same code path can simulate
    different model sizes.
    """

    model_config = ConfigDict(extra="forbid")

    block_size: int = Field(default=16, ge=1)
    kv_bytes_per_token: int = Field(default=512, ge=1)
    prefill_cost_per_token_ms: float = Field(default=0.033, gt=0)
    decode_base_ms: float = Field(default=5.0, gt=0)
    marginal_decode_ms: float = Field(default=0.5, gt=0)


class BandwidthConfig(BaseModel):
    """Inter-tier transfer bandwidths in bytes/second.

    Used to estimate KV-block transfer latency for tier promotion /
    demotion and cross-node migration. Defaults approximate A100-class
    hardware: NVLink GPU↔GPU, PCIe Gen4 GPU↔CPU, NVMe CPU↔Disk.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_to_gpu: float = Field(default=300e9, gt=0)
    gpu_to_cpu: float = Field(default=32e9, gt=0)
    cpu_to_disk: float = Field(default=5e9, gt=0)


class SLOConfig(BaseModel):
    """Per-request SLO targets in milliseconds.

    Schedulers check predicted TTFT / TBT against these targets before
    admission (Mooncake-style early rejection). Targets are global; the
    request-generator stamps each request with these values.
    """

    model_config = ConfigDict(extra="forbid")

    ttft_target_ms: float = Field(default=2000.0, gt=0)
    tbt_target_ms: float = Field(default=100.0, gt=0)


class WorkloadConfig(BaseModel):
    """Synthetic workload knobs for the request generator.

    `prefix_sharing_ratio` controls the fraction of requests that share a
    prompt prefix with an earlier request — the main lever for stressing
    cache-aware scheduling.
    """

    model_config = ConfigDict(extra="forbid")

    request_rate: float = Field(default=50.0, gt=0)
    duration_s: float = Field(default=60.0, gt=0)
    prefix_sharing_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    avg_prompt_len: int = Field(default=1024, ge=1)
    avg_output_len: int = Field(default=256, ge=1)


class NanoKVConfig(BaseModel):
    """Root configuration aggregating every sub-config.

    Loaded once from YAML at CLI entry; every other module receives the
    relevant sub-config rather than reading globals.
    """

    model_config = ConfigDict(extra="forbid")

    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    node: NodeConfig = Field(default_factory=NodeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    bandwidth: BandwidthConfig = Field(default_factory=BandwidthConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)


def load_config(path: str) -> NanoKVConfig:
    """Load and validate a `NanoKVConfig` from a YAML file.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        A fully validated `NanoKVConfig`. Unknown keys raise
        `pydantic.ValidationError` because every sub-model sets
        `extra="forbid"` — typos fail loudly rather than being silently
        ignored.
    """
    raw = Path(path).read_text()
    data = yaml.safe_load(raw) or {}
    cfg = NanoKVConfig.model_validate(data)
    logger.debug("Loaded config from %s", path)
    return cfg

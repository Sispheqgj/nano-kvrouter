from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ClusterConfig(BaseModel):
    """Cluster topology: how many prefill / decode mock nodes to spin up.

    NOTE (v1): ``decode_nodes`` is currently DEAD. ``cli.py`` builds only
    ``prefill_nodes``-many ``MockEngineNode`` instances and every scheduler
    hard-codes ``decode_node == prefill_node``. Real P/D separation lands
    in P2-Infra M5.
    """

    model_config = ConfigDict(extra="forbid")

    prefill_nodes: int = Field(default=4, ge=1)
    decode_nodes: int = Field(
        default=4,
        ge=1,
        description="DEAD until P2-Infra M5. cli.py currently builds only "
                    "prefill_nodes-many MockEngineNodes and all schedulers "
                    "force decode_node == prefill_node.",
    )


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
    kv_bytes_per_token: int = Field(
        default=512,
        ge=1,
        description="DEAD until P2-Infra M5. Will drive KV transfer time "
                    "(prompt_len * kv_bytes_per_token / bandwidth.gpu_to_gpu) "
                    "once P/D separation lands.",
    )
    prefill_cost_per_token_ms: float = Field(default=0.033, gt=0)
    decode_base_ms: float = Field(default=5.0, gt=0)
    marginal_decode_ms: float = Field(default=0.5, gt=0)
    prefill_chunk_size: int = Field(
        default=512,
        ge=1,
        description="LIVE in P2-Infra M3+. Tokens processed per prefill "
                    "chunk per batch step. Smaller = more prefill steps "
                    "(longer TTFT) but less decode interference (lower ITL).",
    )


class BandwidthConfig(BaseModel):
    """Inter-tier transfer bandwidths in bytes/second.

    NOTE (v1): ALL THREE fields are currently DEAD. No scheduler or engine
    code reads them. They will be activated in:
      - P2-Infra M5: ``gpu_to_gpu`` for P/D KV transfer cost
      - P2-Infra M6: ``gpu_to_cpu`` / ``cpu_to_disk`` for HiCache tier
        promotion / demotion latency

    Defaults approximate A100-class hardware: NVLink GPU↔GPU, PCIe Gen4
    GPU↔CPU, NVMe CPU↔Disk.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_to_gpu: float = Field(
        default=300e9, gt=0,
        description="DEAD until P2-Infra M5. Will set KV transfer cost for P/D disaggregation.",
    )
    gpu_to_cpu: float = Field(
        default=32e9, gt=0,
        description="DEAD until P2-Infra M6. Will set GPU→CPU tier demotion bandwidth.",
    )
    cpu_to_disk: float = Field(
        default=5e9, gt=0,
        description="DEAD until P2-Infra M6. Will set CPU→Disk tier demotion bandwidth.",
    )


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


class SchedulerConfig(BaseModel):
    """Scheduler selection and its specific parameters.

    ``name`` selects the scheduler class to instantiate; ``params`` is
    forwarded to its ``__init__`` as keyword arguments. The schema is
    intentionally open (``dict[str, Any]``) — ``cli.py`` is responsible
    for mapping names to concrete classes and validating params.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="round_robin")
    params: dict[str, Any] = Field(default_factory=dict)


class GeneratorConfig(BaseModel):
    """Request generator parameters.

    Controls the K-bucket conversation model: ``num_buckets`` shared
    prefixes are pre-generated at startup; each request randomly draws
    one and appends a per-request suffix. ``seed`` pins the PRNG for
    reproducible experiments.
    """

    model_config = ConfigDict(extra="forbid")

    num_buckets: int = Field(default=10, ge=1)
    vocab_size: int = Field(default=32_000, ge=1)
    seed: int = Field(default=42)


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
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)


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

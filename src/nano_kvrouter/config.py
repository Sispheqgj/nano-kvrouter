from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from nano_kvrouter.simulator.prefix_synthesis import PrefixSynthesisConfig

logger = logging.getLogger(__name__)


class ClusterConfig(BaseModel):
    """Cluster topology: how many prefill / decode mock nodes to spin up.

    M5a (LIVE): ``cli.py`` builds two separate ``MockEngineNode`` pools
    (``p{i}`` prefill / ``d{i}`` decode) and the scheduler returns
    independent ``prefill_node`` / ``decode_node`` IDs. KV cache lives
    only on the decode pool.
    """

    model_config = ConfigDict(extra="forbid")

    prefill_nodes: int = Field(default=4, ge=1)
    decode_nodes: int = Field(
        default=4,
        ge=1,
        description="LIVE in P2-Infra M5+. Sizes the decode-node pool built by cli._run_one().",
    )


class NodeConfig(BaseModel):
    """Per-node resource limits.

    `gpu_blocks` / `cpu_blocks` / `disk_blocks` size the live three-tier
    `BlockPool`. `capacity` caps the number of concurrent running requests
    on a node and therefore affects queueing and decode-side rejection.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_blocks: int = Field(
        default=2_000,
        ge=0,
        description="LIVE in M4+. GPU-tier KV capacity in blocks.",
    )
    cpu_blocks: int = Field(
        default=10_000,
        ge=0,
        description="LIVE in M6+. CPU-tier HiCache capacity in blocks.",
    )
    disk_blocks: int = Field(
        default=100_000,
        ge=0,
        description="LIVE in M6+. Disk-tier HiCache capacity in blocks.",
    )
    capacity: int = Field(
        default=32,
        ge=1,
        description="LIVE in M1+. Per-node concurrent request capacity used by admission and queueing.",
    )


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
        description="LIVE in P2-Infra M5a. Drives KV transfer time "
                    "(prompt_len * kv_bytes_per_token / bandwidth.gpu_to_gpu) "
                    "for the post-prefill KV migration in split P/D.",
    )
    prefill_cost_per_token_ms: float = Field(
        default=0.033,
        gt=0,
        description="LIVE in M1+. Prefill latency slope per uncached token.",
    )
    decode_base_ms: float = Field(
        default=5.0,
        gt=0,
        description="LIVE in M1+. Base decode-step latency component.",
    )
    marginal_decode_ms: float = Field(
        default=0.5,
        gt=0,
        description="LIVE in M1+. Per-batch decode-step marginal latency component.",
    )
    prefill_chunk_size: int = Field(
        default=512,
        ge=1,
        description="LIVE in P2-Infra M3+. Tokens processed per prefill "
                    "chunk per batch step. Smaller = more prefill steps "
                    "(longer TTFT) but less decode interference (lower ITL).",
    )


class BandwidthConfig(BaseModel):
    """Inter-tier transfer bandwidths in bytes/second.

    All three fields are LIVE:
      - ``gpu_to_gpu`` drives prefill->decode KV transfer cost in split P/D.
      - ``gpu_to_cpu`` drives CPU-tier hit reload cost into GPU HBM.
      - ``cpu_to_disk`` drives the Disk->CPU leg of disk-tier hit reload cost.

    Defaults approximate A100-class hardware: NVLink GPU↔GPU, PCIe Gen4
    GPU↔CPU, NVMe CPU↔Disk.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_to_gpu: float = Field(
        default=300e9, gt=0,
        description="LIVE in M5a. KV transfer cost for split P/D = "
                    "(prompt_len * kv_bytes_per_token / gpu_to_gpu) * 1000 ms.",
    )
    gpu_to_cpu: float = Field(
        default=32e9, gt=0,
        description="LIVE in M6+. GPU-facing bandwidth used for CPU-tier reload cost.",
    )
    cpu_to_disk: float = Field(
        default=5e9, gt=0,
        description="LIVE in M6+. Disk-facing bandwidth used for disk-tier reload cost.",
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




class TraceConfig(BaseModel):
    """Trace replay configuration.

    Activate trace mode by setting NanoKVConfig.trace to a TraceConfig
    instance (via YAML). When None, the Poisson RequestGenerator is used
    (backwards compatible).
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        description=(
            "Path to internal JSONL trace file. Resolved by CLI "
            "relative to the config file directory, then handed "
            "to TraceGenerator as absolute path."
        ),
    )
    speedup: float = Field(
        default=1.0, gt=0,
        description="Time-axis speedup. arrival_ms_in_sim = trace.timestamp / speedup",
    )
    max_requests: int | None = Field(
        default=None,
        ge=1,
        description="Stop after replaying this many requests. None = full trace.",
    )
    prefix_mode: Literal["none", "hash_ids", "synthesis"] = Field(
        default="hash_ids",
        description=(
            "How to derive token_ids: none=random, hash_ids=expand "
            "from trace.hash_ids (Mooncake), synthesis=M2 only."
        ),
    )
    prefix_synthesis: PrefixSynthesisConfig | None = Field(
        default=None,
        description="Required when prefix_mode=synthesis. Ignored otherwise.",
    )

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
    trace: TraceConfig | None = None


class SensitivityExperiment(BaseModel):
    """One field-level sensitivity experiment for LIVE acceptance."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    base_config: str = Field(min_length=1)
    scheduler: str = Field(min_length=1)
    values: list[int | float] = Field(min_length=1)
    primary_metrics: list[str] = Field(min_length=1)
    note: str | None = None


class SensitivityConfig(BaseModel):
    """Config-driven collection of sensitivity experiments."""

    model_config = ConfigDict(extra="forbid")

    experiments: list[SensitivityExperiment] = Field(min_length=1)


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


def load_sensitivity_config(path: str) -> SensitivityConfig:
    """Load and validate a ``SensitivityConfig`` from a YAML file."""
    raw = Path(path).read_text()
    data = yaml.safe_load(raw) or {}
    cfg = SensitivityConfig.model_validate(data)
    logger.debug("Loaded sensitivity config from %s", path)
    return cfg

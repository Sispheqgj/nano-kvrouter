# nano-kvrouter

> A KV-cache-centric LLM serving control-plane simulator. Pure Python,
> single-threaded, event-driven, and fully reproducible on a laptop.

## What this repo is for

`nano-kvrouter` models the control plane of LLM serving systems without
running real inference kernels. It keeps the routing and cache-management
decisions, but replaces tensor execution with latency formulas and a
discrete-event engine.

The simulator is useful for comparing scheduler behavior side-by-side on
the same workload:

- cache-blind baselines: `round_robin`, `least_loaded`
- cache-aware routing: `prefix_greedy`, `e2_policy`
- SLO-aware routing: `conductor`

It also models the KV cache as a first-class cluster resource:

- paged GPU block accounting
- split prefill/decode routing
- post-prefill KV transfer
- multi-tier HiCache demotion and tier-aware lookup
- config-driven sensitivity acceptance for LIVE field verification

## Current status

P2-Infra M1-M6 is implemented and verified.

- `uv run pytest -q` -> `377 passed`
- `uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml` -> `13/13 fields PASS`
- repo-outside absolute `--config` sensitivity execution is supported

The repository currently exposes three public CLI workflows:

- `run`: run one scheduler on one config
- `sweep`: run all five schedulers and print a comparison table
- `sensitivity`: run config-driven LIVE field experiments and emit a field-level PASS/FAIL punch list

## Quick start

The project uses [`uv`](https://docs.astral.sh/uv/).

```bash
# Install dependencies, including dev tools
uv sync --extra dev

# Run one scheduler on the default scenario
uv run python -m nano_kvrouter.cli run --config configs/default.yaml

# Compare all five schedulers on the default scenario
uv run python -m nano_kvrouter.cli sweep --config configs/default.yaml

# Stress decode capacity and rejection/throughput tradeoffs
uv run python -m nano_kvrouter.cli sweep --config configs/heavy.yaml

# Exercise multi-tier HiCache behavior
uv run python -m nano_kvrouter.cli run --config configs/hicache.yaml

# Run field-level sensitivity acceptance
uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml

# Optional structured output
uv run python -m nano_kvrouter.cli sensitivity \
  --config configs/sensitivity.yaml \
  --output /tmp/nano-kvrouter-sensitivity.json

uv run python -m nano_kvrouter.cli sensitivity \
  --config configs/sensitivity.yaml \
  --output /tmp/nano-kvrouter-sensitivity.csv

# Full test suite
uv run pytest -q
```

## Implemented P2-Infra capabilities

| Milestone | What is live in code |
| --------- | -------------------- |
| M2 | Continuous batch decode on `MockEngineNode`, driven by `DECODE_BATCH_STEP` |
| M3 | Chunked prefill with `model.prefill_chunk_size` and piggybacked prefill/decode ticks |
| M4 | Paged GPU KV metadata with `RadixTree + BlockPool`, split-aware admit, LRU pressure |
| M5 | Split prefill/decode pools, Mooncake-style post-prefill KV transfer, decode-side back-pressure |
| M6 | Multi-tier HiCache: GPU -> CPU -> Disk demotion chain, tier-aware lookup, CPU/Disk transfer-cost accounting |
| Acceptance | Config-driven `sensitivity` CLI with terminal table + JSON/CSV export |

## Scenario configs

| File | Purpose |
| ---- | ------- |
| [`configs/default.yaml`](configs/default.yaml) | Unsaturated baseline for general scheduler and latency comparisons |
| [`configs/heavy.yaml`](configs/heavy.yaml) | Decode-capacity pressure scenario for rejection and throughput effects |
| [`configs/hicache.yaml`](configs/hicache.yaml) | Multi-tier HiCache scenario for GPU/CPU/Disk tier reuse behavior |
| [`configs/sensitivity.yaml`](configs/sensitivity.yaml) | Acceptance matrix describing field experiments, not hard-coded CLI logic |

## LIVE config matrix

All 13 fields below are LIVE in the current implementation.

| Field | Where it matters |
| ----- | ---------------- |
| `cluster.decode_nodes` | sizes the decode pool built in `cli._run_one()` |
| `node.capacity` | governs `MockEngineNode` admission, queueing, and decode-side rejection pressure |
| `node.gpu_blocks` | sizes tier-1 capacity in `BlockPool` / `CacheManager` |
| `node.cpu_blocks` | sizes tier-2 HiCache capacity in `BlockPool` / `CacheManager` |
| `node.disk_blocks` | sizes tier-3 HiCache capacity in `BlockPool` / `CacheManager` |
| `model.kv_bytes_per_token` | drives post-prefill KV transfer cost and per-block tier load cost |
| `model.prefill_cost_per_token_ms` | drives prefill latency and queue-wait estimation |
| `model.decode_base_ms` | drives decode-step latency, TTFT, TBT, and queue-wait estimation |
| `model.marginal_decode_ms` | drives batch-size-sensitive decode latency and queue-wait estimation |
| `model.prefill_chunk_size` | controls chunk count and chunked-prefill scheduling behavior |
| `bandwidth.gpu_to_gpu` | drives prefill-node -> decode-node KV transfer cost |
| `bandwidth.gpu_to_cpu` | drives CPU-tier hit reload cost into GPU HBM |
| `bandwidth.cpu_to_disk` | drives Disk -> CPU leg of disk-tier hit reload cost |

The public config models live in [src/nano_kvrouter/config.py](src/nano_kvrouter/config.py).

## Sensitivity workflow

Sensitivity is a formal acceptance workflow for LIVE config fields.

Command:

```bash
uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml
```

Behavior:

- `configs/sensitivity.yaml` declares experiments with `field`, `base_config`,
  `scheduler`, `values`, `primary_metrics`, and optional `note`
- each experiment runs one baseline first, then one run per candidate value
- the CLI prints a punch-list table with:
  - field
  - scenario
  - baseline -> candidate value
  - changed metrics
  - candidate PASS/FAIL
  - field-level LIVE PASS/FAIL
- `--output ...json` writes the full structured report
- `--output ...csv` writes long-format rows per `(experiment, candidate, metric)`

Field-level PASS semantics:

- ratio / rate metrics: `abs(delta) >= 0.005`
- latency metrics (`*_ms`): `abs(delta) >= 0.5 ms` or `abs(pct_delta) >= 1%`
- throughput / rejection metrics: absolute or percent delta over threshold
- nested metrics such as `cache_hit_by_tier_ratio.cpu` are checked leaf by leaf
- **Field LIVE = any candidate value changes any primary metric leaf above threshold**

This matters for `cpu_blocks` and `disk_blocks`: expanding an already-large
CPU or Disk tier can land in a platform region where a particular candidate
shows `Candidate FAIL`, while a collapsing candidate such as `400 -> 0` or
`2000 -> 0` still proves that the field is LIVE.

## Paper fidelity matrix

| System | What nano-kvrouter keeps | Current fidelity note |
| ------ | ------------------------ | --------------------- |
| Mooncake FAST'25 | `MooncakeConductor`, split P/D pools, post-prefill KV transfer, TTFT/TBT SLO gate | Disk-tier hit cost uses a Disk -> CPU -> GPU two-hop simulator extrapolation, not a verbatim paper formula |
| SGLang NeurIPS'24 | RadixAttention-style prefix tree and cache-aware routing | Tier-aware lookup is simulator-specific extension on top of prefix matching |
| vLLM v1 / PagedAttention | block metadata abstraction, paged KV accounting, block-pool pressure | No real tensor residency or kernel execution |
| Preble ICLR'25 | E2 exploit-explore prompt-aware scoring | Historical-load + eviction/run-cost logic is intentionally compact |
| Llumnix OSDI'24 | migration / rebalance as a control-plane concept | migration planner remains roadmap work; no full live-migration execution path yet |

## Architecture summary

```text
RequestGenerator
  -> SchedulingPolicy
  -> prefill MockEngineNode pool
  -> KV_TRANSFER_START / KV_TRANSFER_COMPLETE
  -> decode MockEngineNode pool
  -> MetricsCollector

Cache state is owned by CacheManager:
  one RadixTree + one BlockPool per decode node
```

Design invariants:

1. Event-driven, not threaded.
2. No real GPU execution; all time is simulated.
3. `CacheManager` is the cache source of truth.
4. Schedulers are pluggable modules behind `SchedulingPolicy`.
5. Metrics are passive observers and do not mutate state.

See [DESIGN.md](DESIGN.md) for the full design write-up.

## Repository layout

```text
src/nano_kvrouter/
├── cli.py                 # run / sweep / sensitivity entrypoints
├── config.py              # Pydantic config models + sensitivity schema
├── request.py             # Request dataclass
├── kv_cache/
│   ├── radix_tree.py      # Prefix tree
│   ├── block_pool.py      # GPU/CPU/Disk block metadata store
│   └── cache_manager.py   # Unified cache interface
├── engine/
│   └── mock_node.py       # Continuous batching + chunked prefill latency model
├── scheduler/
│   ├── base.py
│   ├── round_robin.py
│   ├── least_loaded.py
│   ├── prefix_greedy.py
│   ├── e2_policy.py
│   └── conductor.py
├── simulator/
│   ├── event.py
│   ├── engine.py
│   └── generator.py
└── metrics/
    └── collector.py
```

## Known simplifications

- No real tensors, tokenizer, or GPU kernels.
- Decode throughput numbers are useful for comparison, not for hardware sizing.
- KV transfer is modeled as bandwidth-bound latency, not a full transport stack.
- Multi-tier tier-hit cost is additive and deterministic.
- Llumnix-style migration remains a control-plane roadmap item, not a shipped execution path.

## Additional docs

- [DESIGN.md](DESIGN.md): current architecture and milestone fidelity
- [doc/code-review/README.md](doc/code-review/README.md): per-file review index
- [doc/review-notes.md](doc/review-notes.md): backlog and cross-file notes

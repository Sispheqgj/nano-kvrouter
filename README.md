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

P2-Infra M1-M6 is implemented and verified. P3-C M1 (real-world trace
replay) is also live.

- `uv run pytest -q` -> `393 passed`
- `uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml` -> `13/13 fields PASS`
- `uv run python -m nano_kvrouter.cli sweep --config configs/trace_mooncake.yaml` -> 5 schedulers on Mooncake FAST'25 real trace, cache-aware ~2× cache_hit vs cache-blind
- repo-outside absolute `--config` sensitivity execution is supported

The repository currently exposes three public CLI workflows:

- `run`: run one scheduler on one config
- `sweep`: run all five schedulers and print a comparison table
- `sensitivity`: run config-driven LIVE field experiments and emit a field-level PASS/FAIL punch list

## Why a simulator

Real LLM serving systems answer four control-plane questions per request:

1. **Which prefill node?** — cache affinity vs. load balance.
2. **Which decode node?** — capacity headroom vs. tier-affinity for the KV transfer destination.
3. **Which storage tier loads the KV cache?** — GPU HBM, CPU DRAM, or disk; the reload cost feeds back into routing.
4. **Should this request be rejected?** — prediction-based early rejection when no node can hit the SLO.

Studying these decisions on a real cluster is slow, expensive, and
non-reproducible. Studying them on synthetic latency models on a single
machine is fast, free, and deterministic — at the cost of not measuring
absolute throughput. `nano-kvrouter` deliberately makes that trade.

## Demo at a glance

A 15-second saturation benchmark on the split P/D cluster (`configs/heavy.yaml`,
`uv run python -m nano_kvrouter.cli sweep --config configs/heavy.yaml`,
numbers as of M6, 2026-06-09):

| scheduler       | TTFT p50 | TTFT p99 | cache_hit | rejection | throughput  |
| --------------- | -------- | -------- | --------- | --------- | ----------- |
| `round_robin`   | 24 ms    | 46 ms    | 0.540     | 19.2%     | 61.0 req/s  |
| `least_loaded`  | 24 ms    | 46 ms    | 0.525     | 21.4%     | 59.3 req/s  |
| `prefix_greedy` | 24 ms    | 44 ms    | **0.582** | **51.8%** | 36.4 req/s  |
| `e2_policy`     | 24 ms    | 45 ms    | 0.564     | 21.6%     | 59.2 req/s  |
| `conductor`     | 24 ms    | 46 ms    | 0.534     | **20.1%** | 60.3 req/s  |

Notable from the same run:

- `prefix_greedy` achieves the highest cache_hit_ratio (0.582) but its high
  rejection (51.8%) reveals an implicit admission-control gap: cache-greedy
  placement concentrates load on hot decode nodes.
- `conductor` balances load best (20.1% rejection) close to the theoretical
  minimum given the request rate vs. decode capacity ratio.
- With P/D split (M5), TTFT is uniform across schedulers — back-pressure is
  absorbed by the separate prefill pool, and rejection is now capacity-driven
  rather than SLO-driven.

The unsaturated demo (`configs/default.yaml`) shows all schedulers completing
100% of requests at ~28 ms TTFT, with cache-aware policies at cache_hit ≈ 0.56
vs. load-balanced at ≈ 0.50.

For a multi-tier HiCache demo (CPU + Disk reuse), see `configs/hicache.yaml`.

### Real-world trace replay (P3-C M1, 2026-06-10)

Replaying the first 2000 requests of Mooncake's FAST'25 `conversation_trace.jsonl`
(real `hash_ids` prefix structure, block_size=512,
`configs/trace_mooncake.yaml`):

| scheduler       | TTFT p50 | TTFT p99 | cache_hit | rejection | throughput |
| --------------- | -------- | -------- | --------- | --------- | ---------- |
| `round_robin`   | 522 ms   | 4100 ms  | 0.075     | 0%        | 3.0 req/s  |
| `least_loaded`  | 521 ms   | 4312 ms  | 0.069     | 0%        | 3.0 req/s  |
| `prefix_greedy` | 457 ms   | 4147 ms  | **0.146** | 0%        | 3.0 req/s  |
| `e2_policy`     | 452 ms   | 4149 ms  | **0.153** | 0%        | 3.0 req/s  |
| `conductor`     | 456 ms   | 4147 ms  | **0.146** | 0%        | 3.0 req/s  |

Cache-aware schedulers (`prefix_greedy` / `e2_policy` / `conductor`) achieve
~2× the cache hit ratio of cache-blind baselines on real Mooncake workload.
TTFT is dominated by the trace's large prompts (~7000 tokens median) rather
than queueing — SLO is loose, no rejections. This is the first scenario
where reported numbers come from a *real* trace rather than a synthetic
Poisson generator.

The `hash_ids` field in the trace gives real block-level prefix identifiers
(block_size=512, verified). Mooncake traces are bundled in-repo
(`traces/mooncake/`, Apache-2.0 license, total ~10 MB).

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

## Schedulers

Each scheduler is a small (~80–150 line) module implementing the
`SchedulingPolicy` protocol. They are intentionally minimal — the
algorithm should be readable in one sitting alongside the matching paper.

| Scheduler           | Paper                          | Key idea                                                                              |
| ------------------- | ------------------------------ | ------------------------------------------------------------------------------------- |
| `RoundRobinPolicy`  | —                              | Rotate across nodes. Cache-blind baseline.                                            |
| `LeastLoadedPolicy` | —                              | Pick the node with lowest `running / capacity`. Cache-blind baseline.                 |
| `PrefixGreedyPolicy`| SGLang (NeurIPS'24)            | Pick the node with the longest cached prefix. Maximises cache reuse, ignores load.    |
| `E2Policy`          | Preble (ICLR'25)               | Score = historical_load + eviction_cost + run_cost. Trades cache hit against pressure.|
| `MooncakeConductor` | Mooncake (FAST'25 Best Paper)  | Three-objective scoring + SLO early rejection. Rejects when no node can hit the SLO.  |

Selecting a scheduler is a one-line YAML change:

```yaml
scheduler:
  name: conductor       # or round_robin / least_loaded / prefix_greedy / e2_policy
  params:
    alpha: 1.0          # cache_benefit weight
    beta: 1.0           # load_penalty weight
    gamma: 1.0          # transfer_penalty weight (GPU-only: 0; M6 multi-tier: CPU/Disk hit cost)
```

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

## Reference papers

The five schedulers in this repository simulate (a simplified version of)
the following systems:

- **Mooncake** — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, FAST'25 Best Paper.
- **Preble** — *Preble: Efficient Distributed Prompt Scheduling for LLM Serving*, ICLR'25.
- **SGLang** — *Efficiently Programming Large Language Models using SGLang* (RadixAttention), NeurIPS'24.
- **Llumnix** — *Llumnix: Dynamic Scheduling for Large Language Model Serving*, OSDI'24 (migration logic — roadmap).
- **vLLM** — *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP'23.

A more detailed paper-to-module mapping lives in
[`doc/code-review/README.md`](doc/code-review/README.md).

## Additional docs

- [DESIGN.md](DESIGN.md): current architecture and milestone fidelity
- [doc/code-review/README.md](doc/code-review/README.md): per-file review index
- [doc/review-notes.md](doc/review-notes.md): backlog and cross-file notes

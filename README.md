# nano-kvrouter

> A KV-cache-centric LLM serving control-plane simulator. Pure Python,
> event-driven, no real GPU — runs on a laptop and reproduces the routing
> behaviour of five published scheduling systems side-by-side.

---

## What this is

`nano-kvrouter` is a research-grade simulator for comparing LLM serving
schedulers under a common, reproducible workload. It replaces the inference
engine with a parameterised latency model so the entire experiment fits on
a Mac in well under a second, while still exercising the real control-plane
decisions: which node to send a request to, what to do with its KV cache,
and when to reject it before it hurts the SLO.

The goal is to make the **routing trade-offs** in recent papers
(Mooncake / Preble / SGLang / Llumnix / vLLM) visible, side-by-side, on
the same workload — so you can read a paper, pull the matching scheduler
out of `src/nano_kvrouter/scheduler/`, and see how its decisions differ
from the others on the same traces.

---

## Why a simulator

Real LLM serving systems answer four control-plane questions per request:

1. **Which prefill node?** — cache affinity vs. load balance.
2. **Which storage tier loads the KV cache?** — GPU HBM, CPU DRAM, or disk.
3. **Should this request be rejected?** — prediction-based early rejection
   when no node can meet the SLO.
4. **Should a KV block be migrated?** — rebalance, defragment, or evict.

Studying these decisions on a real cluster is slow, expensive, and
non-reproducible. Studying them on synthetic latency models on a single
machine is fast, free, and deterministic — at the cost of not measuring
absolute throughput. `nano-kvrouter` deliberately makes that trade.

---

## Demo at a glance

A 15-second saturation benchmark on a 4-node cluster (`configs/heavy.yaml`,
`uv run python -m nano_kvrouter.cli sweep --config configs/heavy.yaml`):

M5a (P/D split, 2026-06-08):

| scheduler       | TTFT p50 | TTFT p99 | cache_hit | rejection | throughput  |
| --------------- | -------- | -------- | --------- | --------- | ----------- |
| `round_robin`   | 24 ms    | 46 ms    | 0.540     | 19.2%     | 61.0 req/s  |
| `least_loaded`  | 24 ms    | 46 ms    | 0.523     | 21.4%     | 59.3 req/s  |
| `prefix_greedy` | 24 ms    | 44 ms    | **0.582** | **51.8%** | 36.4 req/s  |
| `e2_policy`     | 24 ms    | 45 ms    | 0.559     | 21.9%     | 59.0 req/s  |
| `conductor`     | 24 ms    | 46 ms    | 0.522     | **20.4%** | 60.1 req/s  |

SLO target: `ttft ≤ 400 ms`. P/D split (M5a) means all schedulers now
serve within SLO at similar TTFT — the back-pressure is absorbed by the
separate prefill pool. Rejection is now capacity-driven (decode pool
exhausted) rather than SLO-driven, so `conductor` no longer stands alone
in rejection rate.

Compared to the M4 combined-node baseline (`conductor` rejection=0.61, 454 completed),
M5a roughly doubles completed requests to ~926 — the key win from P/D separation.

Notable from the same run:

- `prefix_greedy` still achieves the highest cache_hit_ratio (0.582) but
  its high rejection (51.8%) reveals an implicit admission-control gap:
  cache-greedy placement concentrates load on hot decode nodes.
- `conductor` balances load best (20.4% rejection), close to the
  theoretical minimum given the request rate vs. decode capacity ratio.

The unsaturated demo (`configs/default.yaml`) shows all schedulers completing
100% of requests at ~28ms TTFT, with cache-aware policies at cache_hit≈0.56 vs.
load-balanced at ≈0.50.

---

## Quick start

The project uses [`uv`](https://docs.astral.sh/uv/) exclusively — no
`pip`, no manual venv activation.

```bash
# Install dependencies (incl. dev extras)
uv sync --extra dev

# Run the default demo (single scheduler)
uv run python -m nano_kvrouter.cli run --config configs/default.yaml

# Compare all five schedulers side-by-side
uv run python -m nano_kvrouter.cli sweep --config configs/default.yaml

# Saturation scenario — where conductor's SLO rejection earns its keep
uv run python -m nano_kvrouter.cli sweep --config configs/heavy.yaml

# Full test suite (342 tests, < 1 s real time)
uv run pytest -q
```

---

## What is implemented

As of 2026-06-08 the simulator has shipped P1 + P2-Infra M1-M5a.
The decode engine is now continuous-batched (M2), prefill is
piggyback-chunked (M3), the KV cache uses paged-attention block
ownership (M4), and the cluster is P/D split with explicit KV transfer
events (M5a). **342 tests pass.**

| Module                              | Status | Notes                                                      |
| ----------------------------------- | ------ | ---------------------------------------------------------- |
| `config.py`                         | ✅ M1  | 9 LIVE fields + 4 deferred to M6                            |
| `request.py`                        | ✅     | Request dataclass + factory                                 |
| `kv_cache/radix_tree.py`            | ✅ M4  | block_ids: list[str] per node; mint/free via BlockPool      |
| `kv_cache/block_pool.py`            | ✅ M4  | Active in M4; pool.used('gpu') is the capacity ground truth |
| `kv_cache/cache_manager.py`         | ✅ M4 / M5a | One tree per decode_node only (KV admit only on decode side) |
| `engine/mock_node.py`               | ✅ M2 / M3 | Continuous batching + chunked prefill, single in-flight    |
| `scheduler/{round_robin,least_loaded,prefix_greedy,e2_policy,conductor}.py` | ✅ M5a | schedule(req, prefill_nodes, decode_nodes, cache) |
| `simulator/event.py`                | ✅ M5a | 8 event types including KV_TRANSFER_START/COMPLETE         |
| `simulator/engine.py`               | ✅     | Single-threaded heapq                                       |
| `simulator/generator.py`            | ✅     | Streaming Poisson + K-bucket prefix sharing                |
| `metrics/collector.py`              | ✅ M3 / M5a | + avg_chunked_prefill_steps, kv_transfer_time_avg, dual_phase |
| `cli.py`                            | ✅ M5a | 2 node pools (prefill / decode), transfer_id epoch         |
| `configs/{default,heavy}.yaml`      | ✅     | Two reproducible demo scenarios                             |
| `scheduler/migration.py`            | ⏳     | Llumnix live migration — P3                                 |
| trace replay generator              | ⏳     | ShareGPT / Mooncake replay — P3                             |

### Known dead config (M6 only)

As of M5a (2026-06-08) only 4 config fields are not yet read by any
business logic. They will be activated in P2-Infra M6 (multi-tier
HiCache transfer):

| Field                           | Activated in | Reason |
| ------------------------------- | ------------ | ------ |
| `node.cpu_blocks`               | M6           | No CPU-DRAM tier promotion/demotion yet |
| `node.disk_blocks`              | M6           | Same as above |
| `bandwidth.gpu_to_cpu`          | M6           | No multi-tier transfer cost calculation yet |
| `bandwidth.cpu_to_disk`         | M6           | Same as above |

The previously dead `cluster.decode_nodes`, `bandwidth.gpu_to_gpu`, and
`model.kv_bytes_per_token` are all LIVE as of M5a (see "Sensitivity"
section below).

### gpu_blocks semantics (M4)

`node.gpu_blocks` controls **cache reuse capacity**, not request
execution KV materialisation. When the pool fills, `cm.admit`
silently drops the new KV write but the request still completes
(no KV cached, future prefix lookups miss).

This is a deliberate simplification: the simulator focuses on
scheduling decisions, not paged-attention execution semantics.
`gpu_blocks=50` produces `cache_hit_ratio=0` with all requests
still completing. M6 multi-tier HiCache will not change this —
CPU/Disk tiers will reduce the prefill recompute cost but the
execution path remains the same.

### Sensitivity (P2-Infra M2-M5a)

Every LIVE config field has demonstrable impact on sweep numbers:

| Config | Variation | Effect |
| ------ | --------- | ------ |
| `model.decode_base_ms` | 5 → 10 | ttft_p50 +51%, tbt_avg +95% |
| `model.prefill_chunk_size` | 128/512/2048 | ttft_p50 178/53/53 ms |
| `node.gpu_blocks` | 2000/200/50 | cache_hit 0.55/0.25/0.00 |
| `cluster.decode_nodes` | 4 → 2 | heavy rejection 0.20 → 0.78 |
| `bandwidth.gpu_to_gpu` | 3e11 → 1e7 | kv_transfer 0.0017 → 52.4 ms |
| `model.kv_bytes_per_token` | 512 → 8192 | kv_transfer 0.0017 → 0.028 ms (16x linear) |

---

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
    gamma: 1.0          # transfer_penalty weight (v1 GPU-only: always 0)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  nano-kvrouter control plane                │
│                                                             │
│  RequestGenerator  ──▶  SchedulingPolicy  ──▶  MockNode(s)  │
│   (Poisson,            (round_robin /        (latency model,│
│    bucket prefix        least_loaded /        capacity gate,│
│    sharing)             prefix_greedy /       no tensors)   │
│                         e2_policy /                         │
│                         conductor)            ▲             │
│                              │                │             │
│                              ▼                │             │
│                         CacheManager  ────────┘             │
│                         (RadixTree + BlockPool per node)    │
│                                                             │
│  All glued together by:                                     │
│    SimulationEngine — single-threaded heapq event loop      │
│    MetricsCollector — passive observer, never mutates state │
└─────────────────────────────────────────────────────────────┘
```

The simulator is **event-driven**, not threaded. Every action — a
request arriving, a prefill finishing, a decode step completing — is a
typed `Event` on a priority queue, processed in simulated-time order.
There is no `asyncio`, no threading, no wall-clock waiting. A 15-second
simulated benchmark runs in under 0.5 seconds of real time.

See [`DESIGN.md`](DESIGN.md) for the full design discussion (Chinese).

---

## Project structure

```
src/nano_kvrouter/
├── config.py              # Pydantic models, YAML loader
├── request.py             # Request dataclass + factory
├── cli.py                 # `run` and `sweep` subcommands
├── kv_cache/
│   ├── radix_tree.py      # Prefix tree with LRU eviction
│   ├── block_pool.py      # 3-tier block storage
│   └── cache_manager.py   # Unified per-node interface
├── engine/
│   └── mock_node.py       # Latency model + admission
├── scheduler/
│   ├── base.py            # SchedulingPolicy protocol
│   ├── round_robin.py     # baseline
│   ├── least_loaded.py    # baseline
│   ├── prefix_greedy.py   # SGLang
│   ├── e2_policy.py       # Preble
│   └── conductor.py       # Mooncake
├── simulator/
│   ├── event.py           # Event types
│   ├── engine.py          # heapq event loop
│   └── generator.py       # Streaming Poisson + buckets
└── metrics/
    └── collector.py       # TTFT / TBT / cache-hit / rejection

configs/
├── default.yaml           # Unsaturated demo
└── heavy.yaml             # Saturation scenario (overload + tight SLO)

tests/                     # 342 tests — one file per source module
```

---

## Design principles

These are non-negotiable in this codebase. They are what makes the
simulator small, deterministic, and easy to extend.

1. **Event-driven, not threaded.** Single `heapq` priority queue.
   No `asyncio`, no threads. Reproducibility before performance.
2. **Mock, not real.** Nodes hold no tensors. They compute
   `estimated_time = f(tokens, batch_size, tier)` and emit future events.
3. **Pluggable schedulers.** Every policy is a separate module
   implementing `SchedulingPolicy`. Select via YAML, resolved in `cli.py`.
4. **KV cache is the central abstraction.** `CacheManager` owns the
   `RadixTree + BlockPool`. Schedulers query it; nodes allocate through it.
5. **Metrics are passive observers.** `MetricsCollector` listens to events
   and never mutates simulation state.
6. **All time is simulated.** Use `SimulationEngine.now()`, never
   `time.time()`. No wall-clock dependencies anywhere.

---

## Reference papers

The five schedulers in this repository simulate (a simplified version of)
the following systems:

- **Mooncake** — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, FAST'25 Best Paper.
- **Preble** — *Preble: Efficient Distributed Prompt Scheduling for LLM Serving*, ICLR'25.
- **SGLang** — *Efficiently Programming Large Language Models using SGLang* (RadixAttention), NeurIPS'24.
- **Llumnix** — *Llumnix: Dynamic Scheduling for Large Language Model Serving*, OSDI'24 (migration logic — P2).
- **vLLM** — *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP'23.

A more detailed paper-to-module mapping lives in
[`doc/code-review/README.md`](doc/code-review/README.md).

---

## Roadmap

**P2 — heterogeneity and realistic workloads**

- `simulator/generator.py`: ShareGPT / Mooncake trace replay.
- `engine/mock_node.py`: track per-request `(prompt_len, output_len)`
  so `queue_wait_time` is accurate under mixed-length workloads.
- `metrics/collector.py`: per-request equal-weight TBT (in addition to
  the current step-weighted definition).

**P3 — live migration**

- `scheduler/migration.py`: Llumnix-style KV block migration when load
  becomes unbalanced.
- Tier-aware routing: a GPU-HBM cache miss may load from CPU-DRAM
  instead of recompute, when transfer cost beats prefill cost.

**P4 — multi-region / disaggregated P/D**

- Separate `prefill_node` and `decode_node` decisions in
  `SchedulingDecision`. Currently they are always equal.

---

## Status

P1 + P2-Infra M1-M5a complete. 342 tests pass. Two reproducible demos. Not yet published
as a package; build & install locally with `uv sync --extra dev`.

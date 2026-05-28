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

| scheduler       | TTFT p50  | TTFT p99  | rejection | throughput  |
| --------------- | --------- | --------- | --------- | ----------- |
| `round_robin`   | 8997 ms   | 17787 ms  | 0.0%      | 34.9 req/s  |
| `least_loaded`  | 18730 ms  | 72270 ms  | 0.0%      | 13.1 req/s  |
| `prefix_greedy` | 7046 ms   | 57565 ms  | 0.0%      | 15.7 req/s  |
| `e2_policy`     | 9004 ms   | 17809 ms  | 0.0%      | 34.8 req/s  |
| `conductor`     | **85 ms** | **262 ms**| **54.3%** | 34.4 req/s  |

SLO target: `ttft ≤ 400 ms`. Only `conductor` keeps the served requests
inside the SLO — by deliberately rejecting overload. This is the
Mooncake FAST'25 §4 trade made visible: *early rejection is a contract,
not a failure*.

Two other findings worth noting from the same run:

- `least_loaded` throughput **collapses below `round_robin`** under
  overload — the "least loaded" target oscillates and destroys cache
  locality. Naive load balancing is *worse* than no information here.
- `prefix_greedy` achieves the highest cache hit ratio (0.572) but
  same-bucket pile-up creates hot-spot queues, halving throughput.

The unsaturated demo (`configs/default.yaml`) shows the same five
schedulers behaving similarly on TTFT (no scheduler is stressed), but
cache-aware policies pull ahead on cache hit ratio (0.555 vs. 0.502).

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

# Full test suite (264 tests, < 1 s real time)
uv run pytest -q
```

---

## What is implemented

This is **P1** — a working end-to-end simulator with five schedulers,
streaming Poisson workload generation, and rich-formatted sweep output.

| Module                              | Status | Notes                                           |
| ----------------------------------- | ------ | ----------------------------------------------- |
| `config.py`                         | ✅     | Pydantic v2 models, YAML loader                 |
| `request.py`                        | ✅     | Request dataclass + factory                     |
| `kv_cache/radix_tree.py`            | ✅     | SGLang-style prefix tree, LRU eviction          |
| `kv_cache/block_pool.py`            | ✅     | 3-tier block storage (GPU / CPU / Disk)         |
| `kv_cache/cache_manager.py`         | ✅     | Per-node tree + capacity counter (v1.1)         |
| `engine/mock_node.py`               | ✅     | Latency model + capacity-aware admission        |
| `scheduler/round_robin.py`          | ✅     | Baseline rotation                               |
| `scheduler/least_loaded.py`         | ✅     | Baseline load-aware                             |
| `scheduler/prefix_greedy.py`        | ✅     | SGLang RadixAttention-style                     |
| `scheduler/e2_policy.py`            | ✅     | Preble ICLR'25 exploit-explore                  |
| `scheduler/conductor.py`            | ✅     | Mooncake FAST'25 + SLO early rejection          |
| `simulator/{event,engine}.py`       | ✅     | Single-threaded heapq event loop                |
| `simulator/generator.py`            | ✅     | Streaming Poisson + K-bucket prefix sharing     |
| `metrics/collector.py`              | ✅     | Paper-aligned TTFT / TBT, step-weighted         |
| `cli.py`                            | ✅     | `run` + `sweep` subcommands, rich tables        |
| `configs/{default,heavy}.yaml`      | ✅     | Two reproducible demo scenarios                 |
| `scheduler/migration.py`            | ⏳     | Llumnix live migration — P2                     |
| trace replay generator              | ⏳     | ShareGPT / Mooncake replay — P2                 |
| `engine/latency_model.py`           | ⏳     | Currently inlined in `mock_node`                |

**264 tests pass.** Every public class has a docstring; every scheduler
module header cites its paper.

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

tests/                     # 264 tests — one file per source module
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

P1 complete. 264 tests pass. Two reproducible demos. Not yet published
as a package; build & install locally with `uv sync --extra dev`.

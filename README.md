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

P2-Infra M1-M6 is implemented and verified. P3-C M1+M2 (real-world trace
replay + synthetic prefix sharing on length-only traces), P4-A (per-node
KV transfer lane contention, opt-in), P4-B (schedulers see transfer-lane
backlog + `kv_transfer_queued_avg_ms` metric), and P5-Bidaw M1/M2 (FAST'26
I/O-aware dual-queue scheduling plus metadata-only previous-answer-based
eviction, `bidaw` as the 6th scheduler) are all live.

- `uv run pytest -q` -> `497 passed`
- `uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml` -> `13/13 fields PASS`
- `uv run python -m nano_kvrouter.cli sweep --config configs/trace_mooncake.yaml` -> 6 schedulers on Mooncake FAST'25 real trace, cache-aware ~2× cache_hit vs cache-blind
- `uv run python -m nano_kvrouter.cli prefix-sensitivity --config configs/trace_burstgpt.yaml` -> 4-axis sweep over prefix-synthesis params on BurstGPT replay, with Mooncake real-trace `cache_hit` shown as informational anchor
- `uv run python -m nano_kvrouter.cli sweep --config configs/transfer_contention.yaml` -> per-node KV transfer lane queueing exposes ~30× `kv_transfer_time_avg_ms` inflation vs constant-cost baseline on the same 2p/2d cluster
- `uv run python -m nano_kvrouter.cli sweep --config configs/bidaw.yaml` -> 6-row table with Bidaw `KV_LOAD_*` event path on disk-hit requests; `bidaw_preparing_promotions ≈ disk-hit request count`
- `uv run python -m nano_kvrouter.cli run --config configs/bidaw-interactive.yaml --scheduler bidaw` -> session-history replay + opt-in previous-answer eviction profile plumbing
- repo-outside absolute `--config` sensitivity execution is supported

The repository currently exposes four public CLI workflows:

- `run`: run one scheduler on one config
- `sweep`: run all six schedulers and print a comparison table
- `sensitivity`: run config-driven LIVE field experiments and emit a field-level PASS/FAIL punch list
- `prefix-sensitivity`: scan prefix-synthesis params on a trace and report cache_hit sensitivity (Mooncake real-trace `cache_hit` shown as informational reference, **NOT** a fitting target)

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

### Synthetic prefix sharing on BurstGPT (P3-C M2, 2026-06-10)

BurstGPT (HPMLL, CC-BY-4.0) ships real arrival timestamps + prompt/response
lengths + `session_id`, but no prefix structure. To still exercise cache-aware
schedulers against its workload shape, M2 layers a `PrefixSynthesisModel` on
top of the trace:

- Zipf-distributed bucket selection across `num_buckets` prefix templates
- time-local recency bias via a sliding window (`p_local`, `local_window_s`)
- layered prefix sharing (e.g. 20% long-shared / 50% medium / 30% private)
- bucket prefixes are lazy-extended on demand — no `max_prompt_len` ceiling

Because the prefix model is a HYPOTHESIS layered onto BurstGPT, M2 ships a
`prefix-sensitivity` CLI rather than a fixed sweep. It reports how `cache_hit`
varies across the four synthesis axes and prints Mooncake's real-trace
`cache_hit` as an informational anchor — **NOT** a fitting target:

```bash
uv run python -m nano_kvrouter.cli prefix-sensitivity \
  --config configs/trace_burstgpt.yaml --scheduler conductor
```

Sample numbers on the bundled 1000-record BurstGPT sample with the conductor
scheduler (baseline `zipf_alpha=1.0, p_local=0.6, num_buckets=64,
sharing=mixed` gives `cache_hit=0.069`):

| axis             | range explored               | cache_hit range |
| ---------------- | ---------------------------- | --------------- |
| `zipf_alpha`     | 0.5 — 1.5                    | 0.032 — 0.093   |
| `p_local`        | 0.0 — 0.9                    | 0.047 — 0.073   |
| `num_buckets`    | 16 — 256                     | 0.044 — 0.092   |
| `sharing_layers` | `all_private` — `heavy_shared` | 0.000 — 0.140 |

Reference: Mooncake real-`hash_ids` `cache_hit` on `configs/trace_mooncake.yaml`
with conductor is **0.146** (informational; this is what real prefix reuse
looks like, not what the synthesis is required to match).

### Per-node KV transfer lane contention (P4-A, 2026-06-15)

Until P4-A, simulated KV transfers between prefill and decode nodes paid a
constant per-request cost `kv_bytes / bandwidth.gpu_to_gpu` — 10 transfers
sharing the same `(src, dst)` pair all finished at `now + cost`. That hid
post-prefill transfer queueing, the bottleneck Mooncake FAST'25 calls out
as "per-node KV transfer throughput".

P4-A introduces a `TransferModel` abstraction with two implementations
selected via `bandwidth.contention_model`:

| `contention_model` | Implementation | Behavior |
|--------------------|---------------|----------|
| `"none"` (**default**) | `NoopTransferModel` | constant cost; byte-identical to pre-P4-A baseline. All 7 existing scenario configs run unchanged. |
| `"per_node_lane"` | `PerNodeLaneTransferModel` | each node owns one egress + one ingress lane. A transfer reserves both for `[start, finish)`, where `start = max(now, egress.available_at[src], ingress.available_at[dst])`. Disjoint `(src, dst)` pairs run in parallel; transfers sharing a src or dst serialize. |

Demo scenario `configs/transfer_contention.yaml` (2 prefill × 2 decode,
synthetic 5 MB/s bandwidth so service time exceeds the effective
transfer-producing cadence):

```bash
uv run python -m nano_kvrouter.cli sweep --config configs/transfer_contention.yaml
```

Measured paired diff on the same yaml, only flipping the field:

| `contention_model` | `kv_transfer_time_avg_ms` |
|--------------------|--------------------------:|
| `none` | 104.9 |
| `per_node_lane` | 3137.1 |

A ~30× inflation in observed transfer time when 4 concurrent post-prefill
transfers queue on shared lanes. `kv_transfer_time_avg_ms` becomes
"end-to-end transfer time (service + queue wait)" under `per_node_lane` —
this is the design intent, not a metric regression.

### Schedulers see transfer-lane backlog (P4-B, 2026-06-15)

P4-A left a deliberate divergence: under `per_node_lane`, runtime KV
transfer cost included queue wait, but `compute_est_ttft` returned only
service time, so `MooncakeConductor`'s SLO gate and `E2Policy`'s
`run_cost` were oblivious to lane queueing. P4-B closes the loop.

A new narrow read-only Protocol `TransferBacklogView` exposes only
`peek_backlog(node_id) -> {"egress", "ingress"}`. All five scheduler
constructors now take a `backlog_view`; `compute_est_ttft` includes lane
wait in its estimate:

```
service        = (kv_bytes / bandwidth.gpu_to_gpu) * 1000
src_egress_w   = max(0, peek_backlog(prefill_node)["egress"] - now)
dst_ingress_w  = max(0, peek_backlog(decode_node)["ingress"] - now)
queue_wait     = max(src_egress_w, dst_ingress_w)
kv_transfer    = service + queue_wait
```

`max` (not sum) mirrors the runtime model: a transfer occupies both
lanes simultaneously, so its wait equals the LATER of the two
availabilities. `max(0, ...)` clamps past-backlog to zero. The new
narrow Protocol intentionally hides `request_transfer` from the
estimator so an estimation path can never reserve a lane and corrupt
the simulation (`rg "request_transfer\(" src/nano_kvrouter/scheduler/`
→ 0 hits, enforced by hard guard).

A new metric `kv_transfer_queued_avg_ms` exposes the queue wait
component separately from `kv_transfer_time_avg_ms = service + queued`:

| `contention_model` | `kv_transfer_queued_avg_ms` |
|--------------------|----------------------------:|
| `none` (default)   | `0.000` |
| `per_node_lane`    | `> 0`, scheduler-dependent |

Paired toggle on `configs/transfer_contention.yaml` (same yaml, only
`contention_model` flipped; conductor scheduler):

| metric | `none` | `per_node_lane` |
|--------|------:|----------------:|
| `kv_transfer_time_avg_ms`   | 104.86 | 1507.84 |
| `kv_transfer_queued_avg_ms` |   0.00 | 1402.98 |
| `rejection_rate`            |  0.751 |   0.809 |

Decomposition invariant holds exactly for all 5 schedulers on this
yaml: `lane.total - lane.queued = none.total = 104.86 ms` (service
component is invariant because `avg_prompt_len` is fixed).

Conductor's rejection rises under lane (the **intended primary
effect**: SLO gate sees queue wait → admits less). The other four
schedulers' rejection rates actually fall (about 0.62 → 0.45) — this
is mostly a **second-order side effect**, not scheduler intelligence:
lane queueing delays `KV_TRANSFER_COMPLETE`, giving decode nodes more
time to free slots before B1 (decode-capacity) rejection fires.

How much of `compute_est_ttft` each scheduler actually consumes:

- `conductor`: uses it for both 3-objective scoring AND SLO admission
  gate — full backlog awareness.
- `e2_policy`: uses it as the `run_cost` term in its 3-objective
  score (`w_h*hist + w_e*evict + w_r*run`), so its routing decisions
  ARE shifted by backlog (small but real). No SLO gate.
- `round_robin` / `least_loaded` / `prefix_greedy`: compute the
  estimate and embed it in `SchedulingDecision.estimated_ttft_ms`,
  but routing logic does not branch on it. Informational only.

See DESIGN §12.5 for the full breakdown.

Default `bandwidth.contention_model: "none"` still keeps every old
config and regression number byte-identical — `NoopTransferModel.peek_backlog`
returns zeros, `max(0, 0 - now) = 0`, and `kv_transfer` is unchanged.

### Bidaw I/O-aware scheduling + answer eviction (P5-Bidaw M1/M2, 2026-06-21)

P5 adds a sixth scheduler `bidaw` that simulates the **I/O-aware
request scheduling layer** of Bidaw (FAST'26). Until P5, requests
whose matched prefix lived on the disk tier got "magic"
zero-latency disk hits — `cache_manager.lookup()` reported the disk
hit and `compute_est_ttft` charged a `transfer_cost_ms` estimate,
but `PREFILL_START` did not actually wait for any disk-to-GPU
load. Bidaw treats those disk loads as **real events on the
critical path**:

| state | when | next event |
|-------|------|------------|
| `ready` | matched prefix has 0 disk blocks | `PREFILL_START` immediately |
| `preparing` | matched prefix has ≥ 1 disk block | enqueue + wait for the node's load slot |

Inside each decode node's preparing queue, ordering uses **disk-HRRN**:
`response_ratio = 1 + waiting_ms / max(1, disk_blocks)`. Small KV first
(higher ratio at zero wait), but waiting_ms compensates for large KV
to bound starvation. A **single in-flight load slot per decode node**
serializes the disk loads on that node; cross-node loads run in
parallel. When `KV_LOAD_COMPLETE` fires, the controller marks the
request "ready", `CacheManager` attempts metadata-only disk→CPU promotion
for the matched prefix blocks, and the request can proceed to
`PREFILL_START`. This is still a simulator approximation: no real tensor
copy exists, and CPU/GPU tiers are both treated as Bidaw's ready layer.

The Bidaw branch lives in two new modules
(`scheduler/bidaw.py` + `simulator/bidaw_controller.py`) plus a cli
wiring branch. The five existing schedulers, all eight existing
configs and `TransferModel` are unchanged for non-Bidaw schedulers. Bidaw's
only cache-layer write is the opt-in disk→CPU promotion path on
`KV_LOAD_COMPLETE`.

P5-Bidaw M2 adds an opt-in **previous-answer-based eviction** approximation.
It does not store tensors. Instead, interactive trace records carry
`session_id`, `query_length`, `round_index`, and `previous_answer_length`;
`TraceGenerator(prefix_mode="session_history")` reconstructs growing
conversation prompts; and `CacheManager` tags cached blocks with an
answer-length-derived hit potential. When Bidaw disk→CPU promotion needs CPU
space, low-potential CPU blocks are demoted before high-potential ones. The
profile can be generated from the public Interactive-conversation-workload CSV
with `scripts/convert_interactive_workload.py`.

Run the demo:

```bash
uv run python -m nano_kvrouter.cli run --config configs/bidaw.yaml \
    --scheduler bidaw
uv run python -m nano_kvrouter.cli sweep --config configs/bidaw.yaml
uv run python -m nano_kvrouter.cli run --config configs/bidaw-stress.yaml \
    --scheduler bidaw
uv run python -m nano_kvrouter.cli run --config configs/bidaw-interactive.yaml \
    --scheduler bidaw
```

Bidaw metrics surface in the summary (all default to `0.0`/`0`
on non-bidaw runs):

- `bidaw_preparing_wait_avg_ms` / `_p99_ms` — wait between arriving
  in the preparing queue and `KV_LOAD_START`.
- `bidaw_disk_load_service_avg_ms` — per-request disk load service
  time (the `KV_LOAD_*` interval).
- `bidaw_preparing_promotions` — count of requests that traversed
  the preparing path (≈ disk-hit request count).
- `bidaw_physical_promoted_blocks` / `_skipped_blocks` — block-level
  metadata promotion result after `KV_LOAD_COMPLETE`.
- `bidaw_answer_eviction_count` / `_evicted_blocks` /
  `_cpu_saved_blocks` — answer-aware CPU demotions.
- `bidaw_answer_eviction_hit_potential_avg` — average hit potential of
  answer-aware evicted blocks.
- `bidaw_answer_eviction_cpu_hit_rate` — CPU hits over CPU+Disk session hits.

See `doc/bidaw-deliverable.md` for the full Chinese deliverable
summary (mechanism mapping, simulator approximations, what is NOT
implemented, comparison table, ship verdict) and DESIGN §13 for
the design write-up.

**Known caveats** (documented in DESIGN §13):
1. `configs/bidaw.yaml` is a minimal-stress demo — disk load (~0.37 ms)
   is much faster than the request arrival interval (~67 ms), so the
   preparing queue never backs up and `bidaw_preparing_wait_avg_ms = 0`.
   HRRN ordering is covered by unit tests and exercised by
   `configs/bidaw-stress.yaml`.
2. Bidaw's routing mirrors `least_loaded` for decode-node choice.
   On `bidaw.yaml` this produces lower `cache_hit_ratio` than
   `conductor` (0.69 vs 0.82), but **`bidaw` ends up with lower
   `e2e_avg_ms` (386 vs 413)** because the cache_hit advantage of
   conductor is on paper only — conductor does not pay the real
   disk-load latency that bidaw does. Same-semantics comparison
   would require all schedulers to pay real disk-load time.

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

# Replay Mooncake / BurstGPT traces (P3-C)
uv run python -m nano_kvrouter.cli sweep --config configs/trace_mooncake.yaml
uv run python -m nano_kvrouter.cli sweep --config configs/trace_burstgpt.yaml

# Scan prefix-synthesis sensitivity on a length-only trace (BurstGPT)
uv run python -m nano_kvrouter.cli prefix-sensitivity \
  --config configs/trace_burstgpt.yaml --scheduler conductor

# Exercise per-node KV transfer lane contention (P4-A)
uv run python -m nano_kvrouter.cli sweep --config configs/transfer_contention.yaml

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
| [`configs/trace_mooncake.yaml`](configs/trace_mooncake.yaml) | Mooncake FAST'25 trace replay (real `hash_ids`, block_size=512) |
| [`configs/trace_burstgpt.yaml`](configs/trace_burstgpt.yaml) | BurstGPT trace replay with synthetic prefix sharing (`PrefixSynthesisModel`) |
| [`configs/transfer_contention.yaml`](configs/transfer_contention.yaml) | 2p/2d cluster with `bandwidth.contention_model: per_node_lane` to exercise the P4-A `TransferModel` lane queueing |
| [`configs/bidaw.yaml`](configs/bidaw.yaml) | 2p/4d cluster with `cpu_blocks=0` + `disk_blocks=4000` so disk-tier hits are visible; exercises the P5-Bidaw `KV_LOAD_*` event path on the new `bidaw` scheduler |
| [`configs/bidaw-stress.yaml`](configs/bidaw-stress.yaml) | Single-node Bidaw stress case with slow disk and CPU tier enabled, useful for preparing-queue contention and physical promotion checks |
| [`configs/bidaw-interactive.yaml`](configs/bidaw-interactive.yaml) | Tiny session-history fixture that wires Bidaw's previous-answer eviction profile path; real traces should be produced with `scripts/convert_interactive_workload.py` |

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
| `BidawPolicy`       | Bidaw (FAST'26)                | Routes like `least_loaded`; the contribution is the dual-queue admission controller + disk-HRRN + real `KV_LOAD_*` event path on disk-hit requests. See DESIGN §13. |

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
│   ├── conductor.py
│   └── bidaw.py            # Bidaw I/O-aware routing + disk-HRRN (P5-Bidaw M1)
├── simulator/
│   ├── event.py
│   ├── engine.py
│   ├── generator.py         # Poisson generator
│   ├── trace_generator.py   # JSONL trace replay (Mooncake / BurstGPT)
│   ├── prefix_synthesis.py  # Zipf+locality+layered prefix model (P3-C M2)
│   ├── transfer_model.py    # Noop / per-node lane KV transfer cost models (P4-A)
│   └── bidaw_controller.py  # Bidaw dual-queue admission controller (P5-Bidaw M1)
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
- **Bidaw** — *Bidaw: Interactive LLM Serving with Two-Tier KV Storage*, FAST'26 (I/O-aware dual-queue request scheduling plus metadata-only previous-answer eviction; storage engine + tensor caching remain out of scope, see `doc/bidaw-deliverable.md`).
- **Llumnix** — *Llumnix: Dynamic Scheduling for Large Language Model Serving*, OSDI'24 (migration logic — roadmap).
- **vLLM** — *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP'23.

A more detailed paper-to-module mapping lives in
[`doc/code-review/README.md`](doc/code-review/README.md).

## Additional docs

- [DESIGN.md](DESIGN.md): current architecture and milestone fidelity
- [doc/code-review/README.md](doc/code-review/README.md): per-file review index
- [doc/review-notes.md](doc/review-notes.md): backlog and cross-file notes

# P5-Bidaw M0 preflight — I/O-aware request scheduling

> Status: **M0 preflight, doc only.** No business code or repo config was
> changed in this step. Branch: `feat/bidaw-io-aware-scheduling`.
>
> Repository baseline at preflight time: `HEAD = 3767be9`
> (`docs(p4-b): correct E2 backlog consumption + peek_backlog docstring`),
> working tree clean before this document was added.
>
> Plan source: `/Users/Admin/.claude/plans/tidy-sauteeing-dijkstra.md`
> (P5-Bidaw v2, 2026-06-20).

## 1. Scope decision locked by M0

M1 is a **Bidaw-inspired control-plane simulation**, not a faithful
runtime reproduction:

- Implement: dual ready/preparing queue, disk-HRRN ordering, real
  `KV_LOAD_START` / `KV_LOAD_COMPLETE` event path, single load slot per
  decode node, request-local gating before `PREFILL_START`.
- Do not implement in M1: real tensor allocation, CUDA streams,
  storage-efficient tensor caching, previous-answer eviction,
  cross-node migration, physical disk->cpu promotion.
- CacheManager must remain untouched in M1. Physical disk->cpu
  promotion is M1.5 only.

Important plan drift found during M0: the v2 plan commitments say
M1 uses request-local gating and does **not** mutate `BlockPool`, but
the architecture diagram / API sketch still mention
`promote disk->cpu (best effort)` and
`CacheManager.promote_matched_disk_blocks_to_cpu(...)`. Dispatch must
treat those as stale M1.5 text, not M1 requirements.

## 2. Blast-radius grep

### 2.1 Scheduler registry and 5-scheduler literals

Command:

```bash
rg -n "SCHEDULER_NAMES|5-scheduler|five scheduler|5 scheduler|Run all 5|Run all five|comparison" src tests configs README.md DESIGN.md doc
```

Key current hits:

| file:line | M1 action |
|---|---|
| `src/nano_kvrouter/cli.py:60` | Add `"bidaw"` to `SCHEDULER_NAMES`. |
| `src/nano_kvrouter/cli.py:870` | Replace hard-coded table title `5-scheduler comparison` with `f"{len(SCHEDULER_NAMES)}-scheduler comparison"`. |
| `src/nano_kvrouter/cli.py:1019` | Replace docstring `Run all 5 schedulers...`. |
| `src/nano_kvrouter/cli.py:1254` | Replace argparse help `Run all 5 schedulers and compare`. |
| `README.md:37,45,200,237,276,495` | Docs sync after M1, not part of M1 code commit. |
| `DESIGN.md:532,606` | Docs sync after M1, not part of M1 code commit. |
| `doc/code-review/p4-b-m0-preflight.md:190` | Historical doc; do not edit. |

Tests import `SCHEDULER_NAMES` in `tests/test_cli.py`, so adding
`bidaw` will automatically exercise `_build_scheduler("bidaw", ...)`
where that test loops over the registry.

### 2.2 Event lifecycle and enum tests

Command:

```bash
rg -n "EventType|REQUEST_ARRIVE|PREFILL_START|PREFILL_COMPLETE|KV_TRANSFER_START|KV_TRANSFER_COMPLETE|DECODE_START|DECODE_COMPLETE|lifecycle" src/nano_kvrouter tests doc README.md DESIGN.md
```

M1 must update these live surfaces:

| file:line | M1 action |
|---|---|
| `src/nano_kvrouter/simulator/event.py:39-77` | Add `KV_LOAD_START`, `KV_LOAD_COMPLETE`; update lifecycle docstring with Bidaw branch. |
| `tests/test_event.py:17-29` | Expected enum-name set must include the 2 new events. |
| `src/nano_kvrouter/cli.py:127-537` | Add a Bidaw-only wiring branch; keep the existing 5-scheduler event path unchanged. |
| `src/nano_kvrouter/metrics/collector.py:140-149` | Attach new handlers for load events. |
| `tests/test_metrics_collector.py:250-265` | Attach-registration test must include new load handlers. |
| `src/nano_kvrouter/metrics/collector.py:45-57` | Payload contract docstring must include `KV_LOAD_*`. |

### 2.3 `_build_scheduler` and `_wire_simulator` hidden callers

Command:

```bash
rg -n "def _wire_simulator|_wire_simulator\(|def _build_scheduler|_build_scheduler\(" src tests
```

Current hidden callers:

| area | hits |
|---|---|
| `_build_scheduler` definition | `src/nano_kvrouter/cli.py:65` |
| `_build_scheduler` runtime call | `src/nano_kvrouter/cli.py:576` |
| `_build_scheduler` tests | `tests/test_cli.py:64,70,75,82,89,272,705`; `tests/test_trace_generator.py:133` |
| `_wire_simulator` definition | `src/nano_kvrouter/cli.py:127` |
| `_wire_simulator` runtime call | `src/nano_kvrouter/cli.py:598` |
| `_wire_simulator` tests | `tests/test_cli.py:195,279,389,437,497,563,619,706,744,795,867`; `tests/test_trace_generator.py:136`; `tests/test_metrics_collector.py:610,617` |

M1 should avoid changing the public shape of `_wire_simulator` unless
strictly necessary. If it adds parameters, every hidden caller above
must be updated.

### 2.4 Scheduler schedule call sites

Command:

```bash
rg -n "def schedule\(|schedule\(" src/nano_kvrouter/scheduler tests/test_round_robin.py tests/test_least_loaded.py tests/test_prefix_greedy.py tests/test_e2_policy.py tests/test_conductor.py
```

M1 should add `BidawPolicy.schedule(...)` without changing the existing
`SchedulingPolicy.schedule(...)` signature. The 5 existing schedulers
already require `now=...`; do not reopen this API.

## 3. Proposed `configs/bidaw.yaml` seed

M0 explored temporary configs under `/private/tmp`; no repo config was
created. The best M1 seed is candidate J:

```yaml
cluster:
  prefill_nodes: 2
  decode_nodes: 4

node:
  capacity: 16
  gpu_blocks: 80
  cpu_blocks: 0
  disk_blocks: 4000

model:
  block_size: 16
  kv_bytes_per_token: 4096
  prefill_cost_per_token_ms: 0.04
  decode_base_ms: 5.0
  marginal_decode_ms: 0.5
  prefill_chunk_size: 512

bandwidth:
  gpu_to_gpu: 3.0e11
  gpu_to_cpu: 3.2e10
  cpu_to_disk: 5.0e9
  contention_model: none

slo:
  ttft_target_ms: 600.0
  tbt_target_ms: 50.0

workload:
  request_rate: 15.0
  duration_s: 20.0
  prefix_sharing_ratio: 0.88
  avg_prompt_len: 512
  avg_output_len: 64

generator:
  num_buckets: 4
  vocab_size: 32000
  seed: 42

scheduler:
  name: conductor
  params:
    alpha: 1.0
    beta: 1.0
    gamma: 1.0
```

Rationale:

- `cpu_blocks: 0` makes disk hits visible without relying on physical
  disk->cpu promotion, which is explicitly deferred to M1.5.
- `gpu_blocks: 80` keeps a mix of GPU and disk hits; this is less
  extreme than the colder `gpu_blocks: 70` candidate.
- `prefix_sharing_ratio: 0.88` and `num_buckets: 4` provide enough
  reuse to exercise both cache-aware and cache-blind policies.

Candidate J frozen output path:

```bash
uv run python -m nano_kvrouter.cli sweep \
  --config /private/tmp/bidaw-cand-j.yaml \
  --output /private/tmp/bidaw-cand-j.json
```

Candidate J results:

| scheduler | cache_hit | disk tier share | disk-hit proxy | rejection | ttft_p99 |
|---|---:|---:|---:|---:|---:|
| round_robin | 0.589 | 0.742 | 0.438 | 0.000 | 31.656 |
| least_loaded | 0.693 | 0.877 | 0.608 | 0.000 | 31.670 |
| prefix_greedy | 0.823 | 0.310 | 0.255 | 0.000 | 30.720 |
| e2_policy | 0.723 | 0.639 | 0.462 | 0.000 | 32.890 |
| conductor | 0.823 | 0.310 | 0.255 | 0.000 | 30.720 |

`disk-hit proxy = cache_hit_ratio * cache_hit_by_tier_ratio["disk"]`.
This is not an exact request count, but it is a useful pre-M1 proxy for
how much traffic can enter Bidaw's preparing queue.

Known caveat: if `BidawPolicy` routes exactly like `least_loaded`, the
candidate may be colder than the 30-50% target (`0.608` proxy). That is
acceptable for M1 as a stress test, but M1 dispatch should allow one
round of tuning if the new `bidaw_preparing_wait_*` metrics are too
dominated by preparing traffic.

## 4. Frozen regression baselines

All commands below were run on `HEAD=3767be9` before M1 code changes.
M1 must keep the 5 existing scheduler rows byte-identical except for
expected table-title wording and the addition of the new `bidaw` row.

### 4.1 Existing sweeps

Output files:

```text
/private/tmp/p5-m0-default.json
/private/tmp/p5-m0-heavy.json
/private/tmp/p5-m0-hicache.json
/private/tmp/p5-m0-pd_split.json
/private/tmp/p5-m0-trace_mooncake.json
/private/tmp/p5-m0-trace_burstgpt.json
```

Key rows:

| config | scheduler | cache_hit | rejection | ttft_p50 | ttft_p99 | tbt_avg | throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| default | round_robin | 0.502 | 0.000 | 27.840 | 49.928 | 8.889 | 51.089 |
| default | conductor | 0.560 | 0.000 | 27.831 | 47.986 | 9.652 | 50.986 |
| heavy | round_robin | 0.540 | 0.192 | 24.317 | 46.191 | 6.708 | 61.010 |
| heavy | conductor | 0.528 | 0.191 | 23.772 | 45.596 | 6.781 | 61.011 |
| hicache | prefix_greedy | 0.218 | 0.000 | 25.987 | 32.405 | 6.017 | 14.509 |
| hicache | conductor | 0.218 | 0.000 | 25.987 | 32.405 | 6.017 | 14.509 |
| pd_split | round_robin | 0.536 | 0.372 | 53.648 | 89.742 | 19.694 | 11.146 |
| pd_split | conductor | 0.518 | 0.369 | 54.516 | 91.261 | 19.758 | 11.132 |
| trace_mooncake | round_robin | 0.075 | 0.000 | 521.680 | 4100.494 | 6.240 | 2.978 |
| trace_mooncake | e2_policy | 0.153 | 0.000 | 452.083 | 4149.880 | 6.973 | 2.978 |
| trace_mooncake | conductor | 0.146 | 0.000 | 456.350 | 4147.405 | 7.459 | 2.978 |
| trace_burstgpt | round_robin | 0.050 | 0.000 | 16.902 | 133.443 | 5.500 | 0.016 |
| trace_burstgpt | conductor | 0.069 | 0.000 | 16.886 | 129.671 | 5.514 | 0.016 |

M1 reviewer should compare full JSON, not only this reduced table.

### 4.2 Bidaw candidate baseline

Output file:

```text
/private/tmp/bidaw-cand-j.json
```

These are the 5 existing scheduler rows on the proposed Bidaw scenario.
M1 must preserve these rows and add a 6th `bidaw` row.

| scheduler | cache_hit | rejection | ttft_p50 | ttft_p99 | tbt_avg | throughput | tier |
|---|---:|---:|---:|---:|---:|---:|---|
| round_robin | 0.589 | 0.000 | 12.150 | 31.656 | 5.834 | 14.237 | disk 0.742 / gpu 0.258 |
| least_loaded | 0.693 | 0.000 | 11.046 | 31.670 | 5.901 | 14.246 | disk 0.877 / gpu 0.123 |
| prefix_greedy | 0.823 | 0.000 | 11.671 | 30.720 | 6.355 | 14.226 | disk 0.310 / gpu 0.690 |
| e2_policy | 0.723 | 0.000 | 12.328 | 32.890 | 6.413 | 14.239 | disk 0.639 / gpu 0.361 |
| conductor | 0.823 | 0.000 | 11.671 | 30.720 | 6.355 | 14.226 | disk 0.310 / gpu 0.690 |

## 5. Paper uncertainty and simulator approximation

This M0 did not re-read the Bidaw PDF. For M1 implementation, treat
the user-approved spec as authoritative:

- disk-HRRN formula: `response_ratio = 1 + waiting_time_ms / kv_size`.
- `kv_size` unit in this simulator: **matched disk blocks**. This is
  simpler than bytes/tokens and lines up with the current block-based
  cache accounting.
- If `kv_size_blocks == 0`, use `max(1, kv_size_blocks)` to avoid
  division by zero; no-disk-hit requests should not normally be in the
  preparing queue.
- M1 is request-local gating only. `KV_LOAD_COMPLETE` means "this
  request may start prefill"; it does not mean the underlying block
  moved tier globally.
- Because no physical promotion exists in M1, repeated requests for
  the same disk-resident prefix may repeatedly enter preparing. That is
  a known fidelity gap, not a cache-manager bug.

## 6. M1 dispatch hard gates

Before commit, M1 must prove:

1. `uv run pytest -q` passes.
2. 6 existing sweeps match the JSON baselines above for the 5 existing
   schedulers.
3. `configs/bidaw.yaml` sweep shows 6 rows including `bidaw`.
4. Existing 5 scheduler rows on `configs/bidaw.yaml` match the
   candidate-J baseline unless the config is intentionally retuned and
   M0 baseline is updated.
5. `bidaw_preparing_wait_avg_ms > 0`.
6. `bidaw_disk_load_service_avg_ms > 0`.
7. `bidaw_preparing_promotions` is approximately the number of requests
   with matched disk blocks.
8. Event ordering: disk-hit request has `KV_LOAD_COMPLETE <= PREFILL_START`;
   it must not enter prefill before load completion.
9. Single-slot serialization: for the same decode node, second
   `KV_LOAD_START >= first KV_LOAD_COMPLETE`.
10. Head-of-line absence: a ready/no-disk request arriving behind a
    large preparing request must still reach `PREFILL_START` without
    waiting for that large disk load.

## 7. M0 verdict

YES for dispatch with one correction: M1 instructions must explicitly
delete stale physical-promotion wording from the plan handoff. The
implementation target is request-local Bidaw gating plus metrics, not
`CacheManager` tier mutation.

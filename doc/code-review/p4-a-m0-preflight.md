# P4-A M0 preflight — TransferModel v1 (per-node lane contention)

> Status: **M0 preflight, no code change.** This document captures the
> design surface that M1 (dispatched to Sonnet) will land. Plan source:
> `/Users/Admin/.claude/plans/gentle-routing-cerf.md` (v3, Codex YES on
> 2026-06-15).
>
> Repository baseline at preflight time: `main = b5c7cdd`,
> `git status` clean, `uv run pytest -q` → 439 passed.

## 1. Grep of every existing transfer call site

Every place the current constant-cost transfer model touches code. M1
must keep all "scheduler / `compute_est_ttft`" sites unchanged
(plan v3 §3 hard constraint).

### 1.1 Runtime cost calculation (the one site M1 rewrites)

| file:line | what |
|-----------|------|
| `src/nano_kvrouter/cli.py:335` | `cost_ms = (kv_bytes / bandwidth_cfg.gpu_to_gpu) * 1000.0` |
| `src/nano_kvrouter/cli.py:341` | first `cost_ms` recorded into pending-transfer dict |
| `src/nano_kvrouter/cli.py:344-354` | schedules `KV_TRANSFER_START` at `engine.now()` carrying `cost_ms` |
| `src/nano_kvrouter/cli.py:355-366` | schedules `KV_TRANSFER_COMPLETE` at `engine.now() + cost_ms` carrying `cost_ms` |
| `src/nano_kvrouter/cli.py:368-376` | `on_kv_transfer_start` handler (debug-log-only) |
| `src/nano_kvrouter/cli.py:378-414` | `on_kv_transfer_complete` handler (materializes, admits decode) |

### 1.2 Metric sampling (M1 docstring touches, no behavior change)

| file:line | what |
|-----------|------|
| `src/nano_kvrouter/metrics/collector.py:30-32` | class docstring describing `kv_transfer_time_avg_ms` semantics |
| `src/nano_kvrouter/metrics/collector.py:46-47` | event payload contract (lists `cost_ms` field) |
| `src/nano_kvrouter/metrics/collector.py:67-69` | metric semantics block |
| `src/nano_kvrouter/metrics/collector.py:95` | `self._kv_transfer_cost_samples: list[float]` |
| `src/nano_kvrouter/metrics/collector.py:135-136` | event subscriptions |
| `src/nano_kvrouter/metrics/collector.py:168-171` | `summary()` emits `kv_transfer_time_avg_ms` |
| `src/nano_kvrouter/metrics/collector.py:253-261` | `_on_kv_transfer_start` (transfer_id guard) |
| `src/nano_kvrouter/metrics/collector.py:264-273` | `_on_kv_transfer_complete` (samples `payload["cost_ms"]`) |

**M1 behavior**: collector keeps sampling `payload["cost_ms"]`. With
`NoopTransferModel` the value is unchanged; with
`PerNodeLaneTransferModel` `cost_ms == queued + service`. Docstring
gets a paragraph explaining the semantic shift.

### 1.3 Scheduler TTFT estimate (M1 MUST NOT touch)

`compute_est_ttft` and its 5 callers all read `bandwidth.gpu_to_gpu`
and produce a constant-formula `kv_transfer` term:

| file:line | what |
|-----------|------|
| `src/nano_kvrouter/scheduler/base.py:256` | `kv_transfer = (kv_bytes / bandwidth_bytes_per_s) * 1000.0` |
| `src/nano_kvrouter/scheduler/round_robin.py:112` | passes `bw_cfg.gpu_to_gpu` |
| `src/nano_kvrouter/scheduler/least_loaded.py:107` | passes `bw_cfg.gpu_to_gpu` |
| `src/nano_kvrouter/scheduler/prefix_greedy.py:127` | passes `bw_cfg.gpu_to_gpu` |
| `src/nano_kvrouter/scheduler/e2_policy.py:124` + `:138` | passes `bw_cfg.gpu_to_gpu` |
| `src/nano_kvrouter/scheduler/conductor.py:135` | passes `bw_cfg.gpu_to_gpu` |

**Plan v3 §3 commitment**: schedulers stay on the constant formula.
The estimate diverges from the simulated runtime cost when
contention happens — this is an **acceptable v1 limitation**; backlog
item #39 covers the v2 follow-up (let conductor see `peek_backlog`).

### 1.4 Naming hazard — `transfer_cost_ms` is a different concept

`scheduler/base.py:CacheLookup.transfer_cost_ms`,
`kv_cache/cache_manager.py:lookup` and `conductor.py:transfer_penalty`
all refer to **per-tier reload cost** (CPU/Disk → GPU), not the
prefill→decode KV transfer M1 is modeling. Naming would collide if
the new model exposed a `transfer_cost_ms` API. **M1 keeps the new
API names distinct**: `service_cost_ms`, `queued_cost_ms` (event
payload); `request_transfer`, `peek_backlog` (TransferModel API).

## 2. Final API surfaces (M1 implements these verbatim)

### 2.1 `simulator/transfer_model.py` (new, ~100 LOC)

```python
from __future__ import annotations
from typing import Protocol


class TransferModel(Protocol):
    """Pluggable cost / contention model for KV transfer.

    Two implementations live in this module:

    * NoopTransferModel — constant cost, byte-identical to pre-P4-A
      behavior. Always selected when bandwidth.contention_model == "none".
    * PerNodeLaneTransferModel — per-node egress + ingress lane queue.
      Selected when bandwidth.contention_model == "per_node_lane".

    Implementations must be deterministic given the call sequence
    (request_transfer is the only state-mutating method).
    """

    def request_transfer(
        self,
        src_node_id: str,
        dst_node_id: str,
        now: float,
        cost_ms: float,
    ) -> tuple[float, float]:
        """Reserve lanes and return (start_time, finish_time).

        cost_ms is the *service* cost computed from
        kv_bytes / bandwidth.gpu_to_gpu. Implementations may delay
        start_time past `now` if upstream resources are still busy.
        """

    def peek_backlog(self, node_id: str) -> dict[str, float]:
        """Return {"egress": float, "ingress": float} available_at.

        MUST be side-effect-free: repeated calls before any
        request_transfer between them return identical results.
        Used by future schedulers (out of M1 scope) to factor backlog
        into compute_est_ttft.
        """


class NoopTransferModel:
    """Constant-cost passthrough.

    Returns (now, now + cost_ms) every call. peek_backlog always
    returns {"egress": 0.0, "ingress": 0.0}. With this model,
    KV_TRANSFER_COMPLETE event timing is byte-identical to the
    pre-P4-A baseline.
    """

    def request_transfer(self, src_node_id, dst_node_id, now, cost_ms):
        return (now, now + cost_ms)

    def peek_backlog(self, node_id):
        return {"egress": 0.0, "ingress": 0.0}


class PerNodeLaneTransferModel:
    """Per-node egress + ingress lane queue.

    Each node has one egress lane (used when it's the src of a
    transfer) and one ingress lane (used when it's the dst). A
    transfer occupies BOTH lanes simultaneously for [start, finish):

        start = max(now, egress.available_at[src], ingress.available_at[dst])
        finish = start + cost_ms

    Both lanes' available_at are then advanced to finish. Disjoint
    (src, dst) pairs run in parallel; transfers sharing a src or dst
    serialize. This matches Mooncake's "per-node KV transfer
    throughput is the bottleneck" semantics without modeling RDMA
    link topology.
    """

    def __init__(self) -> None:
        self._egress_available_at: dict[str, float] = {}
        self._ingress_available_at: dict[str, float] = {}

    def request_transfer(self, src_node_id, dst_node_id, now, cost_ms):
        start = max(
            now,
            self._egress_available_at.get(src_node_id, 0.0),
            self._ingress_available_at.get(dst_node_id, 0.0),
        )
        finish = start + cost_ms
        self._egress_available_at[src_node_id] = finish
        self._ingress_available_at[dst_node_id] = finish
        return (start, finish)

    def peek_backlog(self, node_id):
        return {
            "egress": self._egress_available_at.get(node_id, 0.0),
            "ingress": self._ingress_available_at.get(node_id, 0.0),
        }
```

### 2.2 `config.py` — `BandwidthConfig` gains one field

```python
from typing import Literal

class BandwidthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_to_gpu: float = Field(default=300e9, gt=0, description=...)
    gpu_to_cpu: float = Field(default=32e9, gt=0, description=...)
    cpu_to_disk: float = Field(default=5e9, gt=0, description=...)

    # NEW in P4-A M1
    contention_model: Literal["none", "per_node_lane"] = Field(
        default="none",
        description=(
            "LIVE in P4-A M1. 'none' = constant-cost transfer (pre-P4-A "
            "behavior). 'per_node_lane' = each src/dst node has an "
            "egress/ingress lane that serializes simultaneous transfers; "
            "raises kv_transfer_time_avg_ms under contention."
        ),
    )
```

Pydantic's built-in `Literal` validation is sufficient; no extra
`field_validator` needed (plan v3 nit).

### 2.3 `cli.py` wiring diff plan

Two edits in `cli.py`; the only `if` on `contention_model` lives in
`_run_one`.

**Edit A — `_run_one` constructs the model**

After line 549 (`cm = CacheManager(...)`), before `sched = ...`:

```python
from nano_kvrouter.simulator.transfer_model import (
    NoopTransferModel,
    PerNodeLaneTransferModel,
    TransferModel,
)

transfer_model: TransferModel = (
    PerNodeLaneTransferModel()
    if cfg.bandwidth.contention_model == "per_node_lane"
    else NoopTransferModel()
)
```

And pass `transfer_model=transfer_model` to `_wire_simulator(...)`.

**Edit B — `_wire_simulator` signature + `on_prefill_complete`**

Signature gains `transfer_model: TransferModel`.

In `on_prefill_complete`, replace lines 334-366 (compute `cost_ms` →
schedule `KV_TRANSFER_START` + `KV_TRANSFER_COMPLETE`) with:

```python
kv_bytes = len(req.token_ids) * model_cfg.kv_bytes_per_token
service_cost_ms = (kv_bytes / bandwidth_cfg.gpu_to_gpu) * 1000.0
start_t, finish_t = transfer_model.request_transfer(
    decision.prefill_node,
    decision.decode_node,
    engine.now(),
    service_cost_ms,
)
queued_cost_ms = start_t - engine.now()
total_cost_ms = finish_t - engine.now()  # == queued + service

transfer_id = _next_transfer_id(request_id)
_pending_transfers[transfer_id] = {
    "request_id": request_id,
    "src": decision.prefill_node,
    "dst": decision.decode_node,
    "cost_ms": total_cost_ms,
    "request": req,
}
payload_base = {
    "request_id": request_id,
    "transfer_id": transfer_id,
    "src_node_id": decision.prefill_node,
    "dst_node_id": decision.decode_node,
    "service_cost_ms": service_cost_ms,
    "queued_cost_ms": queued_cost_ms,
    "cost_ms": total_cost_ms,
}
engine.schedule(Event(
    time=start_t,
    type=EventType.KV_TRANSFER_START,
    payload={**payload_base},
))
engine.schedule(Event(
    time=finish_t,
    type=EventType.KV_TRANSFER_COMPLETE,
    payload={**payload_base, "request": req},
))
```

**Byte-identical guarantee**: under `NoopTransferModel`,
`start_t = engine.now()`, `queued_cost_ms = 0`,
`total_cost_ms = service_cost_ms`. Event timestamps and `cost_ms`
field value match the pre-change scheduler output exactly.

### 2.4 Event payload — three-field semantic table

| Field | Meaning | Noop value | PerNodeLane value |
|-------|---------|------------|-------------------|
| `service_cost_ms` | Pure transfer time = `kv_bytes / bandwidth.gpu_to_gpu` | same as `cost_ms` | constant per request size |
| `queued_cost_ms` | Wait for lane availability = `start_t - now` | always `0.0` | `≥ 0`; grows under contention |
| `cost_ms` | End-to-end transfer wall clock = `finish_t - now` = service + queued | same as `service_cost_ms` (backward compat) | service + queued |

`MetricsCollector` keeps sampling `payload["cost_ms"]`, so
`kv_transfer_time_avg_ms` becomes "end-to-end transfer time (incl.
queue wait)" under `per_node_lane`. Docstring update goes in
`metrics/collector.py:30-32` and the event payload contract block
at `:46-47`. **No new metric added in M1** (backlog #39 covers
`kv_transfer_queued_avg_ms`).

### 2.5 `configs/transfer_contention.yaml` — only new yaml

Based on `default.yaml`, minimal changes that create contention:

```yaml
cluster:
  prefill_nodes: 2
  decode_nodes: 2

node:
  gpu_blocks: 800
  cpu_blocks: 0
  disk_blocks: 0
  capacity: 8

model:
  block_size: 16
  kv_bytes_per_token: 512
  prefill_cost_per_token_ms: 0.1
  decode_base_ms: 5.0
  marginal_decode_ms: 1.0
  prefill_chunk_size: 512

bandwidth:
  gpu_to_gpu: 30000000000.0   # 10x slower than default to amplify contention
  gpu_to_cpu: 32000000000.0
  cpu_to_disk: 5000000000.0
  contention_model: per_node_lane

slo:
  ttft_target_ms: 2000.0
  tbt_target_ms: 100.0

workload:
  request_rate: 50.0
  duration_s: 5.0
  prefix_sharing_ratio: 0.5
  avg_prompt_len: 1024
  avg_output_len: 128

generator:
  num_buckets: 8
  vocab_size: 32000
  seed: 42

scheduler:
  name: conductor
  params:
    alpha: 1.0
    beta: 1.0
    gamma: 1.0
```

Rationale: 2×2 P/D forces cross-node transfer mixes (p0→d0, p0→d1,
p1→d0, p1→d1) — lets the lane model exhibit both same-egress and
same-ingress contention. Throttling `gpu_to_gpu` 10× makes transfer
cost large enough that queueing is measurable in
`kv_transfer_time_avg_ms`.

## 3. Regression baselines (frozen at b5c7cdd)

M1 must reproduce all numbers below **byte-identical** when
`contention_model="none"` (default for all 7 existing yamls).

### 3.1 Sweep × 6 yaml — cache_hit per scheduler

| yaml | round_robin | least_loaded | prefix_greedy | e2_policy | conductor |
|------|------------:|-------------:|--------------:|----------:|----------:|
| `default.yaml` | 0.502 | 0.496 | 0.560 | 0.558 | 0.560 |
| `heavy.yaml` | 0.540 | 0.525 | 0.582 | 0.564 | 0.528 |
| `hicache.yaml` | 0.040 | 0.026 | 0.218 | 0.197 | 0.218 |
| `pd_split.yaml` | 0.536 | 0.525 | 0.563 | 0.563 | 0.518 |
| `trace_mooncake.yaml` | 0.075 | 0.069 | 0.146 | 0.153 | 0.146 |
| `trace_burstgpt.yaml` | 0.050 | 0.061 | 0.061 | 0.069 | 0.069 |

Notable secondary numbers Sonnet must also match (subset; full
re-run authoritative):

- `default.yaml` / conductor: `ttft_p50≈27.8ms`, `ttft_p99≈47.9ms`,
  `tbt_avg≈9.652`, `throughput≈50.9 req/s`
- `heavy.yaml` / conductor: `rejection=0.191`, `throughput≈61.0 req/s`;
  prefix_greedy: `rejection=0.518`, `throughput≈36.4 req/s`
- `pd_split.yaml` / conductor: `ttft_p50≈54.5ms`, `rejection=0.369`,
  `throughput≈11.1 req/s`
- `trace_mooncake.yaml` / conductor: `ttft_p50≈456.4ms`, `tbt_avg≈7.459`
- `hicache.yaml` / conductor: `ttft_p50≈25.9ms`, `tbt_avg≈6.017`

### 3.2 Sensitivity workflow — `configs/sensitivity.yaml`

```bash
uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml
```

Expected: **13 / 13 fields PASS**.

### 3.3 Prefix-sensitivity — `configs/trace_burstgpt.yaml`

```bash
uv run python -m nano_kvrouter.cli prefix-sensitivity \
  --config configs/trace_burstgpt.yaml --scheduler conductor
```

Expected baseline rows (cache_hit column, scheduler=conductor):

| axis | value | cache_hit |
|------|-------|----------:|
| zipf_alpha | 0.5 | 0.032 |
| zipf_alpha | 1.0 (baseline) | 0.069 |
| zipf_alpha | 1.5 | 0.093 |
| p_local | 0.0 | 0.047 |
| p_local | 0.3 | 0.057 |
| p_local | 0.6 (baseline) | 0.069 |
| p_local | 0.9 | 0.073 |
| num_buckets | 16 | 0.092 |
| num_buckets | 64 (baseline) | 0.069 |
| num_buckets | 256 | 0.044 |
| sharing | all_private | 0.000 |
| sharing | mixed (baseline) | 0.069 |
| sharing | heavy_shared | 0.140 |

### 3.4 pytest

`uv run pytest -q` → **439 passed** at b5c7cdd.

M1 expected delta: +6 mandatory unit tests in `test_transfer_model.py`
+ 1 paired CLI smoke + 1-2 `test_config.py` cases = pytest count after
M1 should land around **446-450 passed**.

## 4. Self-check (re-confirms plan v3)

- [x] All `cost_ms` calculation / sampling / event sites are listed in
      §1 and covered by wiring in §2.3 / §2.4.
- [x] All 7 existing yamls remain untouched (6 sweep + 1 sensitivity
      workflow); only addition is `configs/transfer_contention.yaml`.
- [x] Hard gate test (`test_per_node_lane_serializes_4_simultaneous_transfers`)
      directly calls `PerNodeLaneTransferModel.request_transfer()`; does
      not go through `SimulationEngine` / CLI.
- [x] CLI smoke test uses **paired toggle** on a single yaml
      (`transfer_contention.yaml`), deep-copying the config and
      flipping only `bandwidth.contention_model`.
- [x] `peek_backlog` defined as side-effect-free, with dedicated test
      `test_peek_backlog_is_side_effect_free`.
- [x] Schedulers (`scheduler/*.py`) are not in §2 file list; plan v3
      §3 hard constraint preserved.
- [x] Naming hazard: `transfer_cost_ms` (per-tier reload) and new
      `service_cost_ms` / `queued_cost_ms` (transfer queue) are
      distinct vocabulary — §1.4 calls this out.

## 5. Open items for M1 dispatch package

Items below are reminders Sonnet's dispatch package must capture
verbatim (these aren't decisions, they're harness):

1. Dispatch package path: `/Users/Admin/.claude/jobs/<job-id>/dispatch_p4_a_m1.md`
2. Sonnet must paste real grep outputs (per multi-agent workflow):
   - `grep -c "contention_model" /Users/Admin/nano-kvrouter/src/nano_kvrouter/config.py` → ≥ 2
   - `grep -c "NoopTransferModel\|PerNodeLaneTransferModel" /Users/Admin/nano-kvrouter/src/nano_kvrouter/cli.py` → ≥ 2
   - `grep -c "if.*contention_model\|if.*transfer_model is None" /Users/Admin/nano-kvrouter/src/nano_kvrouter/cli.py` → **exactly 1** (only the `_run_one` factory)
   - `grep -rn "gpu_to_gpu" /Users/Admin/nano-kvrouter/src/nano_kvrouter/scheduler/` → still 5 hits, none changed
   - `cd /Users/Admin/nano-kvrouter && git diff configs/default.yaml configs/heavy.yaml configs/hicache.yaml configs/pd_split.yaml configs/sensitivity.yaml configs/trace_mooncake.yaml configs/trace_burstgpt.yaml` → empty
3. Sonnet must run all 6 sweeps + sensitivity + prefix-sensitivity
   and paste tables for diff against §3 above.
4. Sonnet **must not** execute any git write operations
   (`add`/`commit`/`push`/`stash`) — Opus dispatcher commits after
   independent verification (memory `feedback_fix_review_gate`).
5. After Sonnet's NO/YES report, Codex independently reviews;
   M1.fix loop predicted (1-2 rounds based on prior P3-C M2 cadence).

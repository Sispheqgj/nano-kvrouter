# P4-B M0 preflight — compute_est_ttft sees transfer backlog

> Status: **M0 preflight, no code change.** This document captures
> the design surface that M1 (dispatched to Sonnet) will land. Plan
> source: `/Users/Admin/.claude/plans/quietly-routing-bellman.md`
> (v4, Codex YES on 2026-06-15).
>
> Repository baseline at preflight time: `main = c87aa65`,
> `git status` clean, `uv run pytest -q` → 451 passed.

## 1. Exhaustive grep of every impacted call site

Per `feedback_design_plan_selfcheck` rule 2 (signature-change blast
radius), every caller of every function/class whose signature
changes must be enumerated. P4-A M1 paid the cost of incomplete
grep (hidden `_wire_simulator` callers in two test files); P4-B M0
makes sure that does not recur.

### 1.1 `compute_est_ttft(` — 7 hits, every one updates

| file:line | what |
|-----------|------|
| `src/nano_kvrouter/scheduler/base.py:185` | function definition |
| `src/nano_kvrouter/scheduler/round_robin.py:106` | `ttft_ms = compute_est_ttft(...)` |
| `src/nano_kvrouter/scheduler/least_loaded.py:101` | `ttft_ms = compute_est_ttft(...)` |
| `src/nano_kvrouter/scheduler/prefix_greedy.py:121` | `ttft_ms = compute_est_ttft(...)` |
| `src/nano_kvrouter/scheduler/e2_policy.py:118` | `run = compute_est_ttft(...)` (run_cost term) |
| `src/nano_kvrouter/scheduler/e2_policy.py:132` | `ttft_ms = compute_est_ttft(...)` (final estimate) |
| `src/nano_kvrouter/scheduler/conductor.py:129` | `est_ttft = compute_est_ttft(...)` |
| `tests/test_scheduler_base.py:219` | `est = compute_est_ttft(...)` |

M1 must update all 8 (1 definition + 7 callers) to pass `backlog_view=`
and `now=` explicitly. Note `e2_policy.py` calls **twice** in its
`schedule()` — both must be updated.

### 1.2 Scheduler ctor call sites — 30+ hits across src and tests

```
src/nano_kvrouter/cli.py:91          RoundRobinPolicy(model_config=..., bandwidth_config=...)
src/nano_kvrouter/cli.py:93          LeastLoadedPolicy(model_config=..., bandwidth_config=...)
src/nano_kvrouter/cli.py:96          PrefixGreedyPolicy(...)
src/nano_kvrouter/cli.py:102         E2Policy(...)
src/nano_kvrouter/cli.py:110         MooncakeConductor(...)

tests/test_round_robin.py:24         RoundRobinPolicy(model_config=MODEL, bandwidth_config=BW_INF)
tests/test_least_loaded.py:24        LeastLoadedPolicy(model_config=MODEL, bandwidth_config=BW_INF)
tests/test_prefix_greedy.py:25       PrefixGreedyPolicy(min_hit_ratio=..., bandwidth_config=BW_INF)
tests/test_prefix_greedy.py:69,79,81 PrefixGreedyPolicy(...) — multiple bare/short ctor calls
tests/test_e2_policy.py:71,80,90,92,94,182,205,305,339   E2Policy(...) — 9 ctor calls
tests/test_conductor.py:33,78,88,90,92,414               MooncakeConductor(...) — 6 ctor calls

tests/test_cli.py:192,387,435,495,561,616,742,793,864    RoundRobinPolicy(...) — 9 ctor calls
tests/test_scheduler_base.py:162     RoundRobinPolicy()
tests/test_metrics_collector.py:606  RoundRobinPolicy(model_config=mc, bandwidth_config=bw)
```

M1 must add `backlog_view=NoopTransferModel()` to every one of
these ctor calls. (Test fixtures usually only need Noop because they
don't simulate lane contention.) Some bare `Policy()` calls without
existing kwargs will need `Policy(backlog_view=NoopTransferModel())`.

### 1.3 `sched.schedule(` / `policy.schedule(` — 25+ hits

```
src/nano_kvrouter/cli.py:226              sched.schedule(req, prefill_nodes, decode_nodes, cm)
tests/test_round_robin.py:74,85,86,106,124,150,177,194,212,266   policy.schedule(...)
tests/test_least_loaded.py:168,190,212,233,251                   policy.schedule(...)
tests/test_e2_policy.py:184,207,307,340                          policy.schedule(...)
```

(No `tests/test_conductor.py` or `tests/test_prefix_greedy.py` hits
in this grep — they use a different naming convention; M1 must
re-grep these files explicitly.)

M1 must:
- Add `now=engine.now()` to the cli call site (line 226).
- Add `now=0.0` explicitly to every test fixture caller.

### 1.4 `_build_scheduler(` — 8 hits

```
src/nano_kvrouter/cli.py:65          definition
src/nano_kvrouter/cli.py:566         _build_scheduler(scheduler_name, cfg.scheduler.params, cfg.model, cfg.bandwidth)
tests/test_cli.py:64,70,75,82,89,271,704                _build_scheduler(...)
tests/test_trace_generator.py:132                       _build_scheduler(...)
```

M1 must:
- Add `backlog_view` parameter to definition.
- Update cli call site at line 566 to pass `backlog_view=transfer_model`
  (the P4-A `transfer_model` instance constructed at line 584-587).
- Update every test caller to pass `backlog_view=NoopTransferModel()`.
- Be aware that `test_cli.py:64`, `:70`, `:75`, `:82`, `:89` pass
  only 3 positional args `(name, params, ModelConfig())` — they will
  need updating to the new signature too, not just adding a kwarg.

### 1.5 `request_transfer(` hard guard — currently 0 hits in scheduler/

```bash
$ rg -n "request_transfer\(" src/nano_kvrouter/scheduler/
(empty)
```

This is the invariant P4-B must preserve. Schedulers MUST NOT call
`request_transfer` on the backlog view (TransferBacklogView Protocol
doesn't even expose it). M1 dispatch must include this as a hard
constraint, and the grep verification must keep returning 0.

## 2. Final API surfaces

### 2.1 New narrow Protocol in `scheduler/base.py`

```python
from typing import Protocol


class TransferBacklogView(Protocol):
    """Read-only view of transfer lane backlog for estimation use.

    Implementations: NoopTransferModel and PerNodeLaneTransferModel
    (both in simulator/transfer_model.py) satisfy this Protocol
    structurally — no inheritance change needed.

    Schedulers receive TransferBacklogView, NOT TransferModel.
    Intentional: prevents accidental request_transfer() calls inside
    compute_est_ttft, which would reserve a lane during the estimate
    and corrupt the simulation.
    """

    def peek_backlog(self, node_id: str) -> dict[str, float]:
        """Return {"egress": float, "ingress": float} absolute available_at."""
```

### 2.2 `compute_est_ttft` signature change

```python
def compute_est_ttft(
    prefill_node: MockEngineNode,
    decode_node: MockEngineNode,
    request: Request,
    decode_cache_match: CacheLookup,
    *,
    kv_bytes_per_token: int,
    bandwidth_bytes_per_s: float,
    backlog_view: TransferBacklogView,  # NEW, REQUIRED
    now: float,                          # NEW, REQUIRED
) -> float:
    """... existing docstring ...

    P4-B: kv_transfer term now includes lane queue wait when the
    backlog_view's peek_backlog reports non-zero availability beyond
    `now`. Formula:

        service = (kv_bytes / bandwidth_bytes_per_s) * 1000.0
        src_egress_wait = max(0, backlog_view.peek_backlog(
                                  prefill_node.node_id)["egress"] - now)
        dst_ingress_wait = max(0, backlog_view.peek_backlog(
                                   decode_node.node_id)["ingress"] - now)
        queue_wait = max(src_egress_wait, dst_ingress_wait)
        kv_transfer = service + queue_wait

    Under NoopTransferModel.peek_backlog returns {"egress": 0.0,
    "ingress": 0.0} regardless of node_id, so max(0, 0 - now) = 0,
    queue_wait = 0, kv_transfer = service — byte-identical to
    pre-P4-B output for any caller that passes NoopTransferModel.
    """
```

Body changes from the current line 256 (single line `kv_transfer =
(kv_bytes / bandwidth_bytes_per_s) * 1000.0`) to the formula above.

### 2.3 `SchedulingPolicy.schedule()` Protocol

```python
class SchedulingPolicy(Protocol):
    def schedule(
        self,
        request: Request,
        prefill_nodes: list[MockEngineNode],
        decode_nodes: list[MockEngineNode],
        cache_query: CacheQuery,
        *,
        now: float,  # NEW, REQUIRED keyword-only
    ) -> SchedulingDecision: ...
```

Required, no default. cli passes `now=engine.now()`. Tests pass
`now=0.0` explicitly.

### 2.4 5 scheduler `__init__` gains `backlog_view`

```python
class MooncakeConductor:
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        *,
        model_config: ModelConfig | None = None,
        bandwidth_config: BandwidthConfig | None = None,
        backlog_view: TransferBacklogView,  # NEW, REQUIRED keyword-only
    ) -> None:
        ...
        self._backlog_view = backlog_view
```

Same pattern for `RoundRobinPolicy`, `LeastLoadedPolicy`,
`PrefixGreedyPolicy`, `E2Policy`. Each policy's `schedule()` then
passes `backlog_view=self._backlog_view, now=now` into
`compute_est_ttft(...)` calls.

**`backlog_view` is keyword-only** (after `*,` separator) to keep
the positional arg order stable for any callers using positional
(unlikely but worth defending).

### 2.5 cli wiring

```python
# _run_one (already exists at line 584-587):
transfer_model: TransferModel = (
    PerNodeLaneTransferModel()
    if cfg.bandwidth.contention_model == "per_node_lane"
    else NoopTransferModel()
)

# NEW: pass into _build_scheduler:
sched = _build_scheduler(
    scheduler_name, cfg.scheduler.params, cfg.model, cfg.bandwidth,
    backlog_view=transfer_model,
)

# _build_scheduler signature gains backlog_view kwarg-only:
def _build_scheduler(
    name: str,
    params: dict[str, Any],
    model_cfg: ModelConfig,
    bandwidth_cfg: BandwidthConfig | None = None,
    *,
    backlog_view: TransferBacklogView,
) -> SchedulingPolicy:

# Inside _build_scheduler, every Policy() ctor passes:
RoundRobinPolicy(
    model_config=model_cfg,
    bandwidth_config=bw,
    backlog_view=backlog_view,
)
# ... same for the other 4.

# on_arrive at line 226 passes now:
decision = sched.schedule(
    req, prefill_nodes, decode_nodes, cm,
    now=engine.now(),
)
```

### 2.6 MetricsCollector new metric

```python
# In MetricsCollector.__init__:
self._kv_transfer_queued_samples: list[float] = []

# In _on_kv_transfer_complete (currently at collector.py:264-273):
def _on_kv_transfer_complete(self, event, engine):
    transfer_id = event.payload.get("transfer_id")
    if transfer_id is None or transfer_id not in self._seen_transfer_ids:
        return  # stale guard, existing behavior
    cost_ms = event.payload.get("cost_ms")
    if cost_ms is None:
        return
    self._kv_transfer_cost_samples.append(float(cost_ms))
    # NEW: same if-block, same stale guard already passed.
    queued_cost_ms = event.payload.get("queued_cost_ms", 0.0)
    self._kv_transfer_queued_samples.append(float(queued_cost_ms))

# In summary():
"kv_transfer_queued_avg_ms": (
    statistics.mean(self._kv_transfer_queued_samples)
    if self._kv_transfer_queued_samples
    else 0.0  # NOT None — additive surface visible even with 0 transfers
),
```

The `0.0` (not `None`) default is intentional: even on yamls where
no KV transfer happens, the field should be present and zero so
consumers can detect the feature.

### 2.7 Why `max(egress, ingress)`, not `egress + ingress`

The runtime model at `transfer_model.py:95`:

```python
start = max(now, self._egress_available_at.get(src_node_id, 0.0),
                  self._ingress_available_at.get(dst_node_id, 0.0))
```

A transfer occupies BOTH lanes simultaneously starting at `start`.
So the wait equals the LATER of the two lane availabilities. A sum
would double-count. The estimator must mirror this exactly:

```
queue_wait = max(src_egress_wait, dst_ingress_wait)
```

where both `*_wait` are already `max(0, available_at - now)` to
clamp past-backlog to zero (a transfer that should have already
finished doesn't subtract from current cost).

## 3. Regression baselines (frozen at c87aa65)

Identical to P4-A M0 §3.1 + §3.2 + §3.3. M1 must reproduce all of
these **byte-identical** when no caller passes
`PerNodeLaneTransferModel`. (All 7 existing yamls have
`contention_model: "none"`, so cli constructs `NoopTransferModel`,
which makes `peek_backlog` return all zeros, which clamps to zero
queue wait, which leaves `kv_transfer` unchanged.)

### 3.1 Sweep × 6 yaml — cache_hit per scheduler

| yaml | round_robin | least_loaded | prefix_greedy | e2_policy | conductor |
|------|------------:|-------------:|--------------:|----------:|----------:|
| `default.yaml` | 0.502 | 0.496 | 0.560 | 0.558 | 0.560 |
| `heavy.yaml` | 0.540 | 0.525 | 0.582 | 0.564 | 0.528 |
| `hicache.yaml` | 0.040 | 0.026 | 0.218 | 0.197 | 0.218 |
| `pd_split.yaml` | 0.536 | 0.525 | 0.563 | 0.563 | 0.518 |
| `trace_mooncake.yaml` | 0.075 | 0.069 | 0.146 | 0.153 | 0.146 |
| `trace_burstgpt.yaml` | 0.050 | 0.061 | 0.061 | 0.069 | 0.069 |

### 3.2 Sensitivity workflow

`uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml`
→ **13/13 fields PASS**.

### 3.3 Prefix-sensitivity table

`uv run python -m nano_kvrouter.cli prefix-sensitivity --config configs/trace_burstgpt.yaml --scheduler conductor`
→ identical to P4-A M0 §3.3 (sharing range [0.000, 0.140]).

### 3.4 `transfer_contention.yaml` sweep — EXPECTED to drift

This yaml has `contention_model: "per_node_lane"`, so P4-B changes
its scheduler decisions:

- pre-P4-B (current `main` = `c87aa65`): `compute_est_ttft` returns
  constant kv_transfer = service. Conductor's SLO gate sees an
  optimistic estimate; some requests are admitted that the lane
  queue will push past SLO.
- post-P4-B: estimate includes queue wait; Conductor's SLO gate
  rejects more under contention. e2_policy's `run_cost` term also
  shifts, biasing E2 toward less-backlogged nodes.

M1 dispatch must capture **both pre and post numbers** for this yaml
so the Codex reviewer can sanity-check the drift direction
(rejection should go up; cache_hit may shift in either direction).

### 3.5 pytest

Current: **451 passed**. M1 expected to add at minimum:
- 6 in `tests/test_scheduler_base.py` (4 hard gate + 2 Noop compat)
- 2 in `tests/test_metrics_collector.py` (zero-under-noop + stale-guard)
- 1 in `tests/test_cli.py` (paired decomposition fixed-length)

Plus any updates to existing tests for new ctor kwargs / `now`
parameter. Final pytest count not pinned (per plan v4 nit fix).

## 4. Why decode-side compute backlog awareness is OUT OF SCOPE

`compute_est_ttft` already computes `first_decode_tick =
decode_node.estimate_decode_time(decoding_bs)`. A queue-aware
extension would change `decoding_bs` from "current `len(decoding) +
1`" to "predicted batch size at the time this request enters
decode". That conflates two distinct queue models:

1. **Transfer queue** (lane.available_at) — handled by P4-B.
2. **Decode compute queue** (batch already running on decode node
   when this request's transfer finishes).

Modeling (2) requires predicting batch evolution under chunked
prefill + multi-stream decode, which depends on every other
in-flight request's remaining-output. That's a much bigger model
change and warrants its own milestone (P5 candidate, separate
backlog item if user accepts).

P4-B is scoped to (1) only. The estimator still divergence-bounds
for (2); this is documented as a v1 limit in plan §"关键设计
commitment" §6.

## 5. Self-check (re-confirms plan v4)

- [x] §1 lists every `compute_est_ttft(` caller (8 total: 1 def + 7
      callers including 2 in e2_policy.py).
- [x] §1 lists every Policy ctor call site across `src/` and
      `tests/` (30+ across 8 test files).
- [x] §1 includes `request_transfer\(` hard guard with parenthesized
      regex; currently 0 hits in `src/nano_kvrouter/scheduler/`.
- [x] §2.1 defines `TransferBacklogView` as narrow Protocol with ONLY
      `peek_backlog` — no `request_transfer` exposure.
- [x] §2.2/§2.3 lock signatures with required kwargs (no defaults).
- [x] §2.6 metric sampling is INSIDE the existing `transfer_id`
      stale guard (mirror of `cost_ms` discipline).
- [x] §2.6 metric default is `0.0` (not `None`) on empty samples.
- [x] §2.7 documents `max` vs `sum` rationale.
- [x] §3 baselines fully reused from P4-A M0 (no new measurement).
- [x] §3.4 explicit acknowledgement that `transfer_contention.yaml`
      sweep numbers WILL change.
- [x] §4 documents what's deliberately out of scope and why.

## 6. Open items for M1 dispatch package

Reminders Sonnet's dispatch package must capture verbatim:

1. Dispatch package path:
   `/Users/Admin/.claude/jobs/5292c00d/dispatch_p4_b_m1.md`
2. Sonnet must paste real grep outputs from the 8-step checklist in
   plan §"M1 implementation Grep verification checklist".
3. Sonnet must run all 6 sweeps + sensitivity + prefix-sensitivity
   and paste tables for diff against §3 above.
4. **Plus** Sonnet must run `transfer_contention.yaml` sweep BOTH
   under `contention_model: "none"` (paired toggle on the same yaml
   via memory deep-copy, no yaml edit) AND under `"per_node_lane"`,
   and paste both — so Codex can review the drift direction
   (§3.4).
5. Sonnet **must not** execute any git write operations. Opus
   dispatcher commits after independent verification.
6. After Sonnet's NO/YES report, Codex independently reviews;
   M1.fix loop predicted (1-2 rounds based on P3-C M2 / P4-A M1
   cadence).

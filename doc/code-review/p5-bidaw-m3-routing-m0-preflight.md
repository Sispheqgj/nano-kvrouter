# P5-Bidaw M3 — M0 preflight

> No-code preflight for `plan v4` of P5-Bidaw M3 (routing-aware +
> TTFT SLO gate + session affinity). HEAD=8fa12b7. Working tree
> clean. 497 tests pass.
>
> Plan: `.claude/plans/p5-bidaw-m3-routing-plan-v4.md` (Codex YES).

## 1. Blast-radius grep (live caller lists)

### `_build_scheduler(` — 13 sites

- **Production**: `src/nano_kvrouter/cli.py:75` (def), `:855` (call inside `_run_one`).
- **Tests**: `tests/test_cli.py:64, 70, 75, 82, 89, 272, 705`;
  `tests/test_bidaw_cli.py:125, 254, 358`;
  `tests/test_trace_generator.py:133`.
- **M3 risk**: adding `bidaw_controller: BidawAdmissionController |
  None = None` kwarg with default `None` keeps all 11 hidden callers
  passing — verified by inspection: none pass positional arguments
  beyond `backlog_view`, all subsequent params are kwarg-only.

### `_wire_simulator(` — 18 sites

- **Production**: `src/nano_kvrouter/cli.py:162` (def), `:883` (call).
- **Tests**: `tests/test_cli.py:195, 279, 389, 437, 497, 563, 619,
  706, 744, 795, 867`;
  `tests/test_metrics_collector.py:650, 657`;
  `tests/test_bidaw_cli.py:127, 255, 359`;
  `tests/test_trace_generator.py:136`.
- **M3 risk**: `bidaw_mode=True` direct callers at
  `tests/test_bidaw_cli.py:127, 255, 359` don't pass
  `bidaw_controller`. Plan v4 mandates `_wire_simulator` fallback
  to construct a default `BidawAdmissionController` when
  `bidaw_mode=True and bidaw_controller is None`. Tests stay
  unchanged.

### `BidawAdmissionController(` — 2 sites

- `src/nano_kvrouter/cli.py:580` (inside `_wire_bidaw_branch`).
  M3 removes this construction; controller becomes a passed-in
  argument from `_run_one`.
- `tests/test_bidaw_controller.py:28` (test helper). M3 extends the
  constructor with optional kwargs (`model_config`, `bandwidth_config`,
  `affinity_enabled`), all with defaults; this helper continues to
  pass only positional `node_ids`.

### `SchedulingDecision(` — 18 sites across 5 schedulers + 3 tests

- Each of the 5 existing schedulers constructs `SchedulingDecision`
  with 5 positional/kwarg fields (`prefill_node, decode_node,
  estimated_ttft_ms, estimated_tbt_ms, reject_reason`). Plan v4 adds
  `routing_score: float | None = None` and `affinity_hit: bool =
  False` with defaults → all 18 construction sites remain valid.
- Bidaw constructs at `bidaw.py:121, 184` — M3 will populate the
  two new fields when relevant flags are on.
- Test sites: `tests/test_scheduler_base.py:68, 79, 91` — also
  positional/kwarg without the new fields; defaults preserve.

## 2. CLI `-p` override audit

```bash
$ uv run python -m nano_kvrouter.cli sweep --help
usage: nano-kvrouter sweep [-h] --config CONFIG [--output OUTPUT]
```

**No `-p` flag exists.** The plan-v4 ship-gate sweeps must use
**temporary or named yaml variants** rather than CLI overrides.

M3 dispatch & post-dispatch verification will use:

- **Named yamls landing on disk** (chosen because flag-toggle
  variants are useful for future regression):
  - `configs/bidaw-affinity.yaml` (NEW — A3 only, M2 off) — A3 ship gate
  - `configs/bidaw-m3-stress.yaml` (NEW — all 3 M3 flags on) — A2 ship gate
- **Temporary yamls in `$CLAUDE_JOB_DIR`** for the A1-only
  grid-search variants (5 weight combinations on `bidaw.yaml`).
  Paths recorded in M3 dispatch deliverable.

## 3. Locked baselines (11 yamls, M2 HEAD=8fa12b7)

Sweep JSONs saved to `$CLAUDE_JOB_DIR/m3-baselines/` (job-dir scoped;
re-run via the commands at the bottom of this doc).

**Bidaw-family per-scheduler key metrics** (M3 flags default off →
post-M3 must reproduce these exactly):

### `configs/bidaw.yaml`

| scheduler | cache_hit | ttft_p50 ms | ttft_p99 ms | e2e_avg ms | rejection_rate | completed |
|---|---:|---:|---:|---:|---:|---:|
| round_robin   | 0.589 | 12.15 | 31.66 | 383.93 | 0.000 | 288 |
| least_loaded  | 0.693 | 11.05 | 31.67 | 385.68 | 0.000 | 288 |
| prefix_greedy | 0.823 | 11.67 | 30.72 | 412.59 | 0.000 | 288 |
| e2_policy     | 0.723 | 12.33 | 32.89 | 418.62 | 0.000 | 288 |
| conductor     | 0.823 | 11.67 | 30.72 | 412.59 | 0.000 | 288 |
| **bidaw**     | **0.693** | **11.46** | **32.04** | **385.98** | **0.000** | **288** |

**A1 ship-gate target**: bidaw cache_hit must reach **≥ 0.773**
(within 0.05 of conductor's 0.823).

### `configs/bidaw-interactive.yaml`

All 6 schedulers identical (workload only 6 requests):
cache_hit=0.160, ttft_p50=6.78, ttft_p99=10.58, e2e_avg=192.91,
reject=0, completed=6. Demo fixture only; not used as M3 ship-gate
target. M3 must reproduce these numbers exactly (regression).

### `configs/bidaw-stress.yaml`

| scheduler | cache_hit | ttft_p50 ms | ttft_p99 ms | e2e_avg ms | rejection_rate | completed |
|---|---:|---:|---:|---:|---:|---:|
| 5 non-bidaw (identical rows) | 0.391 | 32.59 | 45.25 | 202.71 | 0.252 | 77 |
| **bidaw** | **0.273** | **139.25** | **445.46** | **341.14** | **0.136** | **89** |

**Drift note**: bidaw row is already very different from the 5
others on this config (bidaw pays real disk-load wait via M1's
KV_LOAD events while the other 5 "magic" through cache_load_ms
estimates without paying it). This pre-existing M1 drift is **not**
a M3 problem; M3 must reproduce the bidaw row above exactly with
flags off. The 5 non-bidaw rows must remain byte-identical.

`bidaw-m3-stress.yaml` (new in M3) will be tuned to **reduce
`slo.ttft_target_ms`** below current bidaw `ttft_p99=445.46`,
forcing A2's storage-aware gate to reject some requests with
`reason="ttft_slo_exceeded"`. Target: `ttft_slo_rejections > 0`.

### Non-bidaw 8 yamls

Full per-scheduler tables saved to JSON. Summary spot-checks:

- `default.yaml`: 5 schedulers (no bidaw row), cache_hit
  conductor=0.560 (matches P4-B baseline).
- `hicache.yaml`: existed pre-Bidaw; M3 flags-off must reproduce.
- `transfer_contention.yaml`: P4-A locked; verify no drift.

## 4. Grid-search responsibilities (post-dispatch verification)

A1's 4-weight tuple `(α matched_blocks, β load, γ preparing, δ
in_flight)` cannot be tuned in M0 because the code doesn't exist
yet. M3 dispatch package mandates that post-implementation
verification produces a grid sweep across these 5 combinations on
`bidaw.yaml` with A1-only flag on:

| Combo | (α, β, γ, δ) | Intent |
|---|---|---|
| G1 | (0.0, 1, 1, 2) | I/O-only baseline (no cache term) |
| G2 | (0.5, 1, 1, 2) | Light cache pull |
| G3 | (1.0, 1, 1, 2) | Balanced (placeholder default) |
| G4 | (2.0, 1, 1, 2) | Cache-heavy |
| G5 | (1.0, 2, 1, 2) | Load-heavy with cache |

**Selection criterion**: pick combo with **highest cache_hit**
satisfying `ttft_p99 ≤ 1.10 × baseline ttft_p99 (=32.04 × 1.10
≈ 35.24 ms)`. If no combo meets the criterion → reopen A1.

Each combo: produce a temporary yaml in `$CLAUDE_JOB_DIR/` cloning
`bidaw.yaml` with the 4 yaml fields set. Sweep + record key metrics
in M3 dispatch deliverable.

## 5. A2 stress validation responsibility

Design `configs/bidaw-m3-stress.yaml` cloning `bidaw-stress.yaml`
with:

- `slo.ttft_target_ms: 100.0` (vs current 5000.0) — below current
  bidaw `ttft_p50=139.25`, so a sizable fraction will trip A2.
- Keep `workload.request_rate: 100.0`, `prefix_sharing_ratio: 0.95`,
  `cpu_to_disk: 1.0e7` unchanged.
- `scheduler.params: { enable_routing_aware: true,
  enable_ttft_slo_gate: true, enable_session_affinity: true }`.

Acceptance for A2 ship gate:
- `ttft_slo_rejections > 0` on bidaw row
- Total `rejection_rate ≤ 5-non-bidaw rejection_rate (0.252)` —
  ensures A2 is not over-aggressive

If the above slo=100ms yields too aggressive rejection (e.g.
`rejection_rate > 0.5`), raise slo to 200ms in dispatch tuning.
Document the chosen value.

## 6. A3 affinity baseline responsibility

Design `configs/bidaw-affinity.yaml`:
- Multi-round session workload via TraceGenerator
  `prefix_mode: session_history` (M2 feature)
- ≥ 20 distinct sessions, each with ≥ 3 rounds
- Decode pool size ≥ 4 (so affinity has multiple choices)
- `node.cpu_blocks` ≥ small (some pressure to test fallback)
- `scheduler.params: { enable_session_affinity: true,
  enable_answer_eviction: false }` — keep M2 off to isolate A3

Acceptance for A3 ship gate:
`bidaw_session_affinity_hits / completed_requests ≥ 0.4`

Use the existing `tests/fixtures/interactive_conversation.jsonl` as
a template; expand to ≥ 60 requests across ≥ 20 sessions.

## 7. Pre-existing formula mismatch — RESOLVED

**Original problem** (M1 era): cli.py KV_LOAD service formula only
charged the Disk→CPU hop, missing the CPU→GPU leg that
`cache_manager.transfer_cost_ms` (and the bidaw double-charging
guard) already counted.

**Resolution**: fixed post-M3 in a dedicated `fix(bidaw)` commit:
- `cli.py:675` `load_service_ms` now uses
  `block_bytes * (1/cpu_to_disk + 1/gpu_to_cpu) * 1000.0` (two-hop)
- `simulator/bidaw_controller.py:287` `peek_projected_preparing_wait_ms`
  per-block service updated to two-hop to stay consistent with the
  event path (plan v4 invariant: A2's projected wait must match
  realized wait)
- `tests/test_bidaw_slo_gate.py:126` formula assertion updated

**Post-fix baseline drift** (verified empirically):
- `bidaw.yaml` bidaw row: ttft_p50 +0.03ms, e2e_avg +0.04ms
  (cache_hit unchanged, no rejections)
- `bidaw-interactive.yaml`, `bidaw-stress.yaml` bidaw rows: no
  measurable change at the reported precision (gpu_to_cpu=3.2e10
  is 3200× faster than cpu_to_disk=1e7; second hop contributes
  ~0.03% to service time, smaller than the rounding floor of these
  configs)
- M3 ship gates re-verified post-fix: A2 ttft_slo_rejections=6
  rate=0.194, A3 hits=40/60=0.667; both still pass.

## 8. Re-lock baseline commands (reproducible)

```bash
mkdir -p "$CLAUDE_JOB_DIR/m3-baselines"
for c in default heavy hicache pd_split trace_burstgpt trace_mooncake \
         transfer_contention bidaw bidaw-interactive bidaw-stress; do
  uv run python -m nano_kvrouter.cli sweep \
    --config configs/$c.yaml \
    --output "$CLAUDE_JOB_DIR/m3-baselines/$c.json"
done
uv run python -m nano_kvrouter.cli sensitivity \
  --config configs/sensitivity.yaml \
  --output "$CLAUDE_JOB_DIR/m3-baselines/sensitivity.json"
uv run pytest -q
```

Expected: 497 tests pass; sweep JSONs match this doc's tables.

## 9. M0 sign-off checklist

- [x] Blast radius greped; all callers enumerated (§1)
- [x] CLI `-p` audit done; M3 will use named/temp yamls (§2)
- [x] 11 baselines locked, bidaw-family key metrics tabulated (§3)
- [x] Grid-search procedure specified for dispatch (§4)
- [x] A2 stress yaml design specified (§5)
- [x] A3 affinity yaml design specified (§6)
- [x] cli.py:641 formula mismatch disclosed as separate backlog (§7)
- [x] Re-lock commands captured for verification (§8)
- [x] 497 baseline tests pass

M3 dispatch package may proceed.

---

## Post-dispatch ratification (added after Codex review of M3 implementation)

Two Important spec drifts surfaced in Codex review of the M3
implementation. Both were ratified as design improvements over
plan v4 (rather than reverted), with rationale:

### Ratified drift D1 — routing-weight units

- **Plan v4**: `cost = β·load + γ·preparing_depth + δ·in_flight_count − α·matched_blocks`
- **Implemented**: `cost = β·load + γ·preparing_disk_blocks + δ·in_flight_disk_blocks − α·matched_blocks`
- **Rationale**: block-weighted is the actual I/O backlog measure.
  Queue of one 50-block request vs five 1-block requests is rated
  identically under `preparing_depth=2` (wrong) but 50 vs 5 under
  `preparing_disk_blocks` (right). Single-slot makes
  `in_flight_count ∈ {0,1}` low-information; `in_flight_disk_blocks`
  reflects remaining service.
- **API impact**: `BidawControllerView.peek_preparing_disk_blocks` /
  `peek_in_flight_disk_blocks` replace the original count methods.
- **Yaml params**: `routing_weight_preparing` / `routing_weight_in_flight`
  names unchanged; semantics now block-weighted (documented in
  scheduler docstring).
- **Empirical**: A1 grid G3 produced `bidaw cache_hit = 0.82335`,
  matching conductor exactly on `bidaw.yaml`.

### Ratified drift D2 — affinity overload threshold

- **Plan v4**: `threshold = max(abs_floor, factor · avg_load)`
- **Implemented**: `threshold = max(factor · min_load, min_load + abs_floor)`
- **Rationale**: anchoring on `min_load` (best alternative) directly
  answers "is there a meaningfully better node?" rather than the
  weaker "is pinned above average?". `factor·min_load` is the
  proportional margin at high load; `min_load + abs_floor` is the
  absolute margin at low load — hybrid avoids oscillation in both
  regimes.
- **Yaml params**: `affinity_overload_factor` / `affinity_overload_abs_floor`
  names unchanged; both still tunable; semantics documented in
  `_affinity_overloaded` docstring.
- **Empirical**: A3 on `bidaw-affinity.yaml` produced
  `bidaw_session_affinity_hits = 40 / completed = 60 = 0.667 ≥ 0.4`
  ship gate.

### Ratification implications for documentation

- README / DESIGN §13 — Bidaw M3 subsection should describe routing
  formula with the *block-based* form and the *min-anchored*
  affinity threshold.
- `doc/bidaw-deliverable.md` — M3 section: mention both ratified
  variants under "simulator design decisions diverging from plan v4
  (improvements)".
- Plan v4 in `.claude/plans/` — annotated as audit-trail; the
  implemented variants are the canonical semantics going forward.


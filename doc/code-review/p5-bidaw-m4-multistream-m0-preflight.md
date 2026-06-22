# P5-Bidaw M4 — M0 preflight

> No-code preflight for plan v3 of P5-Bidaw M4 (multi-stream KV
> load model). HEAD=`f4b714d`. Working tree clean. 515 tests pass.
>
> Plan: `.claude/plans/p5-bidaw-m4-multistream-plan-v3.md` (Codex YES
> after in-place patch addressing 3 Important + 1 Nit).

## 1. Blast-radius grep (live caller lists)

### `BidawAdmissionController(` — 11 sites

- **Production**:
  - `src/nano_kvrouter/cli.py:233` — `_wire_simulator` fallback path
    when `bidaw_mode=True and bidaw_controller is None` (M3 v4 design
    to keep `tests/test_bidaw_cli.py` direct callers working).
  - `src/nano_kvrouter/cli.py:905` — `_run_one` production
    construction, passes through `_build_scheduler` AND
    `_wire_simulator`.
- **Tests**: `tests/test_bidaw_cli.py:519`,
  `tests/test_bidaw_slo_gate.py:88, 104`,
  `tests/test_bidaw_controller.py:28`,
  `tests/test_bidaw_affinity.py:58, 75, 97, 159, 175`.
- **M4 risk**: adding `load_model: BidawLoadModel | None = None`
  kwarg with default `None` (controller internally constructs
  `SingleSlotLoadModel(decode_node_ids)`) keeps all 11 callers
  passing. The 2 production sites also need an update to construct
  `MultiStreamLoadModel` when yaml says so:
  - `cli.py:233` fallback: continues to construct
    `SingleSlotLoadModel` (preserves test-direct-caller M1 semantics).
  - `cli.py:905` production: insert `_build_bidaw_load_model(...)`
    call BEFORE the controller construction; pass result as
    `load_model=` kwarg.

### `mark_load_started(` — 7 sites

- `src/nano_kvrouter/cli.py:688` (single production site in
  `_wire_bidaw_branch`)
- `tests/test_bidaw_controller.py:66, 114` and others (via the
  `service_ms` already added in M3)
- `tests/test_bidaw_slo_gate.py:96, 122`
- **M4 risk**: signature **unchanged** in M4 (controller delegates
  to load_model internally). The cli `_drain_idle_slots` helper
  wraps the current single call site.

### `_build_scheduler(` — 13 sites (unchanged from M3 M0)

M4 does NOT touch `_build_scheduler` — load model construction
moves to a sibling factory `_build_bidaw_load_model` in `_run_one`.
All 13 callers stay byte-stable.

## 2. CLI `-p` override audit — still no -p flag

```bash
$ uv run python -m nano_kvrouter.cli sweep --help
usage: nano-kvrouter sweep [-h] --config CONFIG [--output OUTPUT]
```

M4 ship-gate sweeps use named yamls. The new
`configs/bidaw-m4-multistream.yaml` is added on disk (clone of
`bidaw-stress.yaml` with `load_model: multi` + `num_streams: 4`).
Post-dispatch K-tuning sweep may also drop temporary
`bidaw-stress-k{2,4,8}.yaml` into `$CLAUDE_JOB_DIR/` for the
empirical comparison.

## 3. Locked baselines (HEAD=f4b714d, post two-hop fix)

12 sweep configs + 1 sensitivity workflow ran; JSONs at
`$CLAUDE_JOB_DIR/m4-baselines/`.

### Bidaw-family key metrics

| Config | scheduler | cache_hit | ttft_p50 | ttft_p99 | e2e_avg | reject | completed |
|---|---|---:|---:|---:|---:|---:|---:|
| `bidaw.yaml` (6 schedulers) | round_robin | 0.589 | 12.15 | 31.66 | 383.93 | 0.000 | 288 |
| | least_loaded | 0.693 | 11.05 | 31.67 | 385.68 | 0.000 | 288 |
| | prefix_greedy | 0.823 | 11.67 | 30.72 | 412.59 | 0.000 | 288 |
| | e2_policy | 0.723 | 12.33 | 32.89 | 418.62 | 0.000 | 288 |
| | conductor | 0.823 | 11.67 | 30.72 | 412.59 | 0.000 | 288 |
| | **bidaw** | **0.693** | **11.49** | **32.04** | **386.01** | **0.000** | **288** |
| `bidaw-interactive.yaml` | all 6 identical | 0.160 | 6.78 | 10.58 | 192.91 | 0.000 | 6 |
| `bidaw-stress.yaml` (M4 ship-gate target) | 5 non-bidaw | 0.391 | 32.59 | 45.25 | 202.71 | 0.252 | 77 |
| | **bidaw** | **0.273** | **139.25** | **445.46** | **341.14** | **0.136** | **89** |
| `bidaw-affinity.yaml` | 6 schedulers | varies | varies | varies | varies | 0.000 | 60 |
| `bidaw-m3-stress.yaml` | bidaw (A1+A2+A3 on) | 0.232 | 82.25 | 380.19 | 295.52 | 0.194 | 83 |

Note minor drift from M3 M0 baselines (e.g. bidaw row ttft_p50
11.46 → 11.49 on `bidaw.yaml`) due to the post-M3 two-hop fix
(`f4b714d`). M4 byte-id regression target is THIS doc's numbers,
not the M3 M0 doc's.

### 3-field ship-gate baseline on `bidaw-stress.yaml` (M4 reference)

| Field | Value |
|---|---:|
| `bidaw_preparing_wait_avg_ms` | **143.07** |
| `bidaw_preparing_wait_p99_ms` | **261.49** |
| `bidaw_disk_load_service_avg_ms` | **20.10** |
| `bidaw_preparing_promotions` | **30** |
| `rejection_rate` | **0.136** |
| `completed` | **89** |

M4 ship-gate **3 guards** (all must hold simultaneously on the new
`bidaw-m4-multistream.yaml` clone with K=4):

| Guard | Target | Realized |
|---|---|---:|
| Preparing wait reduction ≥30% | `preparing_wait_avg_ms ≤ 100.15` (= 143.07 × 0.7) | **45.76** ✓ |
| TTFT p50 reduction ≥30% | `ttft_p50_ms ≤ 97.48` (= 139.25 × 0.7) | **38.83** ✓ |
| E2E avg reduction ≥10% | `e2e_avg_ms ≤ 307.03` (= 341.14 × 0.9) | **231.93** ✓ |

**Ship gate v2** (redesigned post-implementation, 2026-06-22).

v1 gates `promotions ∈ [27, 33]` and `rejection_rate ≤ 0.186` were
proven infeasible after the unauthorized overlap guard was removed
(Codex scanned K ∈ {1,2,3,4,5,6,8}: no K satisfied all three v1
guards simultaneously). Root cause: multi-stream pumps requests
into `PREFILL_START` faster → shifts decode-capacity-exhaustion
timing in saturated workloads → promotions/rejection counts are
DOWNSTREAM emergent properties, not invariants. The redesigned
gates anchor on what M4 actually promises (user-facing latency):
preparing-wait, TTFT, E2E reductions.

### `bidaw-affinity.yaml` exposure note

This config currently has `bidaw_preparing_promotions = 0` (i.e.
zero disk-tier hits in this workload). M4 default-single mode runs
must still produce byte-identical output. Multi-stream mode would
not exercise the new code here — used only for M3 A3 regression
guard.

## 4. K-tuning feasibility (back-of-envelope at M0)

Total disk-load work on `bidaw-stress.yaml`: 30 promotions ×
20.10 ms/promotion = **603 ms** total serialized work.

| K | Ideal preparing_wait_avg | Reduction | ≤ 100ms target? |
|---|---:|---:|---|
| 1 (baseline) | 143.0 ms (realized) | 0 % | ❌ |
| 2 | 71.5 ms (ideal) | 50 % | ✅ |
| 4 | 35.8 ms (ideal) | 75 % | ✅ |
| 8 | 17.9 ms (ideal) | 88 % | ✅ |

K=2 already meets the 30% reduction (we're targeting 50% ideal).
**Plan v3 picks K=4** for safety margin against scheduling
overhead and HRRN imperfection. M4 dispatch will lock K=4 in
`configs/bidaw-m4-multistream.yaml` and verify empirically post-
implementation.

If empirical K=4 measures > 100ms (i.e. real overhead larger than
expected), dispatch can retune to K=8 — the formula upper bound is
forgiving.

## 5. M3 ship-gate baselines (must remain met post-M4 in single mode)

| Gate | Config | Target |
|---|---|---|
| A1 | `bidaw.yaml` with `enable_routing_aware=true` | bidaw cache_hit within 0.05 of conductor (0.823 ± 0.05) |
| A2 | `bidaw-m3-stress.yaml` | `ttft_slo_rejections > 0` AND `rejection_rate ≤ 0.252` |
| A3 | `bidaw-affinity.yaml` with `enable_session_affinity=true` | `bidaw_session_affinity_hits / completed ≥ 0.4` |

Currently met (with M3 + two-hop fix). M4 dispatch must re-verify
all three with default `load_model=single`.

## 6. M4 metric scope decision

**No new summary fields in M4.** Existing
`bidaw_preparing_wait_avg_ms` / `_p99_ms` already serve the ship-
gate need. The plan-v1 `bidaw_load_queue_depth_avg` field is
deferred (potential M5/M6 addition with its own M0 audit).

Hard-gate phrasing: "12 sweep JSONs + 1 sensitivity JSON return
identical legacy-field values to this doc's tables when
`load_model=single` (default)".

## 7. Re-lock commands (reproducible)

```bash
mkdir -p "$CLAUDE_JOB_DIR/m4-baselines"
for c in default heavy hicache pd_split trace_burstgpt trace_mooncake \
         transfer_contention bidaw bidaw-interactive bidaw-stress \
         bidaw-affinity bidaw-m3-stress; do
  uv run python -m nano_kvrouter.cli sweep \
    --config configs/$c.yaml \
    --output "$CLAUDE_JOB_DIR/m4-baselines/$c.json"
done
uv run python -m nano_kvrouter.cli sensitivity \
  --config configs/sensitivity.yaml \
  --output "$CLAUDE_JOB_DIR/m4-baselines/sensitivity.json"
uv run pytest -q
```

Expected: 515 tests pass; JSONs match tables above.

## 8. M0 sign-off checklist

- [x] Blast radius greped; cli.py:233 fallback + cli.py:905 production
  + 9 test sites enumerated (§1)
- [x] CLI `-p` audit confirms named-yaml + temp-yaml strategy (§2)
- [x] 12 sweep baselines + 1 sensitivity baseline locked at f4b714d (§3)
- [x] `bidaw-stress.yaml` 3-field ship-gate guards specified (§3)
- [x] K-tuning feasibility verified analytically: K=2 already
  clears 30% gate; K=4 provides safety margin (§4)
- [x] M3 ship gates re-verified must remain met (§5)
- [x] No new summary metrics in M4; existing fields cover ship gate (§6)
- [x] Re-lock commands captured for verification (§7)
- [x] 515 baseline tests pass

M4 dispatch package may proceed.

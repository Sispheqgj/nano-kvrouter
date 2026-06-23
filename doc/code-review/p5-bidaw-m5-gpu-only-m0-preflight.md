# P5-Bidaw M5 — M0 preflight

> No-code preflight for plan v1 (in-place patched after Codex YES
> with revisions) of P5-Bidaw M5 (GPU-only performance mode).
> HEAD=`0640c8e`. Working tree clean. 531 tests pass.
>
> Plan: `.claude/plans/p5-bidaw-m5-gpu-only-plan-v1.md` (Codex YES
> with 4 Important + 2 Nit revisions, all applied in-place).

## 1. Blast-radius grep (live caller lists)

### `controller.on_arrive(` — 15 sites

- **Production**: `src/nano_kvrouter/cli.py:784` (one site, inside
  `_wire_bidaw_branch.on_arrive_bidaw`).
- **Tests**: 14 sites across `test_bidaw_multistream.py`,
  `test_bidaw_slo_gate.py`, `test_bidaw_controller.py` — all use
  positional `matched_disk_blocks=N` form.
- **M5 risk**: plan v1 extends with `*, matched_cpu_blocks: int = 0`
  kwarg-only. Default 0 preserves all 14 existing call sites
  unmodified. The cli.py:784 production site is updated only when
  `performance_layer="gpu_only"`.

### `mark_load_started(` — 16 sites

- **Production**: `src/nano_kvrouter/cli.py:722` (one site).
- **Definition**: `bidaw_controller.py:157`.
- **Tests**: 14 sites across `test_bidaw_multistream.py`,
  `test_bidaw_slo_gate.py`, `test_bidaw_controller.py` — all use
  current `(node_id, req_id, now_ms, service_ms)` signature
  (already extended in M3).
- **M5 design simplification** (recorded here, not yet in plan v1):
  `mark_load_started` signature **does not need extension**. The
  controller already has a `PreparingEntry` for the request in its
  `_preparing` queue (which will include the new `cpu_blocks`
  field). On `mark_load_started`, the controller looks up the entry
  and forwards both `disk_blocks` AND `cpu_blocks` to
  `load_model.start_load(...)` internally. The cli wiring does not
  need to know about cpu_blocks here — it just needs to compute
  the right `load_service_ms` (which it can do from the cache
  lookup before calling mark_load_started).
- **Test impact**: zero — `mark_load_started` signature unchanged,
  all 14 test callers continue to work.

### `load_model.start_load(` — 10 sites

- **Production**: `src/nano_kvrouter/simulator/bidaw_controller.py:178`
  (one site, inside `mark_load_started`).
- **Tests**: 9 sites in `tests/test_bidaw_load_model.py`.
- **M5 risk**: plan v1 extends with `*, cpu_blocks: int = 0`
  kwarg-only. Default 0 preserves M4 byte-id. All 9 test callers
  continue to work.

### `peek_projected_preparing_wait_ms(` — 7 sites

- **Definitions**: `scheduler/bidaw_view.py:25` (Protocol),
  `simulator/bidaw_controller.py:254` (concrete impl).
- **Production caller**: `scheduler/bidaw.py:303` (A2 SLO gate).
- **Tests**: 4 sites total — `test_bidaw_multistream.py:170`,
  `test_bidaw_slo_gate.py:99/134/181`, plus 2 stubs at
  `test_bidaw_slo_gate.py:43` and `test_bidaw_routing.py:48`.
- **M5 risk**: plan v1 extends signature with `*, my_cpu_blocks:
  int = 0`. Both Protocol and impl updated in lockstep. Test
  stubs (`_ProjectedWaitView`) also need the kwarg. All current
  positional callers preserved by default 0.

### `BidawPolicy(` — 11 sites

- **Production**: `src/nano_kvrouter/cli.py:148`.
- **Tests**: 10 sites in `test_bidaw_multistream.py`,
  `test_bidaw_affinity.py`, `test_bidaw_scheduler.py`.
- **M5 risk**: plan v1 adds `*, performance_layer: str = "gpu_and_cpu"`
  kwarg with validation. Default preserves M3/M4 behavior. All
  10 test callers continue to work.

## 2. `mark_load_started` simplification — record as plan addendum

Plan v1 §"Key API contracts" implied a `mark_load_started` extension.
M0 grep shows **the signature does not need to change**. Controller
looks up cpu_blocks from `PreparingEntry` (which gains the field in
M5), forwards to `load_model.start_load(..., cpu_blocks=...)`
internally. This is cleaner than the plan implied.

Dispatch package will record this as a hard constraint: M5
implementation MUST NOT change `mark_load_started` signature; the
cpu_blocks plumbing happens inside the controller body.

## 3. CLI `-p` override audit — still no -p flag

Same conclusion as M3/M4 M0: M5 ship-gate sweeps use a named yaml
file (`configs/bidaw-m5-gpu-only.yaml`). No CLI override needed.

## 4. Locked baselines (HEAD=`0640c8e`)

13 sweep configs (8 non-bidaw + 5 bidaw-family + bidaw-m4-multistream =
**14 yamls actually run via sweep**) + 1 sensitivity workflow. JSONs
at `$CLAUDE_JOB_DIR/m5-baselines/`.

### Bidaw-family key metrics (HEAD=0640c8e, post-M4)

| Config | scheduler | cache_hit | ttft_p50 | ttft_p99 | e2e_avg | reject | completed |
|---|---|---:|---:|---:|---:|---:|---:|
| `bidaw.yaml` | bidaw | 0.693 | 11.49 | 32.04 | 386.01 | 0.000 | 288 |
| `bidaw-interactive.yaml` | bidaw | 0.160 | 6.78 | 10.58 | 192.91 | 0.000 | 6 |
| `bidaw-stress.yaml` | **bidaw (K=1)** | **0.273** | **139.25** | **445.46** | **341.14** | **0.136** | **89** |
| `bidaw-affinity.yaml` | bidaw | varies | varies | varies | varies | 0.000 | 60 |
| `bidaw-m3-stress.yaml` | bidaw (all 3 M3 on) | 0.232 | 82.25 | 380.19 | 295.52 | 0.194 | 83 |
| `bidaw-m4-multistream.yaml` | **bidaw (K=4)** | 0.273 | 38.83 | 260.22 | 231.93 | 0.243 | 78 |

### Ship-gate reference baseline (M5 will compare against `bidaw-stress.yaml`)

| Field | Value |
|---|---:|
| `bidaw_preparing_wait_avg_ms` | **143.07** |
| `bidaw_preparing_wait_p99_ms` | **261.49** |
| `bidaw_disk_load_service_avg_ms` | **20.10** |
| `bidaw_preparing_promotions` | **30** (= count of disk-hit requests) |
| `rejection_rate` | **0.136** |
| `completed` | **89** |
| `cache_hit_ratio` | **0.273** (means ~73% of decode-side blocks ARE cached at some tier) |

## 5. M5 numerical feasibility (back-of-envelope at M0)

Per-block service times on `bidaw-stress.yaml` parameters
(`block_size=16, kv_bytes_per_token=4096, cpu_to_disk=1e7, gpu_to_cpu=3.2e10`):

| Tier hit | Per-block service formula | Per-block service ms |
|---|---|---:|
| Disk | `block_bytes × (1/cpu_to_disk + 1/gpu_to_cpu) × 1000` | **6.556 ms** |
| CPU | `block_bytes × (1/gpu_to_cpu) × 1000` | **0.002 ms (2.048 µs)** |

**Ratio: disk is ~3200× slower than CPU per block.**

### Implications for ship gate design

Plan v1 §"Ship gate" listed `bidaw_preparing_wait_avg_ms > 0` as
a M5 target. With CPU loads ~3200× faster than disk, this gate
becomes trivially satisfied (a single CPU load adds microseconds
to the average). Codex's Nit #2 already noted to add `cpu_blocks`
to event payload for trace clarity; M0 here goes further and
recommends the ship gate language be **promotion-count-centric**,
not wait-time-centric:

**Revised M5 ship gate (M0 recommendation)**:

| Gate | Config | Target |
|---|---|---|
| GPU-only mode enqueues CPU hits | `bidaw-m5-gpu-only.yaml` (K=1) | `bidaw_preparing_promotions` strictly **greater** than `bidaw-stress.yaml` baseline (30) — direct proof CPU-hit requests are now classified as preparing |
| Event-level CPU-only path | `tests/test_bidaw_gpu_only.py` (unit) | Plan v1 §"Event-level hard gate" — verify `KV_LOAD_START` payload for a CPU-only-hit request has `disk_blocks=0`, `cpu_blocks=N`, `load_service_ms = N × block_bytes / gpu_to_cpu × 1000` |
| Cross-product M5 × M4 | `bidaw-m5-gpu-only-k4.yaml` (NEW M0 candidate; clone with `load_model=multi, num_streams=4`) | All-on combo runs deterministically; no slot overflow; reasonable numerical output |

The `bidaw_preparing_wait_avg_ms > 0` target stays but is reframed
as **expected-to-be-met-trivially** (not a meaningful gate). The
preparing_promotions count gate is the meaningful one.

Empirical estimate of M5 effect on bidaw-stress:

- Current K=1 mode: 30 disk-hit promotions out of 89 completed (34%)
- Remaining ~66% (59 requests) are GPU or CPU hits, undistinguished
  in current metric
- With `cpu_blocks=16` capacity + `prefix_sharing_ratio=0.95`, most
  of those 59 likely have substantial CPU prefix hits
- Under M5 gpu_only, plausible promotions count: 30 + most-of-59 →
  ~50-80+ promotions (significantly higher than 30)
- Wait time impact: marginal (CPU loads are µs)

If empirical M5 promotions count is NOT strictly greater than 30,
something is wrong (either workload has no CPU hits at all on
bidaw-stress, or wiring missed the CPU path). Dispatch verifies.

## 6. `bidaw-m5-gpu-only.yaml` design (M0 sketch)

Clone of `bidaw-stress.yaml` (which already has cpu_blocks=16),
change only:

```yaml
scheduler:
  name: bidaw
  params:
    performance_layer: gpu_only   # M5 opt-in
```

All other fields unchanged. The expected effect:

- Same number of arrived requests
- Higher `bidaw_preparing_promotions` (CPU hits now enter preparing)
- Microscopic `bidaw_disk_load_service_avg_ms` shift (CPU loads
  pull the avg toward 0; depends on mix)
- Other fields (completed, rejection_rate, cache_hit_ratio) may shift
  due to changed admission timing — recorded as **observed**, not
  expected-direction (per Nit #1 patch in plan v1)

## 7. M3/M4 ship gates re-verify list (post-M5 implementation)

With `performance_layer="gpu_and_cpu"` (default):

| Gate | Config | Target |
|---|---|---|
| A1 (M3) | `bidaw.yaml` with `enable_routing_aware=true` | cache_hit within 0.05 of conductor 0.823 |
| A2 (M3) | `bidaw-m3-stress.yaml` | `ttft_slo_rejections > 0` AND `rejection_rate ≤ 0.252` |
| A3 (M3) | `bidaw-affinity.yaml` with `enable_session_affinity=true` | `bidaw_session_affinity_hits / completed ≥ 0.4` |
| M4 multi-stream | `bidaw-m4-multistream.yaml` (K=4) | preparing_wait_avg ≤ 100.15 ms; ttft_p50 ≤ 97.48; e2e_avg ≤ 307.03 |

All currently met at HEAD `0640c8e`. M5 dispatch must reproduce
exactly under default flag value.

## 8. Re-lock commands (reproducible)

```bash
mkdir -p "$CLAUDE_JOB_DIR/m5-baselines"
for c in default heavy hicache pd_split trace_burstgpt trace_mooncake \
         transfer_contention bidaw bidaw-interactive bidaw-stress \
         bidaw-affinity bidaw-m3-stress bidaw-m4-multistream; do
  uv run python -m nano_kvrouter.cli sweep \
    --config configs/$c.yaml \
    --output "$CLAUDE_JOB_DIR/m5-baselines/$c.json"
done
uv run python -m nano_kvrouter.cli sensitivity \
  --config configs/sensitivity.yaml \
  --output "$CLAUDE_JOB_DIR/m5-baselines/sensitivity.json"
uv run pytest -q
```

Expected: 531 tests pass; 14 JSONs match this doc's tables.

## 9. M0 sign-off checklist

- [x] Blast radius greped: on_arrive (15), mark_load_started (16),
  start_load (10), peek_projected_preparing_wait_ms (7),
  BidawPolicy (11) (§1)
- [x] `mark_load_started` signature simplification documented (§2)
- [x] CLI `-p` audit (§3)
- [x] 14 baselines locked at f4b714d... no wait, 0640c8e (§4)
- [x] Numerical feasibility analyzed: CPU is 3200× faster than
  disk per block, so ship gate language refined to promotion-count
  focus (§5)
- [x] `bidaw-m5-gpu-only.yaml` design sketched (§6)
- [x] M3/M4 re-verify list (§7)
- [x] Re-lock commands captured (§8)
- [x] 531 baseline tests pass

M5 dispatch package may proceed.

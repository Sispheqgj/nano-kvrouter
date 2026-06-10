# Trace sources verified 2026-06-09 (P3-C M0 preflight)

This file records the verified provenance, schema, license, and integration
constraints for each public trace used by `nano-kvrouter`. Values here are
**facts checked against the upstream repo on 2026-06-09**, not assumptions.

Any future change to upstream trace schema must update this file first,
before any code change.

---

## Mooncake trace (FAST'25 release)

### Source
- **Repo**: https://github.com/kvcache-ai/Mooncake (`main` branch)
- **Directory**: `FAST25-release/traces/`
- **Files**:
  - `conversation_trace.jsonl` — 3.0 MB, 12,031 requests
  - `synthetic_trace.jsonl` — 1.1 MB
  - `toolagent_trace.jsonl` — 4.4 MB
- **Total**: ~8.6 MB combined

### Format
JSONL, one request per line. **Verified first 3 lines of `conversation_trace.jsonl`:**

```json
{"timestamp": 0, "input_length": 6758, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}
{"timestamp": 0, "input_length": 7322, "output_length": 490, "hash_ids": [0, 14, 15, ..., 27]}
{"timestamp": 0, "input_length": 7236, "output_length": 794, "hash_ids": [0, 28, 29, ..., 41]}
```

### Schema (verified)
| Field | Type | Unit / Semantics |
|---|---|---|
| `timestamp` | int | **milliseconds**, relative to start (first record = 0). Confirmed via range 0..3,536,999 ms = 59 minutes ≈ 1 hour (matches paper) |
| `input_length` | int | input tokens |
| `output_length` | int | output tokens |
| `hash_ids` | list[int] | **block-level prefix identifier**, block_size = 512 tokens. Identical leading hash_ids ⇒ identical leading 512×N token prefix |

### Block size: 512 (CRITICAL)
Verified by: `ceil(input_length / 512) == len(hash_ids)` on all 3 sample lines:
- 6758 / 512 = 13.20 → 14 ✓
- 7322 / 512 = 14.30 → 15 ✓
- 7236 / 512 = 14.13 → 15 ✓

**Integration constraint**: `configs/trace_mooncake.yaml` MUST set
`model.block_size: 512`. Other configs (default/heavy/hicache) keep
block_size=16 and are unaffected — trace_mooncake.yaml is a self-contained
scenario faithfully reproducing Mooncake's block granularity.

**Side effect to document**: `prefill_chunk_size` default 512 means
each chunk = 1 block when running this scenario. Chunked prefill behavior
collapses to per-block. Note in `configs/trace_mooncake.yaml` header
comment.

### License
**Apache-2.0** (per `LICENSE-APACHE` in upstream repo).

### Commit strategy
**Commit all 3 trace files directly to `traces/mooncake/`** (8.6 MB total,
well under any reasonable repo threshold; no LFS needed).

No `fetch_traces.sh` needed for Mooncake — bundled in-repo.

### Attribution
Add to `traces/README.md`:
> Mooncake traces (`traces/mooncake/*.jsonl`) are from
> https://github.com/kvcache-ai/Mooncake, released under Apache-2.0 with
> the FAST'25 paper. See upstream for the full preprocessing methodology
> and privacy mechanisms applied to the raw production data.

---

## BurstGPT trace (HPMLL)

### Source
- **Repo**: https://github.com/HPMLL/BurstGPT
- **Release**: `v2.0` (https://github.com/HPMLL/BurstGPT/releases/tag/v2.0)
- **Recommended file**: `BurstGPT_without_fails_3.csv`
  (has Session ID + Elapsed time, with failed requests filtered)
  - Alternate: `BurstGPT_3.csv` (same schema, includes failed requests)

### Format
CSV with header. **Verified first 4 data rows of `BurstGPT_3.csv`:**

```csv
Timestamp,Session ID,Elapsed time,Model,Request tokens,Response tokens,Total tokens,Log Type
19440110.0,1722ac82-0a46-4bf0-aa08-89794e7a2b3f,43,GPT-4,906,446,1352,Conversation log
19440161.0,d5983bf9-4b48-497b-892b-a58995247443,2,ChatGPT,36,29,65,Conversation log
19440192.0,a5c69b35-4fbc-45e9-955e-18715e376d74,8,GPT-4,1779,123,1902,Conversation log
19440254.0,8b74c7d8-1643-4bba-8c69-91cb4548c506,3,ChatGPT,935,178,1113,Conversation log
```

### Schema (verified)
| Field | Type | Unit / Semantics | Map to internal |
|---|---|---|---|
| `Timestamp` | float | **seconds**, NOT zero-based (epoch-like, e.g. 19,440,110) | normalize to `arrival_ms` (×1000, subtract first) |
| `Session ID` | str (UUID) | conversation id; same UUID = same conversation | `session_id` (passthrough) |
| `Elapsed time` | int | seconds, request response time on real system | discarded |
| `Model` | str | "GPT-4" / "ChatGPT" | discarded |
| `Request tokens` | int | input tokens | `input_length` |
| `Response tokens` | int | output tokens | `output_length` |
| `Total tokens` | int | input + output | discarded (derivable) |
| `Log Type` | str | "Conversation log" / "API log" | discarded |

**No `hash_ids` field** — BurstGPT trace has no prefix structure
information. P3-C M2 uses `PrefixSynthesisModel` to synthesize prefix
sharing on top of (timestamp, length).

### Block size
Not bound by trace (no hash_ids). `configs/trace_burstgpt.yaml` can use
any `model.block_size` — keep default 16 for consistency with non-trace
configs.

### Total size
- `BurstGPT_3.csv`: ~5.34M lines, ~188 MB+
- Too large to commit directly.

### License
**CC-BY-4.0** (per upstream repo). Requires attribution.

### Commit strategy
- **Sample only**: `traces/burstgpt/sample.jsonl` ~1000 converted records
  (~70 KB), enough for tests and demo
- **Full trace**: downloaded by `scripts/fetch_traces.sh` into
  `traces/burstgpt/full/`, added to `.gitignore`

### Attribution
Add to `traces/README.md`:
> BurstGPT sample (`traces/burstgpt/sample.jsonl`) is derived from
> https://github.com/HPMLL/BurstGPT v2.0 release, licensed under
> CC-BY-4.0. Original dataset by Wang et al.; see upstream README for
> full citation.

---

## Cross-trace integration notes

### Block size mismatch handling
- `trace_mooncake.yaml`: `model.block_size: 512` (Mooncake-bound)
- `trace_burstgpt.yaml`: `model.block_size: 16` (default)
- Other yaml configs: unchanged

Each scenario is self-contained. We do NOT mix Mooncake hash_ids with
non-512 block_size — that would silently distort prefix matching
semantics.

### timestamp normalization (in converter)
Both converters normalize so first record has `arrival_ms = 0`:
- Mooncake: already 0-based ms, pass through (`arrival_ms = timestamp`)
- BurstGPT: subtract first `Timestamp`, multiply by 1000

### output_length truthfulness
`make_request` (P3-M1 patches) accepts per-request `expected_output_len`.
Both traces have output_length; TraceGenerator passes it through. Decode
pressure now reflects trace truth rather than `config.workload.avg_output_len`.

---

## Plan adjustments resulting from M0

| Original plan assumption | Actual fact | Adjustment |
|---|---|---|
| "small sample commit + fetch script for Mooncake" | Mooncake 8.6 MB total, commit-friendly | **Commit all 3 Mooncake trace files directly**; fetch_traces.sh only needed for BurstGPT |
| "block_size 16 per hash_id" (assumed) | Mooncake block_size = 512 | Document constraint; `trace_mooncake.yaml` pins block_size=512 |
| "BurstGPT timestamp unit unclear" | seconds, not zero-based | Converter does ×1000 + zero-offset |
| "M2 BurstGPT converter skips session_id" | session_id is clean UUID, easy to keep | Converter preserves session_id field for future P3-D use, M2 still ignores it for prefix synth |
| "prefix_mode=hash_ids needs vocab_size guard" | hash_ids in trace exceed any reasonable vocab_size (Mooncake hash_id range is large), but RadixTree only does equality | Document: vocab_size constraint applies only to random suffix tokens, not hash_id-derived ones |

These adjustments are minor and do NOT change the M1/M2 milestone scope.

---

## How to refresh this preflight

If upstream changes (new Mooncake release, BurstGPT v3, etc.), redo
preflight before any related code change. Update both the "Verified
2026-06-09" date in this file's header and the relevant facts.

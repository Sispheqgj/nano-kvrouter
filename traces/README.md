# Trace files for nano-kvrouter

This directory contains workload traces used for trace-driven simulation.
Verified schema, provenance, and integration constraints are documented in
[SOURCES.md](SOURCES.md).

## Mooncake traces (`traces/mooncake/`)

Three JSONL files from the Mooncake FAST'25 release:

| File | Requests | Description |
|---|---|---|
| `conversation_trace.jsonl` | 12,031 | Conversational workload (~59 min) |
| `synthetic_trace.jsonl` | 3,993 | Synthetic workload |
| `toolagent_trace.jsonl` | 23,608 | Tool-agent workload |

Each line is a JSON object:
```json
{
  "timestamp": 0,
  "input_length": 6758,
  "output_length": 500,
  "hash_ids": [0, 1, 2, ...]
}
```

- `timestamp`: milliseconds from trace start (0-based)
- `input_length` / `output_length`: token counts
- `hash_ids`: block-level prefix identifiers, **block_size = 512**

**Integration constraint**: configs using Mooncake traces MUST set
`model.block_size: 512`. See `configs/trace_mooncake.yaml`.

### To run the Mooncake trace simulation

```bash
uv run python -m nano_kvrouter.cli run --config configs/trace_mooncake.yaml
uv run python -m nano_kvrouter.cli sweep --config configs/trace_mooncake.yaml
```

## License

Mooncake traces are released under **Apache License 2.0**. The full
license text is bundled in this repository at
[`LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt). See the upstream
repo https://github.com/kvcache-ai/Mooncake for the original
distribution.

### Attribution

Mooncake traces (`traces/mooncake/*.jsonl`) are from
https://github.com/kvcache-ai/Mooncake, released under Apache-2.0 with
the FAST'25 paper. See upstream for the full preprocessing methodology
and privacy mechanisms applied to the raw production data.

## BurstGPT trace sample

`burstgpt/sample.jsonl` is a 1000-record subset of the `BurstGPT_3.csv`
v2.0 release from <https://github.com/HPMLL/BurstGPT>, with API-log rows
(blank `session_id`) filtered out via
`scripts/convert_burstgpt.py --require-session-id`.

Each line is a JSON object:
```json
{
  "request_id": "burstgpt-0",
  "arrival_ms": 0.0,
  "input_length": 906,
  "output_length": 446,
  "session_id": "1722ac82-0a46-4bf0-aa08-89794e7a2b3f"
}
```

**No `hash_ids` field** — BurstGPT has no prefix structure. Use
`prefix_mode: synthesis` with `configs/trace_burstgpt.yaml` to synthesize
prefix sharing on top of the real arrival/length data.

### To run the BurstGPT trace simulation

```bash
uv run python -m nano_kvrouter.cli sweep --config configs/trace_burstgpt.yaml
uv run python -m nano_kvrouter.cli prefix-sensitivity --config configs/trace_burstgpt.yaml
```

### License: CC-BY-4.0

Full text: <https://creativecommons.org/licenses/by/4.0/legalcode>

### Attribution

Wang, Y., et al. *BurstGPT: A Real-World Workload Dataset to Optimize LLM
Serving Systems.* HPMLL, 2024.
<https://github.com/HPMLL/BurstGPT>

### Full dataset

Full BurstGPT CSVs (188+ MB) are NOT in this repo. Download via:

```bash
mkdir -p traces/burstgpt/full
curl -sL "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv" \
  -o traces/burstgpt/full/BurstGPT_3.csv
```

`traces/burstgpt/full/` is in `.gitignore`.

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

## BurstGPT traces (not yet included)

BurstGPT traces (M2) will be added in a future milestone. The full dataset
is too large to commit directly; `scripts/fetch_traces.sh` will handle
download. See `traces/burstgpt/` (in `.gitignore`) for the download target.

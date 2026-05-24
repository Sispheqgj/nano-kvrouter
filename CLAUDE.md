# nano-kvrouter

KV-cache-centric LLM serving control-plane simulator. Pure Python, event-driven, no real GPU — runs on Mac M3 with mock backends.

## Package manager: uv only

**本项目只用 `uv` 管理依赖和虚拟环境。禁止使用 `pip`、`python -m pip`、`source .venv/bin/activate`、`.venv/bin/pip`、`.venv/bin/python -m pip`。**

All Python invocations must go through `uv run` so they execute inside the project's `.venv`.

## Commands

```bash
# Install / sync dependencies (including dev extras)
uv sync --extra dev

# Add a dependency (runtime / dev)
uv add <package>
uv add --dev <package>

# Tests
uv run pytest                          # all tests
uv run pytest tests/test_radix_tree.py # single module
uv run pytest -x -v                    # stop on first failure, verbose

# Lint & format (ruff/mypy are not yet in dev deps — `uv add --dev ruff mypy` first)
uv run ruff check --fix && uv run ruff format
uv run mypy src/nano_kvrouter

# Run simulation (CLI not yet implemented — see "Implementation status" below)
uv run python -m nano_kvrouter.cli run --config configs/default.yaml
```

## Implementation status

The structure below describes the **target architecture** from `DESIGN.md`. Not all modules exist yet — check the file before importing.

**Implemented:** `config.py`, `request.py`, `kv_cache/{radix_tree,block_pool}.py`, `engine/mock_node.py`, `scheduler/base.py`, `simulator/engine.py`, `metrics/collector.py`.

**Planned (not yet created):** `kv_cache/cache_manager.py`, `engine/latency_model.py`, all concrete schedulers (`round_robin`, `least_loaded`, `prefix_greedy`, `e2_policy`, `conductor`, `migration`), `simulator/{event,generator}.py`, `metrics/dashboard.py`, `cli.py`, `configs/*.yaml`, `traces/*`.

When adding a planned module, also add a `tests/test_<module>.py` in the same change.

## Project structure

```
src/nano_kvrouter/
├── config.py              # Pydantic config models (ClusterConfig, NodeConfig, etc.)
├── request.py             # Request dataclass
├── kv_cache/
│   ├── radix_tree.py      # RadixTree — SGLang-style prefix tree with LRU eviction
│   ├── block_pool.py      # BlockPool — 3-tier block storage (GPU/CPU/Disk)
│   └── cache_manager.py   # CacheManager — unified interface over tree + pool
├── engine/
│   ├── mock_node.py       # MockEngineNode — latency model, no real computation
│   └── latency_model.py   # Parameterized cost functions for prefill/decode/transfer
├── scheduler/
│   ├── base.py            # SchedulingPolicy protocol + SchedulingDecision dataclass
│   ├── round_robin.py     # Baseline: rotate across nodes
│   ├── least_loaded.py    # Baseline: pick node with lowest load
│   ├── prefix_greedy.py   # SGLang-style: maximize prefix cache hit
│   ├── e2_policy.py       # Preble E2: exploit-explore with prompt-aware load
│   ├── conductor.py       # Mooncake Conductor: 3-objective scoring + early rejection
│   └── migration.py       # KV block migration / rebalance logic
├── simulator/
│   ├── event.py           # Event enum + Event dataclass
│   ├── engine.py          # SimulationEngine — priority-queue event loop
│   └── generator.py       # RequestGenerator — Poisson, trace replay, bursty
├── metrics/
│   ├── collector.py       # Per-request + system-wide metrics
│   └── dashboard.py       # Terminal dashboard (rich) + CSV/JSON export
└── cli.py                 # Typer CLI entry point
```

## Code style

- Python 3.11+. Use `from __future__ import annotations` in every file.
- Type hints on all public functions and dataclass fields. Use `Protocol` for interfaces, not ABC.
- Dataclasses for data, not dicts. `@dataclass(slots=True)` when possible.
- Imports: stdlib → third-party → local, separated by blank lines. Use `from __future__ import annotations` as first import.
- Use `ruff` for linting and formatting. Line length 100.
- No `print()` — use `logging` module. Logger per module: `logger = logging.getLogger(__name__)`.
- Prefer composition over inheritance. Schedulers implement `SchedulingPolicy` protocol, not subclass.
- **Comments**: every public class needs a docstring explaining its purpose and key design decisions. Every public method needs a docstring with Args/Returns. Non-obvious logic blocks must have inline comments explaining *why*, not just *what*.

## Architecture rules

- **Event-driven, not threaded.** The simulator is a single-threaded event loop with a `heapq` priority queue. No asyncio, no threads.
- **Mock, not real.** Nodes don't run real inference. They compute `estimated_time = f(tokens, batch_size, tier)` and emit future events. All "time" is simulated.
- **Pluggable schedulers.** Every scheduling strategy is a separate module implementing `SchedulingPolicy`. The strategy is selected via config YAML, resolved in `cli.py`.
- **KV cache is the central abstraction.** `CacheManager` owns `RadixTree` + `BlockPool`. Schedulers query it for prefix match and capacity. Nodes allocate/free blocks through it.
- **Metrics are passive observers.** `MetricsCollector` listens to events, never modifies simulation state.
- **Configs are Pydantic models.** Loaded from YAML via `pydantic-settings`. No magic globals.

## Key domain concepts

- **Block**: fixed-size chunk (e.g. 16 tokens) of KV cache. Has `block_id`, lives on a `node` at a `tier`.
- **Tier**: GPU_HBM (fastest) → CPU_DRAM → DISK (coldest). Transfer time depends on tier pair bandwidth.
- **RadixTree**: prefix tree mapping token sequences to cached blocks. LRU eviction when full.
- **Prefill**: processing input tokens. Cost ∝ `(total - cached) * cost_per_token`.
- **Decode**: autoregressive generation. Per-step cost ∝ `base + batch_size * marginal`.
- **Early rejection**: if predicted TTFT or TBT would violate SLO, reject before prefill (Mooncake).
- **E2 score**: `historical_load + eviction_cost + run_cost` — lower is better (Preble).

## Testing

- Every module gets a `tests/test_<module>.py`. Test the public API, not internals.
- Use `pytest` fixtures for common setup (cluster config, pre-populated cache pool).
- Scheduler tests: assert correct node selection under known scenarios (e.g. "node 0 has prefix cached → PrefixGreedy picks node 0").
- Simulation tests: run short simulations (10-50 requests) and assert metrics are within expected ranges.
- IMPORTANT: run `pytest` after every code change. Fix failures before moving on.

## Reference papers

These are the systems being simulated. Read DESIGN.md for detailed algorithm descriptions.

| System | Key idea we simulate |
|--------|---------------------|
| Mooncake (FAST'25) | Conductor scheduler, 3-objective scoring, early rejection |
| Preble (ICLR'25) | E2 exploit-explore, prompt-aware load |
| SGLang (NeurIPS'24) | RadixAttention prefix tree, cache-aware scheduling |
| Llumnix (OSDI'24) | Live migration, KV-aware rebalancing |
| vLLM v1 | BlockPool design, hash-based prefix caching |

## Common mistakes to avoid

- Don't store actual tensors — blocks are metadata only (id, node, tier, token_hash, ref_count).
- Don't use wall-clock time — all time is simulated via `event.time`. Use `SimulationEngine.now()`.
- Don't add async/threading — the event loop is deliberately single-threaded for reproducibility.
- Don't hardcode parameters — everything flows from the YAML config through Pydantic models.
- Don't forget SLO checks — every scheduling decision must check predicted TTFT/TBT against SLO.
"""Tests for TraceGenerator and _build_token_ids."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from nano_kvrouter.config import NanoKVConfig, TraceConfig
from nano_kvrouter.simulator.trace_generator import TraceGenerator, _build_token_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(block_size: int = 16, vocab_size: int = 32000, seed: int = 42) -> NanoKVConfig:
    """Return a minimal NanoKVConfig with the given block_size."""
    import yaml
    from pathlib import Path
    cfg_dict = yaml.safe_load(Path("configs/default.yaml").read_text())
    cfg_dict.setdefault("model", {})["block_size"] = block_size
    cfg_dict.setdefault("generator", {})["vocab_size"] = vocab_size
    cfg_dict.setdefault("generator", {})["seed"] = seed
    return NanoKVConfig(**cfg_dict)


def _write_trace(tmp_path: Path, records: list[dict]) -> Path:
    """Write records as JSONL to a temp file and return the path."""
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def _make_record(
    timestamp: int = 0,
    input_length: int = 32,
    output_length: int = 10,
    hash_ids: list[int] | None = None,
) -> dict:
    return {
        "timestamp": timestamp,
        "input_length": input_length,
        "output_length": output_length,
        "hash_ids": hash_ids or [],
    }


# ---------------------------------------------------------------------------
# Test 1: len(token_ids) == input_length for non-aligned input_length
# ---------------------------------------------------------------------------


def test_hash_ids_token_length_non_aligned():
    """_build_token_ids must produce exactly input_length tokens even when
    input_length is not a multiple of block_size."""
    block_size = 16
    input_length = 1029   # 64 full blocks + 5 tail tokens
    hash_ids = list(range(10))  # 10 hash_ids, less than 64 full blocks
    rng = random.Random(0)
    toks = _build_token_ids(hash_ids, input_length, block_size, rng, 32000)
    assert len(toks) == input_length, f"Expected {input_length}, got {len(toks)}"


# ---------------------------------------------------------------------------
# Test 2: prefix sharing — same hash_ids prefix → same token prefix
# ---------------------------------------------------------------------------


def test_hash_ids_prefix_sharing():
    """Two records with the same leading hash_ids must produce identical
    leading tokens. The first diverging hash_id must produce different tokens."""
    block_size = 16
    hash_ids_a = [1, 2, 3, 4, 5]
    hash_ids_b = [1, 2, 99, 4, 5]
    input_length = 5 * block_size  # exact fit, no tail

    rng_a = random.Random(0)
    rng_b = random.Random(0)
    toks_a = _build_token_ids(hash_ids_a, input_length, block_size, rng_a, 32000)
    toks_b = _build_token_ids(hash_ids_b, input_length, block_size, rng_b, 32000)

    shared = 2 * block_size  # first 2 hash_ids are identical
    assert toks_a[:shared] == toks_b[:shared], "Shared prefix tokens must be identical"
    # Third block differs (hash_id 3 vs 99)
    assert toks_a[shared : shared + block_size] != toks_b[shared : shared + block_size], (
        "After the shared prefix, token sequences must diverge"
    )


# ---------------------------------------------------------------------------
# Test 3: per-request output_length drives decode (via expected_output_len)
# ---------------------------------------------------------------------------


def test_per_request_output_length_drives_decode(tmp_path: Path):
    """output_length in trace record drives actual decode: expect exactly
    output_length TOKEN_GENERATED events for the single request."""
    import yaml
    from nano_kvrouter.cli import _build_scheduler, _wire_simulator
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.metrics.collector import MetricsCollector
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import EventType

    block_size = 16
    output_length = 42
    records = [_make_record(timestamp=0, input_length=block_size, output_length=output_length,
                            hash_ids=[1])]
    trace_path = _write_trace(tmp_path, records)
    cfg_dict = yaml.safe_load(Path("configs/default.yaml").read_text())
    cfg_dict.setdefault("model", {})["block_size"] = block_size
    cfg_dict.setdefault("generator", {})["vocab_size"] = 32000
    cfg_dict["trace"] = {"path": str(trace_path), "max_requests": 1, "prefix_mode": "hash_ids"}
    from nano_kvrouter.config import NanoKVConfig, TraceConfig
    cfg = NanoKVConfig(**cfg_dict)
    tc = TraceConfig(path=str(trace_path), max_requests=1, prefix_mode="hash_ids")

    import logging
    eng = SimulationEngine()
    prefill_nodes = [MockEngineNode("p0", cfg.model, cfg.node)]
    decode_nodes  = [MockEngineNode("d0", cfg.model, cfg.node)]
    cm = CacheManager(
        node_ids=[n.node_id for n in decode_nodes],
        model_config=cfg.model,
        node_config=cfg.node,
        bandwidth_config=cfg.bandwidth,
        clock=eng.now,
    )
    sched = _build_scheduler("round_robin", {}, cfg.model, cfg.bandwidth)
    all_nodes = {n.node_id: n for n in [*prefill_nodes, *decode_nodes]}
    metrics = MetricsCollector()
    from nano_kvrouter.simulator.transfer_model import NoopTransferModel
    _wire_simulator(eng, sched, cm, prefill_nodes, decode_nodes,
                    logger_=logging.getLogger("test"), model_cfg=cfg.model,
                    bandwidth_cfg=cfg.bandwidth, transfer_model=NoopTransferModel())
    metrics.attach(eng, nodes=all_nodes)
    gen = TraceGenerator(cfg, tc, trace_path)
    gen.attach(eng)

    tokens_per_req: dict[str, int] = {}

    def count_tokens(event, engine):
        rid = event.payload["request_id"]
        tokens_per_req[rid] = tokens_per_req.get(rid, 0) + 1

    eng.on(EventType.TOKEN_GENERATED, count_tokens)
    eng.run()

    assert len(tokens_per_req) == 1, f"Expected 1 request, got {tokens_per_req}"
    count = next(iter(tokens_per_req.values()))
    assert count == output_length, f"Expected {output_length} tokens, got {count}"


# ---------------------------------------------------------------------------
# Test 4: speedup halves arrival time
# ---------------------------------------------------------------------------


def test_speedup_halves_arrival(tmp_path: Path):
    """With speedup=2.0, a record with timestamp=1000 must arrive at t=500."""
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import EventType

    records = [_make_record(timestamp=1000, input_length=16, output_length=10, hash_ids=[1])]
    trace_path = _write_trace(tmp_path, records)
    cfg = _make_config(block_size=16)
    tc = TraceConfig(path=str(trace_path), speedup=2.0, max_requests=1, prefix_mode="hash_ids")

    arrive_times: list[float] = []

    eng = SimulationEngine()
    gen = TraceGenerator(cfg, tc, trace_path)
    gen.attach(eng)

    def capture(event, _engine=None):
        arrive_times.append(eng.now())

    eng.on(EventType.REQUEST_ARRIVE, capture)
    eng.run()

    assert len(arrive_times) == 1
    assert arrive_times[0] == pytest.approx(500.0), (
        f"Expected arrival at 500.0, got {arrive_times[0]}"
    )


# ---------------------------------------------------------------------------
# Test 5: max_requests cap
# ---------------------------------------------------------------------------


def test_max_requests_cap(tmp_path: Path):
    """TraceGenerator must stop scheduling after max_requests ARRIVE events."""
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import EventType

    records = [_make_record(timestamp=i * 10, input_length=16, output_length=5, hash_ids=[i])
               for i in range(10)]
    trace_path = _write_trace(tmp_path, records)
    cfg = _make_config(block_size=16)
    tc = TraceConfig(path=str(trace_path), max_requests=3, prefix_mode="hash_ids")

    arrived: list = []

    eng = SimulationEngine()
    gen = TraceGenerator(cfg, tc, trace_path)
    gen.attach(eng)
    eng.on(EventType.REQUEST_ARRIVE, lambda e, _=None: arrived.append(e))
    eng.run()

    assert len(arrived) == 3, f"Expected 3 arrivals, got {len(arrived)}"


# ---------------------------------------------------------------------------
# P3-M1.fix new tests
# ---------------------------------------------------------------------------


def test_attach_twice_raises(tmp_path: Path):
    """Calling attach() a second time must raise RuntimeError."""
    from nano_kvrouter.simulator.engine import SimulationEngine

    records = [_make_record(timestamp=0, input_length=16, output_length=5, hash_ids=[1])]
    trace_path = _write_trace(tmp_path, records)
    cfg = _make_config(block_size=16)
    tc = TraceConfig(path=str(trace_path), max_requests=1, prefix_mode="hash_ids")

    eng = SimulationEngine()
    gen = TraceGenerator(cfg, tc, trace_path)
    gen.attach(eng)
    with pytest.raises(RuntimeError, match="more than once"):
        gen.attach(eng)


def test_prefix_mode_none(tmp_path: Path):
    """prefix_mode='none' generates exactly input_length random tokens (no hash_ids needed)."""
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import EventType

    records = [{"timestamp": 0, "input_length": 8, "output_length": 3}]  # no hash_ids
    trace_path = _write_trace(tmp_path, records)
    cfg = _make_config(block_size=4)
    tc = TraceConfig(path=str(trace_path), max_requests=1, prefix_mode="none")

    arrived: list = []
    eng = SimulationEngine()
    gen = TraceGenerator(cfg, tc, trace_path)
    gen.attach(eng)
    eng.on(EventType.REQUEST_ARRIVE, lambda e, _=None: arrived.append(e.payload["request"]))
    eng.run()

    assert len(arrived) == 1
    assert len(arrived[0].token_ids) == 8


def test_prefix_mode_synthesis_assigns_prefix(tmp_path: Path):
    """prefix_mode='synthesis' assigns tokens with len == input_length invariant."""
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import EventType
    from nano_kvrouter.simulator.prefix_synthesis import PrefixSynthesisConfig, PrefixSynthesisModel

    block_size = 16
    input_length = 64
    # Use a record with arrival_ms (BurstGPT format)
    record = {"arrival_ms": 0.0, "input_length": input_length, "output_length": 10}
    trace_path = tmp_path / "trace.jsonl"
    import json as _json
    trace_path.write_text(_json.dumps(record) + "\n")

    cfg = _make_config(block_size=block_size)
    tc = TraceConfig(path=str(trace_path), max_requests=1, prefix_mode="synthesis")

    ps_cfg = PrefixSynthesisConfig(
        num_buckets=4,
        sharing_layers=[(1.0, 0.5)],  # always 50% ratio
        seed=0,
    )
    synthesis_model = PrefixSynthesisModel(
        config=ps_cfg,
        block_size=block_size,
        vocab_size=32000,
        initial_prompt_len=256,
    )

    arrived = []
    eng = SimulationEngine()
    gen = TraceGenerator(cfg, tc, trace_path, synthesis_model=synthesis_model)
    gen.attach(eng)
    eng.on(EventType.REQUEST_ARRIVE, lambda e, _=None: arrived.append(e.payload["request"]))
    eng.run()

    assert len(arrived) == 1
    req = arrived[0]
    assert len(req.token_ids) == input_length, (
        f"len(token_ids)={len(req.token_ids)} != input_length={input_length}"
    )


def test_synthesis_requires_synthesis_model(tmp_path: Path):
    """prefix_mode='synthesis' without synthesis_model must raise ValueError."""
    records = [_make_record(timestamp=0, input_length=16, output_length=5)]
    trace_path = _write_trace(tmp_path, records)
    cfg = _make_config(block_size=16)
    tc = TraceConfig(path=str(trace_path), max_requests=1, prefix_mode="synthesis")

    with pytest.raises(ValueError, match="synthesis_model"):
        TraceGenerator(cfg, tc, trace_path)


def test_hash_ids_missing_raises(tmp_path: Path):
    """prefix_mode='hash_ids' with a record missing hash_ids raises ValueError."""
    from nano_kvrouter.simulator.engine import SimulationEngine

    records = [{"timestamp": 0, "input_length": 16, "output_length": 5}]  # no hash_ids
    trace_path = _write_trace(tmp_path, records)
    cfg = _make_config(block_size=16)
    tc = TraceConfig(path=str(trace_path), max_requests=1, prefix_mode="hash_ids")

    eng = SimulationEngine()
    gen = TraceGenerator(cfg, tc, trace_path)
    with pytest.raises(ValueError, match="hash_ids"):
        gen.attach(eng)
        eng.run()


def test_max_requests_closes_file(tmp_path: Path):
    """After max_requests arrivals the trace file handle must be closed."""
    from nano_kvrouter.simulator.engine import SimulationEngine

    records = [_make_record(timestamp=i * 10, input_length=16, output_length=3, hash_ids=[i])
               for i in range(10)]
    trace_path = _write_trace(tmp_path, records)
    cfg = _make_config(block_size=16)
    tc = TraceConfig(path=str(trace_path), max_requests=3, prefix_mode="hash_ids")

    eng = SimulationEngine()
    gen = TraceGenerator(cfg, tc, trace_path)
    gen.attach(eng)
    eng.run()

    assert gen._file is None, "File handle should be closed after max_requests cap"

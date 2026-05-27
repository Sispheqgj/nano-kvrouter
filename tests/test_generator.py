"""Tests for RequestGenerator — streaming Poisson + K-bucket workload."""
from __future__ import annotations

import pytest

from nano_kvrouter.config import (
    GeneratorConfig,
    ModelConfig,
    NanoKVConfig,
    SLOConfig,
    WorkloadConfig,
)
from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import EventType
from nano_kvrouter.simulator.generator import RequestGenerator


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _cfg(
    *,
    rate: float = 100.0,
    duration_s: float = 1.0,
    ratio: float = 0.5,
    prompt_len: int = 64,
    output_len: int = 32,
    block_size: int = 16,
    num_buckets: int = 10,
    vocab_size: int = 32_000,
    seed: int = 42,
) -> NanoKVConfig:
    return NanoKVConfig(
        workload=WorkloadConfig(
            request_rate=rate,
            duration_s=duration_s,
            prefix_sharing_ratio=ratio,
            avg_prompt_len=prompt_len,
            avg_output_len=output_len,
        ),
        model=ModelConfig(block_size=block_size),
        generator=GeneratorConfig(num_buckets=num_buckets, vocab_size=vocab_size, seed=seed),
    )


def _collect_requests(cfg: NanoKVConfig) -> list:
    """Run the engine and collect all Request objects from ARRIVE events."""
    engine = SimulationEngine()
    gen = RequestGenerator(cfg)
    requests = []

    def collect(ev, eng):
        requests.append(ev.payload["request"])

    engine.on(EventType.REQUEST_ARRIVE, collect)
    gen.attach(engine)
    engine.run()
    return requests


# ---------------------------------------------------------------------------
# 1. prefix_len is floored to block_size multiple
# ---------------------------------------------------------------------------


def test_prefix_len_is_block_aligned():
    # ratio=0.5, prompt=100, block=16 → raw=50 → floor(50/16)*16 = 48
    cfg = _cfg(ratio=0.5, prompt_len=100, block_size=16)
    gen = RequestGenerator(cfg)
    assert gen._prefix_len == 48
    assert gen._prefix_len % 16 == 0


# ---------------------------------------------------------------------------
# 2. prefix_len collapses to 0 when ratio × prompt < block_size
# ---------------------------------------------------------------------------


def test_prefix_len_zero_when_ratio_too_low():
    # ratio=0.05, prompt=100, block=16 → raw=5 → floor(5/16)*16 = 0
    cfg = _cfg(ratio=0.05, prompt_len=100, block_size=16)
    gen = RequestGenerator(cfg)
    assert gen._prefix_len == 0


# ---------------------------------------------------------------------------
# 3. prefix_len + suffix_len == avg_prompt_len (always)
# ---------------------------------------------------------------------------


def test_prefix_plus_suffix_equals_prompt_len():
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cfg = _cfg(ratio=ratio, prompt_len=128)
        gen = RequestGenerator(cfg)
        assert gen._prefix_len + gen._suffix_len == 128


# ---------------------------------------------------------------------------
# 4. K buckets have distinct prefixes (extremely likely with large vocab)
# ---------------------------------------------------------------------------


def test_buckets_have_distinct_prefixes_with_high_probability():
    cfg = _cfg(ratio=0.5, prompt_len=64, num_buckets=10, vocab_size=32_000)
    gen = RequestGenerator(cfg)
    # prefix_len = floor(32/16)*16 = 32 tokens; 32000^32 collision space
    prefixes = {tuple(p) for p in gen._bucket_prefixes}
    assert len(prefixes) == 10


# ---------------------------------------------------------------------------
# 5. Seed reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    cfg42a = _cfg(seed=42)
    cfg42b = _cfg(seed=42)
    gen_a = RequestGenerator(cfg42a)
    gen_b = RequestGenerator(cfg42b)
    assert gen_a._bucket_prefixes == gen_b._bucket_prefixes

    cfg999 = _cfg(seed=999)
    gen_c = RequestGenerator(cfg999)
    assert gen_c._bucket_prefixes != gen_a._bucket_prefixes


# ---------------------------------------------------------------------------
# 6. attach schedules the first arrival in the engine queue
# ---------------------------------------------------------------------------


def test_attach_schedules_first_arrival():
    cfg = _cfg(rate=10.0, duration_s=10.0)
    engine = SimulationEngine()
    gen = RequestGenerator(cfg)
    gen.attach(engine)

    assert len(engine._queue) >= 1
    # The first event must be a REQUEST_ARRIVE within the simulation window
    first_ev = min(engine._queue)
    assert first_ev.type == EventType.REQUEST_ARRIVE
    assert first_ev.time < gen._end_time_ms


# ---------------------------------------------------------------------------
# 7. Chain generates ~rate*duration arrivals (Poisson, generous tolerance)
# ---------------------------------------------------------------------------


def test_chain_generates_multiple_requests():
    cfg = _cfg(rate=100.0, duration_s=1.0)   # expect ~100 in 1s
    engine = SimulationEngine()
    gen = RequestGenerator(cfg)

    arrive_count = 0

    def counter(ev, eng):
        nonlocal arrive_count
        arrive_count += 1

    engine.on(EventType.REQUEST_ARRIVE, counter)
    gen.attach(engine)
    engine.run()

    expected = cfg.workload.request_rate * cfg.workload.duration_s  # 100
    assert 50 < arrive_count < 150, (
        f"Expected ~{expected} arrivals, got {arrive_count}"
    )


# ---------------------------------------------------------------------------
# 8. Chain stops: all arrivals have time < end_time_ms
# ---------------------------------------------------------------------------


def test_chain_stops_at_end_time():
    cfg = _cfg(rate=50.0, duration_s=0.5)
    engine = SimulationEngine()
    gen = RequestGenerator(cfg)
    end_ms = gen._end_time_ms

    times = []

    def collect_time(ev, eng):
        times.append(ev.time)

    engine.on(EventType.REQUEST_ARRIVE, collect_time)
    gen.attach(engine)
    engine.run()

    assert len(times) > 0
    assert all(t < end_ms for t in times), (
        f"Found arrival(s) at or beyond end_time_ms={end_ms}: "
        f"{[t for t in times if t >= end_ms]}"
    )


# ---------------------------------------------------------------------------
# 9. request_ids are sequential: req-0, req-1, …
# ---------------------------------------------------------------------------


def test_request_ids_are_sequential():
    cfg = _cfg(rate=50.0, duration_s=0.2)   # ~10 requests
    requests = _collect_requests(cfg)

    assert len(requests) > 0
    for i, req in enumerate(requests):
        assert req.request_id == f"req-{i}", (
            f"Expected req-{i}, got {req.request_id}"
        )


# ---------------------------------------------------------------------------
# 10. Each request's prefix tokens belong to exactly one bucket
# ---------------------------------------------------------------------------


def test_request_belongs_to_one_bucket():
    cfg = _cfg(rate=50.0, duration_s=0.5, ratio=0.5, prompt_len=64)
    gen = RequestGenerator(cfg)
    # Run simulation and collect requests independently so we can inspect gen state
    engine = SimulationEngine()
    requests = []

    def collect(ev, eng):
        requests.append(ev.payload["request"])

    engine.on(EventType.REQUEST_ARRIVE, collect)
    gen.attach(engine)
    engine.run()

    prefix_len = gen._prefix_len
    if prefix_len == 0:
        pytest.skip("prefix_len=0, no prefix to verify")

    bucket_set = {tuple(p) for p in gen._bucket_prefixes}
    for req in requests:
        prefix = tuple(req.token_ids[:prefix_len])
        assert prefix in bucket_set, (
            f"Request {req.request_id} prefix not in any bucket"
        )


# ---------------------------------------------------------------------------
# 11. SLO fields are populated from config
# ---------------------------------------------------------------------------


def test_slo_fields_populated_from_config():
    cfg = _cfg(rate=20.0, duration_s=0.1)
    requests = _collect_requests(cfg)

    assert len(requests) > 0
    for req in requests:
        assert req.slo_ttft == cfg.slo.ttft_target_ms
        assert req.slo_tbt == cfg.slo.tbt_target_ms


# ---------------------------------------------------------------------------
# 12. len(token_ids) == avg_prompt_len for every request
# ---------------------------------------------------------------------------


def test_token_ids_length_equals_avg_prompt_len():
    cfg = _cfg(rate=50.0, duration_s=0.2, prompt_len=64)
    requests = _collect_requests(cfg)

    assert len(requests) > 0
    for req in requests:
        assert len(req.token_ids) == cfg.workload.avg_prompt_len, (
            f"Request {req.request_id}: "
            f"len(token_ids)={len(req.token_ids)}, "
            f"expected {cfg.workload.avg_prompt_len}"
        )


# ---------------------------------------------------------------------------
# 13. prefix_len=0 produces requests with no shared prefix
# ---------------------------------------------------------------------------


def test_zero_prefix_len_produces_pure_cold_start():
    cfg = _cfg(ratio=0.0, prompt_len=32)   # prefix_len forced to 0
    gen = RequestGenerator(cfg)
    assert gen._prefix_len == 0
    assert gen._suffix_len == 32
    # All bucket prefixes are empty lists
    assert all(p == [] for p in gen._bucket_prefixes)


# ---------------------------------------------------------------------------
# 14. expected_output_len stamped from config
# ---------------------------------------------------------------------------


def test_expected_output_len_from_config():
    cfg = _cfg(rate=20.0, duration_s=0.1, output_len=128)
    requests = _collect_requests(cfg)

    assert len(requests) > 0
    for req in requests:
        assert req.expected_output_len == 128


# ---------------------------------------------------------------------------
# 15. Defensive: attach() may only be called once per generator
# ---------------------------------------------------------------------------


def test_attach_twice_raises_runtime_error() -> None:
    """Codex regression: duplicate attach caused exponential arrival
    spawn (~K^N growth), not linear. Guard rails turn it into a hard
    error so user mistakes surface immediately instead of producing
    corrupt simulation data.
    """
    cfg = NanoKVConfig(
        workload=WorkloadConfig(request_rate=10.0, duration_s=1.0),
    )
    eng = SimulationEngine()
    gen = RequestGenerator(cfg)

    gen.attach(eng)   # first attach: OK

    with pytest.raises(RuntimeError, match="more than once"):
        gen.attach(eng)

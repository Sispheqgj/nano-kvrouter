from __future__ import annotations

import pytest

from nano_kvrouter.config import ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.metrics.collector import MetricsCollector
from nano_kvrouter.request import Request
from nano_kvrouter.simulator.engine import SimulationEngine
from nano_kvrouter.simulator.event import Event, EventType


# ------------------------------------------------------------------
# Fixtures & helpers
# ------------------------------------------------------------------

@pytest.fixture
def engine() -> SimulationEngine:
    return SimulationEngine()


@pytest.fixture
def collector() -> MetricsCollector:
    return MetricsCollector()


def _make_request(
    req_id: str,
    n_tokens: int = 100,
    slo_ttft: float = 2000.0,
) -> Request:
    return Request(
        request_id=req_id,
        token_ids=list(range(n_tokens)),
        prefix_hash="x",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=slo_ttft,
        slo_tbt=100.0,
    )


def _arrive(req: Request, t: float) -> Event:
    return Event(time=t, type=EventType.REQUEST_ARRIVE, payload={"request": req})


def _scheduled(req_id: str, t: float, matched: int = 0) -> Event:
    return Event(
        time=t,
        type=EventType.SCHEDULED,
        payload={"request_id": req_id, "matched_tokens": matched},
    )


def _reject(req_id: str, t: float) -> Event:
    return Event(
        time=t,
        type=EventType.REQUEST_REJECTED,
        payload={"request_id": req_id, "reason": "test"},
    )


def _prefill_done(req_id: str, t: float) -> Event:
    return Event(time=t, type=EventType.PREFILL_COMPLETE, payload={"request_id": req_id})


def _decode_step(req_id: str, t: float, step: int = 0) -> Event:
    return Event(
        time=t,
        type=EventType.TOKEN_GENERATED,
        payload={"request_id": req_id, "step_index": step},
    )


def _decode_done(req_id: str, t: float) -> Event:
    return Event(time=t, type=EventType.DECODE_COMPLETE, payload={"request_id": req_id})


# ------------------------------------------------------------------
# Test 1: 初始状态全 0 / None
# ------------------------------------------------------------------

def test_initial_summary_is_empty(collector):
    s = collector.summary()
    assert s["total_arrived"] == 0
    assert s["completed"] == 0
    assert s["rejected"] == 0
    assert s["rejection_rate"] is None
    assert s["ttft_p50_ms"] is None
    assert s["ttft_p99_ms"] is None
    assert s["ttft_avg_ms"] is None
    assert s["tbt_p50_ms"] is None
    assert s["tbt_avg_ms"] is None
    assert s["e2e_p50_ms"] is None
    assert s["e2e_avg_ms"] is None
    assert s["slo_ttft_hit_rate"] is None
    assert s["cache_hit_ratio"] is None
    assert s["throughput_req_per_s"] is None
    assert s["avg_batch_size"] is None
    assert s["decode_throughput_tokens_per_s"] is None
    assert s["avg_chunked_prefill_steps_per_request"] is None
    assert s["prefill_decode_interleave_step_count"] == 0


# ------------------------------------------------------------------
# Test 2: ARRIVE 计数
# ------------------------------------------------------------------

def test_arrive_increments_total(engine, collector):
    collector.attach(engine)
    n = 5
    for i in range(n):
        engine.schedule(_arrive(_make_request(f"r{i}"), t=float(i)))
    engine.run()
    assert collector.summary()["total_arrived"] == n


# ------------------------------------------------------------------
# Test 3: 拒绝计数与 rejection_rate
# ------------------------------------------------------------------

def test_reject_counts(engine, collector):
    collector.attach(engine)
    for i in range(3):
        engine.schedule(_arrive(_make_request(f"r{i}"), t=0.0))
    for i in range(2):  # 2 of 3 rejected
        engine.schedule(_reject(f"r{i}", t=1.0))
    engine.run()
    s = collector.summary()
    assert s["rejected"] == 2
    assert s["rejection_rate"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------
# Test 4 (v2): TTFT 在 TOKEN_GENERATED[step_index=0] 时计算
# ------------------------------------------------------------------

def test_ttft_recorded_on_first_decode_step(engine, collector):
    """v2: TTFT = first TOKEN_GENERATED time - arrival time."""
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=10.0))
    engine.schedule(_decode_step("r0", t=35.0, step=0))
    engine.run()
    s = collector.summary()
    assert s["ttft_p50_ms"] == pytest.approx(25.0)
    assert s["ttft_avg_ms"] == pytest.approx(25.0)


# ------------------------------------------------------------------
# Test 5: E2E 从 ARRIVE → DECODE_COMPLETE 计算（语义不变）
# ------------------------------------------------------------------

def test_e2e_recorded_on_decode_complete(engine, collector):
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=10.0))
    engine.schedule(_decode_done("r0", t=200.0))
    engine.run()
    s = collector.summary()
    assert s["e2e_p50_ms"] == pytest.approx(190.0)
    assert s["e2e_avg_ms"] == pytest.approx(190.0)


# ------------------------------------------------------------------
# Test 6: TBT 跨多个 TOKEN_GENERATED 计算平均
# ------------------------------------------------------------------

def test_tbt_avg_over_multiple_steps(engine, collector):
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=0.0))
    engine.schedule(_prefill_done("r0", t=100.0))   # no-op in v2
    # 5 steps at 5ms intervals starting at t=105.
    # v2: step[0] seeds reference; steps 1-4 each produce one TBT sample.
    # → 4 TBT samples of 5ms each, avg = 5.0.
    for i in range(5):
        engine.schedule(_decode_step("r0", t=105.0 + i * 5.0, step=i))
    engine.schedule(_decode_done("r0", t=130.0))
    engine.run()
    s = collector.summary()
    assert s["tbt_avg_ms"] == pytest.approx(5.0)
    assert s["tbt_p50_ms"] == pytest.approx(5.0)


# ------------------------------------------------------------------
# Test 7: SLO TTFT 命中率（事件序列改为 TOKEN_GENERATED step=0）
# ------------------------------------------------------------------

def test_slo_ttft_hit_rate_all_hit(engine, collector):
    collector.attach(engine)
    for i in range(3):
        req = _make_request(f"r{i}", slo_ttft=2000.0)
        engine.schedule(_arrive(req, t=0.0))
        # TTFT = 100, 200, 300 ms — all < slo 2000
        engine.schedule(_decode_step(f"r{i}", t=float((i + 1) * 100), step=0))
    engine.run()
    assert collector.summary()["slo_ttft_hit_rate"] == pytest.approx(1.0)


def test_slo_ttft_hit_rate_partial(engine, collector):
    collector.attach(engine)
    # r0, r1: slo=2000, TTFT=100, 200 → hit
    for i in range(2):
        req = _make_request(f"r{i}", slo_ttft=2000.0)
        engine.schedule(_arrive(req, t=0.0))
        engine.schedule(_decode_step(f"r{i}", t=float((i + 1) * 100), step=0))
    # r2: slo=200, TTFT=300 → miss
    req2 = _make_request("r2", slo_ttft=200.0)
    engine.schedule(_arrive(req2, t=0.0))
    engine.schedule(_decode_step("r2", t=300.0, step=0))
    engine.run()
    assert collector.summary()["slo_ttft_hit_rate"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------
# Test 8: 缓存命中率（不变）
# ------------------------------------------------------------------

def test_cache_hit_ratio(engine, collector):
    collector.attach(engine)
    for req_id, matched in [("r0", 32), ("r1", 16)]:
        req = _make_request(req_id, n_tokens=100)
        engine.schedule(_arrive(req, t=0.0))
        engine.schedule(_scheduled(req_id, t=1.0, matched=matched))
    engine.run()
    # 48 matched / 200 total = 0.24
    assert collector.summary()["cache_hit_ratio"] == pytest.approx(0.24)


# ------------------------------------------------------------------
# Test 9: 吞吐量 = completed / duration_s（不变）
# ------------------------------------------------------------------

def test_throughput_from_first_arrive_to_last_complete(engine, collector):
    collector.attach(engine)
    # 4 requests, first arrive t=0, last complete t=2000 → 4/2.0 = 2.0 req/s
    for i in range(4):
        req = _make_request(f"r{i}")
        engine.schedule(_arrive(req, t=0.0))
        engine.schedule(_decode_done(f"r{i}", t=500.0 * (i + 1)))
    engine.run()
    s = collector.summary()
    assert s["completed"] == 4
    assert s["throughput_req_per_s"] == pytest.approx(2.0)


# ------------------------------------------------------------------
# Test 10: attach 注册了全部 8 个 handler（M3: 新增 PREFILL_START）
# ------------------------------------------------------------------

def test_attach_registers_all_handlers(engine, collector):
    collector.attach(engine)
    expected = {
        EventType.REQUEST_ARRIVE,
        EventType.SCHEDULED,
        EventType.REQUEST_REJECTED,
        EventType.PREFILL_START,
        EventType.PREFILL_COMPLETE,
        EventType.TOKEN_GENERATED,
        EventType.DECODE_BATCH_STEP,
        EventType.DECODE_COMPLETE,
    }
    assert set(engine._handlers.keys()) == expected
    for et in expected:
        assert len(engine._handlers[et]) == 1


# ------------------------------------------------------------------
# Test 11: PREFILL_COMPLETE 在 v2 中是 no-op（不再记录 TTFT）
# ------------------------------------------------------------------

def test_prefill_complete_does_not_record_ttft(engine, collector):
    """v2: PREFILL_COMPLETE alone must not register a TTFT sample.
    TTFT only counts at first TOKEN_GENERATED."""
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=10.0))
    engine.schedule(_prefill_done("r0", t=35.0))
    # No DECODE_STEP scheduled
    engine.run()
    s = collector.summary()
    assert s["ttft_p50_ms"] is None
    assert s["ttft_avg_ms"] is None


# ------------------------------------------------------------------
# Test 12: TOKEN_GENERATED 缺 step_index 时跳过而不崩溃
# ------------------------------------------------------------------

def test_token_generated_without_step_index_skipped(engine, collector):
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=0.0))
    # TOKEN_GENERATED without step_index in payload — must not crash
    engine.schedule(Event(
        time=10.0, type=EventType.TOKEN_GENERATED,
        payload={"request_id": "r0"},   # no step_index
    ))
    engine.run()
    s = collector.summary()
    assert s["ttft_p50_ms"] is None
    assert s["tbt_p50_ms"] is None


# ------------------------------------------------------------------
# Test 13: TBT 不包含 prefill→first-token 的间隔（v2 核心修复验证）
# ------------------------------------------------------------------

def test_tbt_excludes_prefill_to_first_token_gap(engine, collector):
    """v2 invariant: prefill→step0 latency goes into TTFT, not TBT.

    prefill_complete at t=100 (no-op), step0 at t=200 (huge gap=100ms),
    step1 at t=205, step2 at t=210.
    v2 TBT samples: step1-step0=5, step2-step1=5 → avg=5.0.
    The 100ms prefill→step0 gap is NOT counted in TBT.
    """
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=0.0))
    engine.schedule(_prefill_done("r0", t=100.0))           # no-op
    engine.schedule(_decode_step("r0", t=200.0, step=0))    # TTFT=200, seeds TBT ref
    engine.schedule(_decode_step("r0", t=205.0, step=1))    # TBT=5
    engine.schedule(_decode_step("r0", t=210.0, step=2))    # TBT=5
    engine.schedule(_decode_done("r0", t=210.0))            # flushes TBT samples
    engine.run()
    s = collector.summary()
    assert s["ttft_p50_ms"] == pytest.approx(200.0)
    assert s["tbt_avg_ms"] == pytest.approx(5.0)   # NOT (100+5+5)/3 = 36.67


# ------------------------------------------------------------------
# Test 14: step_index=0 重复到达时 TTFT 幂等（不双重记录）
# ------------------------------------------------------------------

def test_ttft_idempotent_on_repeated_step_zero(engine, collector):
    """Duplicate step_index=0 events must not push a second TTFT sample."""
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=0.0))
    engine.schedule(_decode_step("r0", t=100.0, step=0))   # TTFT=100
    engine.schedule(_decode_step("r0", t=110.0, step=0))   # duplicate — ignored
    engine.run()
    s = collector.summary()
    # Only one TTFT sample recorded; p50 and avg are both 100 (not distorted by 110)
    assert s["ttft_p50_ms"] == pytest.approx(100.0)
    assert s["ttft_avg_ms"] == pytest.approx(100.0)


# ------------------------------------------------------------------
# Test 15: REQUEST_REJECTED without request_id is skipped
# ------------------------------------------------------------------

def test_rejected_without_request_id_skipped(engine, collector):
    collector.attach(engine)
    engine.schedule(_arrive(_make_request("r0"), t=0.0))
    # REJECTED payload missing request_id — rejected count must stay 0
    engine.schedule(Event(
        time=1.0, type=EventType.REQUEST_REJECTED,
        payload={"reason": "test"},  # no request_id
    ))
    engine.run()
    assert collector.summary()["rejected"] == 0


# ------------------------------------------------------------------
# Test 16: Duplicate step_index=0 must not reset the TBT anchor
# ------------------------------------------------------------------

def test_duplicate_step_zero_does_not_reset_tbt_anchor(engine, collector):
    """Second step_index=0 must not move the TBT anchor.
    Subsequent TBT[1] must reflect time since the *first* step 0."""
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=0.0))
    engine.schedule(_decode_step("r0", t=100.0, step=0))   # TTFT=100, anchor=100
    engine.schedule(_decode_step("r0", t=105.0, step=0))   # duplicate — anchor stays 100
    engine.schedule(_decode_step("r0", t=110.0, step=1))   # TBT = 110 - 100 = 10 (NOT 5)
    engine.run()
    s = collector.summary()
    assert s["ttft_p50_ms"] == pytest.approx(100.0)   # TTFT not double-counted
    assert s["tbt_avg_ms"] == pytest.approx(10.0)     # anchor is 100, not 105


# ------------------------------------------------------------------
# Helpers for M2 batch-step tests
# ------------------------------------------------------------------

def _decode_batch_step_event(node_id: str, t: float, batch_size: int) -> Event:
    return Event(
        time=t,
        type=EventType.DECODE_BATCH_STEP,
        payload={"node_id": node_id, "node_pool": "decode", "batch_size": batch_size},
    )


# ------------------------------------------------------------------
# Test 17: avg_batch_size
# ------------------------------------------------------------------

def test_avg_batch_size(engine, collector):
    """avg_batch_size is the arithmetic mean of batch_size values from all DECODE_BATCH_STEP events."""
    collector.attach(engine)
    engine.schedule(_decode_batch_step_event("n0", 10.0, batch_size=4))
    engine.schedule(_decode_batch_step_event("n0", 20.0, batch_size=2))
    engine.run()
    assert collector.summary()["avg_batch_size"] == pytest.approx(3.0)


def test_avg_batch_size_none_when_no_steps(engine, collector):
    """avg_batch_size is None when no DECODE_BATCH_STEP events have fired."""
    collector.attach(engine)
    engine.run()
    assert collector.summary()["avg_batch_size"] is None


def test_avg_batch_size_single_step(engine, collector):
    """Single DECODE_BATCH_STEP with batch_size=5 → avg=5."""
    collector.attach(engine)
    engine.schedule(_decode_batch_step_event("n0", 5.0, batch_size=5))
    engine.run()
    assert collector.summary()["avg_batch_size"] == pytest.approx(5.0)


# ------------------------------------------------------------------
# Test 18: decode_throughput_tokens_per_s
# ------------------------------------------------------------------

def test_decode_throughput_tokens_per_s(engine, collector):
    """3 batch steps × 4 tokens each over 2000 ms → 6.0 tok/s."""
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=0.0))
    engine.schedule(_decode_batch_step_event("n0", 500.0, batch_size=4))
    engine.schedule(_decode_batch_step_event("n0", 1000.0, batch_size=4))
    engine.schedule(_decode_batch_step_event("n0", 1500.0, batch_size=4))
    engine.schedule(_decode_done("r0", t=2000.0))   # sets _end_time
    engine.run()
    s = collector.summary()
    # total_tokens = 12, duration = (2000 - 0) / 1000 = 2s → 6.0 tok/s
    assert s["decode_throughput_tokens_per_s"] == pytest.approx(6.0)


def test_decode_throughput_none_when_no_arrivals(engine, collector):
    """decode_throughput_tokens_per_s is None when no duration can be measured."""
    collector.attach(engine)
    engine.schedule(_decode_batch_step_event("n0", 100.0, batch_size=3))
    engine.run()
    # No REQUEST_ARRIVE → _start_time is None → throughput is None
    assert collector.summary()["decode_throughput_tokens_per_s"] is None


# ------------------------------------------------------------------
# Test 19: execute-time batch_size via node fallback (reversed-order attach)
# ------------------------------------------------------------------


def test_batch_size_uses_execute_time(engine):
    """Collector must record execute-time batch_size via node fallback.

    Simulates the reversed-attach-order case: collector handler runs before
    the simulator handler, so event.payload has no 'batch_size'.  Collector
    must fall back to reading len(node.decoding) at handler-execution time,
    which is the pre-tick execute-time stream count.

    Schedule-time count = 1 (only r0 in decoding at schedule time).
    Execute-time count  = 2 (r1 joined before the event fires).
    Collector must record 2, not 1.
    """
    model = ModelConfig(prefill_cost_per_token_ms=0.1, decode_base_ms=5.0, marginal_decode_ms=1.0)
    node = MockEngineNode("n0", model, NodeConfig(capacity=8))
    # admit puts request into running_requests; start_decode moves it to decoding.
    node.admit("r0")
    node.start_decode("r0")   # 1 stream in decoding at schedule time

    collector = MetricsCollector()
    # Pass nodes so collector can fall back to node state when payload is empty.
    collector.attach(engine, nodes={"n0": node})

    # Schedule the batch step with NO batch_size in payload (reversed-order scenario).
    engine.schedule(Event(
        time=10.0,
        type=EventType.DECODE_BATCH_STEP,
        payload={"node_id": "n0"},   # no batch_size — collector must use node fallback
    ))

    # A 2nd stream joins decoding before the event fires (execute-time count = 2).
    node.admit("r1")
    node.start_decode("r1")

    engine.run()

    # Collector must see 2 (execute-time), not 1 (schedule-time) and not None.
    assert collector.summary()["avg_batch_size"] == pytest.approx(2.0)


# ------------------------------------------------------------------
# Test 20: avg_chunked_prefill_steps_per_request (M3)
# ------------------------------------------------------------------

def test_avg_chunked_prefill_steps_metric(engine, collector):
    """avg_chunked_prefill_steps_per_request is the mean of n_chunks cached at PREFILL_START."""
    collector.attach(engine)
    for req_id in ("a", "b"):
        engine.schedule(_arrive(_make_request(req_id), t=0.0))
    # n_chunks pre-injected into PREFILL_START payload (M3 attach-order-independent pattern)
    engine.schedule(Event(time=0.5, type=EventType.PREFILL_START,
        payload={"request_id": "a", "n_chunks": 2}))
    engine.schedule(Event(time=0.5, type=EventType.PREFILL_START,
        payload={"request_id": "b", "n_chunks": 4}))
    engine.schedule(Event(time=100.0, type=EventType.DECODE_COMPLETE,
        payload={"request_id": "a"}))
    engine.schedule(Event(time=100.0, type=EventType.DECODE_COMPLETE,
        payload={"request_id": "b"}))
    engine.run()
    assert collector.summary()["avg_chunked_prefill_steps_per_request"] == pytest.approx(3.0)


# ------------------------------------------------------------------
# Test 21: prefill_decode_interleave_step_count (M3)
# ------------------------------------------------------------------

def test_interleave_step_count(engine, collector):
    """prefill_decode_interleave_step_count counts DECODE_BATCH_STEP with interleave=True."""
    collector.attach(engine)
    engine.schedule(_arrive(_make_request("r0"), t=0.0))
    engine.schedule(Event(time=10.0, type=EventType.DECODE_BATCH_STEP,
        payload={"node_id": "n0", "batch_size": 2, "interleave": True}))
    engine.schedule(Event(time=20.0, type=EventType.DECODE_BATCH_STEP,
        payload={"node_id": "n0", "batch_size": 2, "interleave": True}))
    engine.schedule(Event(time=30.0, type=EventType.DECODE_BATCH_STEP,
        payload={"node_id": "n0", "batch_size": 2, "interleave": False}))
    engine.run()
    assert collector.summary()["prefill_decode_interleave_step_count"] == 2


# ------------------------------------------------------------------
# Test 22: M3 metrics are attach-order independent
# ------------------------------------------------------------------

def test_m3_metrics_independent_of_attach_order() -> None:
    """avg_chunked_prefill_steps and interleave_count must match for normal
    vs reversed collector/simulator attach order.

    Setup: req_a (0 uncached, enters decode at t=0) + req_b (512 uncached,
    chunk_size=128 → 4 chunks). While b prefills, a decodes → multiple
    interleave steps.  Both orders must report identical metrics.
    """
    import logging
    from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
    from nano_kvrouter.engine.mock_node import MockEngineNode
    from nano_kvrouter.kv_cache.cache_manager import CacheManager
    from nano_kvrouter.scheduler.round_robin import RoundRobinPolicy
    from nano_kvrouter.simulator.engine import SimulationEngine
    from nano_kvrouter.simulator.event import Event, EventType
    from nano_kvrouter.request import Request
    from nano_kvrouter.cli import _wire_simulator

    mc = ModelConfig(
        prefill_cost_per_token_ms=0.1,
        decode_base_ms=5.0,
        marginal_decode_ms=0.5,
        prefill_chunk_size=128,
    )
    nc = NodeConfig(capacity=4, gpu_blocks=200, cpu_blocks=400, disk_blocks=800)

    def _run(wire_first: bool) -> dict:
        eng = SimulationEngine()
        node = MockEngineNode("n0", mc, nc)
        cm = CacheManager(["n0"], mc, nc, BandwidthConfig(), clock=eng.now)
        sched = RoundRobinPolicy()
        coll = MetricsCollector()

        if wire_first:
            _wire_simulator(eng, sched, cm, [node], logger_=logging.getLogger("test"))
            coll.attach(eng, nodes={"n0": node})
        else:
            coll.attach(eng, nodes={"n0": node})
            _wire_simulator(eng, sched, cm, [node], logger_=logging.getLogger("test"))

        # req_a: 0 uncached (cache pre-warmed) → enters decode immediately
        cm.admit([0] * 32, "n0")
        req_a = Request("a", [0] * 32, "ha", expected_output_len=4,
                        arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)
        # req_b: 512 cold tokens, chunk_size=128 → 4 prefill chunks
        req_b = Request("b", list(range(512)), "hb", expected_output_len=2,
                        arrival_time=0.0, slo_ttft=9999.0, slo_tbt=9999.0)

        eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_a}))
        eng.schedule(Event(time=0.0, type=EventType.REQUEST_ARRIVE, payload={"request": req_b}))
        eng.run()
        return coll.summary()

    s_normal = _run(wire_first=True)
    s_reversed = _run(wire_first=False)

    assert s_normal["avg_chunked_prefill_steps_per_request"] is not None, (
        "avg_chunked_prefill_steps should be non-None (b has 4 chunks)"
    )
    assert s_normal["avg_chunked_prefill_steps_per_request"] == pytest.approx(
        s_reversed["avg_chunked_prefill_steps_per_request"]
    ), (
        f"avg_chunks: normal={s_normal['avg_chunked_prefill_steps_per_request']}, "
        f"reversed={s_reversed['avg_chunked_prefill_steps_per_request']}; "
        "metric is attach-order dependent (n_chunks not pre-injected?)"
    )
    assert s_normal["prefill_decode_interleave_step_count"] > 0, (
        "interleave_count should be >0 (req_a decoding while req_b prefills)"
    )
    assert s_normal["prefill_decode_interleave_step_count"] == s_reversed["prefill_decode_interleave_step_count"], (
        f"interleave: normal={s_normal['prefill_decode_interleave_step_count']}, "
        f"reversed={s_reversed['prefill_decode_interleave_step_count']}; "
        "metric is attach-order dependent (interleave not pre-injected?)"
    )

from __future__ import annotations

import pytest

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
        type=EventType.DECODE_STEP,
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
# Test 4 (v2): TTFT 在 DECODE_STEP[step_index=0] 时计算
# ------------------------------------------------------------------

def test_ttft_recorded_on_first_decode_step(engine, collector):
    """v2: TTFT = first DECODE_STEP time - arrival time."""
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
# Test 6: TBT 跨多个 DECODE_STEP 计算平均
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
# Test 7: SLO TTFT 命中率（事件序列改为 DECODE_STEP step=0）
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
# Test 10: attach 注册了全部 6 个 handler（不变）
# ------------------------------------------------------------------

def test_attach_registers_all_handlers(engine, collector):
    collector.attach(engine)
    expected = {
        EventType.REQUEST_ARRIVE,
        EventType.SCHEDULED,
        EventType.REQUEST_REJECTED,
        EventType.PREFILL_COMPLETE,
        EventType.DECODE_STEP,
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
    TTFT only counts at first DECODE_STEP."""
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
# Test 12: DECODE_STEP 缺 step_index 时跳过而不崩溃
# ------------------------------------------------------------------

def test_decode_step_without_step_index_skipped(engine, collector):
    collector.attach(engine)
    req = _make_request("r0")
    engine.schedule(_arrive(req, t=0.0))
    # DECODE_STEP without step_index in payload — must not crash
    engine.schedule(Event(
        time=10.0, type=EventType.DECODE_STEP,
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

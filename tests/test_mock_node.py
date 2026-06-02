from __future__ import annotations

import pytest
from nano_kvrouter.config import ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(prefill_cost_per_token_ms=0.033, decode_base_ms=5.0, marginal_decode_ms=0.5)


@pytest.fixture
def node_config() -> NodeConfig:
    return NodeConfig(capacity=32)


@pytest.fixture
def node(model_config: ModelConfig, node_config: NodeConfig) -> MockEngineNode:
    return MockEngineNode(node_id="node-0", model_config=model_config, node_config=node_config)


# ------------------------------------------------------------------
# estimate_prefill_time
# ------------------------------------------------------------------

def test_prefill_all_uncached(node: MockEngineNode) -> None:
    result = node.estimate_prefill_time(prompt_len=100, cached_tokens=0)
    assert result == pytest.approx(100 * 0.033)


def test_prefill_all_cached(node: MockEngineNode) -> None:
    result = node.estimate_prefill_time(prompt_len=100, cached_tokens=100)
    assert result == pytest.approx(0.0)


def test_prefill_partial_cache(node: MockEngineNode) -> None:
    result = node.estimate_prefill_time(prompt_len=200, cached_tokens=50)
    assert result == pytest.approx(150 * 0.033)


def test_prefill_cached_exceeds_prompt(node: MockEngineNode) -> None:
    # cached_tokens > prompt_len should not go negative
    result = node.estimate_prefill_time(prompt_len=50, cached_tokens=100)
    assert result == pytest.approx(0.0)


def test_prefill_zero_length(node: MockEngineNode) -> None:
    result = node.estimate_prefill_time(prompt_len=0, cached_tokens=0)
    assert result == pytest.approx(0.0)


def test_prefill_single_token(node: MockEngineNode) -> None:
    result = node.estimate_prefill_time(prompt_len=1, cached_tokens=0)
    assert result == pytest.approx(0.033)


# ------------------------------------------------------------------
# estimate_decode_time
# ------------------------------------------------------------------

def test_decode_batch_size_zero(node: MockEngineNode) -> None:
    result = node.estimate_decode_time(batch_size=0)
    assert result == pytest.approx(5.0)


def test_decode_batch_size_one(node: MockEngineNode, model_config: ModelConfig) -> None:
    result = node.estimate_decode_time(batch_size=1)
    assert result == pytest.approx(5.0 + model_config.marginal_decode_ms)


def test_decode_batch_size_large(node: MockEngineNode, model_config: ModelConfig) -> None:
    result = node.estimate_decode_time(batch_size=32)
    assert result == pytest.approx(5.0 + 32 * model_config.marginal_decode_ms)


def test_decode_scales_linearly(node: MockEngineNode, model_config: ModelConfig) -> None:
    t10 = node.estimate_decode_time(10)
    t20 = node.estimate_decode_time(20)
    # difference should equal 10 * marginal
    assert t20 - t10 == pytest.approx(10 * model_config.marginal_decode_ms)


# ------------------------------------------------------------------
# current_load
# ------------------------------------------------------------------

def test_load_initially_zero(node: MockEngineNode) -> None:
    assert node.current_load() == pytest.approx(0.0)


def test_load_after_admit(node: MockEngineNode, node_config: NodeConfig) -> None:
    node.admit("req-1")
    assert node.current_load() == pytest.approx(1 / node_config.capacity)


def test_load_at_full_capacity(node: MockEngineNode, node_config: NodeConfig) -> None:
    for i in range(node_config.capacity):
        node.admit(f"req-{i}")
    assert node.current_load() == pytest.approx(1.0)


def test_load_decreases_after_complete(node: MockEngineNode, node_config: NodeConfig) -> None:
    node.admit("req-1")
    node.admit("req-2")
    node.complete("req-1")
    assert node.current_load() == pytest.approx(1 / node_config.capacity)


# ------------------------------------------------------------------
# queue_wait_time
# ------------------------------------------------------------------

def test_queue_wait_time_empty(node: MockEngineNode) -> None:
    assert node.queue_wait_time() == pytest.approx(0.0)


def test_queue_wait_time_one_queued(node: MockEngineNode, node_config: NodeConfig) -> None:
    for i in range(node_config.capacity):
        node.admit(f"req-{i}")
    node.admit("overflow-1")
    # 32 running == capacity, 1 queued → n_blockers = 1+1 = 2; legacy = 2 * decode_base_ms
    assert node.queue_wait_time() == pytest.approx(2 * 5.0)


def test_queue_wait_time_many_queued(node: MockEngineNode, node_config: NodeConfig) -> None:
    for i in range(node_config.capacity):
        node.admit(f"req-{i}")
    for j in range(3):
        node.admit(f"overflow-{j}")
    # 32 running == capacity, 3 queued → n_blockers = 3+1 = 4; legacy = 4 * decode_base_ms
    assert node.queue_wait_time() == pytest.approx(4 * 5.0)


def test_queue_wait_time_with_args_uses_accurate_formula(
    model_config: ModelConfig
) -> None:
    nc = NodeConfig(capacity=1)
    node = MockEngineNode("n", model_config, nc)
    node.admit("req-0")   # 1 running == capacity → next request is blocked
    # decoding=0 but running=1 → bs = max(0, 1) = 1
    # step_time = decode_base_ms + 1 * marginal_decode_ms = 5.0 + 0.5 = 5.5
    per_req = (
        100 * model_config.prefill_cost_per_token_ms
        + 50 * (model_config.decode_base_ms + model_config.marginal_decode_ms)
    )
    # n_blockers = 0 (queue) + 1 = 1
    assert node.queue_wait_time(prompt_len=100, expected_output_len=50) == pytest.approx(per_req)


def test_queue_wait_time_zero_pending_returns_zero_with_args(node: MockEngineNode) -> None:
    assert node.queue_wait_time(prompt_len=100, expected_output_len=50) == pytest.approx(0.0)


def test_queue_wait_time_returns_zero_when_slot_available(node: MockEngineNode) -> None:
    node.admit("req-0")   # 1 running, capacity=32 → slot still available
    assert node.queue_wait_time() == pytest.approx(0.0)
    assert node.queue_wait_time(prompt_len=100, expected_output_len=50) == pytest.approx(0.0)


def test_queue_wait_time_legacy_counts_n_blockers(
    model_config: ModelConfig
) -> None:
    nc = NodeConfig(capacity=1)
    node = MockEngineNode("n", model_config, nc)
    node.admit("req-0")   # running (at capacity)
    node.admit("req-1")   # queued
    # n_blockers = 1 (queued) + 1 = 2; legacy = 2 * decode_base_ms
    assert node.queue_wait_time() == pytest.approx(2 * model_config.decode_base_ms)


def test_queue_wait_time_full_with_queue_typed() -> None:
    """满载 + 2 queued + typed args → n_blockers=3 × per_req_lifecycle.

    decoding=0 but running=1 → bs = max(0, 1) = 1, step_time = 5.0 + 0.5 = 5.5.
    """
    model = ModelConfig(prefill_cost_per_token_ms=0.1, decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n0", model, NodeConfig(capacity=1))
    node.admit("running-0")   # at capacity
    node.admit("queued-0")    # → queue
    node.admit("queued-1")    # → queue
    assert len(node.running_requests) == 1
    assert len(node.queue) == 2
    # n_blockers = 2 + 1 = 3; bs = max(0, 1) = 1 (1 running, no decoding yet)
    # step_time = 5.0 + 1×0.5 = 5.5
    # per_req = 128×0.1 + 32×5.5 = 12.8 + 176.0 = 188.8
    # expected = 3 × 188.8 = 566.4
    wait = node.queue_wait_time(prompt_len=128, expected_output_len=32)
    assert wait == pytest.approx(566.4)


# ------------------------------------------------------------------
# admit / complete lifecycle
# ------------------------------------------------------------------

def test_admit_up_to_capacity_fills_running(node: MockEngineNode, node_config: NodeConfig) -> None:
    for i in range(node_config.capacity):
        node.admit(f"req-{i}")
    assert len(node.running_requests) == node_config.capacity
    assert len(node.queue) == 0


def test_admit_beyond_capacity_goes_to_queue(node: MockEngineNode, node_config: NodeConfig) -> None:
    for i in range(node_config.capacity + 2):
        node.admit(f"req-{i}")
    assert len(node.running_requests) == node_config.capacity
    assert len(node.queue) == 2


def test_complete_promotes_queued_request(node: MockEngineNode, node_config: NodeConfig) -> None:
    for i in range(node_config.capacity):
        node.admit(f"req-{i}")
    node.admit("queued-0")
    node.complete("req-0")
    assert "queued-0" in node.running_requests
    assert len(node.queue) == 0


def test_complete_running_request_removes_it(node: MockEngineNode, node_config: NodeConfig) -> None:
    node.admit("req-1")
    node.complete("req-1")
    assert "req-1" not in node.running_requests
    assert node.current_load() == pytest.approx(0.0)


# ------------------------------------------------------------------
# custom config values
# ------------------------------------------------------------------

def test_custom_prefill_cost() -> None:
    cfg = ModelConfig(prefill_cost_per_token_ms=0.1, decode_base_ms=5.0)
    n = MockEngineNode("n", cfg, NodeConfig())
    assert n.estimate_prefill_time(10, 0) == pytest.approx(1.0)


def test_custom_decode_base() -> None:
    cfg = ModelConfig(prefill_cost_per_token_ms=0.033, decode_base_ms=10.0)
    n = MockEngineNode("n", cfg, NodeConfig())
    assert n.estimate_decode_time(0) == pytest.approx(10.0)


def test_custom_capacity() -> None:
    cfg = ModelConfig(prefill_cost_per_token_ms=0.033, decode_base_ms=5.0)
    nc = NodeConfig(capacity=8)
    n = MockEngineNode("n", cfg, nc)
    for i in range(8):
        n.admit(f"req-{i}")
    assert n.current_load() == pytest.approx(1.0)
    n.admit("overflow")
    assert len(n.queue) == 1


def test_custom_marginal_decode() -> None:
    cfg = ModelConfig(prefill_cost_per_token_ms=0.033, decode_base_ms=5.0, marginal_decode_ms=1.0)
    n = MockEngineNode("n", cfg, NodeConfig())
    assert n.estimate_decode_time(10) == pytest.approx(15.0)


# ------------------------------------------------------------------
# admit / complete return value tests (new: bool / str | None)
# ------------------------------------------------------------------

def test_admit_returns_true_when_room() -> None:
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=2))
    assert node.admit("r0") is True
    assert node.admit("r1") is True


def test_admit_returns_false_when_full() -> None:
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=1))
    assert node.admit("r0") is True
    assert node.admit("r1") is False
    assert node.queue == ["r1"]


def test_complete_returns_none_when_queue_empty() -> None:
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=2))
    node.admit("r0")
    assert node.complete("r0") is None


def test_complete_returns_promoted_id_when_queue_nonempty() -> None:
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=1))
    node.admit("r0")   # running
    node.admit("r1")   # queued
    result = node.complete("r0")
    assert result == "r1"
    assert "r1" in node.running_requests
    assert node.queue == []


# ------------------------------------------------------------------
# tick_batch_step (M2)
# ------------------------------------------------------------------

def test_tick_batch_step_basic() -> None:
    """3 decode streams: step_time = decode_base + 3 × marginal."""
    model = ModelConfig(prefill_cost_per_token_ms=0.033, decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n", model, NodeConfig(capacity=3))
    node.admit("r0", expected_output_len=10)
    node.admit("r1", expected_output_len=10)
    node.admit("r2", expected_output_len=10)
    node.start_decode("r0")
    node.start_decode("r1")
    node.start_decode("r2")
    next_time, completed = node.tick_batch_step(0.0)
    assert next_time == pytest.approx(5.0 + 3 * 0.5)   # 6.5 ms
    assert completed == []


def test_tick_batch_step_idle_raises() -> None:
    """tick_batch_step on a node with no decoding streams must raise RuntimeError."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=4))
    with pytest.raises(RuntimeError, match="idle"):
        node.tick_batch_step(0.0)


def test_tick_batch_step_completion() -> None:
    """Request with expected_output_len=1 completes after a single tick."""
    model = ModelConfig(decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n", model, NodeConfig(capacity=2))
    node.admit("r0", expected_output_len=1)
    node.admit("r1", expected_output_len=2)
    node.start_decode("r0")
    node.start_decode("r1")
    _, completed = node.tick_batch_step(0.0)
    assert "r0" in completed
    assert "r1" not in completed


def test_tick_batch_step_increments_all_streams() -> None:
    """Each tick advances _output_tokens by 1 for every decoding stream."""
    model = ModelConfig(decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n", model, NodeConfig(capacity=2))
    node.admit("r0", expected_output_len=5)
    node.admit("r1", expected_output_len=5)
    node.start_decode("r0")
    node.start_decode("r1")
    node.tick_batch_step(0.0)
    assert node._output_tokens["r0"] == 1
    assert node._output_tokens["r1"] == 1


def test_tick_batch_step_multiple_completions() -> None:
    """Multiple streams completing in the same tick all appear in completed_ids."""
    model = ModelConfig(decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n", model, NodeConfig(capacity=3))
    node.admit("r0", expected_output_len=1)
    node.admit("r1", expected_output_len=1)
    node.admit("r2", expected_output_len=3)
    node.start_decode("r0")
    node.start_decode("r1")
    node.start_decode("r2")
    _, completed = node.tick_batch_step(0.0)
    assert set(completed) == {"r0", "r1"}
    assert "r2" not in completed


def test_tick_batch_step_next_time_reflects_batch_size() -> None:
    """step_time (and therefore next_time) grows with batch size."""
    model = ModelConfig(decode_base_ms=5.0, marginal_decode_ms=1.0)
    node1 = MockEngineNode("n1", model, NodeConfig(capacity=1))
    node1.admit("r0", expected_output_len=5)
    node1.start_decode("r0")

    node3 = MockEngineNode("n3", model, NodeConfig(capacity=3))
    node3.admit("a0", expected_output_len=5)
    node3.admit("a1", expected_output_len=5)
    node3.admit("a2", expected_output_len=5)
    node3.start_decode("a0")
    node3.start_decode("a1")
    node3.start_decode("a2")

    t1, _ = node1.tick_batch_step(0.0)
    t3, _ = node3.tick_batch_step(0.0)
    # batch=1 → step=6ms; batch=3 → step=8ms
    assert t1 == pytest.approx(6.0)
    assert t3 == pytest.approx(8.0)


# ------------------------------------------------------------------
# admit / complete with output tracking (M2)
# ------------------------------------------------------------------

def test_admit_initializes_output_tracking() -> None:
    """admit() into running_requests initialises _output_tokens and _expected_output."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=4))
    node.admit("r0", expected_output_len=10)
    assert node._output_tokens["r0"] == 0
    assert node._expected_output["r0"] == 10


def test_admit_queued_does_not_init_output_tracking() -> None:
    """Queued requests do NOT get output tracking until promoted via init_promoted."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=1))
    node.admit("r0", expected_output_len=5)   # running
    node.admit("r1", expected_output_len=5)   # queued
    assert "r1" not in node._output_tokens
    assert "r1" not in node._expected_output


def test_complete_cleans_up_output_tracking() -> None:
    """complete() removes output-tracking state and drops from decoding set."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=4))
    node.admit("r0", expected_output_len=10)
    node.start_decode("r0")
    node.complete("r0")
    assert "r0" not in node._output_tokens
    assert "r0" not in node._expected_output
    assert "r0" not in node.decoding


def test_init_promoted_sets_output_tracking() -> None:
    """init_promoted() initialises output tracking for a formerly-queued request."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=1))
    node.admit("r0", expected_output_len=5)   # running
    node.admit("r1", expected_output_len=8)   # queued → not in dicts
    node.complete("r0")                        # promotes r1
    node.init_promoted("r1", expected_output_len=8)
    assert node._output_tokens["r1"] == 0
    assert node._expected_output["r1"] == 8


def test_start_decode_adds_to_decoding_set() -> None:
    """start_decode() adds the request to the decoding set."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=2))
    node.admit("r0", expected_output_len=5)
    assert "r0" not in node.decoding
    node.start_decode("r0")
    assert "r0" in node.decoding


def test_start_decode_raises_if_not_running() -> None:
    """start_decode() must raise RuntimeError when request is not in running_requests."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=1))
    with pytest.raises(RuntimeError, match="not in running_requests"):
        node.start_decode("not-admitted")


def test_init_promoted_raises_if_not_running() -> None:
    """init_promoted() must raise RuntimeError when request is not in running_requests."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig(capacity=1))
    with pytest.raises(RuntimeError, match="not in running_requests"):
        node.init_promoted("not-promoted", expected_output_len=5)


# ------------------------------------------------------------------
# _batch_step_in_flight guards (Critical #1)
# ------------------------------------------------------------------

def test_batch_step_in_flight_initially_false() -> None:
    node = MockEngineNode("n", ModelConfig(), NodeConfig())
    assert not node.is_batch_step_in_flight()


def test_mark_batch_step_scheduled_sets_flag() -> None:
    node = MockEngineNode("n", ModelConfig(), NodeConfig())
    node.mark_batch_step_scheduled()
    assert node.is_batch_step_in_flight()


def test_mark_batch_step_completed_clears_flag() -> None:
    node = MockEngineNode("n", ModelConfig(), NodeConfig())
    node.mark_batch_step_scheduled()
    node.mark_batch_step_completed()
    assert not node.is_batch_step_in_flight()


def test_double_schedule_raises() -> None:
    """Scheduling a batch step twice without completing the first must raise."""
    node = MockEngineNode("n", ModelConfig(), NodeConfig())
    node.mark_batch_step_scheduled()
    with pytest.raises(RuntimeError, match="double-scheduled"):
        node.mark_batch_step_scheduled()


# ------------------------------------------------------------------
# queue_wait_time with active decoding (Important #6)
# ------------------------------------------------------------------

def test_queue_wait_time_uses_actual_batch_size() -> None:
    """With 32 requests actively decoding, step_time uses bs=32, not bs=1."""
    model = ModelConfig(prefill_cost_per_token_ms=0.033, decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n0", model, NodeConfig(capacity=32))
    for i in range(32):
        node.admit(f"r{i}", expected_output_len=64)
        node.start_decode(f"r{i}")
    node.admit("queued-0")
    node.admit("queued-1")
    # n_blockers=3, bs=32, step_time=5+32*0.5=21
    # per_req = 1024*0.033 + 64*21 = 33.792 + 1344 = 1377.792
    # wait = 3 * 1377.792 = 4133.376
    wait = node.queue_wait_time(prompt_len=1024, expected_output_len=64)
    assert wait == pytest.approx(4133.376, rel=1e-4)


# ------------------------------------------------------------------
# queue_wait_time prefill-saturated (Important #1)
# ------------------------------------------------------------------

def test_queue_wait_time_prefill_saturated() -> None:
    """running=32, decoding=0: bs must be max(0,32)=32, not 0.

    Capacity is exactly full but queue is empty → n_blockers = 0+1 = 1.
    """
    model = ModelConfig(prefill_cost_per_token_ms=0.1, decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n0", model, NodeConfig(capacity=32))
    for i in range(32):
        node.admit(f"r{i}", expected_output_len=64)   # fill running; none start_decode
    assert len(node.running_requests) == 32
    assert len(node.queue) == 0   # queue is empty → n_blockers = 0+1 = 1
    # bs = max(decoding=0, running=32) = 32; step_time = 5.0 + 32*0.5 = 21.0
    # per_req = 10*0.1 + 64*21.0 = 1.0 + 1344.0 = 1345.0; wait = 1*1345 = 1345
    expected = 10 * 0.1 + 64 * (5.0 + 32 * 0.5)
    wait = node.queue_wait_time(prompt_len=10, expected_output_len=64)
    assert wait == pytest.approx(expected)


# ------------------------------------------------------------------
# tick_batch_step removes completed from decoding (Critical #1)
# ------------------------------------------------------------------

def test_tick_batch_step_removes_completed_from_decoding() -> None:
    """Critical #1: completed streams must be evicted from decoding immediately
    so a subsequent wakeup cannot advance them a second time."""
    model = ModelConfig(decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n", model, NodeConfig(capacity=2))
    node.admit("fast", expected_output_len=1)
    node.admit("slow", expected_output_len=3)
    node.start_decode("fast")
    node.start_decode("slow")
    _, completed = node.tick_batch_step(0.0)
    assert "fast" in completed
    assert "fast" not in node.decoding   # removed immediately — Critical #1
    assert "slow" in node.decoding       # still active


def test_tick_batch_step_second_tick_excludes_completed() -> None:
    """After a terminal stream is removed, the next tick must not include it."""
    model = ModelConfig(decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n", model, NodeConfig(capacity=2))
    node.admit("fast", expected_output_len=1)
    node.admit("slow", expected_output_len=3)
    node.start_decode("fast")
    node.start_decode("slow")
    node.tick_batch_step(0.0)            # fast completes, removed from decoding
    _, completed2 = node.tick_batch_step(6.0)   # bs=1 now (only slow)
    assert "fast" not in completed2      # fast is NOT re-completed


# ------------------------------------------------------------------
# tick_batch_step determinism (Important #5)
# ------------------------------------------------------------------

def test_tick_batch_step_completed_ids_are_sorted() -> None:
    """completed_ids must be in sorted order for reproducible event scheduling."""
    model = ModelConfig(decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n", model, NodeConfig(capacity=4))
    for rid in ["z", "a", "m", "b"]:
        node.admit(rid, expected_output_len=1)
        node.start_decode(rid)
    _, completed = node.tick_batch_step(0.0)
    assert completed == sorted(completed)

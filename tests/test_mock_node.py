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
    """满载 + 2 queued + typed args → n_blockers=3 × per_req_lifecycle."""
    model = ModelConfig(prefill_cost_per_token_ms=0.1, decode_base_ms=5.0, marginal_decode_ms=0.5)
    node = MockEngineNode("n0", model, NodeConfig(capacity=1))
    node.admit("running-0")   # at capacity
    node.admit("queued-0")    # → queue
    node.admit("queued-1")    # → queue
    assert len(node.running_requests) == 1
    assert len(node.queue) == 2
    # n_blockers = 2 + 1 = 3
    # per_req = 128×0.1 + 32×(5.0+0.5) = 12.8 + 176.0 = 188.8
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

"""Tests for scheduler/bidaw.py — hrrn_priority function and BidawPolicy."""
from __future__ import annotations

import pytest

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.engine.mock_node import MockEngineNode
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheLookup
from nano_kvrouter.scheduler.bidaw import BidawPolicy, hrrn_priority
from nano_kvrouter.simulator.transfer_model import NoopTransferModel


# ---------------------------------------------------------------------------
# hrrn_priority — pure function tests
# ---------------------------------------------------------------------------


def test_hrrn_equal_wait_prefers_small_kv() -> None:
    """With equal waiting_ms, smaller kv_size_blocks yields a higher priority."""
    p_small = hrrn_priority(100.0, 1)   # 1 + 100/1 = 101.0
    p_large = hrrn_priority(100.0, 10)  # 1 + 100/10 = 11.0
    assert p_small > p_large


def test_hrrn_long_wait_overtakes_small_kv() -> None:
    """A large-KV request with much longer wait beats a fresh small-KV request."""
    # size-1 just arrived (waiting 0ms), size-10 has waited 500ms.
    p_fresh_small = hrrn_priority(0.0, 1)    # 1 + 0/1 = 1.0
    p_stale_large = hrrn_priority(500.0, 10)  # 1 + 500/10 = 51.0
    assert p_stale_large > p_fresh_small


def test_hrrn_base_value_is_one() -> None:
    """hrrn_priority always returns ≥ 1.0."""
    assert hrrn_priority(0.0, 1) == 1.0
    assert hrrn_priority(0.0, 100) == 1.0


def test_hrrn_division_by_zero_guard() -> None:
    """max(1, kv_size_blocks) prevents ZeroDivisionError when kv_size=0."""
    result = hrrn_priority(100.0, 0)
    assert result == pytest.approx(1.0 + 100.0 / 1)


def test_hrrn_monotone_in_wait() -> None:
    """For fixed kv_size, priority increases with waiting_ms."""
    assert hrrn_priority(200.0, 5) > hrrn_priority(100.0, 5)


# ---------------------------------------------------------------------------
# BidawPolicy — scheduling decision tests
# ---------------------------------------------------------------------------


def _make_cfg():
    return ModelConfig(), BandwidthConfig(), NodeConfig(
        gpu_blocks=80, cpu_blocks=0, disk_blocks=4000, capacity=16
    )


def _make_nodes(count: int, prefix: str, model_cfg, node_cfg):
    return [MockEngineNode(f"{prefix}{i}", model_cfg, node_cfg) for i in range(count)]


def _make_request(req_id: str = "r0", token_count: int = 32) -> Request:
    return Request(
        request_id=req_id,
        token_ids=list(range(token_count)),
        prefix_hash="deadbeef",
        expected_output_len=16,
        arrival_time=0.0,
        slo_ttft=600.0,
        slo_tbt=50.0,
    )


def test_bidaw_policy_no_disk_hit_routes_to_ready_path() -> None:
    """BidawPolicy returns a valid routing decision when there are no disk hits.

    'Routes to ready path' means: the double-charging guard is NOT applied
    (transfer_cost_ms unchanged) and the policy does not reject the request.
    The actual ready/preparing classification happens in BidawAdmissionController,
    not here.
    """
    model_cfg, bw_cfg, node_cfg = _make_cfg()
    prefill_nodes = _make_nodes(2, "p", model_cfg, node_cfg)
    decode_nodes = _make_nodes(4, "d", model_cfg, node_cfg)
    transfer_model = NoopTransferModel()

    cm = CacheManager(
        node_ids=[n.node_id for n in decode_nodes],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
        clock=lambda: 0.0,
    )

    policy = BidawPolicy(
        model_config=model_cfg,
        bandwidth_config=bw_cfg,
        backlog_view=transfer_model,
    )

    req = _make_request("r0", token_count=32)
    decision = policy.schedule(req, prefill_nodes, decode_nodes, cm, now=0.0)

    # Should not be rejected.
    assert not decision.is_rejected
    assert decision.prefill_node is not None
    assert decision.decode_node is not None
    assert decision.estimated_ttft_ms >= 0.0


def test_bidaw_policy_round_robins_prefill() -> None:
    """BidawPolicy cycles through prefill nodes in order."""
    model_cfg, bw_cfg, node_cfg = _make_cfg()
    prefill_nodes = _make_nodes(3, "p", model_cfg, node_cfg)
    decode_nodes = _make_nodes(2, "d", model_cfg, node_cfg)
    transfer_model = NoopTransferModel()
    cm = CacheManager(
        node_ids=[n.node_id for n in decode_nodes],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
        clock=lambda: 0.0,
    )
    policy = BidawPolicy(
        model_config=model_cfg,
        bandwidth_config=bw_cfg,
        backlog_view=transfer_model,
    )
    selections = []
    for i in range(6):
        d = policy.schedule(_make_request(f"r{i}"), prefill_nodes, decode_nodes, cm, now=0.0)
        selections.append(d.prefill_node)
    # Should cycle p0 p1 p2 p0 p1 p2.
    assert selections == ["p0", "p1", "p2", "p0", "p1", "p2"]


def test_bidaw_policy_picks_least_loaded_decode() -> None:
    """BidawPolicy picks the decode node with the lowest load."""
    model_cfg, bw_cfg, node_cfg = _make_cfg()
    prefill_nodes = _make_nodes(1, "p", model_cfg, node_cfg)
    decode_nodes = _make_nodes(3, "d", model_cfg, node_cfg)
    transfer_model = NoopTransferModel()
    cm = CacheManager(
        node_ids=[n.node_id for n in decode_nodes],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
        clock=lambda: 0.0,
    )
    policy = BidawPolicy(
        model_config=model_cfg,
        bandwidth_config=bw_cfg,
        backlog_view=transfer_model,
    )
    # Artificially busy d0 and d1.
    for i in range(8):
        decode_nodes[0].admit(f"busy_d0_{i}", expected_output_len=16, prompt_len=32, uncached_tokens=32)
    for i in range(4):
        decode_nodes[1].admit(f"busy_d1_{i}", expected_output_len=16, prompt_len=32, uncached_tokens=32)
    # d2 is empty → should be selected.
    decision = policy.schedule(_make_request("r0"), prefill_nodes, decode_nodes, cm, now=0.0)
    assert decision.decode_node == "d2"


def test_bidaw_policy_zeroes_transfer_cost_ms_for_disk_hits() -> None:
    """For a pure disk hit (no CPU blocks), the guard leaves transfer_cost_ms ≈ 0.0.

    Uses a realistic transfer_cost_ms (computed from the two-hop formula) so the
    guard subtracts the full disk_load_ms, leaving only floating-point rounding.
    """
    from unittest.mock import MagicMock, patch
    model_cfg, bw_cfg, node_cfg = _make_cfg()
    prefill_nodes = _make_nodes(1, "p", model_cfg, node_cfg)
    decode_nodes = _make_nodes(1, "d", model_cfg, node_cfg)
    transfer_model = NoopTransferModel()

    # Compute a realistic transfer_cost_ms for 2 disk blocks with no CPU blocks.
    disk_n = 2
    block_bytes = model_cfg.block_size * model_cfg.kv_bytes_per_token
    realistic_disk_load_ms = (
        disk_n * block_bytes
        * (1.0 / bw_cfg.cpu_to_disk + 1.0 / bw_cfg.gpu_to_cpu)
        * 1000.0
    )

    mock_cache = MagicMock()
    mock_cache.lookup.return_value = CacheLookup(
        matched_tokens=32,
        matched_blocks_by_tier={"disk": disk_n},
        transfer_cost_ms=realistic_disk_load_ms,  # all from disk, no CPU
    )

    captured: list[CacheLookup] = []

    original_compute = __import__(
        "nano_kvrouter.scheduler.base", fromlist=["compute_est_ttft"]
    ).compute_est_ttft

    def spy_compute(p, d, req, lookup, **kw):
        captured.append(lookup)
        return original_compute(p, d, req, lookup, **kw)

    with patch("nano_kvrouter.scheduler.bidaw.compute_est_ttft", spy_compute):
        policy = BidawPolicy(
            model_config=model_cfg,
            bandwidth_config=bw_cfg,
            backlog_view=transfer_model,
        )
        policy.schedule(_make_request("r0"), prefill_nodes, decode_nodes, mock_cache, now=0.0)

    assert captured, "compute_est_ttft was not called"
    assert captured[0].transfer_cost_ms == pytest.approx(0.0, abs=1e-9), (
        "For pure disk hit, guard should reduce transfer_cost_ms to ~0"
    )


def test_bidaw_policy_no_disk_hit_preserves_transfer_cost_ms() -> None:
    """When there are no disk blocks, transfer_cost_ms is passed through unchanged."""
    from unittest.mock import MagicMock, patch
    model_cfg, bw_cfg, node_cfg = _make_cfg()
    prefill_nodes = _make_nodes(1, "p", model_cfg, node_cfg)
    decode_nodes = _make_nodes(1, "d", model_cfg, node_cfg)
    transfer_model = NoopTransferModel()

    mock_cache = MagicMock()
    mock_cache.lookup.return_value = CacheLookup(
        matched_tokens=16,
        matched_blocks_by_tier={"gpu": 1},
        transfer_cost_ms=5.0,
    )

    captured: list[CacheLookup] = []
    original_compute = __import__(
        "nano_kvrouter.scheduler.base", fromlist=["compute_est_ttft"]
    ).compute_est_ttft

    def spy_compute(p, d, req, lookup, **kw):
        captured.append(lookup)
        return original_compute(p, d, req, lookup, **kw)

    with patch("nano_kvrouter.scheduler.bidaw.compute_est_ttft", spy_compute):
        policy = BidawPolicy(
            model_config=model_cfg,
            bandwidth_config=bw_cfg,
            backlog_view=transfer_model,
        )
        policy.schedule(_make_request("r0"), prefill_nodes, decode_nodes, mock_cache, now=0.0)

    assert captured, "compute_est_ttft was not called"
    assert captured[0].transfer_cost_ms == pytest.approx(5.0), (
        "transfer_cost_ms should not be zeroed when no disk blocks"
    )


def test_bidaw_policy_zeros_only_disk_portion_of_transfer_cost() -> None:
    """Fix #1: with mixed CPU+Disk hits, only disk_load_ms is subtracted.

    Construct a CacheLookup with cpu_n=2 and disk_n=3 blocks whose
    transfer_cost_ms is the known cpu_load_ms + disk_load_ms sum.
    After the guard, compute_est_ttft must see transfer_cost_ms ≈ cpu_load_ms
    (NOT 0.0 — that would lose the CPU reload cost).
    """
    from unittest.mock import MagicMock, patch
    model_cfg, bw_cfg, node_cfg = _make_cfg()
    prefill_nodes = _make_nodes(1, "p", model_cfg, node_cfg)
    decode_nodes = _make_nodes(1, "d", model_cfg, node_cfg)
    transfer_model = NoopTransferModel()

    cpu_n = 2
    disk_n = 3
    block_bytes = model_cfg.block_size * model_cfg.kv_bytes_per_token

    # Mirror cache_manager.py:231-248 exactly.
    cpu_load_ms = cpu_n * block_bytes / bw_cfg.gpu_to_cpu * 1000.0
    disk_load_ms = (
        disk_n * block_bytes
        * (1.0 / bw_cfg.cpu_to_disk + 1.0 / bw_cfg.gpu_to_cpu)
        * 1000.0
    )
    combined_transfer_cost_ms = cpu_load_ms + disk_load_ms

    mock_cache = MagicMock()
    mock_cache.lookup.return_value = CacheLookup(
        matched_tokens=(cpu_n + disk_n) * model_cfg.block_size,
        matched_blocks_by_tier={"cpu": cpu_n, "disk": disk_n},
        transfer_cost_ms=combined_transfer_cost_ms,
    )

    captured: list[CacheLookup] = []

    original_compute = __import__(
        "nano_kvrouter.scheduler.base", fromlist=["compute_est_ttft"]
    ).compute_est_ttft

    def spy_compute(p, d, req, lookup, **kw):
        captured.append(lookup)
        return original_compute(p, d, req, lookup, **kw)

    with patch("nano_kvrouter.scheduler.bidaw.compute_est_ttft", spy_compute):
        policy = BidawPolicy(
            model_config=model_cfg,
            bandwidth_config=bw_cfg,
            backlog_view=transfer_model,
        )
        policy.schedule(_make_request("r0"), prefill_nodes, decode_nodes, mock_cache, now=0.0)

    assert captured, "compute_est_ttft was not called"
    # CPU reload cost must be preserved — only disk portion is removed.
    assert captured[0].transfer_cost_ms == pytest.approx(cpu_load_ms, rel=1e-6), (
        f"Expected cpu_load_ms={cpu_load_ms:.6f}ms to be preserved; "
        f"got {captured[0].transfer_cost_ms:.6f}ms"
    )
    assert captured[0].transfer_cost_ms > 0.0, (
        "Double-charging guard must NOT zero the CPU reload cost"
    )


def test_bidaw_policy_rejects_empty_pools() -> None:
    """BidawPolicy rejects with no_nodes_available when either pool is empty."""
    model_cfg, bw_cfg, _ = _make_cfg()
    transfer_model = NoopTransferModel()
    policy = BidawPolicy(
        model_config=model_cfg,
        bandwidth_config=bw_cfg,
        backlog_view=transfer_model,
    )
    req = _make_request("r0")

    from unittest.mock import MagicMock
    cache = MagicMock()

    d = policy.schedule(req, [], [MagicMock()], cache, now=0.0)
    assert d.is_rejected
    assert d.reject_reason == "no_nodes_available"

    d2 = policy.schedule(req, [MagicMock()], [], cache, now=0.0)
    assert d2.is_rejected

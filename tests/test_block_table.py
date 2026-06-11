from __future__ import annotations

import pytest

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request


BLOCK_SIZE = 16


def _tokens(n: int, *, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def _req(token_ids: list[int]) -> Request:
    return Request(
        request_id="lookup",
        token_ids=token_ids,
        prefix_hash="00000000",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=2000.0,
        slo_tbt=100.0,
    )


def _manager(*, gpu: int = 16, cpu: int = 0, disk: int = 0) -> CacheManager:
    return CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=gpu, cpu_blocks=cpu, disk_blocks=disk),
        bandwidth_config=BandwidthConfig(),
    )


def _assert_pool_conserved(cm: CacheManager, node_id: str = "n0") -> None:
    stats = cm._pools[node_id].stats()
    for tier_stats in stats.values():
        assert tier_stats["used"] + tier_stats["free"] == tier_stats["capacity"]


def test_materialize_full_cold_prompt_allocates_logical_block_table() -> None:
    cm = _manager(gpu=8)
    tokens = _tokens(BLOCK_SIZE * 4)

    table = cm.materialize_request("r1", tokens, "n0")

    assert table.request_id == "r1"
    assert table.node_id == "n0"
    assert len(table.block_ids) == 4
    assert table.matched_blocks == 0
    assert table.new_blocks == 4
    assert cm._pools["n0"].used("gpu") == 4
    assert all(cm._pools["n0"].pin_count(bid) == 1 for bid in table.block_ids)
    _assert_pool_conserved(cm)


def test_prefix_hit_requests_share_physical_block_ids_in_order() -> None:
    cm = _manager(gpu=8)
    prefix = _tokens(BLOCK_SIZE * 2)
    longer = prefix + _tokens(BLOCK_SIZE * 2, start=1000)

    first = cm.materialize_request("r1", prefix, "n0")
    second = cm.materialize_request("r2", longer, "n0")

    assert second.block_ids[:2] == first.block_ids
    assert second.matched_blocks == 2
    assert second.new_blocks == 2
    assert len(second.block_ids) == 4
    assert all(cm._pools["n0"].pin_count(bid) == 2 for bid in first.block_ids)


def test_release_first_shared_request_keeps_shared_blocks_pinned() -> None:
    cm = _manager(gpu=8)
    prefix = _tokens(BLOCK_SIZE * 2)
    longer = prefix + _tokens(BLOCK_SIZE * 2, start=1000)

    first = cm.materialize_request("r1", prefix, "n0")
    cm.materialize_request("r2", longer, "n0")

    cm.release_request("r1", "n0")

    assert all(cm._pools["n0"].pin_count(bid) == 1 for bid in first.block_ids)
    with pytest.raises(RuntimeError, match="pinned"):
        cm._pools["n0"].free([first.block_ids[0]])
    assert cm.lookup(_req(prefix), "n0").matched_tokens == len(prefix)


def test_release_last_request_unpins_but_keeps_cache_resident() -> None:
    cm = _manager(gpu=8)
    tokens = _tokens(BLOCK_SIZE * 3)
    table = cm.materialize_request("r1", tokens, "n0")

    cm.release_request("r1", "n0")

    assert all(cm._pools["n0"].pin_count(bid) == 0 for bid in table.block_ids)
    assert all(cm._pools["n0"].ref_count(bid) == 0 for bid in table.block_ids)
    assert cm.lookup(_req(tokens), "n0").matched_tokens == len(tokens)
    _assert_pool_conserved(cm)


def test_pinned_blocks_cannot_be_freed_or_demoted() -> None:
    cm = _manager(gpu=4, cpu=4)
    table = cm.materialize_request("r1", _tokens(BLOCK_SIZE * 2), "n0")
    bid = table.block_ids[0]

    with pytest.raises(RuntimeError, match="pinned"):
        cm._pools["n0"].free([bid])
    with pytest.raises(RuntimeError, match="pinned"):
        cm._pools["n0"].demote(bid, "gpu", "cpu")


def test_gpu_pressure_eviction_chooses_unpinned_blocks() -> None:
    cm = _manager(gpu=5, cpu=0, disk=0)
    pinned_tokens = _tokens(BLOCK_SIZE * 2, start=0)
    victim_tokens = _tokens(BLOCK_SIZE * 2, start=1000)
    incoming_tokens = _tokens(BLOCK_SIZE * 2, start=2000)

    cm.materialize_request("active", pinned_tokens, "n0")
    cm.admit(victim_tokens, "n0")
    cm.admit(incoming_tokens, "n0")

    assert cm.lookup(_req(pinned_tokens), "n0").matched_tokens == len(pinned_tokens)
    assert cm.lookup(_req(incoming_tokens), "n0").matched_tokens == len(incoming_tokens)
    assert cm.lookup(_req(victim_tokens), "n0").matched_tokens == 0
    _assert_pool_conserved(cm)


def test_all_pinned_gpu_pressure_raises_memory_error() -> None:
    cm = _manager(gpu=2, cpu=4, disk=4)
    cm.materialize_request("active", _tokens(BLOCK_SIZE * 2), "n0")

    with pytest.raises(MemoryError, match="no demotion candidates"):
        cm.admit(_tokens(BLOCK_SIZE * 2, start=1000), "n0")


def test_partial_prefix_hit_tail_allocate_logical_order() -> None:
    cm = _manager(gpu=8)
    prefix = _tokens(BLOCK_SIZE * 2)
    extended = prefix + _tokens(BLOCK_SIZE * 3, start=1000)

    prefix_table = cm.materialize_request("prefix", prefix, "n0")
    extended_table = cm.materialize_request("extended", extended, "n0")

    assert extended_table.block_ids[:2] == prefix_table.block_ids
    assert len(extended_table.block_ids) == 5
    assert extended_table.matched_blocks == 2
    assert extended_table.new_blocks == 3


def test_duplicate_materialize_raises_value_error() -> None:
    cm = _manager(gpu=4)
    tokens = _tokens(BLOCK_SIZE)
    cm.materialize_request("r1", tokens, "n0")

    with pytest.raises(ValueError, match="already materialized"):
        cm.materialize_request("r1", tokens, "n0")


def test_release_unknown_request_raises_key_error() -> None:
    cm = _manager(gpu=4)

    with pytest.raises(KeyError):
        cm.release_request("missing", "n0")


def test_pool_conservation_after_materialize_and_release_sequence() -> None:
    cm = _manager(gpu=8, cpu=4, disk=4)
    a = _tokens(BLOCK_SIZE * 2)
    b = a + _tokens(BLOCK_SIZE * 2, start=1000)

    cm.materialize_request("a", a, "n0")
    _assert_pool_conserved(cm)
    cm.materialize_request("b", b, "n0")
    _assert_pool_conserved(cm)
    cm.release_request("a", "n0")
    _assert_pool_conserved(cm)
    cm.release_request("b", "n0")
    _assert_pool_conserved(cm)

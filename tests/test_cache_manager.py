"""Tests for CacheManager — v1 GPU-only KV cache interface."""
from __future__ import annotations

import pytest

from nano_kvrouter.config import BandwidthConfig, ModelConfig, NodeConfig
from nano_kvrouter.kv_cache.cache_manager import CacheManager
from nano_kvrouter.request import Request
from nano_kvrouter.scheduler.base import CacheQuery


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16


def _req(token_ids: list[int]) -> Request:
    return Request(
        request_id="r0",
        token_ids=token_ids,
        prefix_hash="00000000",
        expected_output_len=32,
        arrival_time=0.0,
        slo_ttft=2000.0,
        slo_tbt=100.0,
    )


def _tokens(n: int, *, start: int = 0) -> list[int]:
    """Monotonically increasing token sequence of length n starting at start."""
    return list(range(start, start + n))


@pytest.fixture
def small_cluster() -> CacheManager:
    """3-node cluster with small capacities and block_size=16."""
    model_cfg = ModelConfig(block_size=BLOCK_SIZE, kv_bytes_per_token=512)
    node_cfg = NodeConfig(gpu_blocks=100, cpu_blocks=200, disk_blocks=400)
    bw_cfg = BandwidthConfig()
    return CacheManager(
        node_ids=["n0", "n1", "n2"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )


@pytest.fixture
def single_node() -> CacheManager:
    """Single-node cluster, block_size=16."""
    return CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=100),
        bandwidth_config=BandwidthConfig(),
    )


# ---------------------------------------------------------------------------
# 1. CacheQuery Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_cache_query_protocol(small_cluster: CacheManager) -> None:
    assert isinstance(small_cluster, CacheQuery)


# ---------------------------------------------------------------------------
# 2. Initial state — all nodes have zero cache hits
# ---------------------------------------------------------------------------


def test_initial_state_all_nodes_empty(small_cluster: CacheManager) -> None:
    req = _req(_tokens(160))
    for nid in ["n0", "n1", "n2"]:
        lk = small_cluster.lookup(req, nid)
        assert lk.matched_tokens == 0
        assert lk.matched_blocks_by_tier == {}
        assert lk.transfer_cost_ms == 0.0


# ---------------------------------------------------------------------------
# 3. Full cache hit after admit
# ---------------------------------------------------------------------------


def test_admit_then_lookup_full_hit(single_node: CacheManager) -> None:
    tokens = _tokens(320)  # 320 / 16 = 20 blocks
    single_node.admit(tokens, "n0")

    lk = single_node.lookup(_req(tokens), "n0")
    assert lk.matched_tokens == 320
    assert lk.matched_blocks_by_tier == {"gpu": 20}
    assert lk.transfer_cost_ms == 0.0


# ---------------------------------------------------------------------------
# 4. Partial hit — admitted prefix matches leading tokens of a longer request
# ---------------------------------------------------------------------------


def test_admit_then_lookup_partial_hit(single_node: CacheManager) -> None:
    admitted = _tokens(192)  # 12 blocks
    single_node.admit(admitted, "n0")

    # Request extends beyond what was admitted
    longer = admitted + _tokens(96, start=192)
    lk = single_node.lookup(_req(longer), "n0")
    assert lk.matched_tokens == 192
    assert lk.matched_blocks_by_tier == {"gpu": 12}


# ---------------------------------------------------------------------------
# 5. matched_tokens is aligned down to block_size boundary
# ---------------------------------------------------------------------------


def test_matched_tokens_aligned_to_block_boundary(single_node: CacheManager) -> None:
    # 200 tokens: 200 // 16 = 12 blocks = 192 aligned tokens; last 8 are dropped
    single_node.admit(_tokens(200), "n0")
    lk = single_node.lookup(_req(_tokens(200)), "n0")
    assert lk.matched_tokens == 192  # aligned, not 200
    assert lk.matched_blocks_by_tier == {"gpu": 12}


# ---------------------------------------------------------------------------
# 6. lookup_all covers every node
# ---------------------------------------------------------------------------


def test_lookup_all_covers_all_nodes(small_cluster: CacheManager) -> None:
    result = small_cluster.lookup_all(_req(_tokens(32)))
    assert set(result.keys()) == {"n0", "n1", "n2"}


# ---------------------------------------------------------------------------
# 7. Unknown node raises KeyError
# ---------------------------------------------------------------------------


def test_lookup_unknown_node_raises(small_cluster: CacheManager) -> None:
    with pytest.raises(KeyError):
        small_cluster.lookup(_req(_tokens(32)), "ghost")


# ---------------------------------------------------------------------------
# 8. free_blocks reflects pool allocation
# ---------------------------------------------------------------------------


def test_free_blocks_reflects_pool_state(single_node: CacheManager) -> None:
    initial_free = single_node.free_blocks("n0", "gpu")  # 100

    # Admit exactly 5 blocks (80 tokens)
    single_node.admit(_tokens(80), "n0")

    assert single_node.free_blocks("n0", "gpu") == initial_free - 5


# ---------------------------------------------------------------------------
# 9. free_blocks raises for unknown node or tier
# ---------------------------------------------------------------------------


def test_free_blocks_unknown_node_raises(small_cluster: CacheManager) -> None:
    with pytest.raises(KeyError):
        small_cluster.free_blocks("ghost", "gpu")


def test_free_blocks_unknown_tier_raises(small_cluster: CacheManager) -> None:
    with pytest.raises(KeyError):
        small_cluster.free_blocks("n0", "nvme")


# ---------------------------------------------------------------------------
# 10. admit triggers LRU eviction when pool is full
# ---------------------------------------------------------------------------


def test_admit_triggers_eviction_when_full() -> None:
    # Tiny pool: 10 GPU blocks
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=10, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )

    # Fill pool completely with sequence A (10 blocks = 160 tokens)
    seq_a = _tokens(160, start=0)
    cm.admit(seq_a, "n0")
    assert cm.free_blocks("n0", "gpu") == 0

    # Admit a different sequence B (10 blocks) — requires evicting A
    seq_b = _tokens(160, start=1000)
    cm.admit(seq_b, "n0")

    # Eviction happened: A no longer cached (its tree node was removed)
    assert cm.lookup(_req(seq_a), "n0").matched_tokens == 0

    # B is now fully cached
    assert cm.lookup(_req(seq_b), "n0").matched_tokens == 160

    # Pool usage is correct: back to 10 used (B fills the pool)
    assert cm.free_blocks("n0", "gpu") == 0


# ---------------------------------------------------------------------------
# 11. admit raises MemoryError when all blocks are pinned
# ---------------------------------------------------------------------------


def test_admit_raises_memory_error_when_cannot_evict() -> None:
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=2, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )

    # Fill with 2 blocks (32 tokens)
    seq_a = _tokens(32)
    cm.admit(seq_a, "n0")
    assert cm.free_blocks("n0", "gpu") == 0

    # Pin all tree nodes so evict_lru returns nothing
    for node in cm._trees["n0"]._nodes.values():
        node.ref_count = 1

    # Trying to admit a new sequence should raise MemoryError
    seq_b = _tokens(48, start=1000)  # 3 blocks, different prefix
    with pytest.raises(MemoryError):
        cm.admit(seq_b, "n0")


# ---------------------------------------------------------------------------
# 12. transfer_cost_ms is always 0.0 (v1 GPU-only invariant)
# ---------------------------------------------------------------------------


def test_transfer_cost_always_zero_in_v1(single_node: CacheManager) -> None:
    # Before any admit
    lk_cold = single_node.lookup(_req(_tokens(32)), "n0")
    assert lk_cold.transfer_cost_ms == 0.0

    # After admit with a cache hit
    single_node.admit(_tokens(32), "n0")
    lk_hit = single_node.lookup(_req(_tokens(32)), "n0")
    assert lk_hit.transfer_cost_ms == 0.0


# ---------------------------------------------------------------------------
# 13. Non-overlapping admits each consume their own distinct blocks
# ---------------------------------------------------------------------------


def test_two_non_overlapping_admits_use_distinct_blocks() -> None:
    """Two admits with entirely different token prefixes must each consume
    their own block allocation with no cross-contamination."""
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=100),
        bandwidth_config=BandwidthConfig(),
    )
    # Two non-overlapping admits: no splits possible
    cm.admit(_tokens(64, start=0), "n0")
    cm.admit(_tokens(64, start=1000), "n0")
    # Should use exactly 4 + 4 = 8 blocks
    assert cm.free_blocks("n0", "gpu") == 100 - 8

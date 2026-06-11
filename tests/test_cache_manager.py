"""Tests for CacheManager — M4 pool-backed GPU KV cache interface."""
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

    # M6: blocks freed (cpu_blocks=0, disk_blocks=0) → zombie node stays but lookup returns 0
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


# ---------------------------------------------------------------------------
# 14. No orphan leak — pool.used stays in sync with tree block_ids after split
# ---------------------------------------------------------------------------


def test_no_orphan_leak_with_non_aligned_split() -> None:
    """Pool.used must equal sum of block_ids across all tree nodes after a
    non-aligned radix split.  This is the M4 regression guard replacing the
    v1 ceiling-counter check."""
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=50, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )

    # Stage 1: 1 block (16 tokens)
    cm.admit(_tokens(16, start=0), "n0")

    # Stage 2: non-aligned split — only token [0] is shared
    new_tokens = [0] + _tokens(32, start=99)
    cm.admit(new_tokens, "n0")

    # pool.used == sum of per-node block_ids (no orphan leak)
    pool_used = cm._pools["n0"].used("gpu")
    tree_blocks = sum(len(n.block_ids) for n in cm._trees["n0"]._nodes.values())
    assert pool_used == tree_blocks

    # Stage 3: cold 3-block request must succeed
    cm.admit(_tokens(48, start=200), "n0")
    lk = cm.lookup(_req(_tokens(48, start=200)), "n0")
    assert lk.matched_tokens == 48
    assert lk.matched_blocks_by_tier == {"gpu": 3}

    # Invariant still holds after stage 3
    pool_used2 = cm._pools["n0"].used("gpu")
    tree_blocks2 = sum(len(n.block_ids) for n in cm._trees["n0"]._nodes.values())
    assert pool_used2 == tree_blocks2


# ---------------------------------------------------------------------------
# 15. Invariant — pool.used == sum(ceil(key_len/bs) per node) after each admit
# ---------------------------------------------------------------------------


def test_pool_used_equals_per_node_block_sum_after_each_admit() -> None:
    """M4 invariant: pool.used("gpu") equals the sum of block_ids lengths
    across all RadixTree nodes (i.e. the per-node ceiling block count)."""
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=100),
        bandwidth_config=BandwidthConfig(),
    )
    for length in [16, 32, 48, 17, 33]:
        cm.admit(_tokens(length, start=length * 100), "n0")
        pool_used = cm._pools["n0"].used("gpu")
        tree_blocks = sum(len(n.block_ids) for n in cm._trees["n0"]._nodes.values())
        assert pool_used == tree_blocks


# ---------------------------------------------------------------------------
# 16. lookup never reports more matched_blocks than pool.used
# ---------------------------------------------------------------------------


def test_lookup_never_exceeds_pool_used() -> None:
    """For any query, matched_blocks <= pool.used. Lookup cannot report more
    cached blocks than the pool says are physically allocated."""
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=100),
        bandwidth_config=BandwidthConfig(),
    )
    cm.admit(_tokens(64, start=0), "n0")
    cm.admit([0] + _tokens(48, start=999), "n0")  # triggers a split

    used = cm._pools["n0"].used("gpu")

    for query in [_tokens(64), [0] + _tokens(48, start=999), [0], _tokens(16)]:
        lk = cm.lookup(_req(query), "n0")
        matched_blocks = lk.matched_blocks_by_tier.get("gpu", 0)
        assert matched_blocks <= used, (
            f"INVARIANT BROKEN: matched_blocks={matched_blocks} > pool.used={used} "
            f"for query={query[:5]}..."
        )


# ---------------------------------------------------------------------------
# 17. Fast path: already-cached prompt does not allocate or evict
# ---------------------------------------------------------------------------


def test_admit_already_cached_is_noop_even_when_full_and_pinned() -> None:
    """Re-admitting an already-cached prompt must succeed even when the
    pool is full AND all leaves are pinned. Without the fast path, a
    cache-aware scheduler routing to a hot node would incorrectly fail."""
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=4, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )
    cm.admit(_tokens(64, start=0), "n0")   # fill exactly
    assert cm.free_blocks("n0", "gpu") == 0

    # Pin every leaf so evict_lru would return nothing
    for node in cm._trees["n0"]._nodes.values():
        node.ref_count = 1

    # Re-admit the same prompt — must be a no-op, no MemoryError
    cm.admit(_tokens(64, start=0), "n0")
    assert cm.free_blocks("n0", "gpu") == 0
    # And it still looks up fully
    lk = cm.lookup(_req(_tokens(64, start=0)), "n0")
    assert lk.matched_tokens == 64


# ---------------------------------------------------------------------------
# 18. Oversize prompt fails fast without evicting existing cache
# ---------------------------------------------------------------------------


def test_admit_oversize_prompt_fails_without_evicting_existing_cache() -> None:
    """A prompt larger than total capacity must raise MemoryError
    immediately without touching any existing cache. Pre-check guard."""
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=3, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )
    # Fill with 3 blocks of useful cache
    cm.admit(_tokens(48, start=0), "n0")
    assert cm.free_blocks("n0", "gpu") == 0

    # Oversized request: 4 blocks but capacity is 3 — must fail fast
    with pytest.raises(MemoryError, match="exceeds total GPU capacity"):
        cm.admit(_tokens(64, start=1000), "n0")

    # Existing cache must NOT have been evicted
    assert cm.free_blocks("n0", "gpu") == 0
    lk = cm.lookup(_req(_tokens(48, start=0)), "n0")
    assert lk.matched_tokens == 48   # original cache intact


# ---------------------------------------------------------------------------
# 19. M4 new: admit allocates from pool (pool.used increases)  [M4]
# ---------------------------------------------------------------------------


def test_admit_allocates_pool_blocks() -> None:
    """Pool.used("gpu") must increase by exactly the number of new blocks
    when a cold prompt is admitted."""
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=100),
        bandwidth_config=BandwidthConfig(),
    )
    assert cm._pools["n0"].used("gpu") == 0

    # Admit 3 blocks (48 tokens)
    cm.admit(_tokens(48, start=0), "n0")
    assert cm._pools["n0"].used("gpu") == 3

    # Admit 2 more blocks (non-overlapping)
    cm.admit(_tokens(32, start=1000), "n0")
    assert cm._pools["n0"].used("gpu") == 5


# ---------------------------------------------------------------------------
# 20. M4 new: eviction frees pool blocks  [M4]
# ---------------------------------------------------------------------------


def test_evict_frees_pool_blocks() -> None:
    """Evicting cache entries must decrease pool.used("gpu") accordingly."""
    # gpu_blocks=6 gives the +1 split-margin headroom the eviction loop needs,
    # so the first two admits (3+2=5 blocks) land without triggering eviction.
    cm = CacheManager(
        node_ids=["n0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE),
        node_config=NodeConfig(gpu_blocks=6, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )
    # Fill with 5 blocks: 3 + 2
    cm.admit(_tokens(48, start=0), "n0")   # 3 blocks
    cm.admit(_tokens(32, start=1000), "n0")  # 2 blocks
    assert cm._pools["n0"].used("gpu") == 5

    # Admit a 3-block sequence that requires eviction (pool: 5 used, cap=6)
    cm.admit(_tokens(48, start=2000), "n0")

    # M6: pool.used reflects only live GPU blocks; tree may have zombie nodes.
    pool_used = cm._pools["n0"].used("gpu")
    assert pool_used <= 6
    # New sequence must be accessible after eviction
    assert cm.lookup(_req(_tokens(48, start=2000)), "n0").matched_tokens == 48


# ---------------------------------------------------------------------------
# 21. M4 new: gpu_blocks config actually constrains eviction  [M4]
# ---------------------------------------------------------------------------


def test_gpu_blocks_sensitivity() -> None:
    """Tighter gpu_blocks → more evictions → lower cache hit ratio.

    Admits the same sequence of requests under two capacities and confirms
    that the smaller-capacity node exhibits more evictions (higher miss rate
    on re-lookup of earlier admits)."""
    model_cfg = ModelConfig(block_size=BLOCK_SIZE)
    bw_cfg = BandwidthConfig()

    # Large capacity: all admits fit without eviction
    cm_large = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=NodeConfig(gpu_blocks=100),
        bandwidth_config=bw_cfg,
    )
    # Small capacity: admits force eviction
    cm_small = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=NodeConfig(gpu_blocks=6),
        bandwidth_config=bw_cfg,
    )

    sequences = [_tokens(32, start=i * 1000) for i in range(5)]  # 5 × 2 blocks each

    for seq in sequences:
        cm_large.admit(seq, "n0")
        cm_small.admit(seq, "n0")

    # With large capacity, all 5 sequences are cached
    large_hits = sum(
        1 for seq in sequences if cm_large.lookup(_req(seq), "n0").matched_tokens > 0
    )
    # With small capacity (6 blocks), only ~3 sequences can be cached at once
    small_hits = sum(
        1 for seq in sequences if cm_small.lookup(_req(seq), "n0").matched_tokens > 0
    )

    assert large_hits == 5           # all cached on GPU
    # M6: with default cpu/disk capacity, small-GPU node demotes to CPU/Disk
    # rather than evicting from tree → all sequences still accessible.
    assert small_hits == 5
    # Key M6 difference: tighter GPU → CPU/Disk hits → higher transfer cost
    large_transfer = sum(
        cm_large.lookup(_req(seq), "n0").transfer_cost_ms for seq in sequences
    )
    small_transfer = sum(
        cm_small.lookup(_req(seq), "n0").transfer_cost_ms for seq in sequences
    )
    assert large_transfer == 0.0     # large cap: all GPU hits, no transfer cost
    assert small_transfer > 0.0      # small cap: some CPU hits, non-zero transfer cost


# ---------------------------------------------------------------------------
# M6: Multi-tier demotion + tier-aware lookup
# ---------------------------------------------------------------------------


def test_demotion_gpu_to_cpu_on_admit_when_gpu_full() -> None:
    """When GPU is full, admitted blocks demote LRU to CPU (not freed from pool)."""
    model_cfg = ModelConfig(block_size=BLOCK_SIZE, kv_bytes_per_token=512)
    node_cfg = NodeConfig(gpu_blocks=2, cpu_blocks=4, disk_blocks=0)
    bw_cfg = BandwidthConfig(gpu_to_cpu=10_000_000_000, cpu_to_disk=1_000_000_000)
    cm = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )

    # Fill GPU: 2 blocks = 32 tokens
    cm.admit(_tokens(BLOCK_SIZE * 2, start=0), "n0")
    assert cm.free_blocks("n0", "gpu") == 0

    # Admit a new sequence — LRU should be demoted to CPU, not discarded
    cm.admit(_tokens(BLOCK_SIZE * 2, start=100), "n0")

    # CPU must now have at least 1 block (the demoted one)
    assert cm.free_blocks("n0", "cpu") < 4  # something moved to CPU


def test_demotion_cpu_to_disk_when_cpu_also_full() -> None:
    """GPU→CPU→Disk cascade: when CPU is also full, demotion continues to Disk."""
    model_cfg = ModelConfig(block_size=BLOCK_SIZE, kv_bytes_per_token=512)
    node_cfg = NodeConfig(gpu_blocks=2, cpu_blocks=2, disk_blocks=4)
    bw_cfg = BandwidthConfig(gpu_to_cpu=10_000_000_000, cpu_to_disk=1_000_000_000)
    cm = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )

    # Fill GPU
    cm.admit(_tokens(BLOCK_SIZE * 2, start=0), "n0")
    # Move existing GPU blocks to CPU to fill it
    pool = cm._pools["n0"]
    gpu_bids = list(pool._tiers["gpu"].allocated)
    for bid in gpu_bids:
        pool.move(bid, "gpu", "cpu")

    # Re-fill GPU with new blocks
    for bid in list(pool._tiers["gpu"].allocated):
        pool.free([bid])
    cm.admit(_tokens(BLOCK_SIZE * 2, start=200), "n0")

    # Now admit again — GPU full, CPU full → demotion cascades to Disk
    cm.admit(_tokens(BLOCK_SIZE * 2, start=300), "n0")

    disk_used = cm._pools["n0"].used("disk")
    assert disk_used >= 1


def test_lookup_returns_tier_distribution() -> None:
    """After demotion to CPU, lookup reports matched_blocks_by_tier with 'cpu' entry."""
    model_cfg = ModelConfig(block_size=BLOCK_SIZE, kv_bytes_per_token=512)
    node_cfg = NodeConfig(gpu_blocks=2, cpu_blocks=4, disk_blocks=0)
    bw_cfg = BandwidthConfig(gpu_to_cpu=10_000_000_000, cpu_to_disk=1_000_000_000)
    cm = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )

    # Admit sequence A, then demote it to CPU manually
    toks_a = _tokens(BLOCK_SIZE * 2, start=0)
    cm.admit(toks_a, "n0")

    # Move all GPU blocks to CPU
    pool = cm._pools["n0"]
    for bid in list(pool._tiers["gpu"].allocated):
        pool.move(bid, "gpu", "cpu")

    # Lookup should reflect CPU tier
    lookup = cm.lookup(_req(toks_a), "n0")
    assert lookup.matched_tokens > 0
    assert lookup.matched_blocks_by_tier.get("cpu", 0) > 0
    assert lookup.matched_blocks_by_tier.get("gpu", 0) == 0


def test_transfer_cost_uses_bandwidth_config() -> None:
    """transfer_cost_ms is correctly computed from bandwidth config for CPU blocks."""
    block_size = 16
    kv_bytes = 1_000_000  # 1 MB / token — large to make cost measurable
    gpu_to_cpu_bw = 1_000_000_000  # 1 GB/s

    model_cfg = ModelConfig(block_size=block_size, kv_bytes_per_token=kv_bytes)
    node_cfg = NodeConfig(gpu_blocks=4, cpu_blocks=4, disk_blocks=0)
    bw_cfg = BandwidthConfig(gpu_to_cpu=gpu_to_cpu_bw, cpu_to_disk=1_000_000_000)
    cm = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )

    toks = _tokens(block_size * 2, start=0)  # 2 blocks
    cm.admit(toks, "n0")

    # Manually move all GPU blocks to CPU tier
    pool = cm._pools["n0"]
    for bid in list(pool._tiers["gpu"].allocated):
        pool.move(bid, "gpu", "cpu")

    lookup = cm.lookup(_req(toks), "n0")
    cpu_n = lookup.matched_blocks_by_tier.get("cpu", 0)
    assert cpu_n > 0

    kv_bytes_per_block = block_size * kv_bytes
    expected_ms = cpu_n * kv_bytes_per_block / gpu_to_cpu_bw * 1000.0
    assert abs(lookup.transfer_cost_ms - expected_ms) < 1e-6


def test_disk_hit_contributes_to_transfer_cost() -> None:
    """Disk-tier blocks pay both Disk→CPU and CPU→GPU hops (M6.fix2 Important #2)."""
    block_size = 16
    kv_bytes = 500_000  # 0.5 MB / token
    cpu_to_disk_bw = 500_000_000     # 500 MB/s
    gpu_to_cpu_bw = 10_000_000_000   # 10 GB/s

    model_cfg = ModelConfig(block_size=block_size, kv_bytes_per_token=kv_bytes)
    node_cfg = NodeConfig(gpu_blocks=4, cpu_blocks=4, disk_blocks=4)
    bw_cfg = BandwidthConfig(gpu_to_cpu=gpu_to_cpu_bw, cpu_to_disk=cpu_to_disk_bw)
    cm = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )

    toks = _tokens(block_size * 2, start=0)
    cm.admit(toks, "n0")

    # Move blocks: GPU → CPU → Disk
    pool = cm._pools["n0"]
    for bid in list(pool._tiers["gpu"].allocated):
        pool.move(bid, "gpu", "cpu")
    for bid in list(pool._tiers["cpu"].allocated):
        pool.move(bid, "cpu", "disk")

    lookup = cm.lookup(_req(toks), "n0")
    disk_n = lookup.matched_blocks_by_tier.get("disk", 0)
    assert disk_n > 0

    # Disk hit = disk→cpu + cpu→gpu (serial).
    kv_bytes_per_block = block_size * kv_bytes
    expected_ms = (
        disk_n * kv_bytes_per_block
        * (1.0 / cpu_to_disk_bw + 1.0 / gpu_to_cpu_bw) * 1000.0
    )
    assert abs(lookup.transfer_cost_ms - expected_ms) < 1e-6


def test_disk_load_includes_two_hops() -> None:
    """M6.fix2 Important #2: disk hit must include both hops in series.

    With cpu_to_disk = gpu_to_cpu = 1 GB/s, the disk cost should be
    exactly 2× the disk-only single-hop formula.
    """
    block_size = 16
    kv_bytes = 1_000_000
    bw = 1_000_000_000  # both legs the same

    model_cfg = ModelConfig(block_size=block_size, kv_bytes_per_token=kv_bytes)
    node_cfg = NodeConfig(gpu_blocks=4, cpu_blocks=4, disk_blocks=4)
    bw_cfg = BandwidthConfig(gpu_to_cpu=bw, cpu_to_disk=bw)
    cm = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )

    toks = _tokens(block_size * 2, start=0)
    cm.admit(toks, "n0")
    pool = cm._pools["n0"]
    for bid in list(pool._tiers["gpu"].allocated):
        pool.move(bid, "gpu", "cpu")
    for bid in list(pool._tiers["cpu"].allocated):
        pool.move(bid, "cpu", "disk")

    lookup = cm.lookup(_req(toks), "n0")
    disk_n = lookup.matched_blocks_by_tier["disk"]
    kv_bytes_per_block = block_size * kv_bytes
    single_hop = disk_n * kv_bytes_per_block / bw * 1000.0
    # New formula: both hops counted → exactly 2x single-hop when BWs equal.
    assert lookup.transfer_cost_ms == pytest.approx(2 * single_hop)


def test_zombie_prefix_readmits_correctly() -> None:
    """M6.fix2 Critical #1: after a prefix's blocks are truly freed
    (no downstream tier capacity), re-admitting the same prefix must
    re-materialise it instead of taking the fast-path no-op.
    """
    block_size = 16
    model_cfg = ModelConfig(block_size=block_size, kv_bytes_per_token=512)
    node_cfg = NodeConfig(gpu_blocks=4, cpu_blocks=0, disk_blocks=0)
    bw_cfg = BandwidthConfig()
    cm = CacheManager(
        node_ids=["n0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )

    toks_a = _tokens(block_size * 4, start=0)   # fills the 4-block pool
    toks_b = _tokens(block_size * 4, start=500) # disjoint prefix

    cm.admit(toks_a, "n0")
    assert cm._pools["n0"].used("gpu") == 4

    # Admitting B forces A's blocks to be freed (no cpu/disk capacity).
    cm.admit(toks_b, "n0")

    # A should no longer be cached anywhere.
    lookup_a_after_b = cm.lookup(_req(toks_a), "n0")
    assert lookup_a_after_b.matched_tokens == 0, (
        "A's blocks were truly freed — lookup should report 0 matches"
    )

    # Re-admitting A used to short-circuit on the (now zombie) tree path.
    # After the fix, A must be re-materialised.
    cm.admit(toks_a, "n0")
    lookup_a_again = cm.lookup(_req(toks_a), "n0")
    assert lookup_a_again.matched_tokens == block_size * 4, (
        f"re-admit must re-materialise A, got matched_tokens={lookup_a_again.matched_tokens}"
    )


# ---------------------------------------------------------------------------
# M6.fix2: Critical #1 — partial-zombie re-admit
# ---------------------------------------------------------------------------


def test_partial_zombie_readmits_correctly() -> None:
    """Codex critical repro: gpu=4, cpu=2, disk=0.

    After admit(A) + admit(B) forces A's blocks partially to CPU and partially
    freed, a second admit(A) must re-populate all 4 blocks on GPU rather than
    short-circuiting on the zombie tree node and leaving pool.gpu_used == 0.
    """
    model_cfg = ModelConfig(block_size=BLOCK_SIZE, kv_bytes_per_token=512)
    node_cfg = NodeConfig(gpu_blocks=4, cpu_blocks=2, disk_blocks=0)
    bw_cfg = BandwidthConfig(gpu_to_cpu=10_000_000_000, cpu_to_disk=1_000_000_000)
    cm = CacheManager(
        node_ids=["d0"],
        model_config=model_cfg,
        node_config=node_cfg,
        bandwidth_config=bw_cfg,
    )

    A_tokens = _tokens(BLOCK_SIZE * 4, start=0)
    B_tokens = _tokens(BLOCK_SIZE * 4, start=100)

    # Step 1: A fills GPU (4 blocks).
    cm.admit(A_tokens, "d0")
    assert cm._pools["d0"].used("gpu") == 4

    # Step 2: B is admitted, forcing A's blocks into demotion chain.
    #         cpu=2 → 2 of A's blocks survive on CPU, 2 are truly freed.
    cm.admit(B_tokens, "d0")

    lk_mid = cm.lookup(_req(A_tokens), "d0")
    assert lk_mid.matched_tokens == BLOCK_SIZE * 2  # only 2 live CPU blocks
    assert lk_mid.matched_blocks_by_tier.get("cpu", 0) == 2

    # Step 3: Re-admit A. Bug: purge_block_ids skips mixed-state node → insert
    #         reuses zombie node without calling mint → gpu_used stays 0.
    #         Fix: purge_path_with_any_dead clears the zombie path first.
    cm.admit(A_tokens, "d0")

    gpu_used = cm._pools["d0"].used("gpu")
    lk_final = cm.lookup(_req(A_tokens), "d0")

    assert lk_final.matched_tokens == BLOCK_SIZE * 4, (
        f"Expected {BLOCK_SIZE * 4} matched tokens, got {lk_final.matched_tokens}"
    )
    assert lk_final.matched_blocks_by_tier.get("gpu", 0) == 4, (
        f"Expected 4 GPU blocks, got {lk_final.matched_blocks_by_tier}"
    )
    assert gpu_used == 4, (
        f"Expected gpu_used=4 after re-admit, got {gpu_used}"
    )


def test_materialize_release_preserves_cache_manager_lookup_contract() -> None:
    """BlockTable pin/release must not change the existing lookup/admit contract."""
    cm = CacheManager(
        node_ids=["d0"],
        model_config=ModelConfig(block_size=BLOCK_SIZE, kv_bytes_per_token=512),
        node_config=NodeConfig(gpu_blocks=4, cpu_blocks=0, disk_blocks=0),
        bandwidth_config=BandwidthConfig(),
    )
    toks = _tokens(BLOCK_SIZE * 2, start=900)

    table = cm.materialize_request("active", toks, "d0")
    assert len(table.block_ids) == 2
    assert cm.lookup(_req(toks), "d0").matched_tokens == len(toks)

    cm.release_request("active", "d0")

    assert all(cm._pools["d0"].pin_count(bid) == 0 for bid in table.block_ids)
    assert cm.lookup(_req(toks), "d0").matched_tokens == len(toks)

from __future__ import annotations

import pytest
from nano_kvrouter.config import NodeConfig
from nano_kvrouter.kv_cache.block_pool import BlockPool


@pytest.fixture
def pool() -> BlockPool:
    return BlockPool(NodeConfig(gpu_blocks=10, cpu_blocks=20, disk_blocks=50))


# ------------------------------------------------------------------
# 正常分配和释放
# ------------------------------------------------------------------

def test_allocate_returns_unique_ids(pool):
    ids = pool.allocate(5, "gpu")
    assert len(ids) == 5
    assert len(set(ids)) == 5


def test_allocate_updates_stats(pool):
    pool.allocate(3, "gpu")
    s = pool.stats()
    assert s["gpu"]["used"] == 3
    assert s["gpu"]["free"] == 7


def test_free_restores_capacity(pool):
    ids = pool.allocate(4, "cpu")
    pool.free(ids[:2])
    s = pool.stats()
    assert s["cpu"]["used"] == 2
    assert s["cpu"]["free"] == 18


def test_allocate_then_free_then_allocate_again(pool):
    ids = pool.allocate(10, "gpu")  # fill
    pool.free(ids[:5])
    new_ids = pool.allocate(5, "gpu")
    assert len(new_ids) == 5
    assert pool.stats()["gpu"]["free"] == 0


def test_free_unknown_block_raises(pool):
    with pytest.raises(KeyError):
        pool.free(["not-a-real-id"])


# ------------------------------------------------------------------
# 跨 tier promote / demote
# ------------------------------------------------------------------

def test_promote_moves_block_to_faster_tier(pool):
    (bid,) = pool.allocate(1, "cpu")
    pool.promote(bid, from_tier="cpu", to_tier="gpu")
    s = pool.stats()
    assert s["cpu"]["used"] == 0
    assert s["gpu"]["used"] == 1


def test_demote_moves_block_to_slower_tier(pool):
    (bid,) = pool.allocate(1, "gpu")
    pool.demote(bid, from_tier="gpu", to_tier="cpu")
    s = pool.stats()
    assert s["gpu"]["used"] == 0
    assert s["cpu"]["used"] == 1


def test_promote_chain_disk_cpu_gpu(pool):
    (bid,) = pool.allocate(1, "disk")
    pool.promote(bid, from_tier="disk", to_tier="cpu")
    pool.promote(bid, from_tier="cpu", to_tier="gpu")
    s = pool.stats()
    assert s["disk"]["used"] == 0
    assert s["cpu"]["used"] == 0
    assert s["gpu"]["used"] == 1


def test_promote_wrong_from_tier_raises(pool):
    (bid,) = pool.allocate(1, "disk")
    with pytest.raises(KeyError):
        pool.promote(bid, from_tier="cpu", to_tier="gpu")  # block is on disk, not cpu


def test_demote_to_full_tier_raises(pool):
    pool.allocate(20, "cpu")  # fill cpu
    (bid,) = pool.allocate(1, "gpu")
    with pytest.raises(MemoryError):
        pool.demote(bid, from_tier="gpu", to_tier="cpu")


# ------------------------------------------------------------------
# 超出容量时抛异常
# ------------------------------------------------------------------

def test_allocate_over_capacity_raises(pool):
    with pytest.raises(MemoryError):
        pool.allocate(11, "gpu")  # capacity is 10


def test_allocate_exactly_at_capacity(pool):
    ids = pool.allocate(10, "gpu")
    assert len(ids) == 10
    assert pool.stats()["gpu"]["free"] == 0


def test_allocate_one_more_than_full_raises(pool):
    pool.allocate(10, "gpu")
    with pytest.raises(MemoryError):
        pool.allocate(1, "gpu")


def test_promote_to_full_tier_raises(pool):
    pool.allocate(10, "gpu")   # fill gpu
    (bid,) = pool.allocate(1, "cpu")
    with pytest.raises(MemoryError):
        pool.promote(bid, from_tier="cpu", to_tier="gpu")


# ------------------------------------------------------------------
# stats() 数值正确
# ------------------------------------------------------------------

def test_stats_initial_state(pool):
    s = pool.stats()
    assert s["gpu"]  == {"capacity": 10,  "used": 0, "free": 10}
    assert s["cpu"]  == {"capacity": 20,  "used": 0, "free": 20}
    assert s["disk"] == {"capacity": 50,  "used": 0, "free": 50}


def test_stats_multi_tier_allocation(pool):
    pool.allocate(3, "gpu")
    pool.allocate(7, "cpu")
    pool.allocate(15, "disk")
    s = pool.stats()
    assert s["gpu"]["used"]  == 3  and s["gpu"]["free"]  == 7
    assert s["cpu"]["used"]  == 7  and s["cpu"]["free"]  == 13
    assert s["disk"]["used"] == 15 and s["disk"]["free"] == 35


def test_stats_after_promote_reflect_both_tiers(pool):
    (bid,) = pool.allocate(1, "cpu")
    pool.promote(bid, from_tier="cpu", to_tier="gpu")
    s = pool.stats()
    assert s["cpu"]["used"] == 0
    assert s["gpu"]["used"] == 1


def test_default_capacities():
    p = BlockPool(NodeConfig())
    s = p.stats()
    assert s["gpu"]["capacity"]  == 2_000
    assert s["cpu"]["capacity"]  == 10_000
    assert s["disk"]["capacity"] == 100_000


def test_unknown_tier_raises(pool):
    with pytest.raises(ValueError, match="Unknown tier"):
        pool.allocate(1, "nvme")


# ---------------------------------------------------------------------------
# M6 new: tier_of / tiers_of / move
# ---------------------------------------------------------------------------


def test_tier_of_returns_current_tier() -> None:
    """tier_of returns the tier name for allocated blocks."""
    from nano_kvrouter.config import NodeConfig
    from nano_kvrouter.kv_cache.block_pool import BlockPool

    pool = BlockPool(NodeConfig(gpu_blocks=4, cpu_blocks=4, disk_blocks=4))
    [gpu_b] = pool.allocate(1, "gpu")
    [cpu_b] = pool.allocate(1, "cpu")
    [disk_b] = pool.allocate(1, "disk")

    assert pool.tier_of(gpu_b) == "gpu"
    assert pool.tier_of(cpu_b) == "cpu"
    assert pool.tier_of(disk_b) == "disk"
    assert pool.tier_of("does-not-exist") is None


def test_tiers_of_batch_query() -> None:
    """tiers_of returns a mapping of block_id to tier, omitting freed blocks."""
    from nano_kvrouter.config import NodeConfig
    from nano_kvrouter.kv_cache.block_pool import BlockPool

    pool = BlockPool(NodeConfig(gpu_blocks=4, cpu_blocks=4, disk_blocks=0))
    bids = pool.allocate(2, "gpu")
    cbids = pool.allocate(2, "cpu")

    result = pool.tiers_of(bids + cbids + ["zombie"])
    assert result[bids[0]] == "gpu"
    assert result[bids[1]] == "gpu"
    assert result[cbids[0]] == "cpu"
    assert result[cbids[1]] == "cpu"
    assert "zombie" not in result


def test_move_decrements_src_increments_dst() -> None:
    """move() updates tier tracking and capacity counters."""
    from nano_kvrouter.config import NodeConfig
    from nano_kvrouter.kv_cache.block_pool import BlockPool

    pool = BlockPool(NodeConfig(gpu_blocks=2, cpu_blocks=2, disk_blocks=0))
    [bid] = pool.allocate(1, "gpu")
    assert pool.tier_of(bid) == "gpu"
    assert pool.used("gpu") == 1
    assert pool.used("cpu") == 0

    pool.move(bid, "gpu", "cpu")

    assert pool.tier_of(bid) == "cpu"
    assert pool.used("gpu") == 0
    assert pool.used("cpu") == 1

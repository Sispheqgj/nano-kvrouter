import time
import pytest
from nano_kvrouter.kv_cache.radix_tree import RadixTree


# ---------------------------------------------------------------------------
# 1. 完全不命中
# ---------------------------------------------------------------------------

def test_no_match_returns_zero_and_root():
    tree = RadixTree()
    tree.insert([1, 2, 3])
    matched, bid = tree.match_prefix([4, 5, 6])
    assert matched == 0
    assert bid == "root"


def test_partial_edge_counts_as_no_match():
    """A prefix that matches only part of a single edge (not a full node)
    should return 0, because the node's block cannot be used."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5])
    matched, bid = tree.match_prefix([1, 2, 3])
    assert matched == 0
    assert bid == "root"


# ---------------------------------------------------------------------------
# 2. 部分命中（前缀共享）
# ---------------------------------------------------------------------------

def test_partial_match_shared_prefix():
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5])
    tree.insert([1, 2, 3, 6, 7])  # forces a split — creates mid node at [1,2,3]

    matched, bid = tree.match_prefix([1, 2, 3, 9, 9])
    assert matched == 3
    assert bid != "root"


def test_partial_match_longer_query():
    tree = RadixTree()
    tree.insert([10, 20, 30])
    tree.insert([10, 20, 30, 40, 50])
    # Query that extends beyond the shorter inserted sequence
    matched, _ = tree.match_prefix([10, 20, 30, 40, 99])
    assert matched == 3  # only [10,20,30] is a verified node boundary


# ---------------------------------------------------------------------------
# 3. 完全命中
# ---------------------------------------------------------------------------

def test_full_match_returns_correct_length_and_block_id():
    tree = RadixTree()
    bid_insert = tree.insert([10, 20, 30, 40])
    matched, bid_match = tree.match_prefix([10, 20, 30, 40])
    assert matched == 4
    assert bid_match == bid_insert


def test_full_match_same_block_id_on_reinsert():
    tree = RadixTree()
    bid1 = tree.insert([1, 2, 3])
    bid2 = tree.insert([1, 2, 3])
    assert bid1 == bid2


def test_full_match_updates_access_time():
    tree = RadixTree()
    bid = tree.insert([5, 6, 7])
    t_before = tree._nodes[bid].last_access_time
    time.sleep(0.02)
    tree.match_prefix([5, 6, 7])
    assert tree._nodes[bid].last_access_time > t_before


# ---------------------------------------------------------------------------
# 4. LRU 驱逐顺序正确
# ---------------------------------------------------------------------------

def test_lru_eviction_order():
    tree = RadixTree()
    bid1 = tree.insert([1, 2, 3])   # oldest
    time.sleep(0.01)
    bid2 = tree.insert([4, 5, 6])   # middle
    time.sleep(0.01)
    bid3 = tree.insert([7, 8, 9])   # newest

    evicted = tree.evict_lru(2)
    assert len(evicted) == 2
    assert bid1 in evicted
    assert bid2 in evicted
    assert bid3 not in evicted


def test_lru_eviction_survivor_still_matchable():
    tree = RadixTree()
    bid1 = tree.insert([1, 2, 3])
    time.sleep(0.01)
    bid3 = tree.insert([7, 8, 9])

    tree.evict_lru(1)  # removes bid1

    matched, bid = tree.match_prefix([7, 8, 9])
    assert matched == 3
    assert bid == bid3


def test_lru_eviction_respects_ref_count():
    tree = RadixTree()
    bid = tree.insert([1, 2, 3])
    tree._nodes[bid].ref_count = 1   # simulate in-use

    evicted = tree.evict_lru(1)
    assert bid not in evicted


def test_lru_eviction_leaves_before_internal_nodes():
    """Internal nodes (with children) must not be evicted before their leaves."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5])
    time.sleep(0.01)
    tree.insert([1, 2, 3, 6, 7])
    # Tree: root → mid([1,2,3]) → leaf1([4,5])
    #                           → leaf2([6,7])

    evicted = tree.evict_lru(1)
    assert len(evicted) == 1

    # mid node ([1,2,3]) still alive — it had a child remaining
    matched, _ = tree.match_prefix([1, 2, 3])
    assert matched == 3


def test_lru_evict_more_than_available():
    tree = RadixTree()
    tree.insert([1, 2])

    evicted = tree.evict_lru(10)   # only 1 node exists
    assert len(evicted) == 1


# ---------------------------------------------------------------------------
# 5. 可注入仿真时钟
# ---------------------------------------------------------------------------

def test_custom_clock_drives_access_time():
    """注入 fake clock 后 last_access_time 应由 clock 决定而非 wall-clock。"""
    times = iter([10.0, 20.0, 30.0])
    tree = RadixTree(clock=lambda: next(times))

    bid1 = tree.insert([1, 2, 3])
    bid2 = tree.insert([4, 5, 6])
    bid3 = tree.insert([7, 8, 9])

    assert tree._nodes[bid1].last_access_time == 10.0
    assert tree._nodes[bid2].last_access_time == 20.0
    assert tree._nodes[bid3].last_access_time == 30.0


def test_default_clock_is_wall_clock():
    """不传 clock 时，last_access_time 应约等于 time.time()（容差 1 秒）。"""
    t_before = time.time()
    tree = RadixTree()
    bid = tree.insert([100, 200, 300])
    t_after = time.time()

    assert t_before <= tree._nodes[bid].last_access_time <= t_after + 1.0



# ---------------------------------------------------------------------------
# 6. evict_lru_with_lengths — returns token counts not block_ids
# ---------------------------------------------------------------------------

def test_evict_lru_with_lengths_returns_token_counts():
    tree = RadixTree()
    tree.insert([1, 2, 3])      # key_len=3
    time.sleep(0.01)
    tree.insert([4, 5, 6, 7])   # key_len=4
    time.sleep(0.01)
    tree.insert([8, 9])         # key_len=2

    # Evict the 2 oldest leaves; their key lengths must be returned.
    lengths = tree.evict_lru_with_lengths(2)
    assert sorted(lengths) == [3, 4]

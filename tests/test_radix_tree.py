import time
import pytest
from nano_kvrouter.kv_cache.radix_tree import RadixTree


# ---------------------------------------------------------------------------
# 1. 完全不命中
# ---------------------------------------------------------------------------

def test_no_match_returns_zero_and_empty():
    tree = RadixTree()
    tree.insert([1, 2, 3])
    matched, bids = tree.match_prefix([4, 5, 6])
    assert matched == 0
    assert bids == []  # root has no block_ids


def test_partial_edge_counts_as_no_match():
    """A prefix that matches only part of a single edge (not a full node)
    should return 0, because the node's block cannot be used."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5])
    matched, bids = tree.match_prefix([1, 2, 3])
    assert matched == 0
    assert bids == []


# ---------------------------------------------------------------------------
# 2. 部分命中（前缀共享）
# ---------------------------------------------------------------------------

def test_partial_match_shared_prefix():
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5])
    tree.insert([1, 2, 3, 6, 7])  # forces a split — creates mid node at [1,2,3]

    matched, bids = tree.match_prefix([1, 2, 3, 9, 9])
    assert matched == 3
    assert isinstance(bids, list) and len(bids) >= 1


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

def test_full_match_returns_correct_length_and_block_ids():
    tree = RadixTree()
    bids_insert = tree.insert([10, 20, 30, 40])
    matched, bids_match = tree.match_prefix([10, 20, 30, 40])
    assert matched == 4
    assert bids_match == bids_insert


def test_full_match_same_block_ids_on_reinsert():
    tree = RadixTree()
    bids1 = tree.insert([1, 2, 3])
    bids2 = tree.insert([1, 2, 3])
    assert bids1 == bids2


def test_full_match_updates_access_time():
    tree = RadixTree()
    bids = tree.insert([5, 6, 7])
    t_before = tree._nodes[bids[0]].last_access_time
    time.sleep(0.02)
    tree.match_prefix([5, 6, 7])
    assert tree._nodes[bids[0]].last_access_time > t_before


# ---------------------------------------------------------------------------
# 4. LRU 驱逐顺序正确
# ---------------------------------------------------------------------------

def test_lru_eviction_order():
    tree = RadixTree()
    bids1 = tree.insert([1, 2, 3])   # oldest
    time.sleep(0.01)
    bids2 = tree.insert([4, 5, 6])   # middle
    time.sleep(0.01)
    bids3 = tree.insert([7, 8, 9])   # newest

    evicted = tree.evict_lru(2)
    assert len(evicted) == 2
    assert bids1 in evicted
    assert bids2 in evicted
    assert bids3 not in evicted


def test_lru_eviction_survivor_still_matchable():
    tree = RadixTree()
    tree.insert([1, 2, 3])
    time.sleep(0.01)
    bids3 = tree.insert([7, 8, 9])

    tree.evict_lru(1)  # removes [1,2,3]

    matched, bids = tree.match_prefix([7, 8, 9])
    assert matched == 3
    assert bids == bids3


def test_lru_eviction_respects_ref_count():
    tree = RadixTree()
    bids = tree.insert([1, 2, 3])
    tree._nodes[bids[0]].ref_count = 1   # simulate in-use

    evicted = tree.evict_lru(1)
    assert bids not in evicted


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

    bids1 = tree.insert([1, 2, 3])
    bids2 = tree.insert([4, 5, 6])
    bids3 = tree.insert([7, 8, 9])

    assert tree._nodes[bids1[0]].last_access_time == 10.0
    assert tree._nodes[bids2[0]].last_access_time == 20.0
    assert tree._nodes[bids3[0]].last_access_time == 30.0


def test_default_clock_is_wall_clock():
    """不传 clock 时，last_access_time 应约等于 time.time()（容差 1 秒）。"""
    t_before = time.time()
    tree = RadixTree()
    bids = tree.insert([100, 200, 300])
    t_after = time.time()

    assert t_before <= tree._nodes[bids[0]].last_access_time <= t_after + 1.0


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


# ---------------------------------------------------------------------------
# 7. block_ids length = ceil(len(key) / block_size)  [M4 新增]
# ---------------------------------------------------------------------------

def test_insert_mints_one_block_for_single_block_key():
    """With block_size=16, a key of 16 tokens → exactly 1 block_id."""
    tree = RadixTree(block_size=16)
    bids = tree.insert(list(range(16)))
    assert len(bids) == 1


def test_insert_mints_multiple_blocks_for_multi_block_key():
    """With block_size=16, a key of 48 tokens → 3 block_ids."""
    tree = RadixTree(block_size=16)
    bids = tree.insert(list(range(48)))
    assert len(bids) == 3


def test_insert_mints_ceiling_blocks_for_partial_key():
    """With block_size=16, a key of 20 tokens → ceil(20/16)=2 block_ids."""
    tree = RadixTree(block_size=16)
    bids = tree.insert(list(range(20)))
    assert len(bids) == 2


def test_split_mid_node_gets_correct_block_count():
    """After a split at a block boundary, mid-node has 1 block_id."""
    tree = RadixTree(block_size=16)
    # Insert A: 32 tokens → 1 node with 2 blocks
    tree.insert(list(range(32)))
    # Insert B: shares first 16 tokens, diverges → split creates mid(16 tokens)
    tree.insert(list(range(16)) + list(range(100, 116)))

    # mid-node is an internal node (has 2 children)
    internal = [n for n in tree._nodes.values() if n.children]
    assert len(internal) == 1
    mid = internal[0]
    assert len(mid.key) == 16
    assert len(mid.block_ids) == 1  # ceil(16/16)=1


def test_split_block_aligned_no_free():
    """Block-aligned split: block_ids are reassigned, free_blocks callback never called."""
    freed_ids: list[str] = []
    tree = RadixTree(block_size=16, free_blocks=lambda ids: freed_ids.extend(ids))

    # Insert 32 tokens → node with 2 blocks (b0, b1)
    tree.insert(list(range(32)))
    # Split at token 16 (aligned): mid gets b0, child keeps b1 — nothing freed
    tree.insert(list(range(16)) + list(range(100, 116)))

    assert freed_ids == []  # block_ids are transferred, not freed


def test_mint_callback_receives_correct_block_count():
    """Custom mint callback receives the correct block count per allocation."""
    calls: list[int] = []

    def my_mint(n: int) -> list[str]:
        calls.append(n)
        return [f"blk-{i}" for i in range(n)]

    tree = RadixTree(block_size=16, mint=my_mint)
    tree.insert(list(range(32)))  # 2 blocks → mint(2)
    assert calls == [2]


# ---------------------------------------------------------------------------
# 8. Split block_id 归属正确 (M4.fix)
# ---------------------------------------------------------------------------

def test_split_block_aligned_block_ids_inherited():
    """Aligned split: mid inherits old prefix block_ids; child inherits old suffix block_ids."""
    counter = [0]

    def my_mint(n: int) -> list[str]:
        ids = [f"b{counter[0] + i}" for i in range(n)]
        counter[0] += n
        return ids

    freed: list[str] = []
    tree = RadixTree(block_size=16, mint=my_mint, free_blocks=lambda ids: freed.extend(ids))

    # Insert 32 tokens → node([0..31], ['b0', 'b1'])
    tree.insert(list(range(32)))
    assert tree._nodes["b0"].block_ids == ["b0", "b1"]

    # Aligned split at token 16: insert [0..15, 100..115]
    # cp=16, cp % 16 == 0, cp_blocks=1
    # mid([0..15]) → inherits ['b0']
    # child([16..31]) → inherits ['b1']
    # leaf([100..115]) → mints ['b2']
    tree.insert(list(range(16)) + list(range(100, 116)))

    mid = tree._nodes["b0"]
    child = tree._nodes["b1"]
    leaf = tree._nodes["b2"]

    assert mid.key == list(range(16))
    assert mid.block_ids == ["b0"]
    assert child.key == list(range(16, 32))
    assert child.block_ids == ["b1"]
    assert leaf.key == list(range(100, 116))
    assert leaf.block_ids == ["b2"]
    assert freed == []  # no blocks freed during aligned split


def test_split_non_aligned_mints_for_child():
    """Non-aligned split: mid takes old prefix blocks; child gets 1 new head + old tail."""
    counter = [0]

    def my_mint(n: int) -> list[str]:
        ids = [f"b{counter[0] + i}" for i in range(n)]
        counter[0] += n
        return ids

    freed: list[str] = []
    tree = RadixTree(block_size=16, mint=my_mint, free_blocks=lambda ids: freed.extend(ids))

    # Insert 32 tokens → node([0..31], ['b0', 'b1']); pool.used=2
    tree.insert(list(range(32)))

    # Insert [0..9] only (10 tokens, tail=[], cp=10, non-aligned)
    # cp_blocks = ceil(10/16) = 1
    # mid([0..9])     → ['b0']           (old prefix block)
    # child([10..31]) → ['b2', 'b1']     (new head + old tail block)
    # pool.used = 3 (b0 + b1 + b2)
    tree.insert(list(range(10)))

    mid = tree._nodes["b0"]
    child = tree._nodes["b2"]

    assert mid.key == list(range(10))
    assert mid.block_ids == ["b0"]
    assert child.key == list(range(10, 32))
    assert child.block_ids == ["b2", "b1"]
    assert freed == []
    # Total _nodes: mid + child = 2, total block_ids = 1 + 2 = 3 (pool invariant)
    total_bids = sum(len(n.block_ids) for n in tree._nodes.values())
    assert total_bids == 3


def test_split_transactional_on_mint_failure():
    """If the pre-mint for a non-aligned split fails, tree must remain unchanged."""
    import pytest
    counter = [0]
    fail_after = [2]  # succeed for initial insert (mints 2 blocks), then fail

    def my_mint(n: int) -> list[str]:
        if counter[0] >= fail_after[0]:
            raise MemoryError("pool full")
        ids = [f"b{counter[0] + i}" for i in range(n)]
        counter[0] += n
        return ids

    tree = RadixTree(block_size=16, mint=my_mint)
    tree.insert(list(range(32)))  # b0, b1 — consumes mint budget

    # Snapshot tree state before attempted split
    nodes_before = {k: v for k, v in tree._nodes.items()}

    # Non-aligned split requires 1 extra mint → should fail
    with pytest.raises(MemoryError):
        tree.insert(list(range(10)))  # cp=10, non-aligned → pre-mint fails

    # Tree must be completely unchanged
    assert set(tree._nodes.keys()) == set(nodes_before.keys())
    for k, node in nodes_before.items():
        assert tree._nodes[k] is node


def test_split_non_aligned_no_overflow_no_extra_mint():
    """Non-aligned split where cp_blocks + tail_blocks == n_old: no extra block needed.

    Example: block_size=16, insert 17 tokens [0..16] → 2 blocks.
    Split at cp=1 (non-aligned, 1 % 16 != 0):
      cp_blocks   = ceil(1/16) = 1
      tail_blocks = ceil(16/16) = 1
      n_old = ceil(17/16) = 2
      cp_blocks + tail_blocks = 2 == n_old → no overflow → no extra mint.
    """
    counter = [0]

    def my_mint(n: int) -> list[str]:
        ids = [f"b{counter[0] + i}" for i in range(n)]
        counter[0] += n
        return ids

    freed: list[str] = []
    tree = RadixTree(block_size=16, mint=my_mint, free_blocks=lambda ids: freed.extend(ids))

    # Insert 17 tokens → node([0..16], ['b0', 'b1']); pool block count = 2
    tree.insert(list(range(17)))
    assert counter[0] == 2

    # Insert [0] only (1 token, cp=1, non-aligned but no overflow)
    # cp_blocks=1, tail_blocks=ceil(16/16)=1, n_old=2 → needs_extra=False
    tree.insert([0])

    assert counter[0] == 2  # no extra mint
    assert freed == []

    mid = tree._nodes["b0"]
    child = tree._nodes["b1"]

    assert mid.key == [0]
    assert mid.block_ids == ["b0"]     # inherits old prefix block
    assert child.key == list(range(1, 17))
    assert child.block_ids == ["b1"]   # inherits old suffix block (no extra)
    # Invariant: total block_ids == n_old (no change in pool usage)
    total = sum(len(n.block_ids) for n in tree._nodes.values())
    assert total == 2

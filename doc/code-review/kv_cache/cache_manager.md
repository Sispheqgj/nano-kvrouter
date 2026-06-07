# CacheManager — M4 paged Tier 1 design notes

## Status
- ✅ V1 (separate RadixTree + BlockPool with _pool_ids bridge): scrapped
- ✅ V1.1 capacity-counter: superseded by M4
- ✅ **M4 (current): BlockPool active, pool.used("gpu") is the capacity counter**

## M4 design

* **Per-node BlockPool** (`self._pools[nid]`). Ground truth for physical GPU
  block accounting. All KV block allocation and deallocation flows through it.
* **Per-node RadixTree** (`self._trees[nid]`). Wired to its pool via
  `mint = pool.allocate(n, "gpu")` and (effectively no-op) `free_blocks`
  callbacks. Tree structure changes trigger pool changes automatically.
* **`pool.used("gpu")`** replaces the former `_used` / `_tree_ceiling_blocks()`
  derived counter. Capacity signals are always consistent with lookup results
  because both derive from the same pool.
* **`free_blocks(nid, "gpu")`** reads `pool._tiers["gpu"].free` directly.
* **`transfer_cost_ms`** is always `0.0` in v1 — no CPU/Disk tier management.

### Fundamental invariant

After every `admit` or `evict` completes:

```
pool.used("gpu") == sum(len(n.block_ids) for n in tree._nodes.values())
```

Each `RadixNode.block_ids` has length `ceil(len(key) / block_size)` — the
pool is the ground truth for how many physical GPU blocks are in use.

### Split block_id ownership (M4.fix)

When a RadixTree split occurs at offset `cp` into an existing edge:

```
cp_blocks = ceil(cp / block_size)   # blocks owned by the prefix

Case A — cp % block_size == 0 (aligned split):
  mid.block_ids  = old_block_ids[:cp_blocks]   # inherits prefix blocks
  child.block_ids = old_block_ids[cp_blocks:]  # inherits suffix blocks
  → No new mint, no free call.

Case B — cp % block_size != 0 (non-aligned split):
  mid.block_ids   = old_block_ids[:cp_blocks]          # inherits prefix blocks
  child.block_ids = [mint(1)[0]] + old_block_ids[cp_blocks:]  # 1 new head + suffix
  → 1 block minted for child's new head; no free call.
```

The pre-mint in Case B happens **before any tree modification** so that a
MemoryError leaves the tree unchanged (transactional).

This is also why `worst_case_new = min(total_blocks + 1, capacity_gpu)` in
`admit`: the +1 reserves space for the possible extra block in Case B.

### admit() flow

```
1. KeyError check
2. token_ids → aligned_tokens (floor-divide to block_size boundary)
3. Fast path: if already fully cached → no-op, return
4. Pre-check: if total_blocks > capacity_gpu → MemoryError immediately
5. worst_case_new = min(total_blocks + 1, capacity_gpu)
6. Evict loop until pool.used("gpu") + worst_case_new <= capacity_gpu
7. tree.insert(aligned_tokens)   # allocates via mint callback
```

## Known limitations / deferred

1. **P3: CPU/Disk tiers**. `cpu_blocks` and `disk_blocks` are tracked in the
   pool but never used for KV data; promotion/demotion stays unimplemented.
2. **P2 (Codex): worst_case_new over-eviction**. In cases where the insert
   triggers no split, `+1` causes one unnecessary eviction. Proper fix needs
   split dry-run or retry-on-MemoryError, deferred to P3.
3. **Partial-edge match returns 0**. CacheManager.lookup aligns matched tokens
   down to `block_size`; sub-block prefix hits are silently ignored. Tier 2
   (sub-block lookup) is deferred to P3.

## Related code
- `src/nano_kvrouter/kv_cache/cache_manager.py`
- `src/nano_kvrouter/kv_cache/radix_tree.py`
- `src/nano_kvrouter/kv_cache/block_pool.py`
- `tests/test_cache_manager.py`
- `tests/test_radix_tree.py`

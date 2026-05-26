# CacheManager — v1.1 capacity-counter design notes

## Status
- ✅ V1 (separate RadixTree + BlockPool with _pool_ids bridge): scrapped
- ✅ V1.1 capacity-counter: implemented and Codex-reviewed
- ⚠️ Two follow-ups deferred: see Known limitations below

## V1.1 design (current)

* **Per-node RadixTree** (`self._trees[nid]`). All KV cache prefix
  matching goes through it.
* **Capacity counters** (`self._capacity[nid][tier]` and
  `self._used[nid][tier]`). Physical block accounting lives here, not
  in BlockPool.
* `BlockPool` module is retained but **not used in v1.1** — it will
  come back in P2 when we model real CPU/Disk tier promotion / demotion.

### Fundamental invariant

After every `admit` or `evict` completes:

```
_used[nid]["gpu"] == ceil(sum(len(n.key) for n in trees[nid]._nodes.values())
                          / block_size)
```

`free_blocks(nid, "gpu")` is always `_capacity - _used`, derived from
the same tree state that `lookup` reads. **lookup and capacity signals
are always consistent**: scheduler can never see `matched_blocks > _used`.

### admit() flow (after Important fixes)

```
1. KeyError check
2. token_ids → aligned_tokens (drop partial trailing block)
3. Fast path: if already fully cached → no-op, return
4. Pre-check: if total_blocks > capacity → MemoryError immediately
5. Evict loop until free >= worst_case_new (= total_blocks)
6. tree.insert(aligned_tokens)
7. _used = _tree_ceiling_blocks(nid)   # ground truth sync
```

## Why "capacity-counter" instead of "_pool_ids list"?

Earlier v1 maintained `_pool_ids[node_id]: list[str]` to bridge
RadixTree nodes and BlockPool physical block IDs. When admit triggered
a non-block-aligned RadixTree split, the list lost identity:

* Splits created sub-block-sized nodes (`key_len < block_size`)
* `floor(key_len / block_size) = 0` for those, so evicting them freed
  zero pool blocks
* Subsequent evictions of larger nodes freed pool IDs from the *head*
  of `_pool_ids` — but the head ID belonged to an *earlier* admit,
  not the just-evicted node
* Eventual outcome: tree empty, pool counter still > 0,
  **orphan blocks permanently leaked**

Capacity-counter sidesteps the entire issue by deriving `_used` from
the tree itself (ceiling-sum), making "physical block accounting" a
pure function of tree state. No identity binding to maintain, nothing
to leak.

## Known limitations

1. **Implementation duplication** between
   `RadixTree.evict_lru()` and `RadixTree.evict_lru_with_lengths()`.
   Same victim-selection logic, different return type. Refactor to
   share a private helper. Tracked in TaskList #4-eviction-helper
   (informal — file an issue when convenient).

2. **Sub-block-sized tree nodes still consume one full block each**
   under ceiling semantics. This matches real vLLM PagedAttention
   (a partial trailing block occupies a full physical slot), but means
   non-block-aligned prefix sharing inflates `_used`. Mitigation:
   when `simulator/generator.py` lands in P1, enforce block-aligned
   shared prefixes (matches real workloads anyway).

## Related code
- `src/nano_kvrouter/kv_cache/cache_manager.py`
- `src/nano_kvrouter/kv_cache/radix_tree.py` (evict_lru_with_lengths)
- `tests/test_cache_manager.py` (tests #14-#18 cover regressions)

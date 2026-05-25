# CacheManager：v1 设计说明与 rollback 复盘

## 状态

- 已实现 V1（commit 待 add）
- 之前尝试过加入 “split-aware reconcile”，但因为会造成 `lookup()` 和 `pool.used` 不一致，已经选择 rollback。本文件记录原因。
- 相关任务：TaskList #8（已关闭：rollback chosen as the fix）

## V1 当前设计

当前 `CacheManager` 的核心设计是：

- 每个 node 有一棵独立 `RadixTree`。
- 每个 node 有一个独立 `BlockPool`。
- `RadixTree` 和 `BlockPool` 是并行但相互独立的数据结构。
- `_pool_ids[node_id]: list[str]` 负责桥接两者：
  - `admit()` 分配新 KV blocks 后 append pool IDs。
  - eviction 时从 `_pool_ids` 中移除对应数量的 pool IDs 并释放。
- `pool.used` 表示**物理 KV block 分配量**。
- `lookup()` 返回**逻辑 prefix hit**，即 `matched_tokens // block_size`。

这两个视图在非 block 边界 split 时可能相差几个 blocks：

```text
pool.used > lookup matched_blocks
```

但这个偏差是保守的：

```text
pool.used >= matched_blocks_in_tree
```

也就是说，scheduler 最多会看到“剩余容量比理论最大可用量更少”，而不会看到“剩余容量比真实物理容量更多”。这是安全方向的偏差。

## 被 rollback 的尝试：在 `admit()` 末尾 reconcile

之前有一个错误修复尝试：给 `RadixTree` 增加 `total_block_capacity(block_size)`，在 `admit()` 末尾比较：

```text
pool_used - actual_capacity
```

然后从 `_pool_ids` 尾部释放多余 pool IDs。

其中 `total_block_capacity(block_size)` 的含义是：

```python
sum(len(node.key) // block_size for node in tree_nodes)
```

这个 reconcile 是错误的，因为它混淆了两个不同量：

- `pool.used`：物理 KV block 占用量。这是正确的物理 accounting。
- `total_block_capacity`：用 floor 语义计算出来的“每个 radix node key 能装进多少完整 blocks”。它会忽略 non-block-aligned split 产生的 partial-block nodes。

换句话说，`RadixTree` split 只是改变索引结构，不应该改变物理 KV blocks 的占用数量。

## Codex review 反例

下面这个反例打破了 reconcile 的 invariant。

假设：

```text
block_size = 16
```

第一步：

```python
admit([0..15])
```

结果：

```text
1 block
pool.used = 1
```

第二步：

```python
admit([0] + [99..130])
```

这是 32 个对齐 tokens，但它和已有 prompt 只共享第一个 token。

过程：

```text
match_prefix returns 0    # partial edge match
allocate 2                # pool.used = 3
tree.insert splits at cp=1
```

split 后：

| Radix node | key_len | floor capacity |
|------------|---------|----------------|
| `mid([0])` | 1 | 0 |
| `leaf1([1..15])` | 15 | 0 |
| `leaf2([99..130])` | 31 | 1 |

于是：

```text
total_capacity = 0 + 0 + 1 = 1
```

错误 reconcile 会认为：

```text
pool.used = 3
total_capacity = 1
需要释放 2 个 blocks
```

于是强行释放 2 个 pool IDs，让：

```text
pool.used = 1
```

但此时对新请求：

```python
lookup(req=[0, 99..130])
```

仍然会返回：

```text
matched_blocks = 2
```

这就造成严重不一致：

```text
lookup(): GPU 上有 2 个 cache blocks 命中
pool.used/free_blocks(): GPU 上只分配了 1 个 block
```

scheduler 如果用 `lookup()` 做路由、用 `free_blocks()` 做容量检查，就会得到互相矛盾的信号，产生不一致决策。

## 为什么 rollback 是正确修复

没有 reconcile 的 V1 中：

```text
lookup says: 2 blocks cached
pool says: 3 blocks physically occupied
```

这两个说法可以同时为真：

- `lookup()` 表示逻辑上可复用的完整 KV blocks。
- `pool.used` 表示物理上已经分配出去的 KV blocks。

因为 non-block-aligned split 会造成 partial-block waste，所以物理占用可以大于逻辑命中。这是保守偏差。

而 reconcile 后：

```text
lookup says: 2 blocks cached
pool says: 1 block physically occupied
```

这不可能同时为真。它会让 scheduler 的 cache-hit 信号和 capacity 信号互相打架。

所以正确结论是：

```text
不要用 RadixTree.total_block_capacity() 去修正 BlockPool.pool.used。
pool.used 是物理占用来源。
RadixTree split 后的 floor capacity 不是物理占用。
```

## 什么时候再重新处理

### 情况 1：generator 大量产生 non-block-aligned prefix sharing

如果 `simulator/generator.py` 在大规模 workload 中持续生成非 block 对齐的共享前缀，并且物理 block waste 成为可测量问题，可以优先考虑：

```text
在 generator 中强制 shared prefix length 是 block_size 的整数倍。
```

这符合真实 vLLM PagedAttention 的固定大小 block 分页语义，改动也比重写 radix tree 小。

### 情况 2：真实 workload trace 证明 partial-block waste 显著

如果真实生产 trace 显示 partial-block waste 非常明显，再考虑在 `RadixTree` 层实现真正的 block-boundary-aware split。

这会是更大的结构性改动，需要重新审视：

- `RadixTree.insert()` 的 split 语义
- partial edge match
- `match_prefix()` 的 block 对齐行为
- eviction 和 block accounting 的一致性

## 当前问题与差距

| 问题 | 当前处理 | 影响 | 优先级 |
|------|----------|------|--------|
| non-block-aligned split 会造成 partial-block waste | V1 接受，保留物理 `pool.used` | scheduler 可能看到略少的 free capacity | 中 |
| `lookup()` 和 `pool.used` 是不同视图 | 保留差异，不强行 reconcile | 需要在文档和测试中明确语义 | 中 |
| 没有 block-boundary-aware split | 暂不实现 | 如果真实 trace 浪费明显，后续再做 | 低 |
| generator 未强制 block-aligned shared prefix | 暂不处理 | workload 可能产生更多 partial split | 低 |

## 后续 PR 候选

| PR | 改动范围 | 验证方式 |
|----|----------|----------|
| PR-1：增加 rollback regression test | 覆盖 `[0..15]` 和 `[0]+[99..130]` 反例，确认不会 reconcile 到 `pool.used < lookup matched_blocks` | `pytest tests/test_cache_manager.py` |
| PR-2：文档化 `lookup()` vs `pool.used` 语义 | 在源码 docstring 或 DESIGN 中说明逻辑命中和物理占用不同 | 文档 review |
| PR-3：generator block-aligned shared prefix | 如果 workload 需要，约束共享前缀长度对齐 `block_size` | generator 单测 |
| PR-4：block-boundary-aware RadixTree split | 远期结构性改动 | 跑完整 radix/cache manager 测试 |

## 相关代码

- `src/nano_kvrouter/kv_cache/cache_manager.py`：`admit()`、`lookup()`
- `src/nano_kvrouter/kv_cache/radix_tree.py`：`insert()`、split logic
- TaskList #8：closed，rollback chosen as the fix


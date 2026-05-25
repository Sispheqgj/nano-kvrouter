# CacheManager：v1 split-reconcile 修复后的已知局限

## 基本信息

- 源码文件：`src/nano_kvrouter/kv_cache/cache_manager.py`
- 相关文件：`src/nano_kvrouter/kv_cache/radix_tree.py`
- 审阅日期：2026-05-25
- 当前状态：
  - 已实现 V1（commit 待 add）
  - block-aligned split over-allocation 已通过 `admit()` 末尾 reconcile 修复（commit 待 add）
  - non-block-aligned split + small-node floor under-count 仍是残留问题
- 相关任务：TaskList #9

## 文件职责

`CacheManager` 是 scheduler 和 simulator 访问 KV cache 的统一入口。

当前 V1 的职责是：

- 每个 node 维护一棵独立 `RadixTree`。
- 每个 node 维护一个对应的 `BlockPool`。
- `lookup()` / `lookup_all()` 给 scheduler 返回本地 prefix 命中信息。
- `free_blocks()` 给 scheduler 返回 node 的剩余 cache 容量信号。
- `admit()` 在 prefill 完成后把 prompt 的 KV cache 物化到指定 node。

V1 仍然是 GPU-only 简化模型：

- `transfer_cost_ms == 0.0`
- 不做 CPU/Disk tier movement
- 不做跨节点 KV transfer

## 当前已经完成了什么

### 1. V1 CacheManager 已实现

当前实现采用每节点独立缓存视图：

```python
self._trees: dict[str, RadixTree]
self._pools: dict[str, BlockPool]
```

这对应之前确定的 P0 方案：

```text
1B：每个 node 一棵独立 RadixTree
2A：lookup 只考虑同节点本地 cache，不做跨节点 KV 拉取
```

### 2. block-aligned split over-allocation 已修复

V1 曾经存在一种过分配问题：`admit()` 先根据 `match_prefix()` 估算需要新增多少 blocks，再调用 `tree.insert()`。如果 `insert()` 触发 edge split，tree 的实际 block footprint 可能小于预先分配的 block 数。

现在通过 `admit()` 末尾 reconcile pass 修复：

```text
pool.used > tree.total_block_capacity()
```

时，从 `_pool_ids` 尾部释放多余 block，使 `BlockPool` 的占用和 `RadixTree` 的容量统计重新对齐。

这个修复解决了 block-aligned split 场景下的 over-allocation。

## 残留问题：small-node floor under-count

### 症状

当 `admit()` 触发 `RadixTree` 在**非 block 边界**位置切分 edge 时，split 会产生一个 `key_len < block_size` 的 small node。

Sonnet 的 reconcile pass 使用：

```python
len(n.key) // block_size
```

也就是 floor 语义。这样 `key_len < block_size` 的 small node 会被计为 `0 blocks`。

但从物理 KV cache 的角度看，这个 small node 仍然占用 KV cache 空间。

结果是：

- `pool.used` 会低估真实 KV 占用。
- `free_blocks()` 会返回偏高的剩余容量。
- scheduler 可能把请求误调度到一个实际上更满的 node。

误差上界：每次 small-node split 最多低估 1 个 block。

## 复现例子

假设：

```text
block_size = 16
```

第一步：

```python
admit([0..63])
```

这会产生 4 blocks，leaf key 为 `[0..63]`。

第二步：

```python
admit([0..7] + [99..130])
```

这个 prompt 的共同前缀只有 8 tokens，不在 block 边界上。

过程：

```text
match_prefix 返回 0
new_blocks_needed = 3
allocate 后 _pool_ids = 7
tree.insert 在 cp=8 处 split
```

split 后 tree 结构中出现：

| 节点 | key 长度 | floor 计数 |
|------|----------|------------|
| mid `[0..7]` | 8 | 0 blocks |
| leaf1 `[8..63]` | 56 | 3 blocks |
| leaf2 `[99..130]` | 32 | 2 blocks |

于是：

```text
total_block_capacity = 0 + 3 + 2 = 5
```

reconcile 看到：

```text
pool.used = 7
tree.total_block_capacity = 5
```

于是从尾部释放 2 个 pool blocks，把 `pool.used` 调到 5。

但真实语义上：

```text
A 占 4 blocks
B 占 3 blocks
真实总占用应为 7 blocks
```

所以 `free_blocks()` 会比真实情况多报 2 个 blocks。

## 影响评估

| 维度 | 影响 |
|------|------|
| 功能正确性 | 不会 crash，也不会直接造成数据丢失 |
| 调度行为 | scheduler 可能认为某个 node 比实际更空，从而误路由请求 |
| 容量控制 | 可能多 admit 一些请求，随后依赖 reactive eviction 自我修正 |
| 指标质量 | `free_blocks()` 会有噪声，误差通常不超过每次 small-node split 1 个 block |
| 严重程度 | P1，当前不是 P0 阻塞项 |

## 当前问题与差距

| 问题 | 影响 | 证据 | 优先级 |
|------|------|------|--------|
| non-block-aligned split 会产生 small node | small node 物理占用 KV，但 floor 计数为 0 | `total_block_capacity` 使用 `len(key) // block_size` | 高 |
| reconcile 会把真实占用误判为 over-allocation | `pool.used` 被调低，`free_blocks()` 偏高 | split 后 `pool.used=7`，capacity 统计为 5 | 高 |
| `match_prefix()` 对 partial edge match 返回 0 | `admit()` 会高估新增 blocks，再由 reconcile 修正；非对齐时修正可能过头 | 复现例子中共同前缀 8 tokens，但 block-aligned 命中为 0 | 中 |
| 当前 generator 未保证共享前缀 block-aligned | 测试或 workload 可能持续触发非对齐 split | 需要在 `simulator/generator.py` 侧约束 | 中 |

## 缓解路径

### 路径 1：推荐，在 generator 中强制共享前缀 block-aligned

在 `simulator/generator.py` 中保证 prefix sharing length 是 `block_size` 的整数倍。

优点：

- 改动小。
- 符合真实 vLLM 一类系统以固定大小 KV blocks 分页的语义。
- 避免为了仿真 workload 的非对齐 case 重写 radix split。

缺点：

- 只是约束输入分布，不是从数据结构层面彻底修复。

### 路径 2：重写 `RadixTree.insert()`，强制 block 边界 split

让 radix tree 的 edge split 只能发生在 block boundary。

优点：

- 从根上解决 small-node floor under-count。
- tree 的逻辑结构和 block accounting 更一致。

缺点：

- 改动大。
- 可能影响已有 radix 测试。
- 需要重新审视 partial prefix matching、node split、LRU eviction 的语义。

### 路径 3：`total_block_capacity` 改成 ceiling 语义

把：

```python
len(n.key) // block_size
```

改成类似：

```python
ceil(len(n.key) / block_size)
```

优点：

- 改动小。
- small node 不再被计为 0。

缺点：

- 会引入相反方向的偏差。
- 对多个 small nodes 可能高估容量。
- 需要重新校准 reconcile 语义。

## 后续 PR 候选

| PR | 改动范围 | 验证方式 |
|----|----------|----------|
| PR-1：generator 强制 block-aligned prefix sharing | 修改 `simulator/generator.py`，让共享前缀长度对齐 `block_size` | 新增 generator 单测，确认共享前缀长度总是 16/32/48... |
| PR-2：为 small-node under-count 加 regression test | 在 `tests/test_cache_manager.py` 复现 `[0..63]` + `[0..7]+[99..130]` | 先标记当前行为，再为后续修复提供保护 |
| PR-3：评估 block-boundary split | spike `RadixTree.insert()` 的 block-aligned split 版本 | 跑完整 radix/cache manager 测试，确认是否破坏现有语义 |
| PR-4：评估 ceiling capacity semantics | spike `total_block_capacity` ceiling 版本 | 对比 floor/ceil 在 aligned/non-aligned workload 下的 `pool.used` 偏差 |

## 学习笔记

这个问题的本质是：

```text
RadixTree 按 token prefix split，
BlockPool 按 fixed-size KV block 计费。
```

当 split 点刚好落在 block boundary 上时，两者很容易对齐。

当 split 点落在 block 内部时，就会出现 small node：

```text
key_len < block_size
```

这类 node 在 radix tree 语义里是合法的，但在 block accounting 语义里不好处理：

- 用 floor 会低估。
- 用 ceil 会高估。
- 强制 block-aligned split 会改变 radix tree 结构。

所以当前建议是 P1 先从 workload/generator 层面规避非 block-aligned sharing。等后续确实需要模拟任意 token 级共享前缀，再回到 `RadixTree` 层做更彻底的 block-aware split 设计。


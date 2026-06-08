# `src/nano_kvrouter/engine/mock_node.py` Code Review

## 基本信息

- 源码文件：`src/nano_kvrouter/engine/mock_node.py`
- 审阅日期：2026-05-24
- 对照材料：
  - Sarathi-Serve paper: <https://arxiv.org/abs/2403.02310>
  - Sarathi-Serve repo: <https://github.com/microsoft/sarathi-serve>
- 当前状态：已完成第一轮 review

## 文件职责

`MockEngineNode` 是真实 GPU inference engine 的 mock 替身。

它不保存 tensor，也不做真实推理，只负责：

- 维护节点上的 running request 和 waiting queue。
- 根据配置估算 prefill latency。
- 根据 batch size 估算 decode step latency。
- 给 scheduler 暴露节点负载和排队惩罚。
- 在请求完成时从 waiting queue promote 下一个请求。

这个文件属于 simulator 的执行层，但它仍然是 control-plane 项目的一部分：它提供可预测的模拟后端，让 scheduler、metrics 和 SLO 逻辑可以在 Mac M3 上先跑通。

## 当前已经完成了什么

### 1. Prefill latency 估算

当前实现：

```python
def estimate_prefill_time(self, prompt_len: int, cached_tokens: int) -> float:
    uncached = max(0, prompt_len - cached_tokens)
    return uncached * self.model_config.prefill_cost_per_token_ms
```

已经表达了一个重要设计点：KV cache 命中的 token 不需要重新 prefill。

这对 cache-aware scheduler 很关键，因为不同节点上的 cached prefix 长度不同，调度器可以用这个估算值判断把请求发到哪个节点更划算。

### 2. Decode latency 估算

当前实现：

```python
def estimate_decode_time(self, batch_size: int) -> float:
    return self.model_config.decode_base_ms + batch_size * self.model_config.marginal_decode_ms
```

它把 decode step 建模成：

```text
decode_time = base_cost + batch_size * marginal_cost
```

这个模型足够支持早期策略排序，例如比较 least-loaded 和 prefix-greedy 在不同负载下的决策差异。

### 3. 节点负载信号

`current_load()` 使用：

```text
running_requests / capacity
```

这让 scheduler 可以用统一接口读取节点负载。它适合做相对比较，但还不是完整的 GPU workload 模型，因为不同请求的 prompt length、output length、cache hit 情况并不相同。

### 4. 排队惩罚

`queue_wait_time()` 使用：

```text
queue_depth * decode_base_ms
```

这是一个有意简化的相对惩罚。它可以告诉 scheduler “这个节点已经有等待请求”，但不能作为严格 TTFT/TBT 预测。

### 5. 请求生命周期

`admit()` 和 `complete()` 已经提供了最小可用的节点状态机：

- 有容量时进入 `running_requests`。
- 无容量时进入 `queue`。
- 请求完成后从 `queue` promote 下一个请求。

这部分实现适合当前事件驱动 simulator，因为它没有引入线程、async 或 wall-clock time。

## 对照材料说明了什么

Sarathi-Serve 关注的问题是：真实 LLM serving 中，prefill 和 decode 会在同一张 GPU 上争抢执行时间。

如果一个长 prompt 的 prefill 一次性执行，它会长时间占住 GPU，使已有 decode stream 无法及时生成下一个 token。这就是 generation stall。

Sarathi 的关键设计是 chunked prefill：

- 把长 prompt prefill 拆成多个小 chunk。
- 每个 iteration 同时安排 prefill chunk 和 decode tokens。
- prefill chunk tokens 和 decode tokens 共享 token budget。
- 通过限制每轮 token budget，避免长 prefill 阻塞 decode。

可以把它抽象成：

```text
prefill_chunk_tokens + decode_batch_tokens <= token_budget
iteration_cost = f(prefill_chunk_tokens + decode_batch_tokens)
```

所以 Sarathi 不是简单修改 prefill cost 公式，而是改变了 prefill 和 decode 的调度耦合方式。

## 实施状态 (2026-06-08)

- **M2 continuous batching**: 已实现。`tick_batch_step` 整数 token 推进，
  `_batch_step_in_flight` guard 防 lost wakeup/duplicate completion。
- **M3 chunked prefill**: 已实现。`_prefill_remaining` FIFO 队列，
  每步 piggyback 一个 chunk，`prefill_chunk_size` 在 `ModelConfig` 中配置。
- **M5a P/D split**: `admit()` 分别管 prefill_node slot 和 decode_node slot，
  `complete()` 在各自侧 release。KV transfer event 由 cli.py 在
  `PREFILL_COMPLETE` 后调度。

## 当前问题与差距

| 问题 | 影响 | 证据 | 优先级 |
|------|------|------|--------|
| ~~Prefill 被建模成一次性原子阶段~~ | ~~无法模拟长 prompt 阻塞 decode 的 generation stall~~ | ~~`estimate_prefill_time()` 直接返回总 prefill time~~ | ~~高~~ → **M3 已解决** |
| ~~缺少 chunked prefill 抽象~~ | ~~不能对照 Sarathi-Serve 的核心机制做实验~~ | ~~没有 `chunk_size`、`token_budget`、chunk event~~ | ~~高~~ → **M3 已解决** |
| 没有暴露 KV cache 占用/剩余容量 | 调度器无法判断某个节点还能不能容纳新的 prefix，也无法做 capacity-aware cache placement | `MockEngineNode` 只暴露 running/queue load | 高 |
| `queue_wait_time()` 过于乐观 | 不适合用于严格 SLO admission 判断 | 只使用 `queue_depth * decode_base_ms` | 中 |
| latency 公式和节点状态耦合在一个类里 | 后续增加 transfer、stall、chunk 逻辑时文件会变重 | `MockEngineNode` 同时维护状态和公式 | 中 |

## 后续 PR 候选

| PR | 改动范围 | 验证方式 |
|----|----------|----------|
| PR-1：新增 `LatencyModel` | 新建 `src/nano_kvrouter/engine/latency_model.py`，把 prefill/decode/transfer 估算从 `MockEngineNode` 抽出 | 迁移现有 `tests/test_mock_node.py` 中 latency 断言，并新增 `tests/test_latency_model.py` |
| PR-2：暴露节点 KV cache 容量信号 | 给节点或 cache manager 提供可查询的 cache usage / remaining capacity，供 scheduler 做 prefix placement 决策 | scheduler 单测：当一个节点 prefix 命中高但容量不足时，不应继续选择它 |
| PR-3：让 completion 完全事件驱动 | 将节点完成和 promote 逻辑收敛到 simulator 的 `DECODE_COMPLETE` 处理路径 | simulation 单测：完成事件触发后节点状态、metrics、队列 promote 顺序一致 |

## 学习笔记

这个文件最适合用来学习 “mock backend 到真实 serving engine 的距离”。

当前实现回答的是：

```text
一个请求如果放到这个节点，大概需要多少 prefill 和 decode 时间？
```

Sarathi-Serve 进一步要求回答：

```text
这个请求的 prefill 会不会影响正在 decode 的请求？
如果会，怎样切 chunk 才能不让 decode 的 TBT 爆掉？
```

所以后续改进方向不是马上做真实 GPU，而是把 mock latency model 从“阶段级估算”推进到“iteration 级估算”。

另外，`MockEngineNode` 现在只描述计算侧负载，还没有把 KV cache 容量作为节点级信号暴露出来。对 cache-aware scheduler 来说，“这个节点命中了多少 prefix”只是一半问题，另一半是“这个节点还能不能放下新 prefix”。后续这部分更适合通过 `CacheManager` / `BlockPool` 提供统一查询，再让 scheduler 使用，而不是把真实 cache accounting 塞进 `MockEngineNode`。

`complete()` 当前作为同步方法是早期实现的合理简化，但长期应该由 simulator 的 `DECODE_COMPLETE` 事件触发。这样请求生命周期、metrics 记录、队列 promote 和未来的 KV free / demote 都能落在同一条事件时间线上。


## 进一步的参考
Sarathi-Serve GitHub: https://github.com/microsoft/sarathi-serve

  重点看这两个文件：
  - sarathi/core/scheduler/sarathi_scheduler.py：chunk 分配逻辑，对照我们的scheduler/base.py
  - sarathi/core/block_space_manager/sarathi_block_space_manager.py ：chunk 级别的 block分配，，对照我们的block_pool.py
  

  论文: https://arxiv.org/abs/2403.02310 — 重点看 §3（Motivation: Generation Stalls）和
  §4（Stall-Free Batching）里的图2，那张图直观展示了"prefill 堵 decode"的全过程。

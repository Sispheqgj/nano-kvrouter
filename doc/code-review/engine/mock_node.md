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

## 当前问题与差距

| 问题 | 影响 | 证据 | 优先级 |
|------|------|------|--------|
| Prefill 被建模成一次性原子阶段 | 无法模拟长 prompt 阻塞 decode 的 generation stall | `estimate_prefill_time()` 直接返回总 prefill time | 高 |
| Prefill 和 decode 完全独立估算 | 无法表达二者共享 GPU token budget | `estimate_prefill_time()` 和 `estimate_decode_time()` 没有共同状态 | 高 |
| 缺少 chunked prefill 抽象 | 不能对照 Sarathi-Serve 的核心机制做实验 | 没有 `chunk_size`、`token_budget`、chunk event | 高 |
| `queue_wait_time()` 过于乐观 | 不适合用于严格 SLO admission 判断 | 只使用 `queue_depth * decode_base_ms` | 中 |
| latency 公式和节点状态耦合在一个类里 | 后续增加 transfer、stall、chunk 逻辑时文件会变重 | `MockEngineNode` 同时维护状态和公式 | 中 |

## 后续 PR 候选

| PR | 改动范围 | 验证方式 |
|----|----------|----------|
| PR-1：新增 `LatencyModel` | 新建 `src/nano_kvrouter/engine/latency_model.py`，把 prefill/decode/transfer 估算从 `MockEngineNode` 抽出 | 迁移现有 `tests/test_mock_node.py` 中 latency 断言，并新增 `tests/test_latency_model.py` |
| PR-2：引入 chunked prefill 配置 | 在 `ModelConfig` 增加 `prefill_chunk_size`、`token_budget` 等字段 | 配置测试 + latency model 单测，确认长 prompt 被切成多个 chunk |
| PR-3：让 simulator 支持 prefill chunk event | 修改 `PREFILL_START -> PREFILL_COMPLETE` 的原子流程，支持多轮 chunk | 小规模 simulation 测试，确认 TTFT 语义不变 |
| PR-4：建模 generation stall | 让 decode TBT 受同一 iteration 中 prefill chunk 占用影响 | 构造“短 decode 被长 prefill 阻塞”的测试场景 |
| PR-5：输出 Sarathi 对照实验 | 增加一个可复现实验配置，对比 naive prefill 和 chunked prefill | CSV/JSON 指标，观察 TTFT/TBT 差异 |

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


# `src/nano_kvrouter/simulator/generator.py` Code Review

## 基本信息

- 源码文件：`src/nano_kvrouter/simulator/generator.py`
- 审阅日期：2026-05-29
- 对照材料：
  - BurstGPT paper: <https://arxiv.org/abs/2401.17644>
  - BurstGPT repo/schema: <https://github.com/HPMLL/BurstGPT>
  - Preble paper: <https://proceedings.iclr.cc/paper_files/paper/2025/file/5bc342f48de8264779952fac378f96dc-Paper-Conference.pdf>
  - Mooncake paper: <https://www.usenix.org/system/files/fast25-qin.pdf>
  - vLLM benchmark docs: <https://docs.vllm.ai/en/stable/cli/bench/serve.html>
  - NVIDIA AIPerf docs: <https://docs.nvidia.com/aiperf/>
- 当前状态：已完成第一轮 workload-generator review

## 文件职责

`RequestGenerator` 是 simulator 的请求流入口。

它不做真实推理，也不负责调度。它负责把配置里的 workload 参数转成一串
`REQUEST_ARRIVE` 事件，再交给 `SimulationEngine` 驱动后续 scheduler、node、
cache manager 和 metrics。

当前职责可以拆成四层：

- 生成请求到达时间。
- 生成 prompt token 序列。
- 给请求写入 arrival time、SLO、output length 等元数据。
- 把请求封装成 `EventType.REQUEST_ARRIVE` 事件并送入事件队列。

## 当前已经完成了什么

### 1. 流式 Poisson 到达

当前实现通过指数分布采样 inter-arrival time：

```python
self._mean_interarrival_ms = 1000.0 / self._workload.request_rate
return self._rng.expovariate(1.0 / self._mean_interarrival_ms)
```

这对应一个 open-loop Poisson arrival process。它适合做稳定、可复现的基线压测：
给定 `request_rate` 和 `duration_s`，期望请求数约等于二者乘积。

### 2. 请求流是在线生成的

`attach()` 只安排第一个到达事件。之后每个 `REQUEST_ARRIVE` 事件触发
`_on_arrive_chain()`，再安排下一个请求。

这个设计有一个好处：scheduler 在处理当前请求时看不到未来请求，符合在线 serving
系统的决策边界。

### 3. K-bucket prefix sharing

当前实现预先生成 `num_buckets` 个共享前缀，每个请求随机选一个 bucket，再追加随机
suffix：

```text
token_ids = bucket_prefix + random_suffix
```

这让 PrefixGreedy、E2Policy、Conductor 能看到稳定的 prefix cache hit，而不是所有
请求都是 cold start。

### 4. Prefix 长度按 block 对齐

`_prefix_len` 会向下取整到 `model.block_size` 的整数倍：

```python
self._prefix_len = (raw_prefix // bs) * bs
```

这和 block-based KV cache 的模拟边界一致，避免非 block 对齐前缀在 cache accounting
里产生不稳定解释。

### 5. Seed 可复现

生成器使用实例级 `random.Random(seed)`，不会污染全局随机状态。测试里也已经覆盖：
相同 seed 生成相同 bucket prefix，不同 seed 生成不同 workload。

## 对照材料说明了什么

### BurstGPT / Azure OpenAI traces：真实 workload 先是 trace

BurstGPT 的核心价值不是提出新的调度算法，而是提供真实 LLM serving workload 的观测。
它记录的是请求流本身，包括：

- `Timestamp`：请求何时到达。
- `Session ID`：请求属于哪个会话。
- `Request tokens`：输入 token 数，对应 prefill 工作量。
- `Response tokens`：输出 token 数，对应 decode 步数。
- `Total tokens`：总 token 规模。
- `Log Type`：请求类型，例如 conversation/API。
- failure 相关字段：真实服务里有失败和异常请求。

它处理请求的方式更接近 trace replay：每一行记录都是一个请求样本，benchmark 或模拟器
按 timestamp 重放请求，并用 input/output token count 构造请求长度。

对本文件的指导：

- 需要支持 trace replay，而不是只能 synthetic Poisson。
- `prompt_len` 和 `output_len` 应该是逐请求字段，不应该永远等于平均值。
- `Session ID` 应该进入 workload model，用来模拟多轮对话的上下文复用。
- `Log Type` 可以映射成 workload class，例如 chat、API、agent、long-context。
- failure / rejection 应该能进入模拟路径，至少能让 metrics 和 admission policy 区分
  success、reject、fail。

### Preble：请求不是独立样本，而是共享 prompt 的在线调度对象

Preble 研究的是 distributed prompt scheduling。它关心的不是普通负载均衡，而是：
当大量请求共享长 prompt 时，应该把新请求发到哪个 GPU，才能最大化 KV cache reuse，
同时避免某个 GPU 因为热点 prefix 被打满。

它处理请求的大致流程是：

1. 请求到达 global scheduler。
2. scheduler 用全局 prefix tree 查找这个请求在各 GPU 上的 cached prefix。
3. 如果某个 GPU 已有长 prefix，调度器倾向 exploit：复用已有 KV cache。
4. 如果 cache benefit 不够，调度器会考虑 explore/load-balance。
5. 决策里同时考虑 historical load、eviction cost、run cost。

对本文件的指导：

- K 个 bucket 均匀随机不够真实。真实 prefix popularity 更像 hotspot/Zipf：少数 system
  prompt、tool schema、RAG 文档、热门会话前缀会被大量复用。
- Prefix sharing 不应该只有一个固定比例。真实请求可能共享 system prompt、few-shot
  examples、文档片段、会话历史等多层前缀。
- 要评估 E2 这类策略，workload 必须有混合长度，否则 run cost 和 queue cost 的差异
  很难显现。
- 需要保留当前 online chaining 设计，因为 Preble 处理的是在线到达请求，不是提前知道
  全部未来 workload 的离线分配。

### Mooncake：请求生成要能触发 SLO、early rejection 和 KV 层级压力

Mooncake 是 KVCache-centric disaggregated serving 架构。它的 Conductor 不只是路由到
低负载节点，而是根据 TTFT/TBT SLO、prefill/decode 资源、KV cache 位置和传输成本做
全局决策。高负载下，如果预测请求无法满足 SLO，Conductor 会 early reject。

它处理请求的大致流程是：

1. 请求进入全局 Conductor。
2. Conductor 预测 prefill、decode、KV transfer 和排队成本。
3. Conductor 选择 prefill/decode 资源，必要时利用 CPU/DRAM/SSD 中的 KV cache。
4. 如果预测 TTFT/TBT 会违反 SLO，请求在执行前被拒绝。
5. 评估关注 goodput：满足 SLO 的有效吞吐，而不是单纯完成了多少请求。

对本文件的指导：

- 请求应支持不同 SLO profile，例如 premium、standard、batch，而不是所有请求共享同一
  `slo_ttft` 和 `slo_tbt`。
- workload 应支持 overload window 和 burst，这样 early rejection 才有可观察意义。
- prompt/output 长度应覆盖长上下文场景，例如 8K、32K、128K prompt。
- workload type 应影响长度和共享模式：短 chat、agent/tool、long document QA 对 KV
  cache 和 SLO 的压力完全不同。

### vLLM benchmark：工业接口已经把 workload 做成可选数据源和到达过程

`vllm bench serve` 更像一个工业 benchmark 参考。它支持多种 dataset/source，例如
ShareGPT、BurstGPT、random、prefix repetition；到达过程支持 Poisson 和 gamma；同时有
max concurrency、ramp-up、goodput SLO 等参数。

它处理请求的方式是：

1. 从 dataset 中取 prompt/output 样本，或按 random 参数合成样本。
2. 根据 request rate 和 arrival distribution 决定每个请求何时发出。
3. 可选使用 max concurrency 模拟上游限流。
4. 对 streaming response 记录 TTFT、TPOT、E2E latency。
5. 按 SLO 统计 goodput。

对本文件的指导：

- 可以借鉴它的配置语义：`dataset`、`request_rate`、`burstiness`、`max_concurrency`、
  `goodput_slo`。
- 当前 Poisson 可以保留为默认值，但需要增加 gamma arrival 来模拟更突发或更均匀的请求流。
- `prefix_repetition` 这类专门压测 prefix cache 的 workload，也可以作为当前
  K-bucket 模型的增强版本。

### NVIDIA AIPerf：客户端压测和 trace replay 工具

NVIDIA AIPerf 不是 serving engine，也不是调度器。它更像一个面向 LLM inference 的
benchmark/profiling 客户端：

- 向 vLLM、SGLang、TensorRT-LLM、NVIDIA Dynamo 或 OpenAI-compatible endpoint 发请求。
- 支持固定 input/output token 长度，也支持从 trace 中读取长度。
- 支持 streaming 统计 TTFT、inter-token latency、request latency。
- 支持 goodput，即只统计满足 SLO 的有效请求吞吐。
- 支持从服务端 Prometheus metrics 采集 running requests、waiting requests、KV cache
  usage、preemption 等指标。
- 支持 BurstGPT trace replay：按 timestamp 发请求，并用 trace 里的 request/response
  token count 合成请求。

对本文件的指导：

- `nano-kvrouter` 不需要真实文本就能学习真实 workload。只要有 arrival time、
  prompt token count、output token count、session id、prefix group 和 SLO，就能构造
  对调度器有意义的模拟请求。
- 可以把 AIPerf 看成外部真实 benchmark，把 `RequestGenerator` 看成内部离线 simulator
  的 workload adapter。二者应该尽量共享概念：trace、timestamp、input length、output
  length、goodput SLO。

## 当前问题与差距

| 问题 | 影响 | 证据 | 优先级 |
|------|------|------|--------|
| Prompt length 固定 | 不能模拟真实请求长短差异，scheduler 的 run cost 区分度不足 | `_suffix_len + _prefix_len == avg_prompt_len`，每个请求长度相同 | 高 |
| Output length 固定 | decode 压力和排队时间过于同质，无法复现 BurstGPT/ShareGPT 的长尾输出 | `make_request()` 从 `config.workload.avg_output_len` 写入 `expected_output_len` | 高 |
| 只支持 Poisson arrival | 无法表达 burst、ramp-up、overload window 和 Azure/BurstGPT trace 中的非平稳到达 | `_sample_interarrival()` 只有指数分布 | 高 |
| 不支持 trace replay | 无法直接复现 BurstGPT、Mooncake-style real trace 或本地实验 trace | 没有 `trace_path`、trace parser、timestamp replay | 高 |
| Bucket 选择均匀随机 | 低估 hotspot prefix 对 cache placement 和负载倾斜的影响 | `bucket_idx = randint(0, num_buckets - 1)` | 中 |
| Prefix sharing 只有单层固定比例 | 无法表达 system prompt、tool schema、RAG doc、session history 的多层共享 | `_prefix_len` 对所有请求相同 | 中 |
| 没有 session 语义 | 无法模拟多轮对话中上下文逐轮增长和复用 | `Request` 没有 `session_id`，generator 也不维护 session state | 中 |
| 所有请求共享同一 SLO | Mooncake-style early rejection 和 goodput 评估不够真实 | `make_request()` 从全局 `config.slo` 写入 SLO | 中 |
| 没有 workload type | 无法区分 chat、API、agent/tool、long-context QA 等业务形态 | `Request` 没有 workload class 字段 | 低 |
| 生成器和请求构造耦合 | trace replay、synthetic distribution、session-aware prefix 都会堆进 `_build_request()` | 当前只有一个 `RequestGenerator` 类承载所有逻辑 | 低 |

## 后续 PR 候选

| PR | 改动范围 | 验证方式 |
|----|----------|----------|
| PR-1：支持逐请求 output length | 给 `make_request()` 增加 `expected_output_len` 可选参数；generator 传入采样值 | `tests/test_request.py` 覆盖默认值和覆盖值；`tests/test_generator.py` 验证不同请求 output length 可不同 |
| PR-2：支持 prompt/output length distribution | 在 config 中增加 `prompt_len_dist`、`output_len_dist`，先支持 `fixed`、`uniform`、`lognormal` | 固定 seed 下采样可复现；长度满足 min/max 和均值约束 |
| PR-3：支持 gamma arrival / burstiness | 保留 Poisson，新增 `arrival_pattern=gamma` 和 `burstiness` 参数 | 单测比较 inter-arrival 方差；`burstiness=1` 时接近 Poisson |
| PR-4：支持 Zipf bucket popularity | 新增 `bucket_popularity=uniform/zipf`、`zipf_alpha` | 统计前 N 个 bucket 命中率，确认 Zipf 产生热点 |
| PR-5：引入 trace replay adapter | 新增 `trace_path`、`trace_format=burstgpt/jsonl`，按 timestamp 生成到达事件 | 小型 fixture trace：到达时间、prompt_len、output_len、request_id 与文件一致 |
| PR-6：引入 `RequestSample` | 用 dataclass 表达 arrival_time、prompt_len、output_len、session_id、prefix_group、slo_profile，再转换成 `Request` | synthetic 和 trace 两条路径共用 sample-to-request 单测 |
| PR-7：支持 session-aware prefix | generator 维护 session state，同一 session 后续请求复用历史 prefix 并追加 turn token | 构造同一 session 多轮请求，验证后续请求 prefix 命中增长 |
| PR-8：支持 mixed SLO profile | config 增加 SLO profile 分布，每个请求写入自己的 TTFT/TBT 目标 | Conductor 单测：严格 SLO 请求更容易被 reject |
| PR-9：补一个 realistic workload 配置 | 新增 `configs/realistic.yaml`，组合 gamma arrival、lognormal length、Zipf prefix | 跑短 simulation，确认所有 scheduler 能完成且 metrics 可输出 |

## 学习笔记

### 1. 请求生成器不只是压测参数

LLM serving 里的 workload generator 决定了 scheduler 看到的世界。

如果所有请求长度相同、arrival 平稳、prefix 均匀分布，那么调度器之间的差异会被压扁：

```text
RoundRobin / LeastLoaded / PrefixGreedy / E2 / Conductor
```

在这种 workload 下只能验证基本功能，不能充分验证真实系统关心的边界。

更真实的请求至少要有四个维度：

```text
arrival_time: 请求什么时候来
prompt_len: prefill 工作量
output_len: decode 工作量
prefix_shape: 和其他请求共享哪些前缀
```

Mooncake/Preble 还要求再加：

```text
slo_profile: 请求的 TTFT/TBT 目标
cache_location: 命中的 KV 在哪个节点/层级
```

### 2. Poisson 是基线，不是终点

Poisson arrival 的优点是简单、可复现、容易解释。它适合作为 baseline。

但真实 LLM 请求通常有 burst：

- 工作日/夜间有 diurnal pattern。
- 产品活动或脚本调用会制造短时间尖峰。
- 多轮对话会在短时间内连续发出相关请求。
- agent/tool workflow 会产生一串相关调用。

所以后续可以保留当前 Poisson，同时增加：

```yaml
arrival_pattern: poisson | gamma | trace
burstiness: 0.5
```

其中 `gamma` 可以复用 vLLM benchmark 的语义：`burstiness=1` 接近 Poisson，
小于 1 更突发，大于 1 更均匀。

### 3. Trace replay 应该是第一等能力

真实系统论文越来越依赖 trace，而不是只报告 synthetic workload。

对本项目来说，trace replay 不需要真实 tokenizer。最小 trace schema 可以是：

```json
{
  "timestamp_ms": 123.4,
  "prompt_len": 2048,
  "output_len": 128,
  "session_id": "s-001",
  "prefix_group": "doc-17",
  "slo_profile": "standard"
}
```

如果接 BurstGPT，可以做字段映射：

| BurstGPT 字段 | simulator 字段 |
|---------------|-----------------|
| `Timestamp` | `arrival_time` |
| `Request tokens` | `prompt_len` |
| `Response tokens` | `output_len` |
| `Session ID` | `session_id` |
| `Log Type` | `workload_type` |

### 4. Prefix model 要从 bucket 走向树

当前 K-bucket 模型可以理解成：

```text
bucket_prefix_i + random_suffix
```

它能制造 prefix cache hit，但还不是 Preble/SGLang 里的真实 prefix tree。

更接近真实系统的模型应该是分层的：

```text
system_prompt
  + tool_schema
  + few_shot_examples
  + rag_document_prefix
  + session_history
  + user_turn
```

这样才能让不同请求之间出现不同长度的公共前缀，而不是所有共享都停在固定的
`prefix_sharing_ratio * avg_prompt_len`。

### 5. AIPerf 对本项目的定位启发

NVIDIA AIPerf 站在真实服务外面发请求并测量响应。本项目站在模拟器内部生成事件并测量
control-plane 决策。

二者并不冲突：

```text
AIPerf:      真实 serving endpoint 的客户端压测器
nano-kvrouter: 事件驱动 control-plane simulator
```

但它们应该共享 workload 语义。AIPerf 能按 BurstGPT trace 发请求，`RequestGenerator`
也应该能按同样的 trace 生成 `REQUEST_ARRIVE` 事件。这样本项目里的策略实验，才更容易和
真实 benchmark 的结论对照。

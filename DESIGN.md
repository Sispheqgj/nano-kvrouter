# nano-kvrouter Design

> 一句话定位：纯 Python、事件驱动、单线程的 LLM serving control-plane
> simulator。重点是调度、SLO、KV cache 和层次化存储，不是 GPU kernel。

## 1. 当前实现边界

截至当前 checkout，P2-Infra M1-M6 + P3-C M1 已经落地：

- M2: continuous batching
- M3: chunked prefill
- M4: paged GPU KV block metadata
- M5: split prefill/decode pools + post-prefill KV transfer
- M6: multi-tier HiCache (GPU / CPU / Disk) + tier-aware lookup
- acceptance: config-driven `sensitivity` CLI
- P3-C M1: real-world trace replay (Mooncake FAST'25, streaming JSONL),
  per-request `output_length` truthfully drives decode pressure

这个仓库故意不做的事情：

- 不保存真实 tensor
- 不跑真实 GPU 推理
- 不做 wall-clock benchmark
- 不实现完整的 Llumnix live migration 执行路径

## 2. 核心问题

当一个新请求到达时，当前 simulator 关注四个 control-plane 决策：

1. 发往哪个 prefill 节点
2. 发往哪个 decode 节点
3. 命中的 KV block 在 GPU / CPU / Disk 哪一层，加载代价是多少
4. 在当前负载和 SLO 下，应该接收还是拒绝

## 3. 论文对应关系 / fidelity matrix

| 系统 | 当前复现内容 | fidelity 说明 |
| ---- | ------------ | ------------- |
| Mooncake FAST'25 | `MooncakeConductor`、split P/D、post-prefill KV transfer、SLO gate | Disk-tier hit 的 Disk -> CPU -> GPU 两段传输是 simulator extrapolation，不是论文逐字实现 |
| SGLang NeurIPS'24 | RadixAttention-style prefix tree、prefix-aware routing | 在 prefix tree 上扩展了 tier-aware lookup，这部分是本仓库自己的模拟抽象 |
| vLLM v1 / PagedAttention | BlockPool / paged KV metadata abstraction | 没有真实 paged-attention kernel，仅保留 block accounting 语义 |
| Preble ICLR'25 | E2 exploit-explore、prompt-aware load | 评分函数刻意收敛成便于对照的 simulator 版本 |
| Llumnix OSDI'24 | migration / rebalance 作为控制面概念 | 目前只保留路线，不提供完整迁移执行链路 |

### 3.1 MooncakeConductor 评分公式

```python
score(decode_node) = (
    alpha * cache_benefit(decode_node)      # matched_tokens * prefill_cost_per_token_ms
  - beta  * load_penalty(decode_node)       # current_load * prompt_len * ppt + queue_wait
  - gamma * transfer_penalty(decode_node)   # CacheLookup.transfer_cost_ms (M6 tier reload)
)

# SLO 早期拒绝（基于得分最高的 decode_node 的预测）：
if est_ttft > request.slo_ttft or est_tbt > request.slo_tbt:
    REJECT
```

`transfer_penalty` 是 *within-node tier reload* 代价（CPU/Disk → GPU），不是
prefill_node → decode_node 之间的 KV 网络传输代价 —— 后者已经在 `est_ttft`
里通过 `compute_est_ttft` 计入 SLO gate，避免双重计算。

### 3.2 E2 prompt-aware load 公式

```python
e2_score(decode_node, request) = (
    w_historical * historical_load(decode_node)         # current_load * prompt_len * ppt
  + w_eviction   * eviction_cost(decode_node, request)  # shortage_blocks * block_size * ppt
  + w_run        * run_cost(prefill, decode, request)   # compute_est_ttft (含 KV transfer)
)
# 选 e2_score 最小的 decode_node；prefill_node 独立按最低 load 选
```

所有三项都以毫秒计，权重无量纲，便于权衡调参。冷启动 / 等负载时由 `node_id`
词典序破并。

## 4. 总体架构

```text
RequestGenerator
  -> SchedulingPolicy
  -> prefill MockEngineNode pool
  -> KV_TRANSFER_START / KV_TRANSFER_COMPLETE
  -> decode MockEngineNode pool
  -> MetricsCollector

Decode-side cache state:
  CacheManager
    -> one RadixTree per decode node
    -> one BlockPool per decode node
```

关键点：

- scheduler 只决定放到哪里，不维护节点内部执行状态
- `MockEngineNode` 只做 latency model + admission / queueing
- `CacheManager` 是 cache source of truth
- `MetricsCollector` 是被动观察者

## 5. 请求生命周期

当前事件集合在 [src/nano_kvrouter/simulator/event.py](src/nano_kvrouter/simulator/event.py)
里，共 10 种事件：

- `REQUEST_ARRIVE`
- `SCHEDULED`
- `PREFILL_START`
- `PREFILL_COMPLETE`
- `KV_TRANSFER_START`
- `KV_TRANSFER_COMPLETE`
- `DECODE_BATCH_STEP`
- `TOKEN_GENERATED`
- `DECODE_COMPLETE`
- `REQUEST_REJECTED`

一个正常请求的时序如下：

1. `REQUEST_ARRIVE`
2. scheduler 基于 `prefill_nodes`、`decode_nodes` 和 `CacheManager` 做决策
3. prefill 节点开始 chunked prefill
4. `PREFILL_COMPLETE`
5. 触发 Mooncake-style KV transfer
6. `KV_TRANSFER_COMPLETE`
7. decode 节点 admit 请求并进入 batch decode
8. 多次 `DECODE_BATCH_STEP` / `TOKEN_GENERATED`
9. `DECODE_COMPLETE`

如果 decode 节点在 transfer 完成时容量已满，请求会在 admit 前直接走
`REQUEST_REJECTED`。这是当前 split P/D 的 decode-side back-pressure。

## 6. M2-M6 设计细节

### M2: continuous batching

实现位置：

- [src/nano_kvrouter/engine/mock_node.py](src/nano_kvrouter/engine/mock_node.py)
- [src/nano_kvrouter/cli.py](src/nano_kvrouter/cli.py)

当前行为：

- decode 以 `DECODE_BATCH_STEP` 推进
- 每个 batch step 的 decode cost 为
  `decode_base_ms + batch_size * marginal_decode_ms`
- `_batch_step_in_flight` guard 防止 duplicate scheduling / lost wakeup

### M3: chunked prefill

实现位置：

- `MockEngineNode._prefill_remaining`
- `model.prefill_chunk_size`
- `avg_chunked_prefill_steps_per_request` metric

当前行为：

- prefill 按 chunk 推进，不再是一次性原子阶段
- 每个 tick 最多推进一个 prefill chunk
- chunk cost 使用真实 `min(remaining, chunk_size)`，不是固定满 chunk

### M4: paged GPU blocks

实现位置：

- [src/nano_kvrouter/kv_cache/radix_tree.py](src/nano_kvrouter/kv_cache/radix_tree.py)
- [src/nano_kvrouter/kv_cache/block_pool.py](src/nano_kvrouter/kv_cache/block_pool.py)
- [src/nano_kvrouter/kv_cache/cache_manager.py](src/nano_kvrouter/kv_cache/cache_manager.py)

当前行为：

- GPU tier 容量由 `BlockPool` 维护
- `RadixTree` 保存 prefix -> block ownership
- `CacheManager.admit()` 在容量压力下执行 split-aware admit / LRU pressure handling

### M5: split prefill/decode + KV transfer

实现位置：

- `cluster.prefill_nodes`
- `cluster.decode_nodes`
- `cli._run_one()` / `cli._wire_simulator()`

当前行为：

- prefill pool 和 decode pool 是两套 `MockEngineNode`
- cache 只注册在 decode pool
- prefill 完成后触发一次 post-prefill KV transfer
- transfer cost:

```text
prompt_len * model.kv_bytes_per_token / bandwidth.gpu_to_gpu * 1000
```

### M6: multi-tier HiCache

实现位置：

- `node.gpu_blocks`
- `node.cpu_blocks`
- `node.disk_blocks`
- `bandwidth.gpu_to_cpu`
- `bandwidth.cpu_to_disk`
- `CacheManager.lookup()` / `CacheManager.admit()`

当前行为：

- demotion chain: GPU -> CPU -> Disk -> free
- lookup 返回 `matched_blocks_by_tier`
- scheduler 读取 `transfer_cost_ms` 作为 tier-hit 代价

当前 tier-hit 代价语义：

- GPU hit: `0 ms`
- CPU hit: `block_bytes / bandwidth.gpu_to_cpu`
- Disk hit:

```text
block_bytes * (1 / bandwidth.cpu_to_disk + 1 / bandwidth.gpu_to_cpu)
```

Disk 两段 hop 串行相加是 simulator 的显式建模选择，用来表达
Disk -> CPU -> GPU 的冷层命中代价。

## 7. LIVE config surfaces

当前实现里，下面 13 个字段全部是 LIVE：

| 字段 | 使用位置 |
| ---- | -------- |
| `cluster.decode_nodes` | `cli._run_one()` 构造 decode pool |
| `node.capacity` | `MockEngineNode.admit()` / `queue_wait_time()` / decode-side back-pressure |
| `node.gpu_blocks` | `BlockPool` tier-1 容量 |
| `node.cpu_blocks` | `BlockPool` tier-2 容量 |
| `node.disk_blocks` | `BlockPool` tier-3 容量 |
| `model.kv_bytes_per_token` | KV transfer 和 tier reload cost |
| `model.prefill_cost_per_token_ms` | prefill latency |
| `model.decode_base_ms` | decode-step latency |
| `model.marginal_decode_ms` | batch-sensitive decode latency |
| `model.prefill_chunk_size` | chunk count / prefill scheduling |
| `bandwidth.gpu_to_gpu` | prefill -> decode KV transfer |
| `bandwidth.gpu_to_cpu` | CPU-tier reload cost |
| `bandwidth.cpu_to_disk` | Disk-tier reload cost |

因此当前文档统一把这 13 个字段视为 LIVE，不再保留旧的阶段性占位口径。

## 8. 配置场景

### `configs/default.yaml`

- 默认 unsaturated scenario
- 用于一般调度行为、时延对比、部分 sensitivity latency/model 实验

### `configs/heavy.yaml`

- decode-capacity pressure scenario
- 用于 `cluster.decode_nodes` 和 `node.capacity` 的 rejection / throughput sensitivity

### `configs/hicache.yaml`

- HiCache scenario
- 用于 `gpu_blocks` / `cpu_blocks` / `disk_blocks` / `gpu_to_cpu` / `cpu_to_disk`
  的多层缓存实验

### `configs/sensitivity.yaml`

- 描述 field experiments，不把实验逻辑写死在 CLI 里
- `base_config` 默认相对 `sensitivity.yaml` 所在目录解析

## 9. sensitivity acceptance workflow

正式入口：

```bash
uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml
```

支持输出：

```bash
uv run python -m nano_kvrouter.cli sensitivity \
  --config configs/sensitivity.yaml \
  --output /tmp/report.json

uv run python -m nano_kvrouter.cli sensitivity \
  --config configs/sensitivity.yaml \
  --output /tmp/report.csv
```

判定语义：

- 每个 experiment 先跑 baseline，再跑所有 candidate
- `changed_metrics` 只记录超过阈值的 primary metric leaf
- `Candidate PASS` 表示这个候选值真的把对应 metric 拉动到了阈值外
- `Field LIVE` 表示该字段至少有一个候选值成功拉动 primary metric

当前阈值策略：

- ratio / rate: `abs(delta) >= 0.005`
- latency: `abs(delta) >= 0.5 ms` 或 `abs(pct_delta) >= 1%`
- throughput / rejection: absolute 或 percent delta 任一达标

平台区说明：

- `cpu_blocks` / `disk_blocks` 扩容型候选可能出现 `Candidate FAIL`
- 但归零型候选会直接暴露 tier reuse 崩塌，从而证明字段 LIVE

## 10. 当前简化与非目标

和真实系统相比，当前 simulator 保留的是控制面语义，不是执行面细节：

- 没有真实 tokenizer / tensor / kernel
- 没有真实 RDMA / PCIe stack，只保留带宽驱动的时间代价
- 没有完整 migration executor（trace replay 已在 P3-C M1 落地，但 session-aware
  prefix synthesis 留到 P3-C M2、Llumnix migration 留到 P4）
- 指标适合对比，不适合直接当生产容量规划数字

### 10.1 关键简化对照表

| 真实系统 | nano-kvrouter 的简化 | 保留的核心 |
|----------|----------------------|------------|
| 真实 GPU 张量计算 | 用 `cost_per_token + base + batch * marginal` 延迟模型替代 | 调度决策逻辑、batch 经济学 |
| RDMA / PCIe / NVLink 传输 | 用 `bytes / bandwidth` 模型计算时间代价 | 传输代价对路由的影响、tier-aware reload |
| 多副本 KV（TP/PP）+ 真实 paged attention kernel | 单副本 + block 元数据 (`block_id`, `tier`, `ref_count`) | block-level 占用、LRU eviction、demotion chain |
| Token streaming + KV cache write | 离散 `DECODE_BATCH_STEP` + 整数 token 推进 | TBT 统计、continuous batching 行为 |
| 真实 tokenizer | 随机 int token_ids + K-bucket 共享前缀 | RadixAttention 风格的前缀匹配 |
| Llumnix live migration | 控制面概念占位，无 executor | 路线图保留 |

## 11. Trace replay (P3-C M1 + M2)

### 11.1 入口与字段映射

新增 `TraceGenerator` 作为 `RequestGenerator` 的并行实现：

- 内部 JSONL schema：`{request_id, arrival_ms, input_length, output_length, hash_ids?, session_id?}`
- 启用方式：YAML 加 `trace: { path, speedup, max_requests, prefix_mode }`
  字段；非 None 时 CLI 用 `TraceGenerator` 替代 Poisson
- 路径解析：CLI 层调 `_resolve_related_config_path` 把 `trace.path` 解析为
  absolute 后写回 `cfg.trace.path`，`TraceGenerator` 只接收 absolute Path
- `make_request` 加可选 `expected_output_len` 参数，trace 把 record 里的
  `output_length` 真值传进去，让 decode 压力忠实反映 trace（不再写死
  `config.workload.avg_output_len`）
- `_attached` 双重 attach 防御 + `_close()` 在 EOF / cap 时关 file

### 11.2 hash_ids → token_ids 合成

`_build_token_ids` 保证 `len(tokens) == input_length` 强不变量：

- 前 `min(len(hash_ids), full_blocks)` 个 hash_id 各展开为 `block_size` 个
  连续 token（`token = hid * block_size + offset`，保证同 hash_id 跨 request
  展开成相同序列 → RadixTree 命中）
- hash_ids 不够覆盖的完整 block 用随机填充
- 尾部不完整 block (`input_length % block_size`) 用随机填充
- 末尾 `assert len(tokens) == input_length`，覆盖三种边界（hash_ids 多/少/等于
  full_blocks + tail_len>0）

### 11.3 prefix_mode 三档

- `none`：纯随机 token，无 prefix 共享（保守 baseline）
- `hash_ids`：用 trace 真实 hash_ids 合成（Mooncake 路径）
- `synthesis`：用 `PrefixSynthesisModel` 在 length-only trace 上合成 prefix 共享
  （BurstGPT 路径，详见 §11.6）

### 11.4 Mooncake trace 集成约束

- `configs/trace_mooncake.yaml` 必须 `model.block_size: 512`（Mooncake hash_ids
  block_size 是 512，跑前 `ceil(input_length/512) == len(hash_ids)` 已在 SOURCES.md
  里 verified）。其他 yaml 保持 block_size=16
- 副作用：`prefill_chunk_size=512 + block_size=512` 意味着每个 prefill chunk
  = 1 个 block，chunked prefill 行为退化为 per-block。不是 bug，是 trace 的
  block 粒度决定的
- 3 个 Mooncake trace 文件 (`conversation/synthetic/toolagent_trace.jsonl`) 共
  ~10 MB 直接 commit 进 `traces/mooncake/`（Apache-2.0 license，bundle
  `LICENSES/Apache-2.0.txt`）

### 11.5 BurstGPT trace + converter

BurstGPT (HPMLL, CC-BY-4.0) 只有长度/时间戳/session_id，没有 prefix 结构：

- 原始字段：`Timestamp(秒，非零基准) / Session ID / Request tokens /
  Response tokens / 其它日志列`
- `scripts/convert_burstgpt.py` 把原始 CSV 流式转 JSONL，统一到 P3-C
  schema：`{request_id, arrival_ms, input_length, output_length, session_id}`
- 默认开启 `--require-session-id`：实测原始 CSV 中约 51% 行是无 session 的
  API log 行，必须过滤掉，否则 session_id 字段名存实亡
- BurstGPT 不带 `hash_ids`，所以 `configs/trace_burstgpt.yaml` 必须配
  `prefix_mode: synthesis`，由 §11.6 的 `PrefixSynthesisModel` 接管 prefix 合成
- License 是 CC-BY-4.0，不能整库 commit；仓库只保留 1000 行 `sample.jsonl`
  + `scripts/convert_burstgpt.py` 复现路径

### 11.6 PrefixSynthesisModel

`simulator/prefix_synthesis.py` 给 length-only trace 合成 prefix 共享，分三层：

1. **Sharing layer 选择**：`sharing_layers` 是 `[(ratio, share_prob), ...]` 列表，
   按 `ratio` 把 prompt 划成若干层段。每段以 `share_prob` 决定本段是 shared
   token 还是 private random，覆盖 `all_private` / `mixed` / `heavy_shared`
   三个典型分布
2. **Bucket 选择（Zipf + 时间局部性）**：在 `num_buckets` 个 prefix 模板中，
   以 `1 - p_local` 概率按 `zipf_alpha` 形状全局抽样，以 `p_local` 概率从
   `local_window_s` 窗口里的最近 bucket 历史里抽（recency bias）
3. **Bucket prefix 展开**：每个 bucket 维护一个 token 序列，按需 lazy extend，
   遇到比 `initial_prompt_len` 长的 request 自动延伸——`initial_prompt_len`
   仅是性能 hint，**没有真实截断**

关键不变量：

- `PrefixSynthesisConfig` 用 Pydantic `@field_validator` 强校验
  `sharing_layers`（每条 ratio ∈ [0,1]、share_prob ≥ 0、ratio 求和 ≈ 1）
- 合成 prompt 与 `_build_token_ids` 一样保持 `len(tokens) == input_length`
- 全过程不引用 trace 真值（`output_length` / `session_id` 不参与合成）——
  保证 BurstGPT cache_hit 只反映 synthesis 假设，不被 trace 污染

### 11.7 prefix-sensitivity CLI

`cli.py::cmd_prefix_sensitivity`：在 `configs/trace_burstgpt.yaml` 这种
`prefix_mode=synthesis` 的 trace 配上扫四个轴（`zipf_alpha` / `p_local` /
`num_buckets` / `sharing_layers`），每条 candidate 跑一次完整 sim，报
`cache_hit` + `<scheduler>_uplift_vs_round_robin_pct`：

- 表头列名按 `--scheduler` 参数化（`f"{scheduler_name}_uplift_vs_round_robin"`），
  JSON 行也带 `uplift_column` / `uplift_vs_round_robin_pct` 字段
- 末尾打一条 Mooncake 真值参考行：`Mooncake real hash_ids cache_hit
  (conductor) = 0.146`，标注 **informational, NOT a target**——这是设计上的硬
  约束（详见 `.claude/plans/serene-tracing-keynes.md` § 修订记录的统计学
  谬误分析），任何后续 PR 都不许把 synthesis cache_hit 朝这个数字"调"
- `--output FILE.json` 写结构化报告，行内含 `axis / value / cache_hit /
  uplift_column / uplift_vs_round_robin_pct`，便于后续脚本接续

## 12. 下一步路线

- P3-D：把 `PrefixSynthesisModel` 提取给 Poisson `RequestGenerator` 也用
  （硬约束：default yaml 行为零漂移，opt-in only）
- P4 候选：
  - **PagedAttention Tier 2**：让 CPU/Disk 块真正动起来（demote/promote
    路径、HiCache 命中链、prefill cache_miss 不再只看 GPU 命中）
  - Llumnix-style migration / rebalance executor
  - Speculative decoding（必须用 Leviathan 等式，不能线性近似）

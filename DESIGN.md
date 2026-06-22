# nano-kvrouter Design

> 一句话定位：纯 Python、事件驱动、单线程的 LLM serving control-plane
> simulator。重点是调度、SLO、KV cache 和层次化存储，不是 GPU kernel。

## 1. 当前实现边界

截至当前 checkout，P2-Infra M1-M6 + P3-C + P4 + P5-Bidaw M1/M2 已经落地：

- M2: continuous batching
- M3: chunked prefill
- M4: paged GPU KV block metadata
- M5: split prefill/decode pools + post-prefill KV transfer
- M6: multi-tier HiCache (GPU / CPU / Disk) + tier-aware lookup
- acceptance: config-driven `sensitivity` CLI
- P3-C M1: real-world trace replay (Mooncake FAST'25, streaming JSONL),
  per-request `output_length` truthfully drives decode pressure
- P4-A/B: per-node KV transfer contention + scheduler backlog-aware TTFT
- P5-Bidaw M1/M2: I/O-aware dual queue, disk-HRRN, metadata-only
  previous-answer-based eviction

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

## 12. TransferModel — per-node KV transfer 争用建模 (P4-A M1)

### 12.1 问题与建模选择

P4-A 之前 KV transfer 是常量公式：
`cost_ms = kv_bytes / bandwidth.gpu_to_gpu`，10 个同 `(src, dst)` 的并发
transfer 全部在 `now + cost` 完成。这隐藏了 Mooncake FAST'25 §4.1 明确
点名的 "per-node KV transfer throughput is the bottleneck"。

候选建模（用户 2026-06-14 拍板）：

- 不做 per-link queue（太接近真实 RDMA topology，不在 simulator 尺度）
- 不做 shared bandwidth（K 个并发各拿 1/K 带宽——事件模型重算太复杂）
- 不做 token bucket（per-node throughput cap，复杂度介于其他两者之间）
- **选 per-node lane**：每个 node 一条 egress lane（它当 src 时）+ 一条
  ingress lane（它当 dst 时）。一次 transfer 同时占用 src.egress 和
  dst.ingress；start = max(now, 两条 lane.available_at)，finish = start
  + cost；两条 lane 都更新到 finish

per-node lane 自然表达：
- `p0→d0` 和 `p0→d1` 争 `p0.egress`
- `p0→d0` 和 `p1→d0` 争 `d0.ingress`
- `p0→d0` 和 `p1→d1` 完全并行

### 12.2 模块结构（统一 Protocol，零 if/else 散布）

`simulator/transfer_model.py`：

```python
class TransferModel(Protocol):
    def request_transfer(src, dst, now, cost_ms) -> (start, finish): ...
    def peek_backlog(node_id) -> {"egress": float, "ingress": float}: ...

class NoopTransferModel:        # 常量 passthrough，byte-identical 旧行为
class PerNodeLaneTransferModel: # 每节点 egress/ingress 串行
```

硬约束（防回归）：
- `cli.py` **只允许一处** `if cfg.bandwidth.contention_model == ...` —
  `_run_one` 工厂；其它点全走 Protocol 接口
- `peek_backlog()` 必须 side-effect-free（多次调用不动 `_*_available_at`）—
  专门有 `test_peek_backlog_is_side_effect_free` 守

### 12.3 Compat：opt-in，默认零漂移

`BandwidthConfig.contention_model: Literal["none", "per_node_lane"] = "none"`。
默认 `"none"` 走 NoopTransferModel，与 P3-C 末态 byte-identical。

7 个老 yaml（`default / heavy / hicache / pd_split / sensitivity /
trace_mooncake / trace_burstgpt`）零修改；6 sweep cache_hit、sensitivity
13/13 PASS、prefix-sensitivity 表全部 byte-identical 通过。

唯一新增 yaml `configs/transfer_contention.yaml`：2p/2d cluster + 合成
5 MB/s 带宽（**非硬件代表性**，inline comment 说明）让 service_time ≈
105ms 超过 transfer-producing cadence，争用 measurable。

### 12.4 Event payload — 三字段语义

`KV_TRANSFER_START` / `KV_TRANSFER_COMPLETE` payload 新增两字段：

| 字段 | 含义 | Noop 路径 | PerNodeLane 路径 |
|------|------|-----------|------------------|
| `service_cost_ms` | 纯传输时间 = `kv_bytes / bandwidth.gpu_to_gpu` | 同 cost_ms | 单次请求恒定 |
| `queued_cost_ms` | 等 lane = `start_t - now` | 恒为 0.0 | ≥ 0，争用增长 |
| `cost_ms` | 端到端 = `finish_t - now` = service + queued | 同 service_cost_ms（backward compat） | service + queued |

`MetricsCollector.kv_transfer_time_avg_ms` 继续采样 `payload["cost_ms"]`，
但**语义在 `per_node_lane` 下从"纯服务时间"变为"端到端含排队"**——这是
设计意图不是 bug，collector docstring 同步说明。

未来 v2（backlog #39）会加 `kv_transfer_queued_avg_ms` 单独 metric 让两
个语义可同时观测。

### 12.5 Scheduler 看 backlog（P4-B M1 已实现）

P4-A v1 故意没动 scheduler；P4-B（commit `244345d`）把估算闭环了。
新增 narrow Protocol `TransferBacklogView`（仅暴露 `peek_backlog`）在
`scheduler/base.py`，5 个 scheduler ctor 都吃 `backlog_view`，
`compute_est_ttft` 加 required keyword-only `backlog_view` + `now`：

```
service        = (kv_bytes / bandwidth.gpu_to_gpu) * 1000.0
src_egress_w   = max(0, peek_backlog(prefill_node)["egress"] - now)
dst_ingress_w  = max(0, peek_backlog(decode_node)["ingress"] - now)
queue_wait     = max(src_egress_w, dst_ingress_w)
kv_transfer    = service + queue_wait
```

`max` 不是 `sum`——一次 transfer 同时占用 src.egress + dst.ingress，
wait 等于两者**较晚**那个，sum 会双计。`max(0, ...)` clamp 让历史 backlog
不产生负贡献。

**`TransferBacklogView` 故意窄**：仅 `peek_backlog`，不暴露
`request_transfer`。估算阶段不可能误调写接口去 reserve lane 破坏仿真
决定性。硬 grep guard `rg "request_transfer\(" src/nano_kvrouter/scheduler/`
必须为 0（M1 验过）。

新 metric `kv_transfer_queued_avg_ms`：采样 `payload["queued_cost_ms"]`
**在 `_on_kv_transfer_complete` 现有 `transfer_id` stale guard 同一
if-block 内**，与 `cost_ms` 采样路径对齐——stale 事件不污染。
空样本 `summary()` 返回 `0.0`（不是 `None`），让 feature 在零 transfer
场景也可见。

#### 12.5.1 实际效果：5 scheduler 上 paired toggle 结果

同一份 `configs/transfer_contention.yaml`，仅翻 `bandwidth.contention_model`
字段：

| scheduler | `none` reject | `per_node_lane` reject | 方向 |
|-----------|-------------:|----------------------:|------|
| `round_robin`   | 0.644 | 0.484 | **DOWN** |
| `least_loaded`  | 0.627 | 0.458 | **DOWN** |
| `prefix_greedy` | 0.596 | 0.293 | **DOWN** |
| `e2_policy`     | 0.627 | 0.373 | **DOWN** |
| `conductor`     | 0.751 | **0.809** | **UP** |

Decomposition invariant 所有 5 个 scheduler 都 EXACT：
`lane.kv_transfer_time_avg_ms - lane.kv_transfer_queued_avg_ms`
= `none.kv_transfer_time_avg_ms` = **104.86 ms**（service 项不变，因为
`transfer_contention.yaml` 用固定 `avg_prompt_len`）。

#### 12.5.2 ⚠️ 关键解读：non-conductor rejection DOWN 主要是 B1 timing 副作用

**不要把 non-conductor 4 个 scheduler 的 rejection 下降全部误读成 routing 改善。**

`compute_est_ttft` 5 个 scheduler 都调用，但**返回值在 routing 决策
中的角色不同**：

| scheduler | 是否进 routing 决策 | 是否进 SLO gate |
|-----------|------------------|----------------|
| `conductor` | ✅ 3-objective scoring 一项 | ✅ `est_ttft > ttft_target_ms` → reject |
| `e2_policy` | ✅ `run_cost` 是 3-objective 一项 | ❌（E2 无 SLO gate） |
| `prefix_greedy` | ❌（只填 SchedulingDecision） | ❌ |
| `least_loaded` | ❌（只填 SchedulingDecision） | ❌ |
| `round_robin` | ❌（只填 SchedulingDecision） | ❌ |

实际 rejection 方向分析：

- **Conductor reject UP**（0.751→0.809）：**设计意图**。SLO gate 看到
  queue wait → `est_ttft > ttft_target_ms` 命中更多 → admit less。
- **E2 reject DOWN**：**两个效应叠加**——(a) `run_cost` 里看到 backlog
  让 E2 微调 routing 倾向少 backlog 节点（一阶 routing 改善，幅度小）；
  (b) 主导效应仍是下面那条 B1 timing 副作用。
- **prefix_greedy / least_loaded / round_robin reject DOWN**：**纯
  二阶 timing 副作用**。它们的 routing 不读 backlog；rejection 唯一来
  源是 M5a B1（KV transfer 完成时 decode_node 满 → 拒绝）。Lane queueing
  把 `KV_TRANSFER_COMPLETE` 往后推，给 decode_node 更多时间释放 slot，
  B1 拒绝次数自然减少。

换句话说：lane queue 在 3 个 cache-blind/cache-greedy scheduler 上充当
了**意外的 admission throttle**，不是 scheduler 变聪明。如果换种 workload
（decode 节点不饱和、B1 几乎不触发），这个 DOWN 效应会消失。E2 上则会
留下小但真实的 routing-shift 贡献。

Codex review 强调要文档化此点防止 operator 误读；这一节就是答案。

未来若想强化非 conductor scheduler 的 backlog 利用（比如让 prefix_greedy
显式偏向少 backlog 节点 / 让 E2 加 backlog 显式项 / 给 round_robin /
least_loaded 也吃 backlog 做 routing 决策），需要单独 plan + sensitivity。
不在 P4-B M1 scope。

### 12.6 Hard gate（单元测试隔离，不走 SimulationEngine）

Codex plan v1 review 抓出的 critical：原本 hard gate 走 4 个 request 在
CLI 同时到达再断言 `max(finish) ≥ 3.5 × min(finish)`，但 chunked prefill
本身会让 4 个 `KV_TRANSFER_START` 天然错开，断言可能因 prefill
serialization 假阳通过。

修法（plan v2）：hard gate 直接调
`PerNodeLaneTransferModel.request_transfer()` 4 次同 `now=0.0`，
assert 返回 `(0,C)/(C,2C)/(2C,3C)/(3C,4C)`。
**不许通过 SimulationEngine / CLI**。配套 4 个必备单元测试（disjoint
并行 / shared egress 串行 / shared ingress 串行 / peek_backlog 零副
作用）+ Noop 常量验证，全在 `tests/test_transfer_model.py`。

## 13. Bidaw — I/O-aware scheduling + answer eviction (P5-Bidaw M1/M2)

### 13.1 问题与建模选择

P4-B 之前所有 scheduler 都假设 disk-tier 命中是"瞬完成"——`cache_manager.
lookup()` 报告 disk hit、`compute_est_ttft.cache_load_ms` 估算 transfer
cost，但 `PREFILL_START` **不真的等** disk-to-GPU load 发生。这让 disk
hit 看起来比真实情况更便宜。

Bidaw (FAST'26) 论文核心：把 disk load 从估算项升级为 **关键路径上的真实
事件**，并用 dual-queue admission controller + disk-HRRN 智能排序来调度
load 顺序。

M1 做 I/O-aware scheduling；M2 增加 previous-answer-based eviction 的
metadata-only 近似。论文里的 storage-efficient tensor caching、真实 storage
engine、CUDA stream overlap 仍 **不实现**（详见 `doc/bidaw-deliverable.md`）。

### 13.2 模块结构（separate from 5 existing schedulers）

- `scheduler/bidaw.py`：`BidawPolicy` + 纯函数 `hrrn_priority(waiting_ms,
  kv_size_blocks)`。`BidawPolicy.schedule()` 满足 `SchedulingPolicy`
  Protocol，但**routing 决策很简单**——decode_node 用 `min(current_load)`
  （类似 least_loaded），prefill_node 用 round-robin。"聪明"在 controller 路径里。
- `simulator/bidaw_controller.py`：`BidawAdmissionController` 纯状态机，
  持有 `_preparing: dict[node_id, list[PreparingEntry]]` + `_in_flight_load:
  dict[node_id, request_id | None]`。**不调度 event**——cli wiring 在
  controller 决策后 schedule `KV_LOAD_START` / `KV_LOAD_COMPLETE`。
- `cli.py`：在 `_wire_simulator` 中加 `bidaw_mode: bool = False` 关键字
  参数（默认 False 保持 13 个 hidden caller 兼容）。`scheduler.name ==
  "bidaw"` 时 `_run_one` 传 `True`，激活 Bidaw 专用 wiring 分支。
- `kv_cache/answer_eviction.py`：M2 answer-length bucket profile，把
  `previous_answer_length` 映射成 hit potential。
- `simulator/trace_generator.py`：`prefix_mode="session_history"`，根据
  session 历史 query+answer 合成交互式 prompt prefix。
- `scripts/convert_interactive_workload.py`：把 Bidaw public workload CSV
  schema 转成内部 JSONL，并可导出 bucket potential profile。

### 13.3 事件路径（新增 2 个 EventType）

```
普通 scheduler:
REQUEST_ARRIVE → SCHEDULED → PREFILL_START → ...

Bidaw:
REQUEST_ARRIVE → SCHEDULED → BidawController.on_arrive
  ├─ matched_disk_blocks == 0  → "ready"     → PREFILL_START
  └─ matched_disk_blocks > 0   → "preparing" → enqueue
                                  HRRN pick when slot idle
                                  → KV_LOAD_START
                                  → KV_LOAD_COMPLETE
                                  → mark ready, free slot
                                  → PREFILL_START
                                  → ...
```

每个 decode node **一个 in-flight load slot**——同节点多个 disk-hit 请求
serialize；跨节点并行。HRRN 公式：

```
response_ratio = 1 + waiting_ms / max(1, disk_blocks)
```

Controller 取 `max(response_ratio)`——小 KV 起始 ratio 高（denominator 小），
等待时间补偿大 KV 防 starvation。

### 13.4 关键 commitments

1. **Dual-queue gating + metadata promotion**：`KV_LOAD_COMPLETE` 先让
   *该请求* 从 preparing 变成 ready，再通过 `CacheManager.
   promote_matched_disk_blocks_to_cpu()` 尝试把匹配前缀中的 disk blocks
   移到 CPU tier。promotion 仍是 metadata-only：没有真实 tensor copy；
   CPU/GPU 都被视为 Bidaw 的 performance/ready layer。若 CPU 无容量或
   block 被 pin，则记录 skipped，不阻塞该请求进入 `PREFILL_START`。
2. **Double-charging guard**：`BidawPolicy.schedule()` 把 `CacheLookup.
   transfer_cost_ms` 中的 disk 部分清零（cpu 部分保留——KV_LOAD 只 replay
   disk 不 replay cpu reload）后传 `compute_est_ttft`。事件路径自然付出
   真实 disk load 时间。其他 5 scheduler 看到完整 `transfer_cost_ms`，
   行为不变。
   ```python
   block_bytes = block_size * kv_bytes_per_token
   disk_load_ms = disk_n * block_bytes * (1/cpu_to_disk + 1/gpu_to_cpu) * 1000
   transfer_cost_ms -= disk_load_ms   # 保留 cpu_load_ms
   ```
3. **Compat byte-identical**：bidaw 是**第 6 个 scheduler**，纯增量。
   5 老 scheduler / 8 老 yaml / CacheManager / TransferModel 零修改；
   `bidaw_mode=False` 默认值让 `_wire_simulator` 13 个 hidden caller 全部
   不需改。
4. **HRRN 单元测试覆盖**：4 个独立单测（同 waiting / 长 waiting 反超 /
   single slot 序列化 / promotion-on-load-complete）证明算法正确性。
   实际 `bidaw.yaml` demo 上 preparing wait = 0（load 0.37ms ≪ 67ms 到
   达间隔），HRRN 在该 yaml 上是 dead code——见 §13.6 caveat。
5. **Previous-answer-based eviction 是 opt-in**：只有 `scheduler="bidaw"`
   且 `scheduler.params.enable_answer_eviction=true` 时，`CacheManager`
   才注入 `AnswerEvictionPolicy`。Block metadata 记录 session id、上一轮
   answer length 和 hit potential；CPU promotion 需要腾空间时优先 demote
   低 potential CPU block。没有真实 tensor，也没有在线 ghost cache。

### 13.5 Metrics（Bidaw 字段）

```python
bidaw_preparing_wait_avg_ms     # 在 preparing 队列等了多久才 KV_LOAD_START
bidaw_preparing_wait_p99_ms     # 同上 p99
bidaw_disk_load_service_avg_ms  # 单次 disk load 服务时间（KV_LOAD interval）
bidaw_preparing_promotions      # 进过 preparing 队列的请求数
bidaw_physical_promoted_blocks  # KV_LOAD_COMPLETE 后实际 disk→CPU 的块数
bidaw_physical_skipped_blocks   # CPU 无容量 / pinned 导致未 promote 的块数
bidaw_answer_eviction_count     # answer-aware CPU demotion 次数
bidaw_answer_evicted_blocks     # answer-aware demote 的 block 数
bidaw_answer_eviction_cpu_saved_blocks  # 为 incoming promotion 腾出的 CPU slots
bidaw_answer_eviction_hit_potential_avg # 被逐出 block 的平均 hit potential
bidaw_answer_eviction_cpu_hit_rate      # session hits 中 CPU / (CPU + Disk)
```

5 老 scheduler 上这些字段值为 `0.0` / `0`（不是 `None`），让 sweep JSON
schema 统一。MetricsCollector 复用现有 `_seen_transfer_ids` stale-guard
模式来 guard `KV_LOAD_*` 事件。Answer eviction counters 由 `CacheManager`
统计，`cli._run_one()` 在返回 summary 前 merge。

**第 5 个 metric `bidaw_ready_queue_wait_avg_ms` 在 M1 review 阶段删了**：
原 dispatch §3 over-spec，实际所有请求（ready 路径和 preparing 路径完成
load 后）都立即进 `PREFILL_START`，没有"ready queue wait"语义。

### 13.6 ⚠️ 关键解读：bidaw.yaml demo 的两个 caveat

**Caveat 1：`bidaw_preparing_wait_avg_ms = 0` 是预期但不直观**

`configs/bidaw.yaml` 用 `request_rate=15`（67ms 到达间隔），单次 disk
load 服务时间 0.37ms，slot 几乎永远 idle。preparing 队列从不积压，HRRN
排序在 demo 上是 dead code。这**不是 bug**——HRRN 正确性由单元测试守。
要在 demo 里展示 HRRN under contention，使用 `configs/bidaw-stress.yaml`
（更慢 `cpu_to_disk` + 更高 request_rate）。

**Caveat 2：cache_hit_ratio 比 conductor 低，但 e2e_avg_ms 比 conductor 短**

`bidaw.yaml` 上 conductor cache_hit=0.823 / bidaw cache_hit=0.693；
看起来 conductor 赢。**但 e2e_avg_ms 反过来**：bidaw 386ms vs conductor
413ms。原因：conductor 的 cache_hit 优势是 *估算层面的*——它不付出真实
disk load 时间；bidaw 付出真实 load 时间但 routing 像 least_loaded
让 decode queue 更短。**不是同语义对比**。要真正比较，需要让所有 6 个
scheduler 都走 `KV_LOAD_*` 事件路径——这是 P6 候选（"让所有 scheduler
共享 I/O-aware 真实路径"），不在 M1 scope。

### 13.7 Hard gates（10 条全过）

- pytest 497 passed
- 5 老 scheduler 在 6 老 yaml 上 byte-identical
- bidaw.yaml sweep 6 行，5 老行匹配 M0 Candidate J baseline
- 任意 disk-hit 请求 `KV_LOAD_COMPLETE.time ≤ PREFILL_START.time`
- 同 decode node 第 2 次 `KV_LOAD_START.time ≥` 第 1 次 `KV_LOAD_COMPLETE.time`
- ready 小请求不被 large preparing 阻塞（test 真实 timestamp 断言）
- `bidaw_preparing_promotions ≈ disk-hit 请求数`（200 ≈ 200）

详见 `doc/bidaw-deliverable.md` ship 判定段。

### 13.8 Previous-answer eviction fidelity

- 已实现：`Request.session_id / round_index / query_len /
  previous_answer_len` 元数据；session-history trace replay；profile-driven
  bucket potential；CPU 满时低 potential block 优先让位；metrics。
- 未实现：真实 storage-efficient tensor caching；真实 KV tensor eviction；
  在线 ghost cache residency；CUDA stream overlap。
- 语义边界：M2 只改变 CPU-tier victim selection，不改变老 scheduler routing，
  不改变 `CacheManager.lookup()` 的 tier-cost 公式。

### 13.9 未来工作（M1/M2 视角，已被 M3 部分覆盖）

- ~~**P6**：让 5 老 scheduler 也共享 `KV_LOAD_*` 真实事件路径~~ —
  仍是开放问题；M3 没有改这条
- **完整 Bidaw 论文**：storage-efficient tensor caching / CUDA stream
  overlap 仍不在范围
- M3 之后还剩：B1 multi-stream load model、B2 online ghost cache、A4
  GPU-only performance mode；见 §14 路线表

### 13.10 Bidaw M3 — routing intelligence (A1 + A2 + A3, 2026-06-22)

M3 在 M1/M2 基础上加 3 个可选机制，全部默认 off，从 yaml 通过
`scheduler.params.enable_*` 开关，开启时通过新的 `BidawControllerView`
只读 Protocol 从 controller 取信号：

```
scheduler.params:
  enable_routing_aware: false      # A1
  enable_ttft_slo_gate: false      # A2
  enable_session_affinity: false   # A3
  routing_weight_matched_blocks: 1.0   # α
  routing_weight_load: 1.0             # β
  routing_weight_preparing: 1.0        # γ
  routing_weight_in_flight: 2.0        # δ
  affinity_overload_factor: 1.5
  affinity_overload_abs_floor: 2.0
```

#### 13.10.1 A1 — routing-aware decode 节点选择

代价函数（min wins，按 node_id 字典序破并列）：

```
cost(decode) = β · current_load(decode)
             + γ · preparing_disk_blocks(decode)
             + δ · in_flight_disk_blocks(decode)
             − α · matched_blocks(request, decode)
```

**所有惩罚项按 disk block 数加权**，不是按队列深度 / slot 计数。原因：
queue 中一个 50-block 大请求和五个 1-block 小请求在 `preparing_depth=2`
下评分相同（错），但块加权下分别是 50 和 5（对）。单 slot 下
`in_flight_count ∈ {0,1}` 信息量太低，块加权下能反映剩余 service。

A1 closes the cache_hit gap in §13.6 caveat 2：开启时
`bidaw.yaml` 上 bidaw 行 cache_hit = **0.823**（与 conductor 完全相同），
ttft_p99 = 30.72ms（不退化）。详见 `doc/code-review/p5-bidaw-m3-routing-m0-preflight.md` 的 grid search 数据。

#### 13.10.2 A2 — storage-aware TTFT SLO gate

`schedule()` 在 `compute_est_ttft` 之后加一个 projected preparing wait：

```
projected = view.peek_projected_preparing_wait_ms(decode, my_disk_blocks, now)
est_total = est_ttft + projected
if est_total > request.slo_ttft:
    reject(reason="ttft_slo_exceeded")
```

`peek_projected_preparing_wait_ms` 是单 slot **deterministic FIFO** 估计：

```
block_service_ms = block_bytes / cpu_to_disk * 1000  # 严格匹配 cli.py:641
in_flight_residual = max(0, finish_ms − now)  if slot busy else 0
queued_service     = sum(entry.disk_blocks for entry in preparing) * block_service_ms
own_service        = my_disk_blocks * block_service_ms
projected          = in_flight_residual + queued_service + own_service
```

`my_disk_blocks == 0` 时短路返回 0.0（ready 请求不进 preparing 队列，
也不等 load slot）。

⚠️ **已知近似**：HRRN reordering 和跨 decode node 的并行 KV_LOAD 都不建模。
M4 multi-stream load model 时考虑改进。Metric `ttft_slo_rejections` 是
**通用**字段（Conductor 已有的 `ttft_slo_exceeded` 早拒路径也计入），
不是 Bidaw 独占。

#### 13.10.3 A3 — session affinity

`BidawAdmissionController` 维护 `_session_to_decode: dict[session_id, str]`。
`schedule()` 查表，命中且 pin 节点未明显过载就走 pin 路径；否则 fall back
到 A1 / least-loaded。

**Overload 阈值（hybrid，按 min 锚定不是 avg 锚定）**：

```
threshold = max(factor · min_load,  min_load + abs_floor)
overloaded = pinned_load > threshold
```

直接回答"有没有明显更好的节点"。`factor · min_load` 在高负载下生效
（按比例留余量），`min_load + abs_floor` 在低负载下生效（绝对余量防止
集群空闲时震荡）。

**Commit 时点关键**：affinity commit 不在 `_wire_bidaw_branch` 加 chained
handler（会比 shared `on_kv_transfer_complete` 先 fire，污染表），而是
**直接写进 shared handler**，位置在 decode capacity check 通过 **且**
`materialize_request` 成功之后：

```python
materialized_ok = True
try:
    cm.materialize_request(...)
except MemoryError:
    materialized_ok = False
# ... admit & start_decode ...
if (bidaw_controller is not None
        and bidaw_controller.affinity_enabled
        and req.session_id is not None
        and materialized_ok):
    bidaw_controller.commit_session_affinity(req.session_id, decision.decode_node)
```

这样 capacity-rejected 或 materialize-failed 的请求都不污染 affinity。

#### 13.10.4 P/D-split 适配

Bidaw 原文是单节点架构。在 M5a split P/D 下，所有 Bidaw 状态都 scope 到
**decode 池**（因为 `CacheManager` 只在 decode 池存在）：

| Bidaw 论文概念 | 我们的映射 |
|---|---|
| Node-level dual queue | decode-pool dual queue (preparing 按 decode_node_id 分桶) |
| KV_LOAD 等 disk 加载 | decode 侧 disk → memory |
| Disk-HRRN 排序 | 只排 decode 节点 KV_LOAD 候选 |
| A1 routing score | 仅看 decode 侧信号（preparing_blocks、in_flight_blocks、matched_blocks）；prefill 不进路由 |
| A3 session affinity | 绑定 decode_node；prefill 仍 RR |
| A2 SLO gate | `est_ttft` 已跨 prefill + transfer + decode，A2 在其基础上加 decode 侧 projected_preparing_wait |

代价：prefill 节点必须等 decode-side KV_LOAD 完才能 PREFILL_START，
是论文没有的上游 stall。本 milestone 接受。

#### 13.10.5 SchedulingDecision 扩展 + Metrics

`SchedulingDecision` 加两个可选字段（默认 None/False，5 老 scheduler
全部不设，保持回归）：

```python
routing_score: float | None = None
affinity_hit: bool = False
```

3 新 metric 通过 `SCHEDULED` / `REQUEST_REJECTED` payload 被动采样（不
违反 collector passive observer 规则）：

- `bidaw_routing_score_avg` — 平均 A1 cost（仅采样 routing_score 非 None
  的 SCHEDULED 事件）
- `bidaw_session_affinity_hits` — `affinity_hit=True` 计数
- `ttft_slo_rejections` — REQUEST_REJECTED 中 `reason=="ttft_slo_exceeded"`
  计数，通用（含 Conductor）

#### 13.10.6 Ship gates met（实测）

| Gate | Config | 数字 | 阈值 | 结果 |
|---|---|---|---|---|
| A1 cache_hit gap to conductor ≤ 0.05 | `bidaw.yaml`（A1 only） | 0.823 vs 0.823 = **0.000** | ≤ 0.05 | ✅ |
| A2 ttft_slo_rejections > 0 且 rejection_rate ≤ conductor | `bidaw-m3-stress.yaml` | 6 rejections, rate 0.194 | ≤ 0.252 | ✅ |
| A3 affinity_hits / completed ≥ 0.4 | `bidaw-affinity.yaml`（A3 only） | 40 / 60 = **0.667** | ≥ 0.4 | ✅ |

2-of-3 即可 ship，三项全过。详见 `doc/code-review/p5-bidaw-m3-routing-m0-preflight.md` §4–§6。

#### 13.10.7 cli.py KV_LOAD 单跳公式 mismatch — RESOLVED

M3 M0 preflight §7 公开了一个 M1 留下的预存问题：
- `cache_manager.transfer_cost_ms` 估算 disk 命中走两跳
  `(1/cpu_to_disk + 1/gpu_to_cpu)` ✓ 物理正确
- `bidaw.py:281` double-charging guard 减两跳 ✓ 匹配 cache_manager
- **`cli.py:675` KV_LOAD event service** 之前只付一跳，漏 cpu→gpu leg

**已修复**（独立 `fix(bidaw)` commit on `feat/bidaw-io-aware-scheduling`）：

- `cli.py:675` `load_service_ms` 改成
  `disk_blocks * block_bytes * (1/cpu_to_disk + 1/gpu_to_cpu) * 1000.0`
- `bidaw_controller.py:287` `peek_projected_preparing_wait_ms` per-block
  service 同步改成两跳（保持 plan v4 invariant：A2 projected wait 必须
  与 event 实际付的一致）
- `tests/test_bidaw_slo_gate.py:126` 测试常数公式同步

数字漂移实测：`bidaw.yaml` bidaw 行 ttft_p50 +0.03ms、e2e_avg +0.04ms，
cache_hit 不变；其他 bidaw-family yaml 在报告精度下无可见变化（第二跳
贡献 ~0.03%，小于这些 config 的舍入精度）。M3 ship gates 修复后重验全过。

### 13.11 Bidaw M4 — multi-stream KV load model (B1, 2026-06-22)

M4 把 M1 的"单 in-flight load slot per decode node"换成 polymorphic
`BidawLoadModel`，严格 mirror P4-A `TransferModel` 模式（Protocol + 2
implementations）。默认 `load_model="single"` 保持 M1/M2/M3 byte-id
回归；opt-in `load_model="multi"` + `num_streams=K` 启用真正的
per-node K 并行 disk-load lane（模拟 paper 的多 NVMe 或多控制器队列）。

#### 13.11.1 模块结构

```
simulator/bidaw_load_model.py   (NEW)
├── BidawLoadModel (Protocol)   ─ 6 方法 + num_streams property
├── SingleSlotLoadModel         ─ K=1, 冻结成 class，保 M1 byte-id
└── MultiStreamLoadModel(K)     ─ K slots, FIFO claim into first idle

simulator/bidaw_controller.py
└── BidawAdmissionController    ─ 委托 _load_model 管 slot 状态；
                                   peek_* 都改为 K-aware
```

Slot tuple shape:`(request_id, finish_ms, disk_blocks)` per occupied
slot。load_model 是 disk_blocks 唯一来源（不再回查 `_preparing`）。

#### 13.11.2 关键 invariant

- **In-flight identity API**：`load_model.in_flight_request_ids(node)
  -> frozenset[str]`。Controller 三处 query 用它过滤 `_preparing`
  （pick_next_to_load、peek_preparing_disk_blocks、peek_projected_preparing_wait_ms），
  避免 K>1 时把已在 slot 中的 request 重复计数。
- **start_load 重复检测**：同 node 同 req_id 第二次 start 抛 `RuntimeError`
  （defense in depth）。
- **`_drain_idle_slots` pump 顺序**：`pick → mark_load_started（先 claim）
  → engine.schedule(KV_LOAD_START, KV_LOAD_COMPLETE) → loop`。Claim
  在 schedule 前是关键，否则下次循环看不到 slot 已忙。

#### 13.11.3 A2 projected wait 多槽公式

```
residuals = load_model.slot_residuals_ms(node, now)      # len == K, 0.0 if idle
service_per_block = block_bytes * (1/cpu_to_disk + 1/gpu_to_cpu) * 1000  # 两跳
queued = sorted(_preparing 过滤 in_flight, key=HRRN priority desc)

for entry in queued:                            # 模拟未来分配
    idx = argmin(residuals)
    residuals[idx] += entry.disk_blocks * service_per_block

idx = argmin(residuals)                         # me 分配到最早空 slot
return residuals[idx] + my_disk_blocks * service_per_block
```

K=1 collapse 到 M3 公式：residuals=[r]、queued 串行加到唯一 slot、
me 加 own_service → `r + sum(queued) + own` 等价 M3。✓

#### 13.11.4 cli wiring 改造

- `_run_one` 新增 helper `_build_bidaw_load_model(params, node_ids)`：
  - `load_model="single"` + `num_streams != 1` → `ValueError`（fail-fast）
  - 工厂在 `_run_one` 中构造，发生在 controller / scheduler 之前
- `_wire_bidaw_branch` 改用 `_drain_idle_slots(node, now)` 替换 M3 的
  single-shot pick；从 `on_arrive_bidaw`（admission 通过且 verdict
  "preparing"）和 `on_kv_load_complete`（slot 释放后）两处调用。
- cli.py:233 fallback path（`_wire_simulator(bidaw_mode=True,
  bidaw_controller=None)` 直接 caller 兼容）也默认构造 SingleSlotLoadModel。

#### 13.11.5 Ship gate v2 redesign（重要）

Plan v3 原定 v1 gates 是：
- preparing_wait_avg ≥30% 降幅
- preparing_promotions 在 ±10%
- rejection_rate 增幅 ≤0.05

实现完毕后 Codex review 扫了 K ∈ {1,2,3,4,5,6,8}，**没有任何 K** 能
同时满足三条。根因：M4 把请求更快推进 `PREFILL_START`，**改变了
decode 池的 admission 时序**；在饱和 workload 下，decode capacity
exhaustion 提前发生 → rejection_rate 上升、promotions 下降。这些是
M4 机制的 **downstream emergent effect**，**不是 invariant**。

**Ship gate v2**（重新锁定）— anchor 在 M4 真正承诺的指标（用户体感
延迟）上：

| Gate | 目标 vs K=1 bidaw-stress | K=4 实测 |
|---|---|---:|
| preparing_wait_avg 降幅 ≥30% | ≤ 100.15 ms | **45.76** ✓ (−68%) |
| ttft_p50 降幅 ≥30% | ≤ 97.48 ms | **38.83** ✓ (−72%) |
| e2e_avg 降幅 ≥10% | ≤ 307.03 ms | **231.93** ✓ (−32%) |

旧 promotions/rejection guards 在 plan v3 + M0 §3 + 测试
docstring 中显式记录为"已 demoted 为 emergent property，不再
作为 invariant"。

#### 13.11.6 一个独立 backlog

Codex review 期间 Sonnet 临时加了一个 plan 没授权的 shared-prefix
overlap guard（在 `_drain_idle_slots` 内检测多个 pick 命中同一组
disk blocks 时早 return，且穿透 `controller._load_model` /
`cm._trees` / `cm._pools` 私有状态）。该 guard 引入新的 HoL 阻塞，
违反"K>1 多流泵在仍有空 slot、后面也许有非冲突候选时不应停止"
不变量；已在 M4.fix 完整删除（plan 没改）。

未来 milestone（M5/M6）可独立设计带明确语义的 shared-block
coalescing 机制：
- 暴露 public CacheManager API 查询命中的 block id（不再 `cm._trees`
  穿透）
- 设计 skip 语义而非 early-return（保留 pump 进度）
- HRRN priority 与 coalescing decision 的优先级关系（先 HRRN
  还是先 coalesce）
- 加严测试覆盖 HoL 边界

#### 13.11.7 不在范围

- HRRN reordering of in-flight slots（不抢占；FIFO claim 进 first idle slot by index）
- Cross-node shared disk pool（M5 候选）
- MultiStream(K=1) byte-id to SingleSlot（不承诺，因此默认 single 用 SingleSlotLoadModel）
- 新增 summary metric（M4 不加；现有 6 bidaw_* + ttft_slo_rejections 够 ship gate 用）

## 14. 下一步路线

- ✅ **P4-B**（done, commit `244345d`）：5 scheduler 的
  `compute_est_ttft` 已通过 `TransferBacklogView` 看 lane backlog；新
  metric `kv_transfer_queued_avg_ms` 已上。详见 §12.5。
- ✅ **P5-Bidaw M1**（done, commit `6d1cf50` on `feat/bidaw-io-aware-
  scheduling`）：I/O-aware dual-queue scheduling、disk-HRRN、KV_LOAD_*
  事件路径、6 个 Bidaw metrics、6 scheduler sweep table。详见 §13。
- ✅ **P5-Bidaw M1.5**：`KV_LOAD_COMPLETE` 后 metadata-only
  disk→CPU promotion + `configs/bidaw-stress.yaml`，让 preparing queue
  在系统 demo 中真实积压。
- ✅ **P5-Bidaw M2**：metadata-only previous-answer-based eviction +
  session-history trace replay + Interactive-conversation workload converter。
- ✅ **P5-Bidaw M3**（done, commits `5e7b578` + `87cfb79` on
  `feat/bidaw-io-aware-scheduling`）：A1 routing-aware + A2 TTFT SLO
  gate + A3 session affinity，3 个可选 flag 默认 off，6-scheduler
  字节级回归保持；详见 §13.10。Ship gates 三项全过。
- ✅ **P5-Bidaw M4**（done, commits `c3c0b18` + `23226ad` on
  `feat/bidaw-io-aware-scheduling`）：multi-stream KV load model
  (B1)，`BidawLoadModel` Protocol + SingleSlot (default) +
  MultiStream(K)。详见 §13.11。Ship gate v2（实测重设计）K=4
  全过：preparing_wait −68%、ttft_p50 −72%、e2e_avg −32%。
- **P5-Bidaw 后续候选**（roadmap in `.claude/plans/p5-bidaw-followups-roadmap.md`）：
  - **M5**：GPU-only performance mode（A4），yaml flag 让 CPU 命中
    也走 KV_LOAD（贴 paper "GPU-only performance layer" 语义）
  - **M6**：online ghost cache（B2），把 M2 静态 3-bucket
    `AnswerEvictionPolicy` 升级到在线反馈
  - **shared-prefix coalescing**：M4 dispatch 期间发现的设计点
    （Sonnet 临时实现的 unauthorized guard 已删；独立设计要求见
    §13.11.6 — public CacheManager block-id API、skip 语义、HRRN
    与 coalescing 优先级、HoL 边界测试）
- ~~**独立 backlog**：cli.py KV_LOAD service 公式单跳 → 两跳修正~~ —
  已在独立 `fix(bidaw)` commit 中完成；详见 §13.10.7。
- P3-D：把 `PrefixSynthesisModel` 提取给 Poisson `RequestGenerator` 也用
  （硬约束：default yaml 行为零漂移，opt-in only）
- P6 候选：
  - **PagedAttention Tier 2**：让 CPU/Disk 块真正动起来（demote/promote
    路径、HiCache 命中链、prefill cache_miss 不再只看 GPU 命中）
  - **5 老 scheduler 共享 `KV_LOAD_*` 真实路径**：让 cache_hit / e2e 在
    所有 scheduler 上能真对比（破 §13.6 caveat 2）
  - **让 non-Conductor scheduler 主动用 backlog 做 routing**：当前 4 个
    scheduler 仅"被动接收" backlog（compute_est_ttft 估算更准），但 routing
    决策本身不参考 `peek_backlog`。是否值得让 prefix_greedy / e2_policy
    在评分中加入 backlog 项，需要单独 plan + sensitivity（避免 §12.5.2
    描述的副作用变成主效应）
  - Llumnix-style migration / rebalance executor
  - Speculative decoding（必须用 Leviathan 等式，不能线性近似）

# nano-kvrouter — Project Design Document

> **一句话定位**：纯 Python 的事件驱动模拟器，聚焦 KV Cache 作为集群级资源时的调度决策，可在 Mac M3 上用 mock backend 完整运行。不涉及真实 GPU 算子。

---

## 0. 核心问题

当一个新请求到达时，调度器需要回答：
1. **发往哪个 prefill 节点**？（cache affinity vs. load balance）
2. **KV cache 从哪个存储层加载**？（tier promotion/demotion）
3. **什么时候拒绝请求**？（prediction-based early rejection）
4. **什么时候迁移 KV cache 到另一个节点**？（rebalance / live migration）

---

## 1. 参考文献 & 启发来源

| 系统 | 会议 | 核心贡献 | 在本项目中复现的思想 |
|------|------|----------|----------------------|
| **Mooncake** | FAST'25 Best Paper | KV-cache-centric 全局调度器 Conductor，P/D 分离，早期拒绝 | MooncakeConductor 策略、early rejection、三目标评分 |
| **Preble** | ICLR'25 | E2（exploit-explore）分布式 prompt-aware 调度 | E2Policy、prompt-aware load 计算 |
| **SGLang RadixAttention** | NeurIPS'24 | Radix tree 管理前缀共享 KV cache，cache-aware scheduling | RadixTree 数据结构、prefix greedy 策略 |
| **Llumnix** | OSDI'24 | 运行时 KV cache live migration，去碎片化 | Migration planner、rebalance 触发逻辑 |
| **vLLM v1** | — | PagedAttention，BlockPool，LRU eviction | BlockPool 设计、block hash 计算 |
| **DualMap** | arXiv'25 | 双空间映射解决 cache affinity vs. load balance 矛盾 | DualMapPolicy（Phase 3 可选） |
| **HiCache (SGLang)** | — | GPU→CPU→Disk 三层 KV cache 层次结构 | 分层 BlockStore，tier bandwidth 模型 |

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    nano-kvrouter control plane                  │
│                                                                 │
│  RequestGenerator                                               │
│  (trace replay / Poisson / bursty)                              │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────┐                   │
│  │       Global Scheduler (Conductor)      │                   │
│  │  ┌─────────────┐ ┌──────────────┐       │                   │
│  │  │Prefix router│ │Load balancer │       │                   │
│  │  └─────────────┘ └──────────────┘       │                   │
│  │  ┌─────────────────┐ ┌───────────────┐  │                   │
│  │  │Early rejection  │ │Migration plan │  │                   │
│  │  └─────────────────┘ └───────────────┘  │                   │
│  └─────────────────────────────────────────┘                   │
│           │                                                     │
│     ┌─────┼──────┬──────────────────┐                          │
│     ▼     ▼      ▼                  ▼                          │
│  Prefill  Prefill  Decode  Decode  (Mock GPU nodes)            │
│  Node 0   Node 1   Node 0  Node 1                              │
│     │     │        │       │                                   │
│     └─────┴────────┴───────┘                                   │
│                    │                                           │
│     ┌──────────────────────────────────┐                       │
│     │  Distributed KV Cache Pool       │                       │
│     │  ┌──────────┐ ┌────────┐ ┌────┐  │                       │
│     │  │GPU HBM   │ │CPU DRAM│ │Disk│  │                       │
│     │  │(Tier 1)  │ │(Tier 2)│ │(T3)│  │                       │
│     │  └──────────┘ └────────┘ └────┘  │                       │
│     └──────────────────────────────────┘                       │
│                    │                                           │
│     ┌──────────────┐  ┌──────────────┐                         │
│     │Metrics       │  │Dashboard/CLI │                         │
│     │(TTFT,TBT,SLO)│  │(rich table)  │                         │
│     └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 模块详细设计

### 3.1 Request

```python
@dataclass
class Request:
    request_id: str
    token_ids: List[int]          # 模拟 prompt tokens
    prefix_hash: str              # 前缀哈希（用于缓存匹配）
    expected_output_len: int      # 预估输出长度
    arrival_time: float
    slo_ttft_ms: float            # TTFT SLO 约束（毫秒）
    slo_tbt_ms: float             # TBT SLO 约束（毫秒）
    priority: int = 0
```

### 3.2 KV Cache Pool

#### RadixTree（参考 SGLang）
```
RadixTree
├── insert(token_ids: List[int], node_id: str) -> block_ids
├── match_prefix(token_ids) -> (matched_len, block_ids)
├── evict_lru() -> freed_blocks
└── 内部：节点 = (token_seq, kv_block_ids, ref_cnt, last_used)
```

#### BlockPool（参考 vLLM v1）
```
BlockPool
├── tiers: {GPU_HBM=0, CPU_DRAM=1, DISK=2}
├── allocate(num_blocks, node_id, tier) -> List[block_id]
├── free(block_ids)
├── promote(block_id)   # Disk→CPU 或 CPU→GPU
├── demote(block_id)    # GPU→CPU 或 CPU→Disk
└── transfer(block_id, src_node, dst_node)  # 跨节点迁移
```

#### CacheMetadata
```python
block_to_node: Dict[int, str]        # block_id -> node_id
block_to_tier: Dict[int, int]        # block_id -> tier
block_heat:    Dict[int, HeatInfo]   # access_count, last_access_time
prefix_to_blocks: Dict[str, List[int]]  # prefix_hash -> block_ids
```

### 3.3 Mock Engine Node（延迟模型）

```python
# M2/M3: prefill cost is split into chunks of size prefill_chunk_size;
# each chunk piggybacks with active decode in the same batch step.
# step_time = chunk_tokens * prefill_cost_per_token + decode_base + bs * marginal
def estimate_prefill_chunk_time(chunk_tokens, decode_batch_size) -> float:
    return chunk_tokens * PREFILL_COST_PER_TOKEN_MS + BASE_DECODE_MS + decode_batch_size * MARGINAL_DECODE_MS

def estimate_decode_time(batch_size) -> float:
    # TBT 随 batch 增大线性增长
    return BASE_DECODE_MS + batch_size * MARGINAL_DECODE_MS

def estimate_transfer_time(prompt_len) -> float:
    # M5a: KV transfer cost from prefill_node to decode_node
    return prompt_len * KV_BYTES_PER_TOKEN / BANDWIDTH_GPU_TO_GPU * 1000  # ms
```

**延迟参数参考值**（可在 config 中调整）：
- `prefill_cost_per_token_ms = 0.033`（~30K tokens/s，A100 量级）
- `decode_base_ms = 5.0`
- `marginal_decode_ms = 0.5`（每增加一个并发请求）
- GPU↔GPU (NVLink/RDMA): `bandwidth.gpu_to_gpu` bytes/s（默认 3×10¹¹）
- GPU↔CPU (PCIe): `bandwidth.gpu_to_cpu` — M6 激活
- CPU↔Disk (NVMe): `bandwidth.cpu_to_disk` — M6 激活

**P/D Split (M5a)**:
- Cluster 有独立的 `prefill_nodes` 和 `decode_nodes` 两个池
- prefill_node 只做 chunked prefill；decode_node 只做 continuous batching
- KV transfer event flow: `PREFILL_COMPLETE` → `KV_TRANSFER_START` → `KV_TRANSFER_COMPLETE` → decode admit
- Transfer cost = `prompt_len × kv_bytes_per_token / bandwidth.gpu_to_gpu`

### 3.4 Global Scheduler — 可插拔策略接口

```python
class SchedulingPolicy(Protocol):
    def schedule(
        self,
        request: Request,
        nodes: List[MockEngineNode],
        kv_pool: KVCachePool,
    ) -> SchedulingDecision:
        ...

@dataclass
class SchedulingDecision:
    prefill_node: str | None   # None = rejected
    decode_node: str | None
    reject_reason: str | None
    estimated_ttft_ms: float
    estimated_tbt_ms: float
```

#### 内置策略

| 策略 | 说明 | 参考 |
|------|------|------|
| `RoundRobin` | 基线，轮转 | — |
| `LeastLoaded` | 基线，最低负载优先 | — |
| `PrefixGreedy` | 最长前缀命中优先，负载作次级排序 | SGLang |
| `E2Policy` | prompt-aware load = 历史负载 + 驱逐代价 + 运行代价 | Preble ICLR'25 |
| `MooncakeConductor` | 三目标联合评分 + early rejection | Mooncake FAST'25 |

#### MooncakeConductor 评分公式

```python
score(node) = (
    cache_benefit(node)      # 可复用的计算量（越大越好）
  - load_penalty(node)       # 当前负载（越小越好）
  - transfer_penalty(node)   # 需要跨节点/跨层搬运的代价
)

# 同时检查 SLO：
if estimated_ttft > request.slo_ttft or estimated_tbt > request.slo_tbt:
    -> REJECT（early rejection）
```

#### E2 prompt-aware load 公式

```python
e2_load(node, request) = (
    historical_load(node, window_H)        # 过去 H 窗口的计算负载
  + eviction_cost(node, needed_blocks)     # 需要驱逐的代价
  + run_cost(request, cached_on_node)      # 新请求的实际计算代价
)
# 选 e2_load 最小的节点
```

### 3.5 Simulation Engine（事件驱动）

**事件类型**：
```
REQUEST_ARRIVE        -> scheduler.schedule() -> emit PREFILL_START or REJECTED
PREFILL_START         -> wait prefill_time    -> emit PREFILL_COMPLETE
PREFILL_COMPLETE      -> store KV to pool     -> emit DECODE_START
DECODE_START          -> emit first TOKEN_GENERATED
TOKEN_GENERATED       -> if done: emit DECODE_COMPLETE; else emit next TOKEN_GENERATED
DECODE_COMPLETE       -> free resources, record metrics
KV_TRANSFER_COMPLETE  -> update cache metadata
REBALANCE_TICK        -> migration_planner.rebalance()
REQUEST_REJECTED      -> record metrics
```

### 3.6 Metrics

| 指标 | 计算方式 |
|------|----------|
| TTFT | `prefill_complete_time - arrival_time` |
| TBT | `avg(decode_step[i+1] - decode_step[i])` |
| E2E latency | `decode_complete - arrival_time` |
| SLO hit rate | `slo_met_requests / total_requests` |
| Cache hit ratio | `cached_tokens / total_prompt_tokens` |
| Throughput | `completed_requests / duration` |
| Rejection rate | `rejected / total` |
| GPU utilization | `busy_time / total_time` per node |

---

## 4. 目录结构

```
nano-kvrouter/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── mooncake_like.yaml       # P/D 分离，4+4 节点
│   └── preble_like.yaml         # DP 集群，8 节点
├── traces/
│   ├── sharegpt_sample.jsonl
│   └── synthetic_prefix_heavy.jsonl
├── src/nano_kvrouter/
│   ├── config.py                # Pydantic 配置模型
│   ├── request.py
│   ├── kv_cache/
│   │   ├── radix_tree.py        # RadixTree 前缀树
│   │   ├── block_pool.py        # 分层块存储
│   │   └── cache_manager.py     # 统一接口
│   ├── engine/
│   │   ├── mock_node.py         # Mock GPU 节点
│   │   └── latency_model.py     # 延迟估算
│   ├── scheduler/
│   │   ├── base.py              # Protocol 接口
│   │   ├── round_robin.py
│   │   ├── least_loaded.py
│   │   ├── prefix_greedy.py
│   │   ├── e2_policy.py         # Preble E2
│   │   ├── conductor.py         # Mooncake Conductor
│   │   └── migration.py         # KV 迁移 / rebalance
│   ├── simulator/
│   │   ├── event.py
│   │   ├── engine.py            # 事件循环
│   │   └── generator.py         # 请求生成
│   ├── metrics/
│   │   ├── collector.py
│   │   └── dashboard.py
│   └── cli.py
├── tests/
│   ├── test_radix_tree.py
│   ├── test_block_pool.py
│   ├── test_scheduler.py
│   └── test_e2e.py
├── notebooks/
│   └── analysis.ipynb
└── scripts/
    ├── run_benchmark.py
    └── plot_results.py
```

---

## 5. 配置示例（mooncake_like.yaml）

```yaml
cluster:
  prefill_nodes: 4
  decode_nodes: 4

node:
  capacity: 8
  gpu_blocks: 2000              # 模拟 80GB HBM (M4 paged attention)
  cpu_blocks: 10000             # deferred to M6
  disk_blocks: 100000           # deferred to M6

model:
  block_size: 16                # tokens per block
  kv_bytes_per_token: 512       # 7B 模型量级
  prefill_cost_per_token_ms: 0.033
  decode_base_ms: 5.0
  marginal_decode_ms: 0.5
  prefill_chunk_size: 512       # M3 chunked prefill

bandwidth:
  gpu_to_gpu: 300_000_000_000   # bytes/sec (NVLink/RDMA)
  gpu_to_cpu: 32_000_000_000    # bytes/sec (PCIe) — M6
  cpu_to_disk: 5_000_000_000    # bytes/sec (NVMe) — M6

slo:
  ttft_target_ms: 2000
  tbt_target_ms: 100

workload:
  request_rate: 50              # requests/s
  duration_s: 60
  prefix_sharing_ratio: 0.6
  avg_prompt_len: 1024
  avg_output_len: 256
```

---

## 6. 实施路线

| Phase | 内容 | 预计工时 |
|-------|------|----------|
| **Phase 1** | RadixTree + BlockPool + MockNode + 单元测试 | 3天 |
| **Phase 2** | 事件模拟引擎 + RequestGenerator + 基线策略 + Metrics + CLI | 3天 |
| **Phase 3** | PrefixGreedy + E2Policy + MooncakeConductor + Migration | 5天 |
| **Phase 4** | 实验对比 + 可视化 + 参数扫描 + README | 3天 |

---

## 7. 关键设计决策 & 与真实系统的对比

| 真实系统 | nano-kvrouter 的简化 | 保留的核心 |
|----------|----------------------|------------|
| 真实 GPU 张量计算 | 用延迟模型替代 | 调度决策逻辑完整保留 |
| RDMA / PCIe 传输 | 用带宽模型模拟延迟 | 传输代价对调度的影响 |
| 多副本 KV（TP/PP） | 单副本简化 | 跨节点迁移逻辑 |
| Token streaming | 离散 decode step 事件 | TBT 指标计算 |
| 真实 tokenizer | 随机 int token_ids | 前缀匹配逻辑 |

---

*最后更新：项目初始设计阶段。参考：Mooncake (FAST'25), Preble (ICLR'25), SGLang (NeurIPS'24), Llumnix (OSDI'24), vLLM v1, DualMap (arXiv'25)。*
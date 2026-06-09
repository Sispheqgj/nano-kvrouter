# `src/nano_kvrouter/kv_cache/cache_manager.py` Code Review

## 基本信息

- 源码文件：`src/nano_kvrouter/kv_cache/cache_manager.py`
- 审阅日期：2026-06-09
- 对照材料：
  - SGLang RadixAttention
  - vLLM v1 / PagedAttention
  - Mooncake FAST'25
  - HiCache / hierarchical KV cache discussions
- 当前状态：已根据 P2-Infra M6 实现同步

## 文件职责

`CacheManager` 是 decode-side KV cache 的统一读写接口。

它把三件事情收敛到一个模块里：

1. prefix lookup
2. KV block capacity / tier accounting
3. admit / demote / free 的控制面语义

调度器不直接依赖 `RadixTree` 或 `BlockPool`，而是通过 `CacheLookup`
读取：

- `matched_tokens`
- `matched_blocks_by_tier`
- `transfer_cost_ms`

## 当前已经完成了什么

### 1. decode-side cache source of truth

当前每个 decode node 都有：

- 一个 `RadixTree`
- 一个 `BlockPool`

`BlockPool` 是物理 block accounting 的 source of truth，`RadixTree`
负责 prefix -> block ownership 和 LRU path。

### 2. M4 paged GPU block accounting

当前 admit 仍保留了 M4 的核心约束：

- oversized prompt 在 GPU tier 上 fail fast
- split-aware admit 需要为非对齐 split 预留 `+1` block headroom
- 容量压力优先通过 LRU path 释放或降级 block

### 3. M6 multi-tier demotion chain

现在 `cpu_blocks` / `disk_blocks` 已经不是占位字段，而是 live capacity：

```text
GPU -> CPU -> Disk -> free
```

当 GPU tier 满时，旧 block 会沿 demotion chain 下移；如果下游 tier 也满，
才会真正 free 并清理 tree 中的 zombie path。

### 4. tier-aware lookup

`lookup()` / `lookup_all()` 已经表达：

- matched path 上每个 block 在哪个 tier
- CPU hit 和 Disk hit 的 reload 代价
- `matched_blocks_by_tier` 供 metrics / sensitivity 使用

### 5. transfer-cost semantics

当前命中代价语义是：

- GPU hit: `0 ms`
- CPU hit: `block_bytes / bandwidth.gpu_to_cpu`
- Disk hit:

```text
block_bytes * (1 / bandwidth.cpu_to_disk + 1 / bandwidth.gpu_to_cpu)
```

Disk 两段 hop 串行相加是当前 simulator 对冷层命中的明确建模选择。

## 对照材料说明了什么

### SGLang / RadixAttention

SGLang 的关键点是：

- prefix tree 管理共享前缀
- cache-aware scheduler 看的是“哪台机器命中的 prefix 最长”

本仓库保留了这层 prefix-aware control plane，但把“命中后怎么加载”
扩展成了 tier-aware 版本。

### vLLM v1 / PagedAttention

vLLM 的启发不是“真实 tensor 怎么排布”，而是：

- KV cache 用 block metadata 管理
- 物理容量压力需要独立于请求生命周期建模

本仓库保留了 block-level capacity accounting，不做真实 kernel / page table。

### Mooncake

Mooncake 对当前文件的主要启发是：

- split P/D 下，decode side 的 prefix state 是真实调度输入
- transfer cost 应该参与 TTFT / SLO 判断

需要强调的是：Disk-tier hit 的两段传输并不是 Mooncake 论文里的逐字公式，
而是为了模拟层次化 cache load cost 做的 extrapolation。

## 当前问题与差距

| 问题 | 影响 | 证据 | 优先级 |
|------|------|------|--------|
| `worst_case_new = min(total_blocks + 1, capacity_gpu)` 仍可能 over-evict | 某些不触发 split 的 admit 会多驱逐一个 block | admit 逻辑仍保留保守 headroom | 中 |
| tier cost 只建模带宽，不建模并发 contention | CPU/Disk hit 延迟是 deterministic lower-fidelity simulator value | `transfer_cost_ms` 只看 block bytes / bandwidth | 中 |
| cross-node remote tier hit 未单独建模 | 当前 HiCache 只表达 decode node 本地 tier load，不表达更复杂的远端层次化路径 | `CacheLookup.transfer_cost_ms` 聚焦本节点 reload | 低 |

## 后续 PR 候选

| PR | 改动范围 | 验证方式 |
|----|----------|----------|
| PR-1：split dry-run 减少 over-evict | 优化 admit headroom 估算 | 新增 no-split / split 对照测试 |
| PR-2：tier-load contention model | 在 `transfer_cost_ms` 里加入更细的拥塞抽象 | sensitivity 对比不同 tier bandwidth 组合 |
| PR-3：remote tier path 建模 | 明确区分本地 tier hit 和更复杂的远端 cache 路径 | scheduler / metrics 集成测试 |

## 学习笔记

这个文件体现了当前 simulator 的核心取舍：

- tree 负责“逻辑命中”
- pool 负责“物理容量”
- manager 负责把两者粘起来，并把 tier hit 代价转成 scheduler 能消费的数字

这使得 README / DESIGN 里说的 13 个 LIVE 字段能真正映射到实现：

- `gpu_blocks` / `cpu_blocks` / `disk_blocks` 控容量
- `gpu_to_cpu` / `cpu_to_disk` 控 tier hit 代价
- `kv_bytes_per_token` 控 block reload / transfer 尺度

如果未来要继续提高 fidelity，优先级最高的不是把这个文件变成真实 cache runtime，
而是继续把 control-plane 上真正影响调度的信号提纯。

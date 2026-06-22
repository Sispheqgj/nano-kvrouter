# P5-Bidaw M1/M2 交付总结（中文）

> **范围**：Bidaw (FAST'26) 论文的 I/O-aware request scheduling 层
> + previous-answer-based eviction 的 metadata-level 模拟实现。
> **不**包含真实 storage engine、CUDA stream、tensor caching、
> storage-efficient tensor caching 等论文底层模块。
>
> **分支**：`feat/bidaw-io-aware-scheduling`
> **Commit**：M1 为 `6d1cf50`；M2 当前在本工作树实现，待提交。
> **基线**：`origin/main = 3767be9`（P4-B 后）
> **测试**：497 passed。
>
> 完整设计：DESIGN §13。本文档面向用户的 ship 判定 + 论文-实现 mapping。

---

## 1. 实现了哪些 Bidaw 机制

| Bidaw 论文机制 | 本项目实现 | 位置 |
|---|---|---|
| 双队列（ready / preparing） | `BidawAdmissionController` 维护 `_preparing: dict[node_id, list]` + per-node `_in_flight_load` slot | `src/nano_kvrouter/simulator/bidaw_controller.py` |
| disk-HRRN 排序 | 纯函数 `hrrn_priority(waiting_ms, kv_size_blocks) = 1 + waiting_ms / max(1, kv_size_blocks)`；controller `pick_next_to_load()` 取 `max(ratio)` | `src/nano_kvrouter/scheduler/bidaw.py` + `bidaw_controller.py:116-140` |
| KV load 作为关键路径事件 | 新增 `EventType.KV_LOAD_START` / `KV_LOAD_COMPLETE`；preparing 请求必须等 `KV_LOAD_COMPLETE` 才能 `PREFILL_START` | `src/nano_kvrouter/simulator/event.py:94-95` + cli wiring |
| Per-node 单 load slot | `_in_flight_load: dict[str, str \| None]`；`mark_load_started` busy 时 raise；同节点 disk loads 串行，跨节点并行 | `bidaw_controller.py:142-152` |
| Request-local promote to ready | `KV_LOAD_COMPLETE` 触发 `mark_load_completed` → 控制器把请求从 preparing 取出 → cli 调度 `PREFILL_START` | `bidaw_controller.py:154-180` + cli wiring |
| Metadata disk→CPU promotion | `KV_LOAD_COMPLETE` 后调用 `CacheManager.promote_matched_disk_blocks_to_cpu()`，把匹配前缀中的 disk blocks 尝试移动到 CPU tier | `src/nano_kvrouter/kv_cache/cache_manager.py` + cli wiring |
| Previous-answer-based eviction（M2） | request 携带 `session_id/previous_answer_len`；CacheManager 给 blocks 打 hit-potential 元数据；CPU 满时优先 demote 低潜力 block | `src/nano_kvrouter/kv_cache/answer_eviction.py` + `cache_manager.py` |
| Interactive workload converter（M2） | 把 `User_id, Timestamp(seconds), Query_length, Response_length, Round_index` 转成 session-history JSONL，并生成 answer profile JSON | `scripts/convert_interactive_workload.py` |
| Session-history trace replay（M2） | `trace.prefix_mode: session_history` 根据同一 session 的历史 query+answer 合成 prompt prefix | `src/nano_kvrouter/simulator/trace_generator.py` |
| Double-charging guard | `BidawPolicy.schedule` 把 disk 部分从 `transfer_cost_ms` 减掉再传 `compute_est_ttft`，避免估算 + 真实路径双计费 | `bidaw.py:141-160` |
| 选择性 scheduler 启用 | `cli.py` `SCHEDULER_NAMES` 加 `"bidaw"`；`_wire_simulator(bidaw_mode=False)` 默认值保护其他 13 个 hidden caller；只有 `scheduler.name=="bidaw"` 时构造 controller + 接 KV_LOAD handlers | `src/nano_kvrouter/cli.py:62/124/816 area` |
| Bidaw metrics | preparing wait / disk-load service / preparing promotions / physical promoted+skipped blocks / answer eviction counters | `src/nano_kvrouter/metrics/collector.py` + `CacheManager.answer_eviction_summary()` |

## 2. 哪些是 simulator 近似

| 项 | 近似形式 | 与论文差距 |
|---|---|---|
| KV block tier 状态 | `KV_LOAD_COMPLETE` 后会尝试 metadata-only disk→CPU promotion；CPU 无容量或 block pinned 时会 skip | 论文里是真实 tensor residency；本项目只移动 block metadata，不复制 tensor |
| Load 并发度 | **每 decode node 一个 in-flight slot**，跨节点并行 | 论文 figure 7 可能允许多 stream 并发；本项目用 single slot 是 P4-A `TransferModel` 模式的简化复用 |
| Storage 带宽 | 沿用 `bandwidth.cpu_to_disk` 算 service_time = `disk_blocks * block_bytes / cpu_to_disk * 1000` | 论文真实 NVMe / SSD 异步 I/O；本项目是静态带宽公式 |
| Performance layer 映射 | gpu + cpu tier 都视为 "ready"，disk tier 视为 "preparing" | 论文严格按 GPU HBM = performance；本项目把 CPU DRAM 也算 performance（避免 CPU hit 也走 KV_LOAD，与 P4-A/B 的 cache_load_ms 估算冲突） |
| Previous-answer eviction | 用 previous answer length bucket 估计 hit potential，CPU promotion 需要腾空间时按 potential 选 victim | 论文有 ghost cache / weighted reuse distance 分析；本项目只保留 profile 结果和控制面选择，不实现真实 tensor eviction |
| Bidaw 路由策略 | decode_node 选 `min(current_load)`，prefill_node round-robin | 论文 Bidaw 论文重点不在 routing 是 I/O gating；本项目沿用 least_loaded 风格保持简单 |
| Scheduler 估算与真实事件的关系 | `BidawPolicy.schedule` 把 disk 部分清零；保留 cpu 部分；CPU reload 仍由 `compute_est_ttft` 估算 | 论文里 estimate 与 runtime 应一致；本项目保持 5 老 scheduler 估算路径不变 + Bidaw 单独走真实事件 |

## 3. 哪些没有实现（明确不在范围）

| 论文机制 | 不做原因 |
|---|---|
| **Storage-efficient tensor caching** | 需要真实 tensor format conversion + memory layout，超出 metadata 模拟器尺度 |
| **CUDA stream overlap** | 项目 AGENTS.md 明确"no real GPU execution; all time is simulated"，stream 模型与事件循环架构不匹配 |
| **真实 previous-answer tensor eviction** | 已实现控制面元数据近似，但不保存/移动/压缩真实 KV tensor |
| **Ghost cache 在线模拟** | 当前只读离线 profile 的 bucket potential，不维护真实 ghost cache residency |
| **Cross-node KV migration** | Llumnix 领域，不是 Bidaw 论文核心 |
| **真实 disk async I/O / SSD model** | 同上 metadata 边界 |
| **真实 disk→cpu/gpu tensor promotion** | 已实现 metadata-only disk→CPU tier move，但不复制真实 tensor |

## 4. 对比表：round_robin / conductor / bidaw on `configs/bidaw.yaml`

实测自 commit `6d1cf50`（branch `feat/bidaw-io-aware-scheduling`），`uv run python -m nano_kvrouter.cli sweep --config configs/bidaw.yaml`。

| 指标 | round_robin | conductor | **bidaw** | 解读 |
|---|---:|---:|---:|---|
| `cache_hit_ratio` | 0.589 | **0.823** | 0.693 | conductor 看似赢，但**不付真实 disk load 时间**（见 §13.6 caveat 2） |
| `cache_hit_by_tier_ratio` | gpu 0.258 / disk 0.742 | gpu 0.690 / disk 0.310 | gpu 0.123 / disk 0.877 | conductor cache-aware 把请求路由到有 gpu hit 的节点；bidaw 像 least_loaded 不挑节点 |
| `ttft_p50_ms` | 12.150 | 11.671 | **11.459** | bidaw 最低（least_loaded 风格 + 平均 disk load 0.37ms） |
| `ttft_p99_ms` | 31.656 | **30.720** | 32.037 | bidaw 尾部略高（KV_LOAD 事件加了 small 但真实的开销） |
| `tbt_avg_ms` | 5.834 | 6.355 | 5.900 | decode 阶段差异不大 |
| `e2e_p50_ms` | 383.511 | 403.687 | **384.934** | bidaw 接近 round_robin（最快），conductor 因为 routing 集中到 cache 节点导致 queue 长 |
| **`e2e_avg_ms`** | 383.933 | 412.587 | **385.976** | **bidaw 比 conductor 快 6.4%**——cache_hit 的"假"优势被真实 disk load 抹平 |
| `throughput_req_per_s` | 14.237 | 14.226 | 14.246 | 三者相当（系统未饱和） |
| `rejection_rate` | 0.000 | 0.000 | 0.000 | 无 SLO 拒绝 |
| `bidaw_preparing_wait_avg_ms` | 0.000 | 0.000 | **0.000** | load 0.37ms ≪ 67ms 到达间隔 → preparing 永远空（caveat 1） |
| `bidaw_disk_load_service_avg_ms` | 0.000 | 0.000 | **0.367** | 每次 disk load 实际服务时间（disk blocks × block bytes / cpu_to_disk） |
| `bidaw_preparing_promotions` | 0 | 0 | **200** | 进过 preparing 路径的请求数 ≈ 命中 disk 的请求数 |

**两条关键结论**：

1. **bidaw 的 e2e_avg 比 conductor 短 6.4%**（386 vs 413 ms）。conductor 的 cache_hit 优势是估算层面的（不付真实 disk load），而 bidaw 付出真实 load 但用 least_loaded 风格让 decode queue 更短。**不同语义对比要谨慎解读**——同语义对比需要让所有 6 scheduler 都走 `KV_LOAD_*` 真实路径（P6 候选）。
2. **bidaw 的 preparing 队列在 demo 上从不积压**（avg=0, p99=0）。HRRN 的正确性由 4 个单元测试守住（同 waiting 小 KV 优先、长 waiting 反超、single slot 序列化、promotion on complete），但 bidaw.yaml 这个 workload 太轻没把 contention 压出来。要看 HRRN 真正活起来，需要 `bidaw-stress.yaml`（更慢 `cpu_to_disk` 或更高 `request_rate`）。

## 5. 测试覆盖

**新增/扩展的测试覆盖**：

- `tests/test_bidaw_scheduler.py` (~8 tests)：HRRN 数学正确性、BidawPolicy 双计费 guard 在 CPU+Disk mixed hit 下正确（Codex M1.fix 抓到的 important #1）、no-disk fallback、决策签名兼容
- `tests/test_bidaw_controller.py` (~5 tests)：ready/preparing classification、HRRN pick、single slot serialization、`mark_load_started` busy 时 raise（M1.fix nit #4）、promotion on complete
- `tests/test_bidaw_cli.py`：事件路径序列化（`KV_LOAD_COMPLETE ≤ PREFILL_START`）、head-of-line absence、physical promotion、stress yaml preparing wait、no-disk workload 退化、6 scheduler sweep table、metrics 填充、stale-guard 防 unknown KV_LOAD_COMPLETE 污染
- `tests/test_cache_manager.py`：`promote_matched_disk_blocks_to_cpu()` 成功 promotion 与 CPU 容量不足 skip；answer-aware eviction 优先逐出 low-potential CPU block
- `tests/test_trace_generator.py`：`session_history` 模式保留同 session 历史 prompt
- `tests/test_convert_interactive_workload.py`：Interactive-conversation-workload CSV 转 JSONL/profile
- `tests/test_metrics_collector.py`：Bidaw physical promoted/skipped block 计数；answer eviction 默认字段

**回归保护**：5 老 scheduler 仍不进入 Bidaw controller；老 yaml 不需要新增字段；`CacheManager` 只新增 opt-in promotion API，现有 lookup/admit/release 行为由全量测试覆盖。

Hard checks:

- `uv run pytest -q` -> 497 passed
- `uv run python -m nano_kvrouter.cli run --config configs/bidaw.yaml --scheduler bidaw`
- `uv run python -m nano_kvrouter.cli run --config configs/bidaw-stress.yaml --scheduler bidaw`
- `uv run python -m nano_kvrouter.cli run --config configs/bidaw-interactive.yaml --scheduler bidaw`

## 6. Ship 判定

### Critical (block ship)

**无**。M1/M2 后全部 497 测试通过，5 老 scheduler 行为不变。

### Important（当前机制边界 / 后续注意）

1. **`bidaw.yaml` 仍是低压 demo**。该配置 `cpu_blocks=0`，所以物理 promotion 会全部 skipped，且 preparing wait 仍为 0。看 HRRN 积压应使用 `configs/bidaw-stress.yaml`。
2. **物理 promotion 仍是 metadata-only**。它移动 `BlockPool` tier，不复制真实 tensor，也不建 CUDA/storage stream。
3. **bidaw vs conductor 不是同语义对比**。conductor 的 cache_hit 看起来更好但不付真实 disk load；要真对比需要让 5 老 scheduler 也走 `KV_LOAD_*` 路径。**建议**：P6 候选项，需要单独 plan。
4. **previous-answer eviction 是控制面近似**。它依赖 trace/profile 提供 previous answer length 与 bucket potential；没有真实 tensor cache layout，也没有在线 ghost cache。

### Nit

1. Commit `6d1cf50` message 只有标题没 body（Codex 提交时省略），不影响功能，下次注意。
2. ruff / mypy 仍在 backlog #30。

### 综合判定

**✅ Can ship M1/M2 simulator implementation**。论文 I/O-aware scheduling 核心机制（dual queue + disk-HRRN + 真实 KV_LOAD events + single slot per decode node + metadata disk→CPU promotion）已实现；previous-answer-based eviction 作为 metadata/control-plane 近似已实现；5 老 scheduler 行为不变。

后续仍应单独做 P6 全 scheduler 共享 KV_LOAD 路径，避免 bidaw/conductor 对比语义不一致。

## 7. 验证命令

```bash
# 切到 Bidaw 分支
git checkout feat/bidaw-io-aware-scheduling

# 全套测试
uv run pytest -q
# → 497 passed

# Bidaw 单独测试
uv run pytest tests/test_bidaw_scheduler.py tests/test_bidaw_controller.py tests/test_bidaw_cli.py -v
# → 全部 PASS

# 6 scheduler 在 bidaw.yaml 上对比
uv run python -m nano_kvrouter.cli sweep --config configs/bidaw.yaml \
    --output /tmp/bidaw-sweep.json

# 单跑 bidaw（看 5 个新 metrics）
uv run python -m nano_kvrouter.cli run --config configs/bidaw.yaml \
    --scheduler bidaw

# 单跑 interactive fixture（验证 session_history + answer eviction profile wiring）
uv run python -m nano_kvrouter.cli run --config configs/bidaw-interactive.yaml \
    --scheduler bidaw

# 5 老 scheduler byte-identical 回归
uv run python -m nano_kvrouter.cli sweep --config configs/default.yaml
uv run python -m nano_kvrouter.cli sweep --config configs/hicache.yaml
uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml
```

## 8. 文件清单

**新增/主要文件**：
- `src/nano_kvrouter/scheduler/bidaw.py`
- `src/nano_kvrouter/simulator/bidaw_controller.py`
- `src/nano_kvrouter/kv_cache/answer_eviction.py`
- `scripts/convert_interactive_workload.py`
- `src/nano_kvrouter/kv_cache/cache_manager.py`
- `src/nano_kvrouter/cli.py`
- `src/nano_kvrouter/metrics/collector.py`
- `src/nano_kvrouter/simulator/trace_generator.py`
- `src/nano_kvrouter/request.py`
- `src/nano_kvrouter/config.py`
- `configs/bidaw.yaml`
- `configs/bidaw-stress.yaml`
- `configs/bidaw-interactive.yaml`
- `tests/fixtures/interactive_conversation.jsonl`
- `tests/fixtures/interactive_eviction_profile.json`
- `tests/test_bidaw_scheduler.py`
- `tests/test_bidaw_controller.py`
- `tests/test_bidaw_cli.py`
- `tests/test_cache_manager.py`
- `tests/test_metrics_collector.py`

**回归 hard guard**：
5 个老 scheduler 不进入 Bidaw controller；老 configs 不需要 `scheduler.params`
新增字段；answer eviction 默认关闭，只有 `scheduler="bidaw"` 且
`enable_answer_eviction: true` 时才注入 `AnswerEvictionPolicy`。

## 9. M3 — routing intelligence (A1 + A2 + A3, 2026-06-22)

M3 在 M1/M2 基础上加 3 个可选机制，全部默认 off，开启时通过新的
`BidawControllerView` 只读 Protocol 从 controller 取信号。

### 9.1 三个机制

| Flag | 机制 | 公式 / 行为 |
|---|---|---|
| `enable_routing_aware` (A1) | 缓存感知 decode 路由 | `cost = β·load + γ·preparing_disk_blocks + δ·in_flight_disk_blocks − α·matched_blocks`，min cost wins |
| `enable_ttft_slo_gate` (A2) | storage-aware SLO 早拒 | `est_ttft + projected_preparing_wait > slo_ttft` → reject(`ttft_slo_exceeded`) |
| `enable_session_affinity` (A3) | session 粘 decode 节点 | 命中且 pin 未明显过载 → 走 pin；否则 fall back A1 / least-loaded |

### 9.2 与 plan v4 的两个 ratified drift（更贴近实际）

| 维度 | Plan v4 | 实现（ratified） | 理由 |
|---|---|---|---|
| A1 routing penalty 单位 | queue depth / slot count | **disk block 总数** | 队列里一个 50-block 大请求和五个 1-block 小请求在 depth=2 下评分相同（错），块加权下 50 vs 5（对）；单 slot 下 count 信息量太低 |
| A3 overload threshold 锚点 | `factor · avg_load + abs_floor` | **`max(factor · min_load, min_load + abs_floor)`** | min 锚定直接回答"有没有明显更好的节点"；hybrid factor + abs_floor 同时覆盖高低负载场景 |

两点都在 docstring、`doc/code-review/p5-bidaw-m3-routing-m0-preflight.md`
post-ratification section、DESIGN §13.10 中明确记录。

### 9.3 P/D-split 适配（论文 vs 我们的简化）

Bidaw 论文是单节点架构；我们的 M5a split P/D 把 Bidaw 所有 I/O 机制
（dual queue、KV_LOAD slot、HRRN、A1 score、A3 affinity）**scope 到 decode
池**（cache 只在 decode 池）。prefill 仍 round-robin。代价：prefill 节点
必须等 decode-side KV_LOAD 完才 PREFILL_START，是论文没有的上游 stall。
本 milestone 接受；M4 multi-stream + 论文偏离（允许 LOAD/PREFILL 并行）
是未来候选。

### 9.4 Ship gates met（实测）

| Gate | Config | 数字 | 阈值 |
|---|---|---|---|
| A1 cache_hit gap → conductor | `bidaw.yaml`（A1 only） | 0.823 vs 0.823 = **0.000** | ≤ 0.05 ✅ |
| A2 SLO 拒绝触发且不过激 | `bidaw-m3-stress.yaml` | rejections=6, rate=0.194 | rate ≤ 0.252 ✅ |
| A3 session 粘性 | `bidaw-affinity.yaml`（A3 only） | 40/60 = **0.667** | ≥ 0.4 ✅ |

2-of-3 即可 ship，三项全过。Tests: 497 → **515 pass**。

### 9.5 文件清单 / metrics

新增模块 `scheduler/bidaw_view.py` (`BidawControllerView` Protocol)。
`SchedulingDecision` 加两个可选字段 `routing_score`、`affinity_hit`，5 老
scheduler 全部不设，回归 byte-identical。新增 yaml
`configs/bidaw-affinity.yaml`、`configs/bidaw-m3-stress.yaml`。
新增 fixture `tests/fixtures/affinity_workload.jsonl`（20 sessions ×
3 rounds）。

3 个新 metric（通过 SCHEDULED / REQUEST_REJECTED payload 被动采样）：
- `bidaw_routing_score_avg`（A1 cost 均值，可负）
- `bidaw_session_affinity_hits`（A3 命中计数）
- `ttft_slo_rejections`（**通用**字段，Conductor 早拒路径也计入）

### 9.6 cli.py KV_LOAD 两跳 — 已修

M3 之后独立 `fix(bidaw)` commit 修了 M1 留下的 `cli.py:675` 单跳
service 公式（只付 cpu_to_disk，漏 cpu_to_gpu）。同步把
`bidaw_controller.py:287` 的 A2 projected wait 公式改成两跳，保持与
event path 一致。

数字影响：`bidaw.yaml` bidaw 行 ttft 上移 ~0.03ms（修正方向正确），
其他 bidaw-family yaml 在报告精度下无可见变化。M3 ship gates 修复后
重验全过：A2 rejections=6 rate=0.194, A3 hits 40/60=0.667。

详见 DESIGN §13.10.7 + M0 preflight §7（两处均已标 RESOLVED）。

## 10. 后续 milestones

- **M4**：multi-stream load model（B1，单 slot → K 并发，HRRN 真有意义）
- **M5**：GPU-only performance mode（A4，CPU 命中也走 KV_LOAD）
- **M6**：online ghost cache（B2，把 M2 静态 3-bucket 升级到在线反馈）
- ~~**独立 backlog**：cli.py 单跳 → 两跳修正~~（已在独立 fix commit 完成；§9.6）
- **更长远**：让 5 老 scheduler 也共享 `KV_LOAD_*` 真实路径（同语义对比）；
  PagedAttention Tier 2；Llumnix migration；Bidaw 论文 storage-efficient
  tensor caching（需要真实 tensor 支持，超出 metadata 模拟器范围）

详见 DESIGN §14 路线表与
`.claude/plans/p5-bidaw-followups-roadmap.md`。

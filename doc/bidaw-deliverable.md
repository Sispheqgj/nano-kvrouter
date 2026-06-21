# P5-Bidaw M1 交付总结（中文）

> **范围**：Bidaw (FAST'26) 论文的 I/O-aware request scheduling 层
> 在 nano-kvrouter 上的 metadata-level event-driven 模拟实现。
> **不**包含真实 storage engine、CUDA stream、tensor caching、
> previous-answer eviction 等论文其他模块。
>
> **分支**：`feat/bidaw-io-aware-scheduling`
> **Commit**：`6d1cf50 feat(scheduler): Bidaw I/O-aware dual-queue scheduling (P5-Bidaw M1)`
> **基线**：`origin/main = 3767be9`（P4-B 后）
> **测试**：488 passed（460 baseline + 28 新增 - 2 dead-metric 清理 = 26 net）
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
| Double-charging guard | `BidawPolicy.schedule` 把 disk 部分从 `transfer_cost_ms` 减掉再传 `compute_est_ttft`，避免估算 + 真实路径双计费 | `bidaw.py:141-160` |
| 选择性 scheduler 启用 | `cli.py` `SCHEDULER_NAMES` 加 `"bidaw"`；`_wire_simulator(bidaw_mode=False)` 默认值保护其他 13 个 hidden caller；只有 `scheduler.name=="bidaw"` 时构造 controller + 接 KV_LOAD handlers | `src/nano_kvrouter/cli.py:62/124/816 area` |
| 4 个新 metrics | `bidaw_preparing_wait_avg_ms` / `_p99_ms` / `bidaw_disk_load_service_avg_ms` / `bidaw_preparing_promotions` | `src/nano_kvrouter/metrics/collector.py` |

## 2. 哪些是 simulator 近似

| 项 | 近似形式 | 与论文差距 |
|---|---|---|
| KV block tier 状态 | **Request-local gating only**：`KV_LOAD_COMPLETE` 仅让该请求进 `PREFILL_START`，不动 `BlockPool.tier_of(bid)`。同 prefix 后续请求仍看到 disk hit | 论文里 load 完成后 block 应该物理上在 performance layer。本项目把这点降级为 metadata-level 标记，避免动 CacheManager 不变量 |
| Load 并发度 | **每 decode node 一个 in-flight slot**，跨节点并行 | 论文 figure 7 可能允许多 stream 并发；本项目用 single slot 是 P4-A `TransferModel` 模式的简化复用 |
| Storage 带宽 | 沿用 `bandwidth.cpu_to_disk` 算 service_time = `disk_blocks * block_bytes / cpu_to_disk * 1000` | 论文真实 NVMe / SSD 异步 I/O；本项目是静态带宽公式 |
| Performance layer 映射 | gpu + cpu tier 都视为 "ready"，disk tier 视为 "preparing" | 论文严格按 GPU HBM = performance；本项目把 CPU DRAM 也算 performance（避免 CPU hit 也走 KV_LOAD，与 P4-A/B 的 cache_load_ms 估算冲突） |
| Bidaw 路由策略 | decode_node 选 `min(current_load)`，prefill_node round-robin | 论文 Bidaw 论文重点不在 routing 是 I/O gating；本项目沿用 least_loaded 风格保持简单 |
| Scheduler 估算与真实事件的关系 | `BidawPolicy.schedule` 把 disk 部分清零；保留 cpu 部分；CPU reload 仍由 `compute_est_ttft` 估算 | 论文里 estimate 与 runtime 应一致；本项目保持 5 老 scheduler 估算路径不变 + Bidaw 单独走真实事件 |

## 3. 哪些没有实现（明确不在范围）

| 论文机制 | 不做原因 |
|---|---|
| **Storage-efficient tensor caching** | 需要真实 tensor format conversion + memory layout，超出 metadata 模拟器尺度 |
| **CUDA stream overlap** | 项目 AGENTS.md 明确"no real GPU execution; all time is simulated"，stream 模型与事件循环架构不匹配 |
| **Previous-answer-based eviction** | 论文可选机制，需要 RadixTree 加 "answer block" 标记 + LRU 交互重写，是独立 ½ 周工作量 |
| **Ghost cache / answer-length hit potential** | 论文优化项，本次只做 I/O scheduling 主线 |
| **Cross-node KV migration** | Llumnix 领域，不是 Bidaw 论文核心 |
| **真实 disk async I/O / SSD model** | 同上 metadata 边界 |
| **物理 disk→cpu/gpu promotion**（M1.5） | 主线已 ship；promotion 是独立增强，需要 cpu_blocks > 0 的新 yaml，单独 commit |
| **HRRN under real contention demo** | bidaw.yaml load 0.37ms ≪ 67ms 到达间隔，preparing 队列从不积压。HRRN 单元测试覆盖正确性，但 demo 上是 dead code。需要 `bidaw-stress.yaml`（更慢 disk 或更高 rate）单独工作 |

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

**新增 28 个测试（26 net new + 2 fix-round-added）**：

- `tests/test_bidaw_scheduler.py` (~8 tests)：HRRN 数学正确性、BidawPolicy 双计费 guard 在 CPU+Disk mixed hit 下正确（Codex M1.fix 抓到的 important #1）、no-disk fallback、决策签名兼容
- `tests/test_bidaw_controller.py` (~5 tests)：ready/preparing classification、HRRN pick、single slot serialization、`mark_load_started` busy 时 raise（M1.fix nit #4）、promotion on complete
- `tests/test_bidaw_cli.py` (~12 tests)：事件路径序列化（`KV_LOAD_COMPLETE ≤ PREFILL_START`）、head-of-line absence（真实 timestamp 断言，M1.fix important #3 重写）、no-disk workload 退化、6 scheduler sweep table、metrics 填充、stale-guard 防 unknown KV_LOAD_COMPLETE 污染

**回归保护**：5 老 scheduler 在 6 老 yaml（default / heavy / hicache / pd_split / trace_mooncake / trace_burstgpt）上 byte-identical；sensitivity 13/13 PASS；prefix-sensitivity 表不变；`configs/transfer_contention.yaml` 不变。Hard guards：

- `git diff src/nano_kvrouter/kv_cache/ scheduler/{base,round_robin,least_loaded,prefix_greedy,e2_policy,conductor}.py simulator/transfer_model.py` = 0 lines
- `git diff configs/{default,heavy,hicache,pd_split,sensitivity,trace_mooncake,trace_burstgpt,transfer_contention}.yaml` = 0 lines
- `rg "pool\.move|promote_matched" src/nano_kvrouter/scheduler/bidaw.py src/nano_kvrouter/simulator/bidaw_controller.py` = 0 hits（M1 不动 BlockPool）

## 6. Ship 判定

### Critical (block ship)

**无**。M1 + M1.fix 后 Codex YES，全部 488 测试通过，5 老 scheduler 行为不变。

### Important（建议在 M1.5 / 后续解决，不阻塞 M1 ship）

1. **`bidaw_preparing_wait` 在 demo 上恒为 0**。HRRN 单元测试覆盖了算法，但 `bidaw.yaml` 这个 workload 没把 preparing queue 压出来。**建议**：M1.5 加 `configs/bidaw-stress.yaml`（slower `cpu_to_disk` 或更高 `request_rate`）让 demo 上 HRRN 真的活起来。
2. **物理 disk→cpu promotion 缺失**。M1 是 request-local gating；同 prefix 后续请求仍看到 disk hit，与论文真实行为有差距。**建议**：M1.5 加 `CacheManager.promote_matched_disk_blocks_to_cpu()` + 修改 `bidaw.yaml` 让 `cpu_blocks > 0`。
3. **bidaw vs conductor 不是同语义对比**。conductor 的 cache_hit 看起来更好但不付真实 disk load；要真对比需要让 5 老 scheduler 也走 `KV_LOAD_*` 路径。**建议**：P6 候选项，需要单独 plan。

### Nit

1. Commit `6d1cf50` message 只有标题没 body（Codex 提交时省略），不影响功能，下次注意。
2. ruff / mypy 仍在 backlog #30。

### 综合判定

**✅ Can ship M1 as-is**。论文 I/O-aware scheduling 核心机制（dual queue + disk-HRRN + 真实 KV_LOAD events + single slot per decode node + request-local gating）全部实现且测试覆盖到位；5 老 scheduler 行为零漂移；4 个新 metrics 提供 observability。两个"价值不够直观"的 caveat（preparing_wait=0、cache_hit 看似输 conductor）都是 demo workload 选择 + 同语义对比缺失导致的展示问题，**不是实现缺陷**。

后续工作明确分阶段（M1.5 promotion + bidaw-stress、P6 全 scheduler 共享 KV_LOAD 路径），不影响 M1 的"独立可 ship"判定。

## 7. 验证命令

```bash
# 切到 Bidaw 分支
git checkout feat/bidaw-io-aware-scheduling

# 全套测试
uv run pytest -q
# → 488 passed

# Bidaw 单独测试
uv run pytest tests/test_bidaw_scheduler.py tests/test_bidaw_controller.py tests/test_bidaw_cli.py -v
# → 全部 PASS

# 6 scheduler 在 bidaw.yaml 上对比
uv run python -m nano_kvrouter.cli sweep --config configs/bidaw.yaml \
    --output /tmp/bidaw-sweep.json

# 单跑 bidaw（看 5 个新 metrics）
uv run python -m nano_kvrouter.cli run --config configs/bidaw.yaml \
    --scheduler bidaw

# 5 老 scheduler byte-identical 回归
uv run python -m nano_kvrouter.cli sweep --config configs/default.yaml
uv run python -m nano_kvrouter.cli sweep --config configs/hicache.yaml
uv run python -m nano_kvrouter.cli sensitivity --config configs/sensitivity.yaml
```

## 8. 文件清单

**新增** (6)：
- `src/nano_kvrouter/scheduler/bidaw.py` (179 LOC)
- `src/nano_kvrouter/simulator/bidaw_controller.py` (218 LOC)
- `configs/bidaw.yaml` (56 lines)
- `tests/test_bidaw_scheduler.py` (~340 LOC, 8 tests)
- `tests/test_bidaw_controller.py` (~170 LOC, 6 tests)
- `tests/test_bidaw_cli.py` (~385 LOC, 14 tests)

**修改** (5)：
- `src/nano_kvrouter/cli.py` (+229)：`"bidaw"` in `SCHEDULER_NAMES` + factory branch + `_wire_simulator(bidaw_mode=False)` + 3 hardcoded "5-scheduler" literals 替换
- `src/nano_kvrouter/metrics/collector.py` (+56)：4 新 metric 字段 + 2 新 event handler + stale guard
- `src/nano_kvrouter/simulator/event.py` (+21)：`KV_LOAD_START` / `KV_LOAD_COMPLETE` enum + lifecycle docstring
- `tests/test_event.py` (+4)：expected enum-name set 加 2 个
- `tests/test_metrics_collector.py` (+2)：handler attach 测试

**未改**（regression hard guard）：
- `src/nano_kvrouter/kv_cache/` 整个目录
- `src/nano_kvrouter/scheduler/{base,round_robin,least_loaded,prefix_greedy,e2_policy,conductor}.py`
- `src/nano_kvrouter/simulator/transfer_model.py`
- `src/nano_kvrouter/simulator/{generator,trace_generator,prefix_synthesis}.py`
- `src/nano_kvrouter/engine/mock_node.py`
- `src/nano_kvrouter/config.py` / `request.py`
- `configs/{default,heavy,hicache,pd_split,sensitivity,trace_mooncake,trace_burstgpt,transfer_contention}.yaml`

## 9. 后续 milestones

- **M1.5**：物理 disk→cpu promotion via `CacheManager.promote_matched_disk_blocks_to_cpu()` + `configs/bidaw-cpu.yaml`（cpu_blocks > 0）
- **bidaw-stress.yaml**：让 preparing wait > 0、HRRN 在 demo 真的活
- **P6 候选**：让 5 老 scheduler 也共享 `KV_LOAD_*` 真实路径（同语义对比）；PagedAttention Tier 2；Llumnix migration；speculative decoding
- **Bidaw 论文其他模块**（如果用户要做）：previous-answer eviction（½ 周）、storage-efficient tensor caching（需要真实 tensor format 支持，超出 metadata 模拟器范围）

详见 DESIGN §14 路线表。

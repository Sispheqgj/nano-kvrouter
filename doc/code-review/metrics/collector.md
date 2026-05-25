# MetricsCollector：v2 定义修正记录

## 基本信息

- 源码文件：`src/nano_kvrouter/metrics/collector.py`
- 测试文件：`tests/test_metrics_collector.py`
- 审阅日期：2026-05-25
- 当前状态：
  - 已实现 V1（commit 待 add）
  - TTFT/TBT v2 paper-aligned definition 已应用（commit 待 add）
  - TBT 权重从 per-request 改为 per-step，本文件记录这个语义变化
- 相关任务：TaskList #10

## 文件职责

`MetricsCollector` 是 simulator 的被动观察者。

它监听事件，但不修改 simulation state。当前主要负责汇总：

- 到达请求数
- 完成请求数
- 拒绝请求数
- rejection rate
- TTFT
- TBT
- end-to-end latency
- SLO 命中率
- cache hit ratio
- throughput

## V2 当前定义

当前定义与 Mooncake §3 和 Sarathi-Serve OSDI'24 对齐：

```text
TTFT = DECODE_STEP[step_index=0].time - REQUEST_ARRIVE.time

TBT = average over all DECODE_STEP[step_index >= 1] of:
      current_decode_step.time - previous_decode_step.time
```

也就是说：

- 第一个 `DECODE_STEP` 代表首 token 产出时间，用于计算 TTFT。
- 从第二个 `DECODE_STEP` 开始，相邻 decode step 的时间差才是 TBT sample。
- `PREFILL_COMPLETE` 不再参与 TTFT/TBT 计算。

`PREFILL_COMPLETE` handler 保留为 no-op extension point，后续如果要统计 prefill duration histogram，可以继续使用这个事件。

## TBT 权重变化：v1 -> v2

### v1：per-request equal weighting

旧版本逻辑：

- 每个 request 的 TBT samples 在 `DECODE_COMPLETE` 时做平均。
- 每个 request 的平均值 append 到 `_tbt_per_request`。
- `summary()` 对这些 per-request mean 求 mean / median。

这意味着：

```text
短请求 = 长请求
```

每个 request 对最终 TBT 统计的权重相同。

### v2：per-step equal weighting

当前逻辑：

- `_tbt_samples[request_id]` 收集每个 decode step interval。
- `summary()` 调用 `_flat_tbt()` 时动态 flatten 所有 request 的 samples。
- 不再依赖 `DECODE_COMPLETE` flush TBT。

这意味着：

```text
每个 decode step interval 权重相同
```

长请求会贡献更多 TBT samples，因此对整体 TBT 分布有更高权重。

## 为什么发生这个变化

dispatch spec 的 smoke test 要求：

```text
tbt_avg_ms == 5.0
```

并且这个 smoke test 没有发出 `DECODE_COMPLETE` 事件。

Sonnet B 为了满足这个 smoke test，把 TBT 聚合移动到 `summary()` 时间点：只要 decode step events 已经到达，即使 request 还没有 complete，也能汇总 TBT。

这个实现顺带改变了 TBT 权重语义：从 per-request weighting 变成了 per-step weighting。

## 为什么保留 v2 语义

保留 per-step weighting 的理由：

- vLLM、Mooncake、Sarathi-Serve 一类 serving benchmark 通常报告 token-level TBT distribution。
- token-level / step-level 分布更能反映用户实际看到的连续生成体验。
- v1 的短请求和长请求同权会放大短请求影响，这是早期实现的疏漏，不是明确设计目标。
- v2 不需要在 `DECODE_COMPLETE` 时 flush TBT，状态管理更简单。
- `summary()` 可以在 simulation 中途调用，仍然能看到当前已有的 TBT samples。

## 当前问题与差距

| 问题 | 影响 | 当前选择 | 优先级 |
|------|------|----------|--------|
| v2 改变了 TBT 权重语义 | 历史 v1 指标和 v2 指标不能直接横向比较 | 接受，v2 更贴近 token-level benchmark | 中 |
| 没有 per-request weighted TBT 视图 | 如果后续想分析“每个请求体验是否公平”，当前 summary 不够 | 未来可新增 `tbt_per_request_avg_ms` | 低 |
| `PREFILL_COMPLETE` 暂时 no-op | 当前无法直接从 metrics 看 prefill duration 分布 | 保留 extension point，后续补 prefill metrics | 低 |

## 后续 PR 候选

| PR | 改动范围 | 验证方式 |
|----|----------|----------|
| PR-1：新增 per-request TBT 视图 | 在 `summary()` 中增加 `tbt_per_request_avg_ms`，与当前 per-step TBT 并存 | 构造长短请求混合测试，断言两个指标不同 |
| PR-2：补充 prefill duration metrics | 使用 `PREFILL_START` / `PREFILL_COMPLETE` 或现有事件记录 prefill latency | 单测确认 prefill histogram 不影响 TTFT/TBT |
| PR-3：文档化 metrics 定义 | 在 README 或 DESIGN 中明确 TTFT/TBT 当前定义 | 文档 review，不改变代码 |

## 学习笔记

TTFT 和 TBT 的定义很容易混淆。

当前项目采用：

```text
TTFT 看首个 decode token 出现的时间。
TBT 看后续 token 与前一个 token 之间的间隔。
```

这比用 `PREFILL_COMPLETE` 计算 TTFT 更贴近用户体验，因为用户真正感知到的是第一个输出 token，而不是 prefill 内部完成。

TBT 的 per-step weighting 也更贴近 serving benchmark 的常见做法：长请求生成更多 token，就自然贡献更多 token interval 样本。

如果后续实验关注 request-level fairness，可以并行增加 per-request weighted TBT，而不是把当前 v2 定义改回去。

## Update：REQUEST_REJECTED 和重复 `step_index=0` 防御修复

rollback commit 中合入了两个小的防御性修复：

- `_on_rejected` 在 payload 缺少 `request_id` 时会 `logger.warning` 后跳过，和其他 handlers 的防御风格保持一致。
- 对同一个 request 收到重复的 `step_index=0` 事件时，整个 step-0 分支会 early return，避免重置 `_last_decode_step_time`。

第二点的语义是：

```text
TBT[1] 锚定到第一次 step_index=0 的时间戳，
而不是后续重复 step_index=0 的时间戳。
```

对应测试：

```text
test_duplicate_step_zero_does_not_reset_tbt_anchor
```

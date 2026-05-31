# nano-kvrouter 学习与 Code Review 总览

这个文档用来沉淀本项目的学习审阅过程。详细 review 按源码文件拆分到 `doc/code-review/`，这里只保留项目定位、索引和跨文件 PR backlog。

## 记录方式

每个模块按四类信息整理：

1. **当前已经完成了什么**：只记录本仓库已经实现、可以从源码验证的内容。
2. **对照材料说明了什么**：记录对应论文或开源仓库的关键设计点。
3. **当前差距是什么**：指出本项目和真实系统之间还没有建模的部分。
4. **后续 PR 候选**：把改进拆成尽量小、可单独提交和测试的任务。

## 项目当前定位

nano-kvrouter 是一个 KV-cache-centric LLM serving control-plane simulator。

当前边界：

- 纯 Python、事件驱动、单线程模拟。
- 不运行真实 GPU 推理，不存实际 tensor。
- 节点通过 mock latency model 模拟 prefill、decode、排队和容量压力。
- 重点不是复刻 vLLM/SGLang 的底层执行引擎，而是在 control plane 层学习调度、admission、cache-aware routing、metrics 和 SLO 判断。

## 文档索引

- Code review 索引：[`doc/code-review/README.md`](code-review/README.md)
- Per-file 模板：[`doc/code-review/TEMPLATE.md`](code-review/TEMPLATE.md)
- `mock_node.py` review：[`doc/code-review/engine/mock_node.md`](code-review/engine/mock_node.md)
- `cache_manager.py` review：[`doc/code-review/kv_cache/cache_manager.md`](code-review/kv_cache/cache_manager.md)
- `collector.py` review：[`doc/code-review/metrics/collector.md`](code-review/metrics/collector.md)
- `generator.py` review：[`doc/code-review/simulator/generator.md`](code-review/simulator/generator.md)

## 待审阅模块 Backlog

下面是后续可以继续对照论文和源码学习的模块。每完成一个模块，就在“已完成审阅”中新增小节。

| 模块 | Review 文档 | 对照系统 | 重点问题 |
|------|-------------|----------|----------|
| Mock node | [`doc/code-review/engine/mock_node.md`](code-review/engine/mock_node.md) | Sarathi-Serve | chunked prefill、token budget、generation stall |
| Cache manager | [`doc/code-review/kv_cache/cache_manager.md`](code-review/kv_cache/cache_manager.md) | RadixTree / BlockPool | split-aware reconcile rollback，区分逻辑命中和物理占用 |
| Request generator | [`doc/code-review/simulator/generator.md`](code-review/simulator/generator.md) | BurstGPT / Preble / Mooncake / vLLM / AIPerf | trace replay、混合长度、burst、session-aware prefix |
| Radix prefix cache | 待建 | SGLang RadixAttention | prefix match、node split、LRU eviction 是否表达清楚 |
| Block pool | 待建 | vLLM v1 / HiCache | block metadata、tier movement、capacity pressure |
| Scheduler protocol | 待建 | SGLang / Mooncake / Preble | policy interface 是否能承载 cache-aware 和 SLO-aware 决策 |
| Event engine | 待建 | serving simulator pattern | event loop 是否能支持 chunked prefill 和 migration |
| Metrics collector | [`doc/code-review/metrics/collector.md`](code-review/metrics/collector.md) | Mooncake / Sarathi-Serve | TTFT/TBT v2 定义，TBT per-step weighting |

## PR 拆分原则

后续提 PR 时尽量遵守：

- 一个 PR 只解决一个清晰问题。
- 先补测试，再改行为，或者至少在同一个 PR 中补齐测试。
- 文档 PR 和行为 PR 可以分开，避免 review 负担过大。
- 涉及指标语义变化时，要在文档中写清楚旧口径和新口径。
- 涉及 scheduler 决策变化时，要给出一个可复现的小场景测试。

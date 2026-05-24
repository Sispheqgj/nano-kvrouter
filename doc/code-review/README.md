# Code Review 文档索引

这个目录按源码文件拆分学习与 review 记录。每个 Markdown 文件对应一个主要代码文件，方便后续把问题拆成小 PR。

## 目录规则

- 文档路径尽量镜像源码路径。
- 例如 `src/nano_kvrouter/engine/mock_node.py` 对应 `doc/code-review/engine/mock_node.md`。
- 一个文档只讨论一个代码文件的职责、当前实现、对照材料、问题和 PR 候选。
- 跨文件问题可以在相关文件中互相链接，也可以临时记录在根目录的 `doc/review-notes.md`。

## 已建立文档

| 源码文件 | Review 文档 | 状态 | 对照材料 |
|----------|-------------|------|----------|
| `src/nano_kvrouter/engine/mock_node.py` | [`engine/mock_node.md`](engine/mock_node.md) | 已审阅 | Sarathi-Serve |

## 待补文档

| 源码文件 | 建议对照材料 | 重点问题 |
|----------|--------------|----------|
| `src/nano_kvrouter/kv_cache/radix_tree.py` | SGLang RadixAttention | prefix match、node split、LRU eviction |
| `src/nano_kvrouter/kv_cache/block_pool.py` | vLLM v1 / HiCache | block metadata、tier movement、capacity pressure |
| `src/nano_kvrouter/scheduler/base.py` | SGLang / Mooncake / Preble | policy interface、SLO-aware decision |
| `src/nano_kvrouter/simulator/engine.py` | serving simulator pattern | event loop、chunked prefill、migration |
| `src/nano_kvrouter/metrics/collector.py` | LLM serving metrics | TTFT、TBT、reject rate、cache hit rate |


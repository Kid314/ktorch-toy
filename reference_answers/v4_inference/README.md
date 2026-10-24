# v4：推理与 KV Cache 参考答案

v4 把重点从“训练参数”转到“生成时如何复用历史计算”。完整因果注意力每生成一个 token 都会重算前缀；KV Cache 保存历史 Key/Value，只为当前 Query 计算一行注意力。

## 为什么这样改

训练时我们有完整序列，可以并行算所有位置。生成时未来 token 不存在，只能逐步 decode。历史 token 的 K/V 已经不会变化，因此可以缓存：

```text
当前 Query q_t
→ 与 cache 中 K_0...K_t 做匹配
→ softmax
→ 加权 cache 中 V_0...V_t
```

这一版提供一个可独立理解的单头、单层 cache 演示。它与 v1 的训练模型保持同样的注意力公式，但先把 cache 的数学和状态边界单独验证清楚，再考虑把它嵌回模型推理路径。

## 结构

```text
python/ktorch/cache.py       KVCache、完整前缀基线、逐token decode
examples/cache_demo.py       打印完整路径与缓存路径的输出
tests/test_cache.py          数值等价、快照恢复、clear和错误shape
design.md                    prefill/decode、位置、内存和边界说明
```

运行：

```powershell
python examples/cache_demo.py
python -m unittest discover -s tests -v
```

本版不默认训练、不做大模型延迟压测。它先证明：对相同 Q/K/V，完整因果注意力与缓存逐token计算逐元素一致。

## 边界

- Cache 属于一个请求；开始新请求必须 `clear()`。
- K/V 的宽度和长度必须成对；错误追加不能留下半行。
- 当前 demo 使用窗口内绝对顺序，未处理滑动窗口后的模型绝对位置编码迁移。
- 它是单头教学实现，未加入多层、批处理、分页内存或并发服务。

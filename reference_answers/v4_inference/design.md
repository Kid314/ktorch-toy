# v4 整体设计说明

## 计算量变化

若当前前缀长度为 T，朴素生成每一步都重新处理完整前缀；随着 T 增长，历史位置被反复计算。缓存后，历史 K/V 只计算一次，当前 decode 只计算新 Query 与已有 K/V 的匹配和加权。

单行公式仍是：

```text
score_j = q_t · k_j / sqrt(d)
p_j = softmax(score_0...score_t)
out_t = Σ_j p_j v_j
```

cache 保存的是 `k_j` 和 `v_j`，不是 Query，也不是最终输出。Query 每个新位置只使用一次；输出依赖当前 Query 与全部历史状态。

## prefill 与 decode

- **prefill**：给定一段已有前缀，按位置建立 K/V 历史。教学实现可以逐行建立，生产实现通常使用批量矩阵运算。
- **decode**：每次收到一个新 token，计算它的 Q/K/V，把新 K/V 追加，再用当前 Q 读取 cache。

本版 `cached_causal_attention` 用逐行循环模拟两者，目的在于和 `full_causal_attention` 对照。它没有声称达到生产推理性能。

## 位置和滑动窗口

v1 使用可学习绝对位置表。若上下文超过窗口，简单截取最近 token 会让位置重新编号；这与保留旧绝对位置的 cache 不是同一个语义。真正集成前必须决定：重新 prefill 当前窗口，或使用支持增量绝对位置的设计。

## 优化方向

后续可以把每层的 cache 组织成连续 `[T, d]`，减少Python列表和对象分配；再处理批量请求、页式分配和并发。每次优化都要先以完整重算结果做逐元素校验，并记录内存字节数和单token延迟。

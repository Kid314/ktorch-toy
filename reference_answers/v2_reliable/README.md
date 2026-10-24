# v2：可靠的小型训练框架参考答案

v2 从 v1 的“能训练一次”开始，增加让实验可以暂停、恢复、校验和排错的能力。模型结构保持单头、单 Block 和 CPU 约束，避免把可靠性问题和 GPU/大模型问题混在一起。

## 相对 v1 增加了什么

```text
v1：参数 + SGD + schema1 checkpoint
v2：AdamW + 优化器状态 + schema2完整性摘要 + Trainer恢复 + 微批梯度平均
```

v2 的新增点集中在：

- `native/adam.cpp`：在 C++ 中执行一阶矩、二阶矩、偏差修正和解耦权重衰减。
- `python/ktorch/optim.py`：保存/加载 SGD 或 AdamW 的超参数、步数和矩状态。
- `python/ktorch/checkpoint.py`：原子写入、schema2、SHA-256 完整性摘要，以及 schema1 推理兼容。
- `python/ktorch/training.py`：Trainer 保存随机采样状态、数据摘要、batch_size 和优化器状态。

## 为什么状态必须一起保存

AdamW 下一步需要：

```text
m_t = β₁ m_(t−1) + (1−β₁) g_t
v_t = β₂ v_(t−1) + (1−β₂) g_t²
θ_t = θ_(t−1)(1−ηλ) − η · m̂_t/(sqrt(v̂_t)+ε)
```

只保存模型权值而丢弃 `m`、`v` 和步数，重新开始的更新轨迹就不再是原来训练的延续。Trainer 还保存随机采样器状态和语料 SHA-256，防止用户误把另一份数据当作“恢复训练”。

## 微批的边界

v2 仍然只支持二维 Tensor。`batch_size` 表示 Python 依次取多个 `[T]` 窗口，每个 loss 先除以 batch_size，再调用 backward，最后只执行一次 optimizer step。这实现了平均梯度，但没有把 Tensor 扩展到 `[B,T,d]`。

这样安排的原因是先学习 loss 归约和梯度累计，再决定是否需要真正的三维 batch。增加批维会同时影响 shape、Embedding、LayerNorm、因果 mask、内存布局和检查点，不能只改一个循环。

## checkpoint 设计

schema1 保存模型配置、词表和参数，可用于推理。schema2 增加训练状态和 `sha256`：

```text
format / schema / config / vocab / parameters
metadata / training / sha256
```

保存先写同目录临时文件，再 `os.replace`。加载前检查文件大小、格式、schema、摘要、词表、参数名称和 shape。所有参数通过已有 Parameter 身份加载，保持共享 E 的关系。

## 当前接入边界

v2 的原生 AdamW 和 Python Trainer 代码已经写入，但继续工作时需要先把示例 `train.py` 接到 AdamW/Trainer，再做恢复等价测试。代码存在不等于 v2 已验收。

## 可优化方向

1. 将 checkpoint 从 JSON 扩展为更适合大参数的二进制格式，同时保留可读元数据。
2. 将参数状态和优化器状态的写入做更严格的临时文件清理、版本迁移和损坏恢复。
3. 增加配置文件、实验日志、早停和验证集，但保持训练目标和数据来源透明。
4. 只有在 v2 的可靠性检查通过后，才进入 v3 的 CPU 测量。

## 轻量验收目标

短文本、少量步数下检查：

1. AdamW 的参数和矩状态能序列化。
2. 保存后恢复，下一步使用相同采样状态、模型参数和优化器状态。
3. 连续训练 N 步与 A 步保存后再恢复训练 N−A 步，在同一环境下得到相同参数。
4. schema1 模型仍能推理；损坏 schema2 摘要会被拒绝。

大规模数据、长时间训练和性能压力测试不属于本版默认验证。

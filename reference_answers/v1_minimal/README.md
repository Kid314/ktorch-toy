# v1：最小可训练 GPT 参考答案

这是一份独立的本地参考实现。它的目标是把一条最短的文本训练链路真正跑通：

```text
中文文本
→ Unicode 字符 ID
→ token / position embedding
→ 一个 Pre-LN Transformer Block
→ 因果单头注意力 + MLP
→ logits
→ 稳定交叉熵
→ C++ 反向传播
→ SGD 更新
→ 逐字符生成
```

本版使用 CPU、C++17、Python 标准库 `ctypes`、二维连续 `double` Tensor、batch=1、一个注意力头和一个 Transformer Block。模型默认 `d=8`、MLP 宽度 `32`、上下文 `8`。这些选择把重点放在计算和梯度上，避免先引入 GPU、任意 rank Tensor 或完整 PyTorch API。

## 为什么这样分成 C++ 和 Python

Python 擅长处理 Unicode 文本、词表、训练循环、checkpoint 和命令行。C++ 负责真正的数值运算、计算图、局部反向和参数写回。两边通过 `extern "C"` 的窄接口连接，Python 只保存一个整数句柄。

句柄表的设计让 Python 不需要理解 `std::vector`、C++ 异常或节点内部结构。C++ 负责句柄的创建和释放，`ctypes` 负责声明参数类型和检查错误。这样一旦 `matmul` 或 LayerNorm 写错，测试会直接落到对应的 C++ 算子，而不是被 Python 的另一套计算掩盖。

## 为什么 Tensor 只有二维

本版的训练数据一次只有一段文本，因此主要形状已经足够表达完整模型：

```text
ids       [T]       整数索引
E         [V, d]    字符表
P         [C, d]    位置表
X/Q/K/V   [T, d]    隐藏表示
scores/A  [T, T]    注意力分数与比例
MLP       [T, h]
logits    [T, V]
loss      [1, 1]
```

引入三维 batch Tensor 会同时带来 stride、广播、loss 归约和内存布局问题。v1 先用多个二维窗口依次累加梯度，保持每一步都能对应数学手算。后续版本可以增加微批，但需要重新定义平均损失和形状。

## 计算图怎样工作

每个算子创建一个结果节点，结果节点持有输入节点。节点同时保存前向值、梯度、输入版本和局部 `backward` 函数。调用标量 `loss.backward()` 时，C++ 先拓扑排序，再从输出向输入反向执行。

梯度使用 `+=`，因为同一节点可能有多条路径。残差连接会把输入直接传到后面；共享字符表 E 同时出现在输入查表和输出投影；重复 Embedding ID 也会多次命中同一行。覆盖赋值会丢掉其中一条贡献。

节点保存 `version`。如果前向之后直接修改参数或输入，反向会拒绝使用过期图。这条限制帮助学习者看到“前向缓存和参数值必须属于同一轮计算”，也避免第一版悄悄得到错误梯度。

## 模型结构为何是 Pre-LN

一个 Block 不是一个 Linear，而是下面这条链：

```text
X
→ LayerNorm1
→ Q/K/V 投影
→ QKᵀ / √d
→ 因果 softmax
→ A @ V
→ 输出投影
→ 加回 X
→ LayerNorm2
→ Linear → GELU → Linear
→ 加回第一次残差
```

数学形式是：

```text
U = X + Attention(LayerNorm1(X))
H = U + MLP(LayerNorm2(U))
logits = LayerNormFinal(H) @ E.T
```

LayerNorm 对每个位置自己的 `d` 个维度计算均值和总体方差。因果遮罩把 `j > i` 的注意力概率和梯度设为零。位置编号每次窗口都从 `0` 开始；窗口滑动时重新编号，避免超过可学习位置表。

## 文本训练怎样得到监督

假设原文是：

```text
我 爱 猫 。
```

训练输入和标签是：

```text
x = [我, 爱, 猫]
y = [爱, 猫, 。]
```

一次前向可以并行得到三个位置的 logits；因果遮罩保证位置0只能读“我”，位置1只能读“我爱”。交叉熵直接从 logits 计算：

```text
loss_t = logsumexp(logits[t]) - logits[t, target[t]]
loss = mean(loss_t)
```

训练循环依次做 `zero_grad → forward → loss → backward → clip → SGD`。生成时只取最后一行 logits，选择一个字符追加回上下文；生成不保存计算图，也不修改参数。

## 文件结构

```text
native/core.hpp       Tensor、节点、算子声明
native/tensor.cpp     生命周期、拓扑反向、SGD
native/ops.cpp        add/mul/matmul/gather/LN/GELU/softmax/CE
native/kernel.cpp     清晰的 i-j-k 矩阵内核
native/c_api.cpp      uint64 句柄和错误边界
python/ktorch/        ctypes、Tensor、nn、模型、优化器、训练工具
examples/train.py     训练并保存 checkpoint
examples/generate.py  独立加载并生成
examples/inspect_step.py  打印一个小模型的真实 logits/loss/梯度
tests/test_core.py    8 个小规模行为测试
```

## 本版验证与局限

历史验证使用本机 CLion MinGW GCC、Python 3.12：动态库成功编译；8 个标准库测试通过，覆盖非方阵矩阵乘法、重复查表梯度、稳定交叉熵、因果遮罩、LayerNorm/GELU 有限差分、因果性、共享参数、checkpoint、生命周期和短文本训练。

后续继续时应重新运行验证，因为源码注释展开和 v2 复制可能已经改变文件。没有执行长训练、大矩阵性能测试或多平台构建。这个模型适合学习和记住小语料中的字符模式，不能当作聊天助手或通用语言模型。

## 可以怎样优化

先保持数学结果不变，再测量改动的影响。可以依次考虑：

1. 把矩阵内核从 `i-j-k` 与缓存更友好的 `i-k-j` 做小矩阵对比。
2. 减少 ctypes 的复制和句柄调用次数，但要先记录数据所有权和生命周期。
3. 增加 AdamW、训练恢复、配置校验和更完整的错误测试。
4. 再讨论 batch、KV Cache、量化或 GPU。每个扩展都要先重新定义 shape、loss 归约和验收。

这些优化属于后续版本。改变循环顺序或数值类型不能因为更快就自动算作正确，必须重新对拍前向和梯度。

## 运行入口

```powershell
python scripts/build.py
python -X utf8 -m unittest discover -s tests -v
python -X utf8 examples/train.py --steps 25
python -X utf8 examples/generate.py --prompt 我 --tokens 8
```

后两条训练命令在本地参考答案里需要小语料和足够的上下文；实际输出以当前机器运行结果为准。长训练默认不执行。

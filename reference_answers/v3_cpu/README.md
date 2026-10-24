# v3：CPU 计算与测量参考答案

v3 在 v1 的正确性基线上增加一个可以观察的 CPU 计算问题：**同一个矩阵乘法，循环顺序改变后，内存访问模式会怎样影响时间。** 这一版保持模型结构与训练算法不变，把学习重点放到测量边界和可复现对比。

## 为什么这样改

v1 的 `gemm` 采用最直观的 `i-j-k`：先选输出格，再沿共享维度累加。它容易对应公式 `C[i,j] += A[i,k] * B[k,j]`，适合第一版学习。

v3 增加 `i-k-j`：固定 `A[i,k]` 后连续更新 C 的一整行。C++ 的 vector 是行连续存储，连续访问通常更容易利用缓存。这里不提前承诺一定快多少，实际收益取决于矩阵形状、编译器和机器。

矩阵内核根据环境变量 `KTORCH_GEMM` 选择策略：

```powershell
$env:KTORCH_GEMM = "ijk"
python examples/benchmark.py --rows 16 --inner 16 --cols 16 --repeats 5

$env:KTORCH_GEMM = "ikj"
python examples/benchmark.py --rows 16 --inner 16 --cols 16 --repeats 5
```

这只是小规模教学测量。它不会启动长训练，也不会把一次小矩阵计时外推成生产性能结论。

## 结构

```text
native/kernel.cpp          i-j-k 与 i-k-j 两种可切换实现
scripts/build.py           与 v1 相同的本地构建入口
examples/benchmark.py      固定小矩阵、预热、重复计时、检查结果一致
examples/train.py          仍可训练 v1 同样的单层文本模型
tests/test_cpu.py          策略切换的数值等价和计时输出格式检查
design.md                  循环顺序、缓存、计时边界和下一步优化
```

## 运行

先在本版目录中构建：

```powershell
python scripts/build.py
python -m unittest discover -s tests -v
python examples/benchmark.py --rows 8 --inner 8 --cols 8 --repeats 3
```

Windows PowerShell 中改变策略后重新运行 benchmark。若没有设置变量，默认使用 `ijk`。

## 验收边界

- 两种策略在相同输入上输出逐元素一致。
- benchmark 明确记录矩阵 shape、策略、重复次数和计时范围。
- 本版不添加 GPU、SIMD intrinsic、线程池或大矩阵压测。
- 只有在得到真实改前/改后数据后，才能讨论下一步分块或并行化。

"""在小矩阵上比较v3的两种C++ GEMM循环策略。"""  # 不运行大规模压力测试。
from _bootstrap import ROOT  # 确保使用v3自己的Python包。
import argparse  # 允许把形状和重复次数缩小。
import os  # 设置策略环境变量。
import time  # 使用高分辨率计时器。
from ktorch import Tensor, no_grad  # 通过同一个模型后端触发matmul。

def run(rows, inner, cols, repeats, strategy):  # 返回平均毫秒和输出快照。
    os.environ["KTORCH_GEMM"] = strategy  # C++内核每次调用读取策略。
    left = Tensor([[((i * inner + k) % 7) / 7.0 for k in range(inner)] for i in range(rows)])  # 固定输入A。
    right = Tensor([[((k * cols + j) % 5) / 5.0 for j in range(cols)] for k in range(inner)])  # 固定输入B。
    with no_grad():  # benchmark只测前向矩阵乘法，不记录图。
        expected = (left @ right).tolist()  # 第一次调用作为数值快照。
        for _ in range(2):  # 预热两次，减少首次加载影响。
            left @ right  # 仍然执行真实C++内核。
        started = time.perf_counter()  # 记录正式计时起点。
        for _ in range(repeats):  # 重复相同规模的计算。
            output = (left @ right).tolist()  # 取回结果会包含ctypes复制开销，需在报告中说明。
        elapsed = (time.perf_counter() - started) * 1000.0 / repeats  # 计算每次平均毫秒。
    error = max(abs(a - b) for row_a, row_b in zip(expected, output) for a, b in zip(row_a, row_b))  # 检查输出一致。
    return elapsed, error  # 不把一次计时外推成通用性能结论。

parser = argparse.ArgumentParser(description=__doc__)  # 生成可读命令帮助。
parser.add_argument("--rows", type=int, default=8)  # 默认小矩阵。
parser.add_argument("--inner", type=int, default=8)  # 默认共享维度。
parser.add_argument("--cols", type=int, default=8)  # 默认输出列数。
parser.add_argument("--repeats", type=int, default=3)  # 慢机器可保持三次。
args = parser.parse_args()  # 读取本次实验配置。
for strategy in ("ijk", "ikj"):  # 两种循环都在同一输入和进程中测试。
    elapsed, error = run(args.rows, args.inner, args.cols, args.repeats, strategy)  # 执行并检查。
    print(f"strategy={strategy} shape=({args.rows},{args.inner})@({args.inner},{args.cols}) repeats={args.repeats} average_ms={elapsed:.6f} max_abs_error={error:.3e}")  # 打印完整边界。

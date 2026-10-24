"""v3只做小规模数值等价检查，不做性能门槛判断。"""  # 性能会随机器变化。
import os  # 切换C++循环策略。
import sys  # 加载本版Python包。
from pathlib import Path  # 定位版本根目录。
import unittest  # 标准库测试。
ROOT = Path(__file__).resolve().parents[1]  # 当前v3目录。
sys.path.insert(0, str(ROOT / "python"))  # 不导入v1或主项目。
from ktorch import Tensor, no_grad  # 真正通过C++后端计算。

class CpuTests(unittest.TestCase):  # 每个策略都必须保持数学结果。
    def test_ijk_and_ikj_are_equal(self):  # 小矩阵足以发现索引错误。
        a = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3输入。
        b = Tensor([[1, 0], [0, 1], [1, 1]])  # 3×2输入。
        with no_grad():  # benchmark不记录计算图。
            os.environ["KTORCH_GEMM"] = "ijk"  # 使用直观基线。
            expected = (a @ b).tolist()  # 保存基线数值。
            os.environ["KTORCH_GEMM"] = "ikj"  # 切换缓存友好循环。
            actual = (a @ b).tolist()  # 相同输入再次计算。
        self.assertEqual(expected, actual)  # 策略改变不应改变数学结果。

if __name__ == "__main__":  # 支持直接运行测试文件。
    unittest.main()  # 输出实际测试数量。

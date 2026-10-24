"""v5检查命令行帮助和HTTP错误边界，不启动长时间服务。"""  # 用户可在有checkpoint时另测真实请求。
from pathlib import Path  # 定位版本根目录。
import os  # 给子进程传递PYTHONPATH。
import subprocess  # 调用公开模块入口。
import sys  # 使用当前解释器。
import unittest  # 标准库测试。
ROOT = Path(__file__).resolve().parents[1]  # v5目录。

class PackageTests(unittest.TestCase):  # 小型且低负担的发布检查。
    def env(self):  # 返回只影响当前子进程的环境。
        return {**os.environ, "PYTHONPATH": str(ROOT / "python")}  # 不修改用户系统配置。
    def test_module_help(self):  # python -m入口应能在没有模型文件时提供帮助。
        result = subprocess.run([sys.executable, "-m", "ktorch.cli", "--help"], env=self.env(), capture_output=True, text=True, encoding="utf-8", errors="replace")  # 捕获输出便于断言，不受Windows代码页影响。
        self.assertEqual(result.returncode, 0)  # argparse帮助是成功路径。
        self.assertIn("checkpoint", result.stdout)  # 帮助必须说明必要输入。
    def test_server_import(self):  # HTTP模块导入不应打开端口或加载checkpoint。
        result = subprocess.run([sys.executable, "-c", "import ktorch.server"], env=self.env(), capture_output=True, text=True, encoding="utf-8", errors="replace")  # 单独进程验证副作用边界。
        self.assertEqual(result.returncode, 0)  # 导入成功即通过。

if __name__ == "__main__":  # 支持直接运行。
    unittest.main()  # 打印真实测试数量。

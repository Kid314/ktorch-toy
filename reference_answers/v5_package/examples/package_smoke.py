"""v5离线检查包入口、CLI帮助和HTTP处理器是否能导入。"""  # 不要求checkpoint或启动后台服务。
from pathlib import Path  # 定位本版Python目录。
import subprocess  # 运行CLI帮助确认console逻辑可导入。
import sys  # 当前Python解释器。
ROOT = Path(__file__).resolve().parents[1]  # v5目录。
env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "python")}  # 临时让子进程加载本版包。
subprocess.run([sys.executable, "-m", "ktorch.cli", "--help"], env=env, check=True)  # help不需要模型文件。
subprocess.run([sys.executable, "-c", "import ktorch.server; print('server import ok')"], env=env, check=True)  # 服务模块可被加载。
print("package smoke ok")  # 只有两个命令都成功才到达这里。

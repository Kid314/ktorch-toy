"""让示例从任何工作目录都能找到本版源码。"""  # 各版不会导入别的版本。
from pathlib import Path  # 用文件位置定位版本根目录。
import sys  # 调整示例自身的模块搜索路径与日志编码。
ROOT = Path(__file__).resolve().parents[1]  # examples的父目录就是当前版本。
sys.path.insert(0, str(ROOT / "python"))  # 优先使用这版的ktorch包。
if hasattr(sys.stdout, "reconfigure"):  # 重定向到日志时也使用统一UTF-8。
    sys.stdout.reconfigure(encoding="utf-8")  # 避免中文输出被系统代码页误解。

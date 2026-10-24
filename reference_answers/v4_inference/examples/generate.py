"""从保存的参数继续逐字符生成。"""  # 可以在独立Python进程运行。
from _bootstrap import ROOT  # 使用当前版本的包与目录。
import argparse  # 接收提示文本和生成长度。
from ktorch.checkpoint import load  # 复用配置/词表/参数加载规则。
from ktorch.training import generate  # 使用首版greedy生成函数。
parser = argparse.ArgumentParser(description=__doc__)  # 创建说明清楚的CLI。
parser.add_argument("--checkpoint", default=str(ROOT / "build" / "checkpoint.json"))  # 默认读取本版训练产物。
parser.add_argument("--prompt", default="我")  # 提示字符必须在训练词表中。
parser.add_argument("--tokens", type=int, default=20)  # 最大新增字符数。
args = parser.parse_args()  # 读取用户输入。
model, vocab, _ = load(args.checkpoint)  # 重新创建模型并加载真实权值。
print(generate(model, vocab, args.prompt, args.tokens))  # 只执行推理，不更新参数。

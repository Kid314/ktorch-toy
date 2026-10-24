"""v2从schema1或schema2 checkpoint加载模型并逐字符生成。"""  # 生成不恢复优化器、不建立计算图。
from _bootstrap import ROOT  # 先把当前版本加入Python路径。
import argparse  # 接收checkpoint、prompt和新增token数。
from ktorch.checkpoint import load  # schema1兼容推理，schema2附带校验。
from ktorch.training import generate  # 复用窗口截取和greedy规则。
parser = argparse.ArgumentParser(description=__doc__)  # 创建可读帮助。
parser.add_argument("--checkpoint", default=str(ROOT / "build" / "checkpoint.json"))  # 默认v2训练产物。
parser.add_argument("--prompt", default="我")  # 必须包含在保存的词表中。
parser.add_argument("--tokens", type=int, default=20)  # 最多新增多少字符。
args = parser.parse_args()  # 读取参数。
model, vocab, payload = load(args.checkpoint)  # 完整校验后返回模型和词表。
print(generate(model, vocab, args.prompt, args.tokens))  # 只读取参数，不改变checkpoint。

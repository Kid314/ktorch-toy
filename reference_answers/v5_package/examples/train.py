"""用真实中文语料训练一个单层字符GPT。"""  # 默认只有小模型与短训练。
from _bootstrap import ROOT  # 先配置本版Python路径。
import argparse  # 提供可复现的实验参数。
from pathlib import Path  # 用普通文件接口读取本地语料。
from ktorch import Config, GPT  # 使用自己的模型实现。
from ktorch import checkpoint  # 保存训练后的参数。
from ktorch.training import tokenize, train, generate  # 训练与生成共用同一个模型。

def main():  # 示例入口保留清楚的读取、训练、保存顺序。
    parser = argparse.ArgumentParser(description=__doc__)  # 命令行帮助来自文件说明。
    parser.add_argument("--data", default=str(ROOT / "data" / "tiny.txt"))  # 使用本版的小型真实文本。
    parser.add_argument("--steps", type=int, default=100)  # 用户可选择更短或更长训练。
    parser.add_argument("--lr", type=float, default=0.05)  # SGD学习率。
    parser.add_argument("--checkpoint", default=str(ROOT / "build" / "checkpoint.json"))  # 产物留在本版build。
    args = parser.parse_args()  # 读取本次实验配置。
    text = Path(args.data).read_text(encoding="utf-8").strip()  # 去掉文件末尾换行，中文保持完整。
    vocab, ids = tokenize(text)  # 得到词表和整数语料流。
    model = GPT(Config(len(vocab)))  # 使用固定seed初始化小模型。
    print("词表：", vocab, "；参数量：", sum(p.shape[0] * p.shape[1] for p in model.parameters()))  # 打印真实模型大小。
    report = train(model, ids, steps=args.steps, lr=args.lr)  # 实际前向、反向和更新都在此发生。
    checkpoint.save(args.checkpoint, model, vocab, report)  # 保存普通参数数值与验证信息。
    print(f"固定窗口loss：{report['initial_loss']:.6f} → {report['final_loss']:.6f}")  # 对比同一组评估窗口。
    print("生成：", generate(model, vocab, text[0], 20))  # 输出来自模型分数，不是固定字符串。
    print("已保存：", args.checkpoint)  # 告诉用户生成脚本需要哪个文件。

if __name__ == "__main__":  # 导入示例不会启动训练。
    main()  # 只在直接运行时执行。

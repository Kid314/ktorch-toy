"""v2训练入口：选择SGD或AdamW，并能从schema2 checkpoint继续。"""  # CLI只负责调度，数值仍由模型和C++后端完成。
from _bootstrap import ROOT  # 初始化当前版本的Python路径。
import argparse  # 提供可复现且可检查的命令行参数。
from pathlib import Path  # 读取UTF-8文本和构造默认路径。
from ktorch import Config, GPT  # 创建单层参考模型。
from ktorch import checkpoint  # 负责schema和原子保存。
from ktorch.optim import SGD, AdamW  # v2允许用户选择更新规则。
from ktorch.training import Trainer, tokenize, generate  # Trainer保存采样与优化器状态。

def main():  # 保持入口顺序明显，便于逐行阅读。
    parser = argparse.ArgumentParser(description=__doc__)  # 让--help成为接口的一部分。
    parser.add_argument("--data", default=str(ROOT / "data" / "tiny.txt"))  # 训练语料路径。
    parser.add_argument("--steps", type=int, default=10)  # 本次新增的优化器步数。
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="adamw")  # v2默认展示可靠优化器。
    parser.add_argument("--lr", type=float, default=0.003)  # AdamW默认学习率；SGD可显式调整。
    parser.add_argument("--batch-size", type=int, default=2)  # 二维微批累计数量，不创建三维Tensor。
    parser.add_argument("--checkpoint", default=str(ROOT / "build" / "checkpoint.json"))  # schema2目标文件。
    parser.add_argument("--resume", action="store_true", help="从已有schema2 checkpoint恢复精确训练状态")  # 恢复需要同一语料和结构。
    args = parser.parse_args()  # 读取本次实验的显式配置。
    text = Path(args.data).read_text(encoding="utf-8").strip()  # UTF-8字符保持为Python Unicode字符。
    if args.resume:  # 恢复路径必须先加载完整训练状态。
        trainer = Trainer.restore(args.checkpoint, text)  # 该函数校验摘要、语料、参数顺序和优化器状态。
        if args.optimizer.lower() != trainer.optimizer.__class__.__name__.lower():  # 不允许用另一种优化器悄悄续训。
            raise ValueError("--optimizer必须与checkpoint中的优化器一致")  # 修正配置后再恢复。
    else:  # 新训练从固定模型和随机源开始。
        vocab, _ = tokenize(text)  # 建立本次训练的词表。
        model = GPT(Config(len(vocab)))  # 参数初始化由Config.seed决定。
        optimizer = SGD(model.parameters(), lr=args.lr, clip=1.0) if args.optimizer == "sgd" else AdamW(model.parameters(), lr=args.lr, clip=1.0)  # 只在Python层选择策略。
        trainer = Trainer(model, vocab, text, optimizer, seed=7, batch_size=args.batch_size)  # 保存采样器与语料摘要。
    report = trainer.fit(args.steps, log_every=max(1, min(25, args.steps)))  # 继续指定数量的优化器更新。
    trainer.save(args.checkpoint, metadata={"cli_optimizer": trainer.optimizer.__class__.__name__, "cli_steps": args.steps})  # 保存模型与可恢复训练状态。
    print(f"固定窗口loss：{report['initial_loss']:.6f} → {report['final_loss']:.6f}")  # 对比同一评估窗口。
    print("生成：", generate(trainer.model, trainer.vocab, text[0], 20))  # 输出来自当前模型logits。
    print("累计优化器步数：", trainer.steps)  # 恢复时可检查是否接在旧步数之后。
    print("已保存：", args.checkpoint)  # 便于另一个进程调用generate.py。

if __name__ == "__main__":  # 导入模块不会自动训练。
    main()  # 直接执行CLI入口。

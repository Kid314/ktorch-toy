"""v5的命令行入口：读取checkpoint并生成文本。"""  # CLI只负责参数与错误呈现。
from pathlib import Path  # 默认路径可相对于当前包计算。
import argparse  # 标准库命令行解析。
import sys  # 返回清晰的非零错误码。
from .checkpoint import load  # 复用经过schema和完整性检查的加载器。
from .training import generate  # 生成本身仍使用模型logits。

def main(argv=None):  # 允许控制台脚本和测试直接调用。
    parser = argparse.ArgumentParser(description=__doc__)  # 让新用户可以先看CLI帮助。
    parser.add_argument("--checkpoint", required=True, help="训练后JSON checkpoint路径")  # 不默认为不可知的当前目录。
    parser.add_argument("--prompt", required=True, help="词表中已有的Unicode字符前缀")  # 由用户明确输入提示。
    parser.add_argument("--tokens", type=int, default=20, help="最多新增字符数量")  # 控制请求资源。
    args = parser.parse_args(argv)  # 解析当前调用的参数列表。
    try:  # 将用户可修复的输入错误转成短错误。
        model, vocab, _ = load(Path(args.checkpoint))  # checkpoint负责配置、词表和权值校验。
        print(generate(model, vocab, args.prompt, args.tokens))  # 输出纯文本，便于shell管道使用。
        return 0  # 成功退出。
    except (OSError, ValueError, RuntimeError) as exc:  # 不暴露Python内部堆栈给普通CLI用户。
        print(f"ktorch-reference-generate: {exc}", file=sys.stderr)  # 错误仍保留可检索的前缀。
        return 2  # 约定参数/文件错误返回2。

if __name__ == "__main__":  # 支持python -m ktorch.cli。
    raise SystemExit(main())  # 将返回码交给操作系统。

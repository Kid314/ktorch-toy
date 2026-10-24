"""JSON保存配置、词表和参数；不保存激活图。"""  # 首版不承诺精确恢复优化器过程。
import json  # 使用可阅读且不执行任意代码的格式。
import os  # 用原子替换避免写到一半留下损坏检查点。
from pathlib import Path  # 统一路径处理。
import tempfile  # 临时文件与目标放在同一目录。
from .model import Config, GPT  # 加载时先重建模型结构。

def atomic_json(path, payload):  # 所有保存都先写临时文件。
    path = Path(path)  # 接受字符串或Path。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建明确的输出目录。
    temporary = None  # 用于失败时清理本次临时文件。
    try:  # 完成写入后才替换目标。
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as stream:  # 放在同一文件系统中。
            temporary = Path(stream.name)  # 记住只属于本次保存的文件名。
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)  # NaN和Inf不应进入合法检查点。
            stream.flush()  # 将Python缓冲交给操作系统。
            os.fsync(stream.fileno())  # 要求实际刷新文件内容。
        os.replace(temporary, path)  # 原子地换入完整新文件。
    finally:  # 保存失败时不残留半成品。
        if temporary is not None and temporary.exists():  # 成功replace后临时路径已经不存在。
            temporary.unlink()  # 只删除本次函数创建的临时文件。

def save(path, model, vocab, metadata=None):  # 保存可独立加载的推理模型。
    payload = {"format": "ktorch-reference", "schema": 1, "config": model.config.to_dict(), "vocab": list(vocab), "parameters": model.state_dict(), "metadata": metadata or {}}  # 字段含义都可直接检查。
    atomic_json(path, payload)  # 统一使用可靠写入路径。

def load(path):  # 返回模型、词表以及原始元信息。
    with Path(path).open(encoding="utf-8") as stream:  # 明确按UTF-8处理中文词表。
        payload = json.load(stream)  # JSON只解码数据。
    if payload.get("format") != "ktorch-reference" or payload.get("schema") != 1:  # 不静默猜测未知格式。
        raise ValueError("不支持的checkpoint格式或schema")  # 新版兼容策略会在后续版本说明。
    cfg = Config(**payload["config"])  # 配置验证发生在构造时。
    vocab = payload["vocab"]  # 生成必须使用训练时的同一词表。
    if not isinstance(vocab, list) or len(vocab) != cfg.vocab_size or len(set(vocab)) != len(vocab) or any(not isinstance(c, str) or len(c) != 1 for c in vocab):  # 每个Unicode码点对应唯一token。
        raise ValueError("词表无效或与模型尺寸不一致")  # 阻止编号含义错位。
    model = GPT(cfg)  # 建立相同的参数身份和E绑定关系。
    model.load_state_dict(payload["parameters"])  # 全部名称、shape、数值检查通过后写入。
    return model, vocab, payload  # 调用者可读取训练元信息但不把它当图。

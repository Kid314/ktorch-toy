"""一个注意力头、一个Pre-LN Block的字符GPT。"""  # 本文件呈现完整模型公式。
from dataclasses import dataclass, asdict  # 配置能够和checkpoint一起保存。
import math  # 注意力分数除以sqrt(d)。
import random  # 初始化只使用Python标准库随机源。
from .nn import Module, Linear, Embedding, LayerNorm  # 复用层封装。
from . import functional as F  # 复用无状态数学函数。

@dataclass  # 明确配置字段并提供可读表示。
class Config:  # 首版只改变尺寸，不改变单层单头结构。
    vocab_size: int  # 词表大小由数据决定。
    width: int = 8  # 每个token的隐藏向量宽度。
    context: int = 8  # 学习位置表和训练窗口的上限。
    hidden: int = 32  # MLP内部特征宽度，默认4d。
    seed: int = 42  # 可复现的参数初始化种子。
    def __post_init__(self):  # 在分配参数之前检查配置。
        for name in ("vocab_size", "width", "context", "hidden"):  # 这些字段都必须为正整数。
            value = getattr(self, name)  # 读取当前字段。
            if type(value) is not int or value <= 0 or value > 4096:  # 设置教学框架的明确容量界限。
                raise ValueError(f"{name}必须是1到4096的整数")  # 提前拒绝明显误配置。
        if self.width < 2 or type(self.seed) is not int:  # d=1的LayerNorm失去有效特征变化。
            raise ValueError("width至少为2，seed必须是整数")  # 报出可理解的原因。
    def to_dict(self):  # 序列化不包含任何Tensor。
        return asdict(self)  # 返回纯配置字典。

class Attention(Module):  # 单头注意力把可见上下文汇总到每个位置。
    def __init__(self, cfg, rng):  # 全部投影共享模型的随机源。
        self.width = cfg.width  # 缩放因子等于单头的维度。
        self.q = Linear(cfg.width, cfg.width, rng)  # Query投影。
        self.k = Linear(cfg.width, cfg.width, rng)  # Key投影，参数独立于Q。
        self.v = Linear(cfg.width, cfg.width, rng)  # Value投影，负责被汇总的内容。
        self.out = Linear(cfg.width, cfg.width, rng)  # 注意力汇总后的输出投影。
    def forward(self, x):  # x形状[T,d]，来自Block的第一次LayerNorm。
        q, k, v = self.q(x), self.k(x), self.v(x)  # 三次投影保持[T,d]。
        scores = (q @ k.T) / math.sqrt(self.width)  # 得到[T,T]的匹配分数。
        weights = F.softmax(scores, causal=True)  # 每行只读取自己和之前位置。
        return self.out(weights @ v)  # 按比例汇总，再投影回d维。

class MLP(Module):  # 对每个位置独立做非线性特征处理。
    def __init__(self, cfg, rng):  # 使用与其他层相同的初始化规则。
        self.up = Linear(cfg.width, cfg.hidden, rng)  # d扩展到hidden。
        self.down = Linear(cfg.hidden, cfg.width, rng)  # hidden压回d。
    def forward(self, x):  # 所有位置共享这两组权值。
        return self.down(F.gelu(self.up(x)))  # Linear→GELU→Linear。

class Block(Module):  # 一个Block内部仍包含多次线性运算。
    def __init__(self, cfg, rng):  # 构建两次归一化和两类子层。
        self.norm1 = LayerNorm(cfg.width)  # 注意力前的归一化。
        self.attention = Attention(cfg, rng)  # 跨位置读取信息。
        self.norm2 = LayerNorm(cfg.width)  # MLP前的归一化。
        self.mlp = MLP(cfg, rng)  # 位置内部的非线性变换。
    def forward(self, x):  # 使用Pre-LN结构。
        x = x + self.attention(self.norm1(x))  # 第一次残差加回归一化前输入。
        return x + self.mlp(self.norm2(x))  # 第二次残差加回第一次残差后的结果。

class GPT(Module):  # 整个模型将字符ID转换成下一字符分数。
    def __init__(self, cfg):  # 构造期间只创建长期参数。
        self.config = cfg  # 保留可序列化配置。
        rng = random.Random(cfg.seed)  # 不干扰全局随机状态。
        self.token = Embedding(cfg.vocab_size, cfg.width, rng)  # 字符表E[V,d]。
        self.position = Embedding(cfg.context, cfg.width, rng)  # 可学习位置表P[context,d]。
        self.block = Block(cfg, rng)  # 只有一个Transformer Block。
        self.final_norm = LayerNorm(cfg.width)  # 输出投影前的最后归一化。
    def forward(self, ids):  # 输入是当前窗口的整数序列。
        ids = list(ids)  # 将range等对象固定为本次输入。
        if not 0 < len(ids) <= self.config.context:  # 不越过位置表容量。
            raise ValueError("输入长度必须介于1和context之间")  # 超长窗口由生成器负责截取。
        x = self.token(ids) + self.position(range(len(ids)))  # 每次窗口位置从0重新编号。
        h = self.final_norm(self.block(x))  # 经过单Block和最后归一化。
        return h @ self.token.weight.T  # 与输入共用E，梯度将合并两条路径。

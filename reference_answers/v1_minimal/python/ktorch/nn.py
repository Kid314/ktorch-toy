"""Module管理参数身份，具体层组合已经验证的原生算子。"""  # 模块不保存每轮激活图。
import math  # 依据输入宽度设置初始化尺度。
from .tensor import Parameter, matrix  # 所有权值都使用同一种Parameter。
from . import functional as F  # 将数值规则留给无状态函数与C++。

class Module:  # 基类只组织调用与参数，不约束forward必须接收同一类型。
    def __call__(self, *args, **kwargs):  # 使layer(x)等价于layer.forward(x)。
        return self.forward(*args, **kwargs)  # 子类定义真实输入含义。
    def named_parameters(self):  # 递归收集唯一的Parameter身份。
        output, seen_parameters, seen_modules = [], set(), set()  # 分别记录参数与模块的已访问对象。
        def visit(value, name):  # 支持模块、简单容器以及Parameter。
            if isinstance(value, Parameter):  # 只有Parameter进入优化器列表。
                if id(value) not in seen_parameters:  # 共享权值不能重复更新。
                    seen_parameters.add(id(value))  # 登记对象身份而不是数值内容。
                    output.append((name, value))  # 保存稳定的参数路径。
            elif isinstance(value, Module) and id(value) not in seen_modules:  # 避免模块别名导致递归重复。
                seen_modules.add(id(value))  # 先登记，再访问子对象。
                for key, child in vars(value).items():  # 层的属性决定参数名。
                    visit(child, f"{name}.{key}" if name else key)  # 构造层级名称。
            elif isinstance(value, (list, tuple)):  # 给以后按列表组织模块留下简单支持。
                for i, child in enumerate(value):  # 用稳定序号标识容器元素。
                    visit(child, f"{name}.{i}")  # 继续收集内部Parameter。
            elif isinstance(value, dict):  # 字典中的模块也可被发现。
                for key, child in value.items():  # 保持插入顺序。
                    visit(child, f"{name}.{key}")  # 保留可读参数路径。
        visit(self, "")  # 从当前模型根节点开始。
        return output  # 每个共享Parameter只出现一次。
    def parameters(self):  # 优化器只需要Parameter对象。
        return [value for _, value in self.named_parameters()]  # 去掉名称但保留顺序。
    def state_dict(self):  # 保存普通数值，不保存计算图。
        return {name: value.tolist() for name, value in self.named_parameters()}  # 字典可直接编码为JSON。
    def load_state_dict(self, state):  # 加载前先验证所有项，避免部分写入。
        expected = dict(self.named_parameters())  # 当前模型结构决定合法参数集合。
        if not isinstance(state, dict) or set(state) != set(expected):  # 参数名必须完整一致。
            raise ValueError("checkpoint参数名称与模型结构不一致")  # 不静默忽略缺失或多余参数。
        for name, value in expected.items():  # 第一遍只验证候选值。
            rows, cols, _ = matrix(state[name])  # 检查矩形、有限值和大小。
            if (rows, cols) != value.shape:  # 对每个参数检查形状。
                raise ValueError(f"参数{name}的shape不匹配")  # 在任何写入前报错。
        for name, value in expected.items():  # 全部合法后才执行加载。
            value.set_data(state[name])  # 复用原Parameter身份，保持绑定关系。

class Linear(Module):  # 一个线性层拥有自己的权值和可选偏置。
    def __init__(self, inputs, outputs, rng, bias=False):  # 随机源由模型统一传入。
        scale = 1.0 / math.sqrt(inputs)  # 使用清楚的零均值初始化尺度。
        self.weight = Parameter([[rng.gauss(0.0, scale) for _ in range(inputs)] for _ in range(outputs)])  # 存储布局为[out,in]。
        self.bias = Parameter([[0.0] * outputs]) if bias else None  # 首版GPT投影默认没有bias。
    def forward(self, x):  # 对所有位置使用同一套参数。
        return F.linear(x, self.weight, self.bias)  # 不在层里重复实现矩阵循环。

class Embedding(Module):  # Embedding拥有一张可训练参数表。
    def __init__(self, rows, width, rng):  # rows可以是词表大小或上下文容量。
        self.weight = Parameter([[rng.gauss(0.0, 0.2) for _ in range(width)] for _ in range(rows)])  # 初值随机且不预设语义。
    def forward(self, ids):  # IDs是普通整数序列。
        return self.weight.gather(ids)  # 反向会将梯度累加到被访问的行。

class LayerNorm(Module):  # 每个LayerNorm拥有独立的仿射参数。
    def __init__(self, width):  # width是特征维度d。
        self.gamma = Parameter([[1.0] * width])  # 初始缩放为1。
        self.beta = Parameter([[0.0] * width])  # 初始平移为0。
    def forward(self, x):  # 对每行单独计算归一化。
        return F.layer_norm(x, self.gamma, self.beta)  # epsilon使用统一默认值。

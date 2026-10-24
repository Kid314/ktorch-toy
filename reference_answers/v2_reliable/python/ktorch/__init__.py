"""ktorch参考实现的公开入口。"""  # 导入包不会训练、下载或编译。
from .tensor import Tensor, Parameter, no_grad  # 提供张量、参数与推理上下文。
from . import functional, nn, optim  # 保留接近常见框架的模块组织。
from .model import Config, GPT  # 提供完整的单层字符模型。
__version__ = "2.0.0"  # 此版本号描述本地参考实现阶段。

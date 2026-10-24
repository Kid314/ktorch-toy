"""无状态函数只组合参数和算子，不拥有模型权值。"""  # 有状态层放在nn.py。
from . import _backend as B  # 复用经过声明的ABI。
from .tensor import Tensor, indices  # 使用统一的句柄与索引规则。

def linear(x, weight, bias=None):  # weight采用[out,in]布局。
    result = x @ weight.T  # [T,in]@[in,out]得到[T,out]。
    return result if bias is None else result + bias  # 偏置沿位置维广播。

def layer_norm(x, gamma, beta, eps=1e-5):  # gamma和beta也是可训练Tensor。
    return Tensor._wrap(B.lib.kt_layer_norm(x._handle, gamma._handle, beta._handle, float(eps)))  # 原生实现均值、方差和完整反向。

def gelu(x):  # 对应MLP的非线性。
    return Tensor._wrap(B.lib.kt_gelu(x._handle))  # 原生采用tanh近似及对应导数。

def softmax(x, causal=False):  # 普通词表概率和因果注意力共享稳定算法。
    return Tensor._wrap(B.lib.kt_softmax(x._handle, int(causal)))  # 因果模式仅用于方阵分数。

def cross_entropy(logits, targets):  # 直接输入分数，不提前softmax。
    array, count = indices(targets)  # 一行对应一个正确下一token。
    return Tensor._wrap(B.lib.kt_cross_entropy(logits._handle, array, count))  # 返回平均损失1×1。

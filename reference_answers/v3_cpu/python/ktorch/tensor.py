"""二维Tensor句柄、Parameter与no_grad的Python入口。"""  # Python不重新计算原生算子的结果。
from contextlib import contextmanager  # 让no_grad在异常退出时也能恢复状态。
import math  # 检查用户输入是否有限。
from . import _backend as B  # 所有原生调用经过同一ABI声明。

def matrix(data):  # 将用户数据规范化为二维有限double数组。
    if not isinstance(data, (list, tuple)) or not data:  # 要求显式的非空二维列表。
        raise ValueError("Tensor需要非空二维列表")  # 避免0维/1维含义不清。
    rows = len(data)  # 外层列表长度对应行数。
    if not all(isinstance(row, (list, tuple)) for row in data):  # 所有行必须是序列。
        raise ValueError("Tensor只支持二维数据")  # 第一版不隐式添加维度。
    cols = len(data[0])  # 第一行确定特征宽度。
    if cols == 0 or any(len(row) != cols for row in data):  # 拒绝空列和锯齿列表。
        raise ValueError("每行长度必须相等且非零")  # 防止形状与实际内存不一致。
    if rows * cols > 10000000:  # 与原生接口的教学容量上限保持一致。
        raise ValueError("Tensor过大，请缩小本次实验")  # 防止无意间分配巨大数组。
    flat = [float(x) for row in data for x in row]  # 按行展开成连续顺序。
    if not all(math.isfinite(x) for x in flat):  # 普通Tensor不接收NaN/Inf。
        raise ValueError("Tensor数值必须有限")  # 将错误定位在输入处。
    return rows, cols, flat  # 返回形状和可交给ctypes的数值。

def indices(values):  # 检查词表索引而不对其求导。
    values = list(values)  # 允许range等有限迭代对象。
    if not values or any(type(i) is not int or not -(2**31) <= i < 2**31 for i in values):  # bool也不当作合法token ID。
        raise ValueError("索引必须是非空int32序列")  # 越界词表由C++进一步检查。
    return (B.C.c_int * len(values))(*values), len(values)  # 显式分配C整型数组。

class Tensor:  # 外部对象只拥有一个原生句柄。
    def __init__(self, data, requires_grad=False):  # 创建叶子张量，默认不求导。
        rows, cols, flat = matrix(data)  # 验证形状并展开数值。
        buffer = (B.C.c_double * len(flat))(*flat)  # 临时连续数组供原生复制。
        self._handle = B.handle(B.lib.kt_create(buffer, rows, cols, int(requires_grad)))  # 构造真实C++节点。
    @classmethod  # 运算结果已有原生节点，无需重新复制数值。
    def _wrap(cls, handle):  # 把原生返回的所有权接到Python。
        value = cls.__new__(cls)  # 跳过创建叶子张量的构造函数。
        value._handle = B.handle(handle)  # 先验证调用成功再保存ID。
        return value  # 返回包装好的结果。
    def close(self):  # 需要确定性释放时可显式调用。
        if getattr(self, "_handle", 0):  # 已释放对象不会再次释放。
            h, self._handle = self._handle, 0  # 先清句柄，保证异常时也不会重复释放。
            B.checked(B.lib.kt_release(h))  # 减少一个原生共享引用。
    def __del__(self):  # 正常离开作用域时回收原生句柄。
        try:  # Python解释器关闭时模块可能已销毁。
            self.close()  # 平时与显式close走同一条释放路径。
        except Exception:  # 析构函数不能把异常传播到解释器清理过程。
            pass  # 正常使用中的错误应通过显式调用检查。
    @property  # 将shape作为只读属性。
    def shape(self):  # 从原生节点查询当前形状。
        out = (B.C.c_int * 2)()  # 分配两个整数的输出缓冲。
        B.checked(B.lib.kt_shape(self._handle, out))  # 无效句柄在这里报错。
        return int(out[0]), int(out[1])  # 返回Python元组。
    @property  # requires_grad是节点状态而不是Python副本。
    def requires_grad(self):  # 读取原生标记。
        value = B.lib.kt_requires_grad(self._handle)  # -1代表错误。
        if value < 0:  # 不能把失败误解成True。
            raise B.failure()  # 给出失效句柄等原因。
        return bool(value)  # 正常结果只有0和1。
    def _read(self, gradient=False):  # 统一读取数值和梯度。
        rows, cols = self.shape  # 先知道需要多少空间。
        out = (B.C.c_double * (rows * cols))()  # 分配Python拥有的复制缓冲。
        B.checked(B.lib.kt_data(self._handle, out, rows * cols, int(gradient)))  # 数据复制后不依赖原生数组地址。
        return [[float(out[i * cols + j]) for j in range(cols)] for i in range(rows)]  # 还原二维列表。
    def tolist(self):  # 便于打印、checkpoint和手算对照。
        return self._read(False)  # 返回数值副本，不暴露可修改的原生指针。
    @property  # grad属性同样返回副本。
    def grad(self):  # 查看损失对当前张量的梯度。
        return self._read(True)  # 读取原生梯度数组。
    def item(self):  # 将1×1损失转为日志中的普通数。
        if self.shape != (1, 1):  # 非标量不能静默只取第一格。
            raise ValueError("item仅用于1×1 Tensor")  # 避免错误的loss打印。
        return self.tolist()[0][0]  # 不在日志里保存整个计算图。
    def set_data(self, data):  # 用于参数加载和有限差分检查。
        rows, cols, flat = matrix(data)  # 验证所有候选数值。
        if (rows, cols) != self.shape:  # 不能通过加载数据改变结构。
            raise ValueError("参数shape不匹配")  # 调用者应先重建正确模型。
        B.checked(B.lib.kt_replace(self._handle, (B.C.c_double * len(flat))(*flat), len(flat)))  # 原生会记录版本变化。
    def backward(self):  # 触发整张图的原生反向。
        B.checked(B.lib.kt_backward(self._handle))  # C++负责排序与梯度累加。
    def zero_grad(self):  # 清除当前参数已有梯度。
        B.checked(B.lib.kt_zero_grad(self._handle))  # 不创建计算图。
    def __add__(self, other):  # Python的加法对应广播add。
        other = other if isinstance(other, Tensor) else Tensor([[other]])  # 标量转为1×1常量。
        return Tensor._wrap(B.lib.kt_add(self._handle, other._handle))  # 数学运算在C++完成。
    __radd__ = __add__  # 标量在左侧时也使用加法规则。
    def __mul__(self, other):  # 星号表示逐元素乘法或常数缩放。
        return Tensor._wrap(B.lib.kt_mul(self._handle, other._handle)) if isinstance(other, Tensor) else self.scale(float(other))  # 明确区别于@。
    __rmul__ = __mul__  # 支持常数乘Tensor。
    def __truediv__(self, scalar):  # 第一版除法只支持非零常数。
        if not math.isfinite(float(scalar)) or scalar == 0:  # 检查分母。
            raise ValueError("除数必须是有限非零常数")  # 不创建不合法运算。
        return self.scale(1.0 / scalar)  # 除常数等价于乘倒数。
    def scale(self, factor):  # 提供显式缩放接口。
        return Tensor._wrap(B.lib.kt_scale(self._handle, float(factor)))  # 常数不作为求导节点。
    def __matmul__(self, other):  # @只接受另一个二维Tensor。
        if not isinstance(other, Tensor):  # 提前定位类型错误。
            raise TypeError("matmul需要Tensor输入")  # 不进行隐式数组转换。
        return Tensor._wrap(B.lib.kt_matmul(self._handle, other._handle))  # 原生检查共享维度。
    @property  # .T保持接近常见张量库的写法。
    def T(self):  # 返回转置结果。
        return Tensor._wrap(B.lib.kt_transpose(self._handle))  # 当前实现分配连续副本。
    def gather(self, values):  # 为字符和位置查表提供入口。
        array, count = indices(values)  # 不把整数索引变成求导Tensor。
        return Tensor._wrap(B.lib.kt_gather(self._handle, array, count))  # 原生复制索引并连接参数表。

class Parameter(Tensor):  # Parameter是需被优化器登记的叶子Tensor。
    def __init__(self, data):  # 初始数值由层决定。
        super().__init__(data, requires_grad=True)  # 对模型权值启用梯度。

@contextmanager  # 自动执行进入/退出操作。
def no_grad():  # 推理阶段关闭计算图记录。
    previous = B.lib.kt_grad_mode(0)  # 保存原状态，支持嵌套。
    try:  # 用户推理代码可能抛异常。
        yield  # 在这个作用域内使用无图算子。
    finally:  # 正常或异常退出都必须恢复。
        B.lib.kt_grad_mode(previous)  # 不影响其他线程的记录状态。

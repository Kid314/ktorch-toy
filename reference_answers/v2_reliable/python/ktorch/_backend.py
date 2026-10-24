"""ctypes只负责传递句柄与数组，数值计算和反向传播留在C++。"""  # ABI错误在这里统一转成Python异常。
import ctypes as C  # 使用Python标准库加载动态库。
import os  # 判断Windows平台。
from pathlib import Path  # 相对包文件定位库。
import sys  # 区分macOS与其他Unix平台。

H = C.c_uint64  # 不透明句柄与C++的uint64_t完全对应。
D = C.POINTER(C.c_double)  # double数组指针的ABI类型。
I = C.POINTER(C.c_int)  # 整数索引数组指针的ABI类型。
suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"  # 匹配构建脚本的文件名。
library = Path(__file__).with_name("_native" + suffix)  # 每个版本加载自己的动态库。
if not library.is_file():  # 缺失时不尝试静默退回Python数值实现。
    raise RuntimeError(f"缺少 {library}；请先运行本版 scripts/build.py。")  # 告诉学习者先完成编译。
lib = C.CDLL(str(library))  # 使用C调用约定加载原生实现。

def bind(name, args, result):  # 为每个ABI函数显式声明类型。
    function = getattr(lib, name)  # 查找导出的未改编符号。
    function.argtypes = args  # 防止Python把64位句柄当成32位整数。
    function.restype = result  # 正确解释返回值。
    return function  # 便于以后增加版本专属接口。

bind("kt_abi_version", [], C.c_int)  # 先识别ABI版本。
bind("kt_last_error", [], C.c_char_p)  # 错误消息由C++提供。
bind("kt_create", [D, C.c_int, C.c_int, C.c_int], H)  # 创建会复制输入double数组。
bind("kt_release", [H], C.c_int)  # 每个返回句柄应释放一次。
bind("kt_shape", [H, I], C.c_int)  # 写入两个整数形状。
bind("kt_data", [H, D, C.c_int, C.c_int], C.c_int)  # 最后一个参数选择data或grad。
bind("kt_replace", [H, D, C.c_int], C.c_int)  # 安全替换叶子数值。
bind("kt_requires_grad", [H], C.c_int)  # 查询是否记录梯度。
for name in ("kt_add", "kt_mul", "kt_matmul"):  # 这些算子都接受两个Tensor句柄。
    bind(name, [H, H], H)  # 统一声明双输入算子。
for name in ("kt_transpose", "kt_gelu"):  # 这些算子只有一个Tensor输入。
    bind(name, [H], H)  # 统一声明单输入算子。
bind("kt_scale", [H, C.c_double], H)  # 缩放系数是普通常数。
bind("kt_gather", [H, I, C.c_int], H)  # 明确索引数组长度。
bind("kt_layer_norm", [H, H, H, C.c_double], H)  # gamma和beta也是Tensor。
bind("kt_softmax", [H, C.c_int], H)  # 第二个参数开启因果模式。
bind("kt_cross_entropy", [H, I, C.c_int], H)  # 每个位置一个整数标签。
bind("kt_backward", [H], C.c_int)  # 只接受标量损失句柄。
bind("kt_zero_grad", [H], C.c_int)  # 清除当前参数梯度。
bind("kt_sgd", [H, C.c_double, C.c_double], C.c_int)  # 接收学习率和统一梯度缩放。
bind("kt_grad_mode", [C.c_int], C.c_int)  # 返回旧记录状态。
bind("kt_live_nodes", [], H)  # 读取生命周期诊断计数。
bind("kt_adamw", [H, H, H, C.c_int, *([C.c_double] * 6)], C.c_int)  # v2将参数、矩状态和超参数一起交给C++更新。
if lib.kt_abi_version() != 1:  # 阻止前端与错误版本后端混用。
    raise RuntimeError("原生ABI版本不匹配，请重新构建这一版。")  # 要求使用本版源码构建。

def failure():  # 立即复制错误消息，避免被后续原生调用覆盖。
    message = lib.kt_last_error()  # 获取当前线程最后一次错误。
    return RuntimeError(message.decode("utf-8", errors="replace") if message else "native call failed")  # 转为Python可读异常。

def checked(status):  # 状态型ABI以0为成功。
    if status != 0:  # 任何非0都意味着调用失败。
        raise failure()  # 保留原生错误原因。

def handle(value):  # Tensor返回型ABI以0为失败。
    if not value:  # 不允许无效句柄流入Python对象。
        raise failure()  # 在出错的调用处立即抛异常。
    return int(value)  # Python整数能够完整保存64位句柄。

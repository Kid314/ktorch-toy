/*
任务 K05：C ABI 的待实现边界。
ktorch-toy C++ / Python 边界的规划骨架。

文件职责
    将内部 C++ 张量与算子暴露为 ctypes 可以调用的 C 风格接口。
    统一句柄所有权、二维形状、连续 double 缓冲区和错误传递合同。
    Python 包装层在此接口之上组织易读的类和模型代码。

未来输入与输出
    输入：不透明张量句柄、double/int 数组指针、数组长度及简单标量配置。
    输出：新张量句柄、复制到调用方缓冲区的数据/梯度，或明确的状态码。
    错误：通过可读取的错误字符串返回上下文；C++ 异常不得跨越 C ABI。

规划 API 分组（签名待双方实现前统一）
    生命周期：kt_create、kt_release。
    属性与数据：kt_rows、kt_cols、kt_requires_grad、kt_copy_data、kt_copy_grad。
    参数写入：kt_set_data，需规定是否只允许修改叶子张量。
    基础算子：kt_add、kt_mul、kt_scale、kt_matmul、kt_transpose、kt_gather。
    模型算子：kt_layer_norm、kt_gelu、kt_softmax、kt_cross_entropy。
    求导与更新：kt_backward、kt_zero_grad、kt_sgd、kt_set_grad_enabled。
    错误读取：kt_last_error。

计划依赖
    tensor.h、tensor.cpp、ops.cpp 与 C++17 标准库。
    Windows / Unix 的动态库符号导出方式需分别规划。
    接口只传递约定好的 C 数据，避免跨语言传递 C++ 容器、异常或对象布局。

TODO：建议后续补写顺序
    1. 确定失败时的空句柄、状态码、错误字符串与所有权规则。
    2. 实现创建、复制数据、读取形状与释放，先打通最小 ctypes 调用。
    3. 逐项包装已完成的基础算子。
    4. 最后加入梯度访问、反向传播、no_grad 与 SGD。

未来验收
    Python 创建张量后能读回相同的形状和数据，并按约定释放。
    两端 double/int 类型、指针签名、数组长度与矩阵布局保持一致。
    对空句柄、非法形状与越界 ID 返回明确错误。
    释放一个 Python 句柄时，仍被计算图引用的内部节点有清楚的生命周期。
    通过 tests/test_native.py 的后续接口与数值测试。
*/

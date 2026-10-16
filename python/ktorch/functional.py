"""
任务 K07：封装无状态计算函数（当前仅为说明）。
计划 API：linear、gather、layer_norm、gelu、softmax、cross_entropy。
输入：Tensor、整数索引、目标 token 或必要的标量选项。
输出：由原生运算创建的结果 Tensor。
依赖：K06 和对应的 C++ 前向及反向运算。
TODO：linear 采用 weight[out_features, in_features]。
TODO：gather 接收字符或位置索引，索引本身不求梯度。
TODO：softmax 支持逐行归一化和因果遮罩。
TODO：cross_entropy 直接接收 logits 与下一 token 标签。
TODO：补充每个函数的形状约定和错误行为。
验收：不保存模型参数、不缓存激活或计算图。
验收：固定输入结果与独立数值参考一致。
进度：未实现；本文件没有运算函数。
"""

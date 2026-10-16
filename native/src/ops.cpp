/*
任务 K03：数值算子与局部梯度。
ktorch-toy CPU 基础算子与局部梯度的规划骨架。

文件职责
    为单层、单头 GPT 提供最少的二维张量计算。
    每个算子先用小数字讲清输入、输出与形状，再逐步补写前向及反向。

未来输入与输出 / 规划 API
    add、multiply：两个形状兼容的二维张量 -> 逐元素结果。
    scale：张量与标量系数 -> 同形状结果。
    matmul：[m,k] 与 [k,n] -> [m,n]。
    transpose：[m,n] -> [n,m]。
    gather_rows：[词表大小,宽度] 与 token ID 列表 -> [序列长度,宽度]。
    layer_norm：输入、逐列 gamma、beta 与 epsilon -> 与输入同形状的结果。
    gelu：输入张量 -> 同形状激活结果，近似公式需在教程与实现中保持一致。
    softmax：分数矩阵与是否使用因果遮罩 -> 每行归一化概率。
    cross_entropy：[位置数,词表大小] 的 logits 与每位置目标 ID -> 平均标量损失。

计划依赖
    tensor.h 中约定的存储、梯度缓冲区和计算图接口。
    仅使用 C++17 标准库；第一版使用 CPU double，不要求 GPU 或第三方数值库。

TODO：建议后续补写顺序
    1. add、scale、multiply 与清楚限定的广播规则。
    2. 非方阵 matmul 与 transpose。
    3. gather_rows，随后验证重复 ID 的梯度累加。
    4. LayerNorm 与 GELU。
    5. 稳定 softmax、因果遮罩与 logits 交叉熵。
    6. 分别加入局部反向传播，再组成一个完整的 GPT 前向与反向链路。

实现前需要统一的合同
    广播的反向传播需按被扩展维度累加。
    因果遮罩应在归一化之前生效，未来位置的概率与相应梯度应为零。
    softmax 与交叉熵需处理较大的有限 logits，避免直接指数运算溢出。
    非法形状、越界 ID 与不合法配置要给出可被 C API 捕获的错误。

未来验收
    每个算子均能用小矩阵核对形状和前向结果。
    可训练输入的梯度可通过中央差分检查。
    完整单头单 Block 示例的数值与文本教程一致。
    验收测试规划位于 tests/test_native.py。
*/

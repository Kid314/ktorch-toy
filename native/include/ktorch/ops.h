/*
任务 K03：C++ 算子接口（当前只有说明，尚未实现）。
职责：声明二维 add/mul/scale、matmul、transpose、gather、LayerNorm、
      GELU、softmax/因果遮罩、稳定交叉熵等实际所需运算。
输入/输出：Tensor 句柄及明确参数，返回新 Tensor；每个接口写清 shape。
依赖：K02 的 tensor.h；局部反向规则与 K04 协调。
验收：非方阵、广播归约、重复 ID、因果性及数值稳定用例。
*/

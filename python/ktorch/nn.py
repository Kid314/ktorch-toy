"""
任务 K07：规划层与参数管理（当前仅为说明）。
计划 API：Module、Linear、Embedding、LayerNorm、Parameter。
输入：层配置、Tensor 或整数 token ID。
输出：层的前向结果，以及可遍历的模型参数。
依赖：K06、functional.py 的无状态接口和相应原生运算。
TODO：Module 递归收集子模块参数，并按参数身份去重。
TODO：Linear 保存权值和偏置，内部调用 functional.linear。
TODO：Embedding 保存查表矩阵，LayerNorm 保存缩放与偏移。
TODO：state_dict 仅保存参数数值，不保存激活或计算图。
TODO：load_state_dict 检查名称与尺寸后再加载。
验收：同一共享参数仅被优化器更新一次。
验收：不同层可组合，层不会实现自己的数值后端。
进度：未实现；不含任何类或 forward 定义。
"""

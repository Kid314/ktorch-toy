"""
任务 K06：规划 ktorch 的 Python 公共入口（当前仅为说明，无实现）。
目标：给学习者提供接近 PyTorch 的简洁导入方式。
计划公开名称：Tensor、Parameter、nn、functional、optim、no_grad。
输入：用户的 import 语句和后续 API 调用。
输出：未来各模块的公开对象；本文件当前不会导出这些对象。
依赖：tensor.py、nn.py、functional.py、optim.py 逐步完成。
TODO：等对应任务实现后再添加导入，避免空模块导致误导。
TODO：明确 no_grad 的原生开关和嵌套恢复规则。
TODO：保持入口简单，数学计算应交给 C++ 后端。
验收：文档中的导入名称与已完成任务一致。
验收：不得在导入时执行训练或下载依赖。
进度：未实现；暂不能 import 后创建 Tensor 或训练模型。
"""

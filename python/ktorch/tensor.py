"""
任务 K06：设计 Python Tensor 句柄包装（当前仅为说明）。
计划 API：Tensor、Parameter、shape、tolist、grad、item、backward。
输入：二维浮点列表；Parameter 表示需要更新的模型参数。
输出：持有原生句柄的 Python 对象或复制回来的数值列表。
依赖：K05 和原生 Tensor、自动求导任务。
TODO：校验矩阵非空、矩形且维度匹配。
TODO：封装 +、*、@、转置和标量缩放。
TODO：全部数值计算及梯度计算委托 C++。
TODO：明确句柄释放与临时对象寿命，避免重复释放。
TODO：把 data 与 grad 区分；暴露必要的只读查看接口。
验收：二维小矩阵的结果与手算一致，出错时有明确信息。
验收：重复引用和临时 Tensor 不造成悬空指针。
进度：未实现；这里只列出计划名称，不含类定义。
"""

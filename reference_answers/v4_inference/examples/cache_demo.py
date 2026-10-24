"""演示完整因果注意力与逐token KV Cache的数值等价。"""  # 只用很短的固定列表。
from pathlib import Path  # 通过bootstrap定位版本包。
import sys  # 修改当前进程的模块路径。
ROOT = Path(__file__).resolve().parents[1]  # v4目录。
sys.path.insert(0, str(ROOT / "python"))  # 使用v4自己的cache模块。
from ktorch.cache import full_causal_attention, cached_causal_attention  # 导入两种路径。

queries = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  # 每个位置的查询向量。
keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  # 实际模型会由K投影产生这些值。
values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 实际模型会由V投影产生这些值。
full, full_weights = full_causal_attention(queries, keys, values)  # 逐行重算的基线。
cached, cached_weights, cache = cached_causal_attention(queries, keys, values)  # 逐token共享历史K/V。
max_error = max(abs(a - b) for left, right in zip(full, cached) for a, b in zip(left, right))  # 比较全部输出元素。
print("完整计算：", full)  # 展示基线输出。
print("KV Cache：", cached)  # 展示缓存路径输出。
print("最大绝对误差：", max_error)  # 正确实现应为0或浮点舍入量级。
print("cache长度：", len(cache.keys))  # 缓存包含3个历史位置。
print("最后一行权重：", cached_weights[-1])  # 最后位置能读取全部三行。

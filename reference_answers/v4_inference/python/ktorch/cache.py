"""v4的最小KV Cache：只负责一层单头注意力的推理缓存。"""  # 它不参与训练反向。
import math  # 计算缩放点积和指数。

def softmax(values):  # 对一维分数做稳定softmax。
    maximum = max(values)  # 先减最大值避免exp溢出。
    exponentials = [math.exp(value - maximum) for value in values]  # 所有指数保持有限。
    total = sum(exponentials)  # 分母是指数总和。
    return [value / total for value in exponentials]  # 返回和为1的注意力比例。

def attention_row(query, keys, values):  # 给一个查询和历史K/V计算一个输出向量。
    if not keys or len(keys) != len(values):  # Cache必须成对保存K和V。
        raise ValueError("keys和values必须非空且长度相等")  # 不返回部分状态。
    width = len(query)  # 单头的特征维度。
    if width == 0 or any(len(key) != width for key in keys) or any(len(value) != len(values[0]) for value in values):  # 检查矩阵行宽。
        raise ValueError("query/key/value的shape不一致")  # cache边界错误应尽早暴露。
    scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(width) for key in keys]  # QKᵀ/sqrt(d)。
    weights = softmax(scores)  # 在所有已经可见的历史位置上分配比例。
    output = [0.0] * len(values[0])  # 输出宽度等于Value宽度。
    for weight, value in zip(weights, values):  # 将每个历史Value按比例汇总。
        for index, element in enumerate(value):  # 遍历Value的每个特征。
            output[index] += weight * element  # 完成A×V的一行。
    return output, weights  # 返回输出与权重，便于教学和验证。

class KVCache:  # 每个请求拥有自己的缓存实例，不能跨请求混用。
    def __init__(self, width):  # width是Key的最后一维。
        if type(width) is not int or width <= 0:  # 防止空向量导致缩放无意义。
            raise ValueError("width必须是正整数")  # 明确缓存配置。
        self.width = width  # 保存预期Key/Query宽度。
        self.keys = []  # 历史Key按生成顺序保存。
        self.values = []  # 历史Value与Key一一对应。
    def clear(self):  # 新请求或上下文重建必须清空旧状态。
        self.keys.clear()  # 先清Key。
        self.values.clear()  # 再清Value。
    def append(self, key, value):  # prefill或decode都通过同一个入口追加。
        if len(key) != self.width or not value:  # Key宽度必须稳定，Value不能空。
            raise ValueError("KV shape不符合cache配置")  # 错误状态不写入半行。
        if self.values and len(value) != len(self.values[0]):  # 同一层Value宽度必须一致。
            raise ValueError("Value宽度发生变化")  # 需要新cache而不是修改旧cache。
        self.keys.append(list(key))  # 复制输入，避免调用者之后原地修改缓存。
        self.values.append(list(value))  # 同样复制Value。
    def decode(self, query):  # 将一个新位置的Q与当前历史K/V做注意力。
        if len(query) != self.width:  # 查询宽度必须和Key相同。
            raise ValueError("Query宽度不匹配")  # 先检查再计算。
        return attention_row(query, self.keys, self.values)  # decode只计算当前行。
    def snapshot(self):  # 复制缓存用于测试或诊断，不暴露内部列表。
        return {"keys": [row[:] for row in self.keys], "values": [row[:] for row in self.values]}  # JSON可保存普通数字。
    def restore(self, state):  # 从同层的快照恢复。
        keys, values = state.get("keys"), state.get("values")  # 读取显式字段。
        if not isinstance(keys, list) or not isinstance(values, list) or len(keys) != len(values):  # 防止损坏状态。
            raise ValueError("KV snapshot格式无效")  # 不部分恢复。
        self.clear()  # 新状态替代旧历史。
        for key, value in zip(keys, values):  # 逐行通过append重新执行shape检查。
            self.append(key, value)  # 保证恢复和正常追加有同一不变量。

def full_causal_attention(queries, keys, values):  # 作为无缓存基线计算完整下三角输出。
    outputs, rows = [], []  # 分别保存每个位置输出与注意力权重。
    for index, query in enumerate(queries):  # 第i个位置只能看到前i+1行。
        output, weights = attention_row(query, keys[:index + 1], values[:index + 1])  # 用完整历史的前缀计算。
        outputs.append(output)  # 保存第i行。
        rows.append(weights)  # 保存便于逐行对照的比例。
    return outputs, rows  # 返回完整prefill参考。

def cached_causal_attention(queries, keys, values):  # 逐位置追加K/V并计算decode输出。
    cache = KVCache(len(queries[0]))  # 新请求从空cache开始。
    outputs, rows = [], []  # 保存每次decode的结果。
    for query, key, value in zip(queries, keys, values):  # 模拟prefill后逐token decode。
        cache.append(key, value)  # 当前token的K/V先进入历史。
        output, weights = cache.decode(query)  # Query读取包含当前token的缓存。
        outputs.append(output)  # 保留当前token的输出。
        rows.append(weights)  # 保存当前可见长度。
    return outputs, rows, cache  # 返回cache供长度和状态检查。

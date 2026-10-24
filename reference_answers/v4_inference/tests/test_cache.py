"""v4只验证小规模缓存等价和状态边界，不做延迟压力测试。"""  # 性能测量留给用户选择。
from pathlib import Path  # 定位版本根目录。
import sys  # 加载本版Python模块。
import unittest  # 标准库测试框架。
ROOT = Path(__file__).resolve().parents[1]  # 当前版本目录。
sys.path.insert(0, str(ROOT / "python"))  # 不依赖v1/v2源码。
from ktorch.cache import KVCache, full_causal_attention, cached_causal_attention  # 导入真实cache实现。

class CacheTests(unittest.TestCase):  # 逐项验证cache不变量。
    def test_full_and_cached_are_equal(self):  # cache不能改变注意力数学结果。
        queries = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  # 固定短序列。
        keys, values = queries, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # K/V与Q长度一致。
        full, _ = full_causal_attention(queries, keys, values)  # 完整前缀参考。
        cached, _, cache = cached_causal_attention(queries, keys, values)  # 逐token路径。
        self.assertEqual(len(cache.keys), 3)  # 每个token恰好追加一次。
        for left, right in zip(full, cached):  # 对照每个位置的输出。
            for a, b in zip(left, right):  # 对照每个输出维度。
                self.assertAlmostEqual(a, b, places=12)  # 结果应仅有浮点舍入差异。
    def test_snapshot_restore_and_clear(self):  # 状态可保存、恢复和清空。
        cache = KVCache(2)  # 创建独立请求缓存。
        cache.append([1, 0], [2, 3])  # 追加第一行。
        cache.append([0, 1], [4, 5])  # 追加第二行。
        state = cache.snapshot()  # 取得普通Python状态。
        restored = KVCache(2)  # 新请求实例。
        restored.restore(state)  # 重新执行shape检查并恢复。
        self.assertEqual(restored.snapshot(), state)  # 两个缓存的内容一致。
        restored.clear()  # 新请求开始前必须清空历史。
        self.assertEqual(restored.keys, [])  # Key不应泄漏到下一请求。
    def test_invalid_shape_is_rejected(self):  # 错误缓存不能产生半状态。
        cache = KVCache(2)  # 只接受宽度2的Key。
        with self.assertRaises(ValueError):  # 让调用者看到明确错误。
            cache.append([1], [2, 3])  # 宽度1应被拒绝。
        self.assertEqual(cache.keys, [])  # 失败后缓存仍为空。

if __name__ == "__main__":  # 支持直接运行。
    unittest.main()  # 输出真实测试数量。

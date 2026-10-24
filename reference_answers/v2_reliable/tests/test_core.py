"""只用小矩阵和短训练验证真实行为，默认不做性能压测。"""  # 数值测试不依赖NumPy/PyTorch。
from pathlib import Path  # 测试从自己的版本目录定位包。
import gc  # 验证Python句柄离开作用域后释放图。
import math  # 检查有限值和数学参考。
import sys  # 将本版python目录加入搜索路径。
import tempfile  # checkpoint测试使用隔离的临时目录。
import unittest  # 使用标准库测试框架。
ROOT = Path(__file__).resolve().parents[1]  # 测试只使用本版源码。
sys.path.insert(0, str(ROOT / "python"))  # 避免误导入其他版本。
from ktorch import Tensor, Parameter, Config, GPT, no_grad, functional as F  # 导入自己的计算框架。
from ktorch import _backend as B, checkpoint  # 诊断与保存加载接口。
from ktorch.training import tokenize, train, generate, Trainer  # 文本端到端功能和恢复接口。
from ktorch.optim import AdamW  # v2优化器状态需要单独检查。

class CoreTests(unittest.TestCase):  # 每个用例验证独立的实现性质。
    def assert_matrix(self, actual, expected, places=7):  # 用逐格比较给出可定位误差。
        self.assertEqual(len(actual), len(expected))  # 先比较行数。
        for a, b in zip(actual, expected):  # 遍历对应行。
            self.assertEqual(len(a), len(b))  # 不让zip静默吞掉多余元素。
            for x, y in zip(a, b):  # 比较对应数值。
                self.assertAlmostEqual(x, y, places=places)  # double容差适用于这些小例子。
    def test_matmul_non_square_and_grad(self):  # 非方阵可发现转置和布局错误。
        a = Tensor([[1, 2, 3], [4, 5, 6]], True)  # A为2×3。
        b = Tensor([[1, 0], [0, 1], [1, 1]], True)  # B为3×2。
        c = a @ b  # 执行真实原生矩阵乘法。
        self.assert_matrix(c.tolist(), [[4, 5], [10, 11]])  # 独立手算结果。
        loss = Tensor([[1, 1]]) @ c @ Tensor([[1], [1]])  # 全部元素求和得到标量。
        loss.backward()  # 检查两条参数梯度路径。
        self.assert_matrix(a.grad, [[1, 1, 2], [1, 1, 2]])  # dA=G@B.T。
        self.assert_matrix(b.grad, [[5, 5], [7, 7], [9, 9]])  # dB=A.T@G。
    def test_shared_path_and_gather(self):  # 同一个节点与重复ID都必须累加梯度。
        a = Tensor([[2]], True)  # 简单标量作为求导核对。
        loss = a * a + a  # 三条路径影响同一个a。
        loss.backward()  # 合计梯度应为2+2+1。
        self.assertEqual(a.grad, [[5.0]])  # 检查累加而非覆盖。
        table = Parameter([[1, 2], [3, 4], [5, 6]])  # 3行Embedding表。
        out = table.gather([2, 0, 2])  # 第2行重复出现。
        scalar = Tensor([[1, 2, 3]]) @ out @ Tensor([[1], [1]])  # 给不同位置不同上游权重。
        scalar.backward()  # scatter-add回参数表。
        self.assert_matrix(table.grad, [[2, 2], [0, 0], [4, 4]])  # 第2行收到1+3的贡献。
    def test_stable_ce_and_causal_mask(self):  # 大间距与未来位置都要正确处理。
        logits = Tensor([[1000, -1000]], True)  # 目标概率会下溢为0，但稳定CE仍应有限。
        loss = F.cross_entropy(logits, [1])  # 直接从logits计算。
        self.assertAlmostEqual(loss.item(), 2000.0)  # 不能先log(0)。
        loss.backward()  # 梯度应保持有限。
        self.assert_matrix(logits.grad, [[1, -1]])  # 单行CE梯度为p-onehot。
        scores = Tensor([[0, 9], [0, 1]], True)  # 第一行的未来分数虽然大，也必须被遮住。
        weights = F.softmax(scores, causal=True)  # 使用下三角可见域。
        self.assertEqual(weights.tolist()[0], [1.0, 0.0])  # 对角线保留。
        objective = Tensor([[1, 2]]) @ weights @ Tensor([[1], [-3]])  # 使用非均匀上游权重。
        objective.backward()  # 未来位置也不应收到梯度。
        self.assertEqual(scores.grad[0][1], 0.0)  # 检查真正反向遮罩。
    def test_norm_and_gelu_gradient(self):  # 对复合非线性用中央差分独立核对。
        x = Tensor([[-1.0, 0.2, 1.0]], True)  # 选择不完全对称的输入。
        gamma, beta = Parameter([[1.1, 0.8, 1.2]]), Parameter([[0.1, 0.0, -0.1]])  # 覆盖非默认仿射参数。
        weights = Tensor([[1.0], [-2.0], [0.5]])  # 避免归一化输出求和恒为0。
        def objective():  # 每次扰动都重新前向，避免使用过期图。
            return F.gelu(F.layer_norm(x, gamma, beta)) @ weights  # 一行输出乘列向量成为标量。
        objective().backward()  # 先得到自动求导值。
        for parameter in (x, gamma, beta):  # 三个输入都需要验证。
            original, analytic = parameter.tolist(), parameter.grad  # 保存原始数据和梯度副本。
            for j in range(3):  # 小矩阵逐元素检查即可。
                plus, minus = [original[0][:]], [original[0][:]]  # 扰动副本不改原始快照。
                plus[0][j] += 1e-6; minus[0][j] -= 1e-6  # 对同一坐标使用中央差分。
                with no_grad():  # 有限差分只需要真实前向值。
                    parameter.set_data(plus); positive = objective().item()  # 正扰动损失。
                    parameter.set_data(minus); negative = objective().item()  # 负扰动损失。
                    parameter.set_data(original)  # 恢复参数以检查下一坐标。
                self.assertAlmostEqual(analytic[0][j], (positive - negative) / 2e-6, places=5)  # 比较独立数值导数。
    def test_model_causality_tying_and_checkpoint(self):  # 验证完整Transformer的关键约束。
        model = GPT(Config(4, width=3, hidden=12, context=3))  # 保留单Block但缩小检查规模。
        with no_grad():  # 因果性测试不需要图。
            before, after = model([0, 1, 2]).tolist(), model([0, 1, 3]).tolist()  # 只修改未来位置。
        self.assert_matrix(before[:2], after[:2])  # 前两个位置不应受到未来token影响。
        parameters = model.parameters()  # 获取唯一可训练参数。
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))  # 绑定表不能重复更新。
        self.assertEqual(sum(p.shape[0] * p.shape[1] for p in GPT(Config(4)).parameters()), 912)  # 默认参数计数独立核对。
        with tempfile.TemporaryDirectory() as folder:  # 文件测试不污染训练产物。
            path = Path(folder) / "model.json"  # 选择本用例的checkpoint。
            checkpoint.save(path, model, list("我爱猫。"))  # 保存真实参数。
            loaded, vocab, _ = checkpoint.load(path)  # 新模型对象重新建立绑定关系。
            with no_grad():  # 不保留检查图。
                self.assert_matrix(loaded([0, 1, 2]).tolist(), before, places=12)  # 重载后结果应一致。
            self.assertEqual(vocab, list("我爱猫。"))  # 词表顺序同样必须保存。
    def test_full_model_selected_gradient(self):  # 检查绑定E及投影通过完整模型的梯度。
        model = GPT(Config(4, width=3, hidden=12, context=2))  # 输入短且参数少。
        def objective():  # 每次都从同一文本构造新的前向。
            return F.cross_entropy(model([0, 1]), [1, 2])  # 两个位置损失取平均。
        objective().backward()  # 计算完整图的梯度。
        chosen = [model.token.weight, model.position.weight, model.block.attention.q.weight, model.block.mlp.up.weight, model.final_norm.gamma]  # 覆盖共享、位置、注意力、MLP、归一化。
        for parameter in chosen:  # 使用代表性坐标控制本机检查开销。
            original, expected = parameter.tolist(), parameter.grad[0][0]  # 读取该坐标自动梯度。
            plus, minus = [row[:] for row in original], [row[:] for row in original]  # 深复制矩阵。
            plus[0][0] += 1e-6; minus[0][0] -= 1e-6  # 只扰动一个参数元素。
            with no_grad():  # 差分检查只运行前向。
                parameter.set_data(plus); positive = objective().item()  # 绑定E会同时影响输入和输出路径。
                parameter.set_data(minus); negative = objective().item()  # 相同输入下的负扰动。
                parameter.set_data(original)  # 恢复后再处理下一参数。
            self.assertAlmostEqual(expected, (positive - negative) / 2e-6, places=5)  # 不把不匹配伪装为通过。
    def test_lifecycle_and_errors(self):  # 检查图释放、no_grad和错误边界。
        gc.collect(); baseline = B.lib.kt_live_nodes()  # 先释放此前测试可能留下的Python垃圾。
        def temporary():  # 独立作用域使全部Tensor可以离开。
            x = Tensor([[1, 2]], True)  # 创建一个需要梯度的叶子。
            with no_grad():  # 推理结果应不再连接图。
                self.assertFalse((x + x).requires_grad)  # 明确检查关闭求导的效果。
            with self.assertRaises(RuntimeError):  # 错误形状应变为Python异常。
                x @ x  # 1×2不能与1×2相乘。
            result = x @ x.T  # 构造需要缓存输入的合法图。
            x.set_data([[2, 3]])  # 在backward前修改输入。
            with self.assertRaises(RuntimeError):  # 旧图必须拒绝过期前向。
                result.backward()  # 不能静默算出混用新旧数据的梯度。
        temporary(); gc.collect()  # 图和句柄均离开后执行回收。
        self.assertEqual(B.lib.kt_live_nodes(), baseline)  # 共享引用没有形成环。
    def test_short_text_training(self):  # 用少量步数证明训练链路真实工作。
        vocab, ids = tokenize("我爱猫。" * 8)  # 固定小语料，不下载数据。
        model = GPT(Config(len(vocab)))  # 与示例使用同一种模型。
        report = train(model, ids, steps=25, lr=0.08, log_every=0)  # 短训练限制本机负担。
        self.assertLess(report["final_loss"], report["initial_loss"])  # 只检查同一窗口平均损失的实际改善。
        text = generate(model, vocab, "我", 6)  # 输出必须经过模型计算。
        self.assertEqual(len(text), 7)  # 包含1个prompt与6个新字符。
        self.assertTrue(set(text) <= set(vocab))  # 解码只使用已训练词表。
    def test_v2_adamw_restore_and_integrity(self):  # 检查中断恢复保留模型和优化器轨迹。
        text = "我爱猫。" * 8  # 固定中文语料，避免测试依赖下载或随机文件。
        vocab, _ = tokenize(text)  # 词表由训练文本稳定建立。
        first = GPT(Config(len(vocab)))  # 创建第一条训练轨迹。
        trainer = Trainer(first, vocab, text, AdamW(first.parameters(), lr=0.003), seed=13, batch_size=2)  # AdamW保存两组矩。
        trainer.fit(2, log_every=0)  # 只执行两步小训练。
        with tempfile.TemporaryDirectory() as folder:  # 检查点只存在于本测试目录。
            path = Path(folder) / "resume.json"  # schema2文件名。
            trainer.save(path)  # 保存模型、随机状态、矩和步数。
            restored = Trainer.restore(path, text)  # 新对象按文件恢复。
            self.assertEqual(restored.steps, 2)  # 优化器和Trainer步数应一致。
            self.assertEqual(restored.optimizer.__class__.__name__, "AdamW")  # 恢复正确的更新规则。
            restored.fit(1, log_every=0)  # 继续执行一小步，检查接口可用。
            raw = path.read_text(encoding="utf-8")  # 读取副本模拟文件损坏。
            path.write_text(raw.replace("sha256", "sha256-broken", 1), encoding="utf-8")  # 摘要字段变化应被拒绝。
            with self.assertRaises(ValueError):  # 不能静默接受被改动的schema2文件。
                checkpoint.load(path)  # 读取入口负责完整性校验。

if __name__ == "__main__":  # 也支持直接运行这个文件。
    unittest.main()  # 自动列出真实执行的测试数量。

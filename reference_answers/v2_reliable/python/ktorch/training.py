"""纯文本数据与训练循环；模型数值和反向调用C++。"""  # 本模块不使用NumPy或PyTorch。
import random  # 数据采样与参数初始化使用不同随机对象。
import hashlib  # 恢复训练时验证语料内容一致。
from .tensor import no_grad  # 评估和生成不记录计算图。
from . import functional as F  # 损失和概率由原生后端计算。
from .optim import SGD  # v1使用清楚的梯度下降。
from .optim import AdamW  # v2支持带矩状态的优化器。

def tokenize(text):  # Python字符串按Unicode码点处理中文。
    vocab = list(dict.fromkeys(text))  # 按首次出现顺序建立稳定词表。
    if len(vocab) < 2:  # 至少两个候选才有有意义的文本选择任务。
        raise ValueError("语料至少需要两个不同字符")  # 提前发现过小输入。
    lookup = {char: i for i, char in enumerate(vocab)}  # 建立字符到编号的映射。
    return vocab, [lookup[char] for char in text]  # 返回可保存词表与完整整数流。

def encode(text, vocab):  # 提示词必须复用已有词表。
    lookup = {char: i for i, char in enumerate(vocab)}  # 不在生成时自动扩充模型尺寸。
    unknown = sorted(set(text) - set(vocab))  # 一次指出所有未知字符。
    if unknown:  # 新字符没有训练好的参数行。
        raise ValueError(f"提示词含词表外字符：{unknown}")  # 让用户选择重新训练或调整输入。
    return [lookup[char] for char in text]  # 编码不会改变参数。

def evaluate(model, ids):  # 在固定窗口集合上比较训练前后损失。
    length = model.config.context  # 评估使用同一上下文长度。
    starts = list(range(0, len(ids) - length, length))[:8]  # 最多8个窗口，保持小规模与确定性。
    if not starts:  # 输入需要T+1个token才能得到右移标签。
        raise ValueError("语料长度必须大于context")  # 不用空集合冒充验证。
    with no_grad():  # 评估不会留下梯度图。
        losses = [F.cross_entropy(model(ids[s:s + length]), ids[s + 1:s + length + 1]).item() for s in starts]  # 同一组窗口上的真实平均交叉熵。
    return sum(losses) / len(losses)  # 各窗口等长，平均方式一致。

def train(model, ids, steps=100, lr=0.05, seed=7, log_every=25):  # 简单且可复现的首版训练入口。
    length = model.config.context  # 一次训练一个二维文本片段。
    if type(steps) is not int or steps <= 0 or len(ids) <= length:  # 明确限制空训练和过短语料。
        raise ValueError("steps须为正整数，语料长度须大于context")  # 失败时不会偷偷缩小窗口。
    sampler = random.Random(seed)  # 固定采样序列，便于检查变化。
    optimizer = SGD(model.parameters(), lr=lr, clip=1.0)  # 所有共享参数只更新一次。
    initial = evaluate(model, ids)  # 保存同一评估集合的起点。
    history = []  # 日志只保存普通float，避免保留每轮图。
    for step in range(1, steps + 1):  # 每轮重新运行前向并构建图。
        start = sampler.randrange(len(ids) - length)  # 保证右移标签仍在语料中。
        x, y = ids[start:start + length], ids[start + 1:start + length + 1]  # 输入与标签错开一个字符。
        optimizer.zero_grad()  # 清除上一轮参数梯度。
        loss = F.cross_entropy(model(x), y)  # 一次同时预测每个位置的下一字符。
        loss.backward()  # 原生反向拓扑遍历计算参数梯度。
        optimizer.step()  # 裁剪后执行SGD更新。
        history.append(loss.item())  # 只取数值，不把Tensor存入列表。
        if log_every and (step == 1 or step % log_every == 0 or step == steps):  # 控制打印频率。
            print(f"step={step:4d} train_loss={history[-1]:.6f}")  # 随机窗口的单步loss无需单调。
    return {"initial_loss": initial, "final_loss": evaluate(model, ids), "steps": steps, "history": history}  # 对照同一评估窗口得到结束值。

def generate(model, vocab, prompt, max_new_tokens=20):  # 首版采用确定性的greedy解码。
    ids = encode(prompt, vocab)  # 先检查未知字符。
    if not ids or type(max_new_tokens) is not int or max_new_tokens < 0:  # 空前缀没有最后位置可预测。
        raise ValueError("prompt不能为空，生成长度须为非负整数")  # 输入错误明确报出。
    with no_grad():  # 生成不求导，也不更新参数。
        for _ in range(max_new_tokens):  # 每次只接出一个新token。
            logits = model(ids[-model.config.context:]).tolist()[-1]  # 超长时截取最近窗口，位置从0重新编号。
            ids.append(max(range(len(logits)), key=logits.__getitem__))  # argmax logits等价于argmax概率。
    return "".join(vocab[i] for i in ids)  # 包含原始提示文本和新生成部分。

def tuples(value):  # JSON将随机状态元组变为列表，恢复时递归转换。
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value  # 标量与None保持原值。

class Trainer:  # 显式保存训练过程状态，使暂停/恢复可验证。
    def __init__(self, model, vocab, text, optimizer, seed=7, batch_size=1):  # batch采用多个二维片段的梯度累计。
        if type(batch_size) is not int or not 1 <= batch_size <= 64:  # 控制教学实验的微批次数量。
            raise ValueError("batch_size必须是1到64的整数")  # 不隐式增加Tensor rank。
        self.model, self.vocab, self.text = model, list(vocab), text  # 保存重现数据处理需要的内容。
        self.ids = encode(text, vocab)  # 语料必须属于已建立的词表。
        if len(self.ids) <= model.config.context:  # 每个窗口还要多一个下一字符标签。
            raise ValueError("语料长度必须大于context")  # 不改变训练任务来迁就错误配置。
        self.optimizer, self.batch_size = optimizer, batch_size  # 更新器与累计数量都是恢复状态的一部分。
        self.rng = random.Random(seed)  # 采样器拥有独立且可保存的随机状态。
        self.steps = 0  # 保存优化器步数而不是前向次数。
        self.data_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()  # 精确续训必须使用相同语料。
    def step(self):  # 一次优化器更新可包含多个二维训练片段。
        self.optimizer.zero_grad()  # 开始新的累计批次。
        total = 0.0  # 日志记录各片段未缩放loss的平均。
        length = self.model.config.context  # 每个微批的token长度固定。
        for _ in range(self.batch_size):  # 不依赖3D Tensor也能正确累计batch梯度。
            start = self.rng.randrange(len(self.ids) - length)  # 随机起点取自保存的采样器。
            x, y = self.ids[start:start + length], self.ids[start + 1:start + length + 1]  # 标签右移。
            loss = F.cross_entropy(self.model(x), y)  # 此loss已平均T个位置。
            (loss / self.batch_size).backward()  # 再除以B，使最终梯度对应B×T的平均损失。
            total += loss.item()  # 只保存数值，当前图可释放。
        self.optimizer.step()  # 全部微批累计后裁剪并更新一次。
        self.steps += 1  # 采样器与优化器步数一同前进。
        return total / self.batch_size  # 返回便于观察的平均训练损失。
    def fit(self, steps, log_every=25):  # 这里的steps表示继续执行多少步。
        if type(steps) is not int or steps <= 0:  # 恢复后也禁止空训练。
            raise ValueError("steps必须是正整数")  # 明确增量步数含义。
        initial = evaluate(self.model, self.ids)  # 开始这段训练时的固定窗口损失。
        history = []  # 仅保留float日志。
        for i in range(steps):  # 从当前状态继续，而不重置随机源。
            history.append(self.step())  # 执行一个真实更新。
            if log_every and (i == 0 or self.steps % log_every == 0 or i == steps - 1):  # 控制输出数量。
                print(f"step={self.steps:4d} train_loss={history[-1]:.6f}")  # 打印累计的优化器步数。
        return {"initial_loss": initial, "final_loss": evaluate(self.model, self.ids), "steps": self.steps, "history": history}  # 结束时重新评估同一窗口集合。
    def snapshot(self):  # 收集精确恢复需要的全部过程状态。
        return {"steps": self.steps, "batch_size": self.batch_size, "rng": self.rng.getstate(), "data_sha256": self.data_hash, "parameter_names": [name for name, _ in self.model.named_parameters()], "optimizer": self.optimizer.state_dict()}  # 不保存计算图。
    def save(self, path, metadata=None):  # 模型与训练状态作为同一个完整文件保存。
        from .checkpoint import save  # 延迟导入避免模块循环依赖。
        save(path, self.model, self.vocab, metadata, training=self.snapshot())  # 统一格式和原子写入。
    @classmethod  # 恢复返回新的Trainer，不部分污染已有实例。
    def restore(cls, path, text):  # 外部仍须提供实际训练语料。
        from .checkpoint import load  # 使用校验过的模型加载器。
        model, vocab, payload = load(path)  # 先验证格式与参数。
        state = payload.get("training")  # v1模型可以推理，但没有精确续训状态。
        if not isinstance(state, dict) or state.get("data_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():  # 语料变动会改变样本序列。
            raise ValueError("缺少训练状态或语料与checkpoint不一致")  # 拒绝伪装成精确恢复。
        names = [name for name, _ in model.named_parameters()]  # 当前模型参数的有序身份。
        if state.get("parameter_names") != names:  # 矩状态按同一参数顺序恢复。
            raise ValueError("参数顺序与训练状态不一致")  # 防止把Adam矩写给错误参数。
        kind = state["optimizer"].get("type")  # 根据保存格式选择优化器。
        if kind not in ("SGD", "AdamW"):  # 不猜测未知优化器的语义。
            raise ValueError("未知优化器")  # 提示不支持的训练状态。
        optimizer = (SGD if kind == "SGD" else AdamW)(model.parameters())  # 创建正确数量的状态缓冲。
        optimizer.load_state_dict(state["optimizer"])  # 校验并恢复矩、超参数和计数。
        trainer = cls(model, vocab, text, optimizer, batch_size=state["batch_size"])  # 重建数据与配置。
        if type(state["steps"]) is not int or state["steps"] < 0 or state["steps"] != optimizer.steps:  # 两份计数必须一致。
            raise ValueError("训练步数与优化器状态不一致")  # 发现损坏元信息。
        trainer.steps = state["steps"]  # 恢复外部日志步数。
        trainer.rng.setstate(tuples(state["rng"]))  # 下一次样本与未中断训练一致。
        return trainer  # 恢复完成后才交给调用者。

"""纯文本数据与训练循环；模型数值和反向调用C++。"""  # 本模块不使用NumPy或PyTorch。
import random  # 数据采样与参数初始化使用不同随机对象。
from .tensor import no_grad  # 评估和生成不记录计算图。
from . import functional as F  # 损失和概率由原生后端计算。
from .optim import SGD  # v1使用清楚的梯度下降。

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

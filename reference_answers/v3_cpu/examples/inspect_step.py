"""从一小段文本看完整前向与一次反向的真实数值。"""  # 这里不手填模型输出。
from _bootstrap import ROOT  # 初始化当前版本路径。
from ktorch import Config, GPT, functional as F  # 导入真实计算模块。
text = "我爱猫。"  # 只用四个字符使输出容易检查。
vocab = list(text)  # 给这个固定例子建立可读词表。
model = GPT(Config(len(vocab), width=3, context=3, hidden=12))  # 使用更小尺寸展示同一结构。
x, y = [0, 1, 2], [1, 2, 3]  # 输入与正确下一字符错开一位。
logits = model(x)  # 真正调用C++前向算子。
loss = F.cross_entropy(logits, y)  # 从分数得到平均交叉熵。
loss.backward()  # 通过计算图获得参数梯度。
print("词表：", vocab, "输入IDs：", x, "标签：", y)  # 先明确监督任务。
print("logits：", logits.tolist())  # 展示每个位置的候选分数。
print("概率：", F.softmax(logits).tolist())  # 第二种softmax是在词表维分配概率。
print("loss：", loss.item())  # 展示真实平均损失。
print("共享字符表梯度：", model.token.weight.grad)  # 这份梯度合并输入查表与输出投影路径。

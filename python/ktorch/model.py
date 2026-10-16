"""
任务 K08：规划单层、单头文本 GPT（当前仅为说明）。
计划 API：GPTConfig、CausalSelfAttention、MLP、Block、GPT。
输入：长度 T 不超过上下文长度的整数 token ID 序列。
输出：形状 [T, vocab_size] 的下一 token logits。
依赖：K07；先完成并验证底层 Tensor 与自动求导。
TODO：使用字符 Embedding 加可训练的位置 Embedding。
TODO：一个 Block：Pre-LN 注意力、残差、Pre-LN MLP、残差。
TODO：单头注意力计算 QK 转置、缩放、因果 softmax 和加权 V。
TODO：MLP 使用 Linear、GELU、Linear。
TODO：最终归一化后复用同一个字符 Embedding 参数作为输出权值。
建议起点：context=8、embedding=8、hidden=32、batch=1。
验收：未来 token 不改变较早位置 logits，形状逐步可解释。
进度：未实现；尚无可训练或可生成文本的模型。
"""

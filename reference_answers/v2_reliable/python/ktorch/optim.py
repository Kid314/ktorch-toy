"""优化器只组织已登记参数的更新，实际数值写回在C++完成。"""  # 参数去重避免绑定E更新两次。
import math  # 计算全局梯度范数并检查有限值。
from . import _backend as B  # 调用原生SGD和梯度清理。
from .tensor import Tensor, matrix  # 用无梯度Tensor保存原生优化器状态。

class SGD:  # 第一版先使用最容易理解的梯度下降。
    def __init__(self, parameters, lr=0.05, clip=1.0):  # clip为None时不裁剪。
        self.parameters = list({id(p): p for p in parameters}.values())  # 按对象身份去重并保留顺序。
        if not self.parameters or any(not p.requires_grad for p in self.parameters):  # 优化器需要真实可训练参数。
            raise ValueError("优化器需要非空可训练参数列表")  # 不静默接受空模型。
        if not math.isfinite(lr) or lr <= 0 or (clip is not None and (not math.isfinite(clip) or clip <= 0)):  # 检查更新超参数。
            raise ValueError("lr和clip必须为正有限数")  # 避免错误的更新方向或非法裁剪。
        self.lr, self.clip = float(lr), clip  # 保存训练超参数。
        self.steps = 0  # 保存已经执行的优化器步数。
    def zero_grad(self):  # 每个优化器步之前清除旧梯度。
        for parameter in self.parameters:  # 每个唯一参数只操作一次。
            parameter.zero_grad()  # 调用原生清零。
    def gradient_scale(self):  # 对全部参数一起计算全局裁剪系数。
        norm = 0.0  # 用hypot稳定累积，减少平方溢出风险。
        for parameter in self.parameters:  # 包含所有可训练参数。
            for row in parameter.grad:  # 梯度复制到Python用于诊断与统一裁剪。
                for value in row:  # 每个元素均参与范数。
                    if not math.isfinite(value):  # 在任何参数更新之前拒绝非法梯度。
                        raise FloatingPointError("梯度含NaN或Inf")  # 不把问题隐藏为零梯度。
                    norm = math.hypot(norm, value)  # 稳定更新L2范数。
        return min(1.0, self.clip / (norm + 1e-12)) if self.clip is not None else 1.0  # 保持所有梯度的相对方向。
    def step(self):  # backward之后执行一次参数更新。
        factor = self.gradient_scale()  # 先统一检查和计算缩放。
        for parameter in self.parameters:  # 逐参数调用原生更新。
            B.checked(B.lib.kt_sgd(parameter._handle, self.lr, factor))  # theta_new=theta-lr*factor*grad。
        self.steps += 1  # 整个优化器步完成后才增加计数。
    def state_dict(self):  # SGD仍需记录超参数和步数以便恢复。
        return {"type": "SGD", "lr": self.lr, "clip": self.clip, "steps": self.steps}  # 只保存普通数据。
    def load_state_dict(self, state):  # 先验证再替换训练状态。
        if not isinstance(state, dict) or state.get("type") != "SGD":  # 不把其他优化器状态塞进SGD。
            raise ValueError("需要SGD状态")  # 格式不一致时明确拒绝。
        self._load_common(state)  # 复用通用超参数验证。
    def _load_common(self, state):  # AdamW也使用相同的基础字段。
        lr, clip, steps = state["lr"], state["clip"], state["steps"]  # 先读取候选值。
        if not math.isfinite(lr) or lr <= 0 or (clip is not None and (not math.isfinite(clip) or clip <= 0)) or type(steps) is not int or steps < 0:  # 同时验证大小与类型。
            raise ValueError("无效优化器超参数或步数")  # 拒绝损坏检查点。
        self.lr, self.clip, self.steps = float(lr), clip, steps  # 验证全部通过后统一更新。

class AdamW(SGD):  # 复用参数去重、清梯度和全局裁剪逻辑。
    def __init__(self, parameters, lr=0.003, clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8, decay=0.01):  # 教学版对全部参数统一衰减。
        super().__init__(parameters, lr, clip)  # 基础参数列表和超参数先通过校验。
        self._validate(beta1, beta2, eps, decay)  # 再检查AdamW专属设置。
        self.beta1, self.beta2, self.eps, self.decay = beta1, beta2, eps, decay  # 保存当前配置。
        self.m = [Tensor([[0.0] * p.shape[1] for _ in range(p.shape[0])]) for p in self.parameters]  # 一阶矩无梯度。
        self.v = [Tensor([[0.0] * p.shape[1] for _ in range(p.shape[0])]) for p in self.parameters]  # 二阶矩也是独立的常量Tensor。
    @staticmethod  # 验证不依赖实例数据。
    def _validate(beta1, beta2, eps, decay):  # 负数或NaN都应失败。
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or not math.isfinite(eps) or eps <= 0 or not math.isfinite(decay) or decay < 0:  # 衰减允许为0。
            raise ValueError("无效AdamW超参数")  # 先报错再分配状态。
    def step(self):  # 真实AdamW数值更新在native/adam.cpp。
        factor = self.gradient_scale()  # 所有参数共享一个裁剪系数。
        step = self.steps + 1  # 偏差修正从第1步开始。
        for p, m, v in zip(self.parameters, self.m, self.v):  # 每个参数有两份独立矩状态。
            B.checked(B.lib.kt_adamw(p._handle, m._handle, v._handle, step, self.lr, self.beta1, self.beta2, self.eps, self.decay, factor))  # C++执行偏差修正和解耦衰减。
        self.steps = step  # 所有参数更新完成后保存步数。
    def state_dict(self):  # 矩、步数与超参数一起决定下一次更新。
        return {"type": "AdamW", "lr": self.lr, "clip": self.clip, "steps": self.steps, "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps, "decay": self.decay, "m": [x.tolist() for x in self.m], "v": [x.tolist() for x in self.v]}  # 不保存原生地址或计算图。
    def load_state_dict(self, state):  # 恢复同样的数值状态以继续训练。
        if not isinstance(state, dict) or state.get("type") != "AdamW":  # 优化器种类不能混用。
            raise ValueError("需要AdamW状态")  # 阻止静默丢失矩信息。
        self._validate(state["beta1"], state["beta2"], state["eps"], state["decay"])  # 先校验专属超参数。
        if len(state["m"]) != len(self.m) or len(state["v"]) != len(self.v):  # 每个参数都必须有状态。
            raise ValueError("优化器状态数量不匹配")  # 不允许只恢复部分参数。
        for key, buffers in (("m", self.m), ("v", self.v)):  # 先完整验证所有矩形状和数值。
            for tensor, values in zip(buffers, state[key]):  # 与参数顺序逐一对应。
                rows, cols, flat = matrix(values)  # 检查有限性和矩形布局。
                if (rows, cols) != tensor.shape or (key == "v" and any(x < 0 for x in flat)):  # 二阶矩不允许负数。
                    raise ValueError("优化器矩状态无效")  # 在写入前停止。
        self._load_common(state)  # 基础状态通过后再改变实例。
        self.beta1, self.beta2, self.eps, self.decay = (state[k] for k in ("beta1", "beta2", "eps", "decay"))  # 恢复同一更新公式。
        for key, buffers in (("m", self.m), ("v", self.v)):  # 写回已验证的矩。
            for tensor, values in zip(buffers, state[key]):  # 每个状态都是叶子Tensor。
                tensor.set_data(values)  # 用原生安全写入口。

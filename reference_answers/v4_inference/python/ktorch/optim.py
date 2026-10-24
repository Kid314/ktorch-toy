"""优化器只组织已登记参数的更新，实际数值写回在C++完成。"""  # 参数去重避免绑定E更新两次。
import math  # 计算全局梯度范数并检查有限值。
from . import _backend as B  # 调用原生SGD和梯度清理。

class SGD:  # 第一版先使用最容易理解的梯度下降。
    def __init__(self, parameters, lr=0.05, clip=1.0):  # clip为None时不裁剪。
        self.parameters = list({id(p): p for p in parameters}.values())  # 按对象身份去重并保留顺序。
        if not self.parameters or any(not p.requires_grad for p in self.parameters):  # 优化器需要真实可训练参数。
            raise ValueError("优化器需要非空可训练参数列表")  # 不静默接受空模型。
        if not math.isfinite(lr) or lr <= 0 or (clip is not None and (not math.isfinite(clip) or clip <= 0)):  # 检查更新超参数。
            raise ValueError("lr和clip必须为正有限数")  # 避免错误的更新方向或非法裁剪。
        self.lr, self.clip = float(lr), clip  # 保存训练超参数。
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

from __future__ import annotations

from collections.abc import Callable, Iterable
import math
import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    A1 §4.1：数值稳定的交叉熵损失（Cross-Entropy Loss）
    logits:  (..., vocab_size) —— 模型输出的未归一化打分
    targets: (...)             —— 真实下一个词的 token ID
    公式：loss_i = -log(softmax(o_i)[y_i]) = log(sum(exp(o_i - max))) - (o_i[y_i] - max)
    """
    # 1. 沿最后一维减去最大值，防止 exp 溢出
    max_val = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_val

    # 2. 消去 log 和 exp 后的简化计算：log(sum(exp(shifted))) - shifted[target]
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 3. 对整个 batch 所有位置取平均
    return (log_sum_exp - target_logits).mean()


class AdamW(torch.optim.Optimizer):
    """
    A1 §4.3：带解耦权重衰减的 AdamW 优化器
    对每个参数 p 维护：
      - m：梯度的一阶矩估计（类似“动量 / 平均方向”）
      - v：梯度平方的二阶矩估计（类似“梯度波动幅度”，用来给每个参数自适应缩放步长）
      - t：更新步数（从 1 开始，用于早期偏差校正）
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 首次更新时初始化该参数的 m, v, t
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # 1. 更新一阶矩 m（动量）和二阶矩 v（梯度平方滑动平均）
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 2. 偏差校正学习率 alpha_t
                alpha_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

                # 3. 按自适应步长更新参数，再做解耦权重衰减（把权重往 0 拉一点，防过拟合）
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    A1 §4.4：带线性预热（Warmup）的余弦退火学习率调度
    - 阶段 1 (it < warmup_iters)：线性从 0 爬升到 max_lr
    - 阶段 2 (warmup_iters <= it <= cosine_cycle_iters)：按余弦曲线平滑降到 min_lr
    - 阶段 3 (it > cosine_cycle_iters)：保持 min_lr
    """
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (
            max_learning_rate - min_learning_rate
        )
    return min_learning_rate


def clip_gradient(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    A1 §4.5：全局 L2 范数梯度裁剪（原地修改 p.grad）
    把所有参数的梯度看成一个超大向量，若总长度 > max_l2_norm，就按比例整体缩小到上限内。
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)

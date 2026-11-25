from __future__ import annotations

import os
from typing import BinaryIO, IO

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一维词元数组 dataset 中随机采样 batch_size 条长度为 context_length 的样本：
      - x: 输入序列，形状 (batch_size, context_length)
      - y: 目标序列，整体右移 1 位，形状 (batch_size, context_length)
    """
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=(batch_size,))
    offsets = np.arange(context_length)

    x_np = dataset[starts[:, None] + offsets]
    y_np = dataset[starts[:, None] + offsets + 1]

    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """将模型权重、优化器状态和当前迭代步数序列化保存到磁盘。"""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """从磁盘恢复模型与优化器状态，并返回之前保存的迭代步数。"""
    payload = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["iteration"])

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.data import get_batch, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, clip_gradient, cross_entropy, get_lr_cosine_schedule


def build_model(
    vocab_size: int = 4096,
    context_length: int = 256,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    d_ff: int = 768,
    device: str = "cuda",
) -> TransformerLM:
    """实例化一个小规模现代 TransformerLM（约 5.2M 参数，RTX 5060 上几分钟即可收敛）。"""
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=torch.device(device),
    )
    return model


def train_step(
    model: TransformerLM,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    max_grad_norm: float = 1.0,
) -> float:
    """
    第 1 块核心：一次完整的训练步（6 行闭环）。
    - 输入：x (B, S), y (B, S)
    - 输出：当前步标量 loss，并原地更新 model 参数
    """
    # 0. 按余弦调度更新当前步学习率
    for group in optimizer.param_groups:
        group["lr"] = lr

    # 1. 前向传播：(B, S) -> (B, S, V)
    logits = model(x)
    # 2. 计算交叉熵损失：标量
    loss = cross_entropy(logits, y)
    # 3. 清空上一步梯度，防止梯度累加失控
    optimizer.zero_grad()
    # 4. 反向传播，计算所有参数的梯度 p.grad
    loss.backward()
    # 5. 全局梯度裁剪，防止偶发大梯度把权重冲爆
    clip_gradient(model.parameters(), max_l2_norm=max_grad_norm)
    # 6. AdamW 按自适应步长 + 权重衰减更新参数
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate_loss(
    model: TransformerLM,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int = 20,
) -> float:
    """在验证集（或训练集）上抽 eval_batches 批算平均 Loss，不建计算图、不更新权重。"""
    model.eval()
    total_loss = 0.0
    for _ in range(eval_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        total_loss += float(cross_entropy(logits, y).item())
    model.train()
    return total_loss / eval_batches


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_overfit_check(
    train_data: npt.NDArray,
    out_file: Path,
    vocab_size: int = 4096,
    context_length: int = 128,
    batch_size: int = 16,
    steps: int = 200,
    lr: float = 3e-3,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """
    第 2 块（自检）：固定 1 个 batch 连训 200 步，验证初始 Loss ≈ ln(V) 且能过拟合到接近 0。
    """
    set_seed(seed)
    model = build_model(
        vocab_size=vocab_size,
        context_length=context_length,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    num_params = sum(p.numel() for p in model.parameters())

    # 只抽一次固定不变的 (x, y)
    x, y = get_batch(train_data, batch_size, context_length, device)
    with torch.no_grad():
        init_loss = float(cross_entropy(model(x), y).item())

    lines = [
        f"Device: {device} | Params: {num_params:,} | Vocab: {vocab_size}",
        f"Expected random init loss ln({vocab_size}) = {math.log(vocab_size):.4f}",
        f"Step    0 (init): loss = {init_loss:.4f}",
    ]
    print("\n".join(lines))

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        loss_val = train_step(model, optimizer, x, y, lr=lr)
        if step in (1, 10, 25, 50, 100, 150, steps):
            msg = f"Step {step:4d}: loss = {loss_val:.4f}"
            print(msg)
            lines.append(msg)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / steps
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.startswith("cuda") else 0.0
    )
    summary = f"Avg step time: {elapsed_ms:.2f} ms | Peak GPU mem: {peak_mb:.1f} MB"
    print(summary)
    lines.append(summary)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"自检日志已保存至: {out_file}")


def run_training(
    train_data: npt.NDArray,
    val_data: npt.NDArray,
    out_dir: Path,
    ckpt_dir: Path,
    run_name: str = "endpoint1",
    vocab_size: int = 4096,
    context_length: int = 256,
    batch_size: int = 32,
    max_steps: int = 1200,
    eval_interval: int = 100,
    max_lr: float = 1e-3,
    min_lr: float = 1e-4,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """
    第 2 块（正式训练）：完整训练主循环，定期评估 Val Loss、存 Checkpoint 并保存日志与曲线图。
    """
    set_seed(seed)
    model = build_model(
        vocab_size=vocab_size,
        context_length=context_length,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"开始训练 [{run_name}] | Params: {num_params:,} | Device: {device} | Steps: {max_steps}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 保存第 0 步未训练权重，方便后续对比“第 0 步 vs 中期 vs 末期”生成效果
    save_checkpoint(model, optimizer, 0, ckpt_dir / f"{run_name}_step0.pt")

    history = {"steps": [], "train_loss": [], "val_loss": []}
    mid_step = max_steps // 2

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    # 第 0 步基准评估
    tr0 = evaluate_loss(model, train_data, batch_size, context_length, device)
    va0 = evaluate_loss(model, val_data, batch_size, context_length, device)
    history["steps"].append(0)
    history["train_loss"].append(tr0)
    history["val_loss"].append(va0)
    print(f"Step    0 | train_loss: {tr0:.4f} | val_loss: {va0:.4f}")

    for step in range(1, max_steps + 1):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=max_steps,
        )
        x, y = get_batch(train_data, batch_size, context_length, device)
        train_step(model, optimizer, x, y, lr=lr)

        if step == mid_step:
            save_checkpoint(model, optimizer, step, ckpt_dir / f"{run_name}_mid.pt")

        if step % eval_interval == 0 or step == max_steps:
            tr_loss = evaluate_loss(model, train_data, batch_size, context_length, device)
            va_loss = evaluate_loss(model, val_data, batch_size, context_length, device)
            history["steps"].append(step)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            print(f"Step {step:4d} | lr: {lr:.5f} | train_loss: {tr_loss:.4f} | val_loss: {va_loss:.4f}")

    total_sec = time.perf_counter() - t_start
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.startswith("cuda") else 0.0
    )
    save_checkpoint(model, optimizer, max_steps, ckpt_dir / f"{run_name}_final.pt")

    metrics = {
        "run_name": run_name,
        "num_params": num_params,
        "total_sec": round(total_sec, 2),
        "avg_step_ms": round(total_sec * 1000.0 / max_steps, 2),
        "peak_gpu_mb": round(peak_mb, 1),
        "history": history,
    }
    (out_dir / f"{run_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # 绘制 Train / Val Loss 曲线图
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(history["steps"], history["train_loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(history["steps"], history["val_loss"], label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{run_name} Train vs Val Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / f"{run_name}_loss_curve.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"训练完成！耗时 {total_sec:.1f}s | 峰值显存 {peak_mb:.1f} MB | 曲线已保存至: {fig_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["overfit", "train"], default="overfit")
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="endpoint1")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    out_dir = repo_root / "outputs"
    ckpt_dir = repo_root / "checkpoints"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = np.load(data_dir / "train.npy")
    val_data = np.load(data_dir / "val.npy")

    if args.mode == "overfit":
        run_overfit_check(
            train_data=train_data,
            out_file=out_dir / "endpoint1_overfit_check.txt",
            device=device,
        )
    else:
        run_training(
            train_data=train_data,
            val_data=val_data,
            out_dir=out_dir,
            ckpt_dir=ckpt_dir,
            run_name=args.run_name,
            max_steps=args.max_steps,
            max_lr=args.max_lr,
            seed=args.seed,
            device=device,
        )


if __name__ == "__main__":
    main()

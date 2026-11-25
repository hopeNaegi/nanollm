from __future__ import annotations

import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.generate import generate_text
from cs336_basics.optim import AdamW, get_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import build_model, evaluate_loss, set_seed, train_step


def run_one_lr(
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer: Tokenizer,
    max_lr: float,
    max_steps: int = 600,
    eval_interval: int = 50,
    batch_size: int = 32,
    context_length: int = 256,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """
    第 1 块：严格锁死随机种子与总训练步数，单独跑一组 max_lr 并返回曲线与生成样例。
    """
    set_seed(seed)
    model = build_model(context_length=context_length, device=device)
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

    min_lr = max_lr * 0.1
    warmup_steps = max(1, max_steps // 10)
    hist = {"steps": [], "train_loss": [], "val_loss": []}
    diverged = False

    tr0 = evaluate_loss(model, train_data, batch_size, context_length, device)
    va0 = evaluate_loss(model, val_data, batch_size, context_length, device)
    hist["steps"].append(0)
    hist["train_loss"].append(tr0)
    hist["val_loss"].append(va0)
    print(f"[lr={max_lr:<7g}] Step   0 | train={tr0:.4f} | val={va0:.4f}", flush=True)

    for step in range(1, max_steps + 1):
        lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)
        x, y = get_batch(train_data, batch_size, context_length, device)
        step_loss = train_step(model, optimizer, x, y, lr=lr)

        if math.isnan(step_loss) or math.isinf(step_loss):
            print(f"[lr={max_lr:<7g}] Step {step:3d} | 发散 (NaN/Inf)，提前终止！", flush=True)
            diverged = True
            break

        if step % eval_interval == 0 or step == max_steps:
            tr = evaluate_loss(model, train_data, batch_size, context_length, device)
            va = evaluate_loss(model, val_data, batch_size, context_length, device)
            hist["steps"].append(step)
            hist["train_loss"].append(tr)
            hist["val_loss"].append(va)
            print(
                f"[lr={max_lr:<7g}] Step {step:3d} | lr_now={lr:.5f} | train={tr:.4f} | val={va:.4f}",
                flush=True,
            )

    torch.manual_seed(seed)
    sample = (
        "[发散无法生成]"
        if diverged
        else generate_text(model, tokenizer, "Once upon a time", max_new_tokens=60, temperature=0.8, device=device)
    )
    print(f"[lr={max_lr:<7g}] 生成样例: {sample}\n", flush=True)

    return {
        "max_lr": max_lr,
        "diverged": diverged,
        "final_val_loss": None if diverged else hist["val_loss"][-1],
        "history": hist,
        "sample": sample,
    }


def run_lr_sweep(
    lrs: list[float],
    max_steps: int = 500,
    eval_interval: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    第 2 块：固定总步数与随机种子，顺序跑完多档 max_lr，
    输出同框曲线图、汇总 JSON 与生成样例对比。
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = np.load(data_dir / "train.npy")
    val_data = np.load(data_dir / "val.npy")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(data_dir / "vocab.json"),
        merges_filepath=str(data_dir / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )

    total_tokens = max_steps * 32 * 256
    print(f"=== 开始单变量对照实验：Learning Rate Sweep ===")
    print(f"锁死条件: seed={seed} | steps={max_steps} | tokens={total_tokens:,} | device={device}\n")

    results: list[dict] = []
    for lr in lrs:
        res = run_one_lr(
            train_data=train_data,
            val_data=val_data,
            tokenizer=tokenizer,
            max_lr=lr,
            max_steps=max_steps,
            eval_interval=eval_interval,
            seed=seed,
            device=device,
        )
        results.append(res)

    # 1. 保存汇总指标 JSON
    (out_dir / "endpoint2_lr_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2. 保存多组生成文本对比
    sample_blocks = [f"Locked: steps={max_steps}, tokens={total_tokens:,}, seed={seed}\n"]
    for res in results:
        val_str = "DIVERGED" if res["diverged"] else f"{res['final_val_loss']:.4f}"
        sample_blocks.append(f"=== max_lr = {res['max_lr']} (final val_loss = {val_str}) ===\n{res['sample']}\n")
    (out_dir / "endpoint2_lr_samples.txt").write_text("\n".join(sample_blocks), encoding="utf-8")

    # 3. 绘制 4 档学习率同框 Val Loss 对比图（左：Val Loss，右：Train Loss）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for res in results:
        label = f"lr={res['max_lr']}" + (" (diverged)" if res["diverged"] else "")
        h = res["history"]
        axes[0].plot(h["steps"], h["val_loss"], marker="o", markersize=3, label=label)
        axes[1].plot(h["steps"], h["train_loss"], marker="o", markersize=3, label=label)

    axes[0].set_title("Validation Loss vs Step")
    axes[1].set_title("Training Loss vs Step")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_ylim(1.5, 9.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(f"Endpoint 2: Learning Rate Controlled Comparison ({max_steps} steps, {total_tokens:,} tokens)")
    plt.tight_layout()
    fig_path = out_dir / "endpoint2_lr_sweep.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # 4. 终端打印最终成绩汇总表
    print("=== 最终汇总表 ===")
    for res in results:
        val_str = "发散 (NaN/Inf)" if res["diverged"] else f"{res['final_val_loss']:.4f}"
        print(f"  max_lr = {res['max_lr']:<7g} -> final val_loss = {val_str}")
    print(f"\n对比图已保存至: {fig_path}")
    print(f"生成对比已保存至: {out_dir / 'endpoint2_lr_samples.txt'}")
    return results


if __name__ == "__main__":
    run_lr_sweep(lrs=[1e-4, 1e-3, 5e-3, 2e-2], max_steps=500, eval_interval=50, seed=42)

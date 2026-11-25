from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cs336_basics.run_lr_sweep import run_one_lr
from cs336_basics.tokenizer import Tokenizer


def main() -> None:
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

    target_lrs = [1e-3, 5e-3]
    new_seeds = [123, 777]
    max_steps = 500

    # 先读取终点 2 已跑过的 seed=42 结果，避免重复算力浪费
    by_lr: dict[float, dict[int, float]] = {lr: {} for lr in target_lrs}
    prev_path = out_dir / "endpoint2_lr_results.json"
    if prev_path.exists():
        for item in json.loads(prev_path.read_text(encoding="utf-8")):
            lr_val = float(item["max_lr"])
            if lr_val in by_lr and item["final_val_loss"] is not None:
                by_lr[lr_val][42] = float(item["final_val_loss"])

    print("=== 步骤 3.2：终点 2 结论的多随机种子复核（seed=42 复用 + 补跑 123, 777）===\n")
    for lr in target_lrs:
        for seed in new_seeds:
            print(f"--- Running max_lr={lr}, seed={seed} ---")
            res = run_one_lr(
                train_data=train_data,
                val_data=val_data,
                tokenizer=tokenizer,
                max_lr=lr,
                max_steps=max_steps,
                eval_interval=100,
                seed=seed,
                device=device,
            )
            by_lr[lr][seed] = float(res["final_val_loss"])

    summary = {}
    print("=== 多种子稳定性汇总表 (500 steps) ===")
    for lr in target_lrs:
        seeds_sorted = sorted(by_lr[lr].keys())
        vals = [by_lr[lr][s] for s in seeds_sorted]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[str(lr)] = {
            "seeds": {str(s): round(by_lr[lr][s], 4) for s in seeds_sorted},
            "mean": round(mean, 4),
            "std": round(std, 4),
        }
        detail = ", ".join(f"seed {s}={by_lr[lr][s]:.4f}" for s in seeds_sorted)
        print(f"  max_lr={lr:<6g} | mean = {mean:.4f} ± {std:.4f}  ({detail})")

    (out_dir / "endpoint3_seed_check.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # 画均值 ± 标准差柱状图 + 每个种子散点
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [f"lr={lr}" for lr in target_lrs]
    means = [summary[str(lr)]["mean"] for lr in target_lrs]
    stds = [summary[str(lr)]["std"] for lr in target_lrs]
    x_pos = np.arange(len(target_lrs))

    ax.bar(x_pos, means, yerr=stds, capsize=8, width=0.45, color=["tab:orange", "tab:green"], alpha=0.8)
    for i, lr in enumerate(target_lrs):
        pts = list(by_lr[lr].values())
        ax.scatter([i] * len(pts), pts, color="black", zorder=5, s=35, label="Single seed" if i == 0 else None)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final Validation Loss (500 steps)")
    ax.set_title("Endpoint 3: Multi-Seed Stability Check (seeds = 42, 123, 777)")
    ax.set_ylim(2.2, 3.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig_path = out_dir / "endpoint3_seed_check.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n稳定性对比图已保存至: {fig_path}")


if __name__ == "__main__":
    main()

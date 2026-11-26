from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs"

    base = json.loads((out_dir / "endpoint4_eval_base.json").read_text(encoding="utf-8"))
    sft = json.loads((out_dir / "endpoint4_eval_sft.json").read_text(encoding="utf-8"))

    both_right, both_wrong, base_only, sft_only = [], [], [], []
    for b, s in zip(base["math_records"], sft["math_records"]):
        if b["correct"] and s["correct"]:
            both_right.append((b, s))
        elif (not b["correct"]) and (not s["correct"]):
            both_wrong.append((b, s))
        elif b["correct"] and (not s["correct"]):
            base_only.append((b, s))
        else:
            sft_only.append((b, s))

    print("=== GSM8K 100 题逐题翻转统计 (Base vs LoRA SFT) ===")
    print(f"  都做对 (Right -> Right): {len(both_right)}")
    print(f"  SFT 救回 (Wrong -> Right): {len(sft_only)}")
    print(f"  SFT 改错 (Right -> Wrong): {len(base_only)}")
    print(f"  都做错 (Wrong -> Wrong): {len(both_wrong)}")

    # 1. 绘制前后对比图（左：准确率 & 格式率对比，右：逐题翻转拆解）
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    metrics = ["Exact Match (%)", "#### Format Rate (%)"]
    base_vals = [base["accuracy_pct"], base["format_rate_pct"]]
    sft_vals = [sft["accuracy_pct"], sft["format_rate_pct"]]
    x = np.arange(len(metrics))
    w = 0.34

    b1 = axes[0].bar(x - w / 2, base_vals, w, label="Base (Before SFT)", color="tab:gray")
    b2 = axes[0].bar(x + w / 2, sft_vals, w, label="LoRA SFT (After SFT)", color="tab:purple")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 115)
    axes[0].set_title("GSM8K Independent Test (n=100): Before vs After SFT")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    cats = ["Both Right", "Wrong -> Right\n(Improved)", "Right -> Wrong\n(Regressed)", "Both Wrong"]
    counts = [len(both_right), len(sft_only), len(base_only), len(both_wrong)]
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
    b3 = axes[1].bar(cats, counts, color=colors, alpha=0.85, width=0.55)
    for bar in b3:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1.0, str(int(h)), ha="center", fontsize=10)
    axes[1].set_ylim(0, max(counts) + 12)
    axes[1].set_ylabel("Number of Questions")
    axes[1].set_title("Per-Question Outcome Shift (n=100)")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Endpoint 4: Qwen2.5-0.5B LoRA SFT Evaluation & Regression Analysis")
    plt.tight_layout()
    fig_path = out_dir / "endpoint4_sft_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # 2. 导出代表性 Bad Cases（含数学回归 Right->Wrong、持续算错 Wrong->Wrong、以及通用常识退化）
    lines = [
        f"Base Acc: {base['accuracy_pct']}% (fmt {base['format_rate_pct']}%) | "
        f"SFT Acc: {sft['accuracy_pct']}% (fmt {sft['format_rate_pct']}%)",
        f"Shift: BothRight={len(both_right)}, Improved(W->R)={len(sft_only)}, "
        f"Regressed(R->W)={len(base_only)}, BothWrong={len(both_wrong)}\n",
        "=== 典型数学退化案例 (Base 对 -> SFT 错) ===",
    ]
    for b, s in base_only[:3]:
        lines.append(
            f"[Q#{b['idx']}] GT={b['gt']} | Base Pred={b['pred']} (OK) | SFT Pred={s['pred']} (WRONG)\n"
            f"Question: {b['question']}\n"
            f"--- Base Output ---\n{b['completion']}\n"
            f"--- SFT Output ---\n{s['completion']}\n"
        )

    lines.append("=== 典型数学推理失败案例 (Base 错 -> SFT 仍错，但格式已遵守) ===")
    for b, s in both_wrong[:3]:
        lines.append(
            f"[Q#{b['idx']}] GT={b['gt']} | SFT Pred={s['pred']} (Format={s['has_format']})\n"
            f"Question: {b['question']}\n"
            f"--- SFT Output ---\n{s['completion']}\n"
        )

    lines.append("=== 通用常识退化案例 (Before vs After SFT) ===")
    for gb, gs in zip(base["general_checks"], sft["general_checks"]):
        lines.append(
            f"{gb['prompt'].splitlines()[0]}\n"
            f"  [Base]: {gb['response']}\n"
            f"  [SFT ]: {gs['response']}\n"
        )

    bad_path = out_dir / "endpoint4_bad_cases.txt"
    bad_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"对比图已保存至: {fig_path}\n案例归因源文件已保存至: {bad_path}")


if __name__ == "__main__":
    main()

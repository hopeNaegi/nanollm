from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.generate import generate_text
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, get_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import evaluate_loss, set_seed, train_step


def run_single_variant(
    train_data: np.ndarray,
    val_data: np.ndarray,
    use_causal_mask: bool,
    max_steps: int = 400,
    eval_interval: int = 50,
    batch_size: int = 32,
    context_length: int = 256,
    max_lr: float = 1e-3,
    min_lr: float = 1e-4,
    warmup_steps: int = 40,
    seed: int = 42,
    device: str = "cuda",
) -> tuple[TransformerLM, dict]:
    set_seed(seed)
    model = TransformerLM(
        vocab_size=4096,
        context_length=context_length,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=768,
        use_causal_mask=use_causal_mask,
        device=torch.device(device),
    )
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    hist = {"steps": [], "train_loss": [], "val_loss": []}

    label = "causal_mask=True" if use_causal_mask else "causal_mask=False"
    tr0 = evaluate_loss(model, train_data, batch_size, context_length, device)
    va0 = evaluate_loss(model, val_data, batch_size, context_length, device)
    hist["steps"].append(0)
    hist["train_loss"].append(tr0)
    hist["val_loss"].append(va0)
    print(f"[{label}] Step   0 | train={tr0:.4f} | val={va0:.4f}")

    for step in range(1, max_steps + 1):
        lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)
        x, y = get_batch(train_data, batch_size, context_length, device)
        train_step(model, optimizer, x, y, lr=lr)

        if step % eval_interval == 0 or step == max_steps:
            tr = evaluate_loss(model, train_data, batch_size, context_length, device)
            va = evaluate_loss(model, val_data, batch_size, context_length, device)
            hist["steps"].append(step)
            hist["train_loss"].append(tr)
            hist["val_loss"].append(va)
            print(f"[{label}] Step {step:3d} | train={tr:.4f} | val={va:.4f}")

    return model, hist


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

    m_causal, h_causal = run_single_variant(train_data, val_data, use_causal_mask=True, device=device)
    m_nomask, h_nomask = run_single_variant(train_data, val_data, use_causal_mask=False, device=device)

    # 画同框对比图
    plt.figure(figsize=(8, 4.8))
    plt.plot(h_causal["steps"], h_causal["train_loss"], label="With Mask - Train", linestyle="--", color="tab:blue")
    plt.plot(h_causal["steps"], h_causal["val_loss"], label="With Mask - Val", marker="o", markersize=4, color="tab:blue")
    plt.plot(h_nomask["steps"], h_nomask["train_loss"], label="No Mask - Train", linestyle="--", color="tab:red")
    plt.plot(h_nomask["steps"], h_nomask["val_loss"], label="No Mask - Val", marker="s", markersize=4, color="tab:red")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Endpoint 1 Ablation: Causal Mask ON vs OFF (400 steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "endpoint1_ablation.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # 对比自回归生成效果
    prompt = "Once upon a time"
    torch.manual_seed(42)
    gen_causal = generate_text(m_causal, tokenizer, prompt, max_new_tokens=80, temperature=0.8, device=device)
    torch.manual_seed(42)
    gen_nomask = generate_text(m_nomask, tokenizer, prompt, max_new_tokens=80, temperature=0.8, device=device)

    report = (
        f"Prompt: {prompt!r}\n\n"
        f"=== With Causal Mask (Step 400 | train={h_causal['train_loss'][-1]:.4f}, val={h_causal['val_loss'][-1]:.4f}) ===\n"
        f"{gen_causal}\n\n"
        f"=== Without Causal Mask (Step 400 | train={h_nomask['train_loss'][-1]:.4f}, val={h_nomask['val_loss'][-1]:.4f}) ===\n"
        f"{gen_nomask}\n"
    )
    (out_dir / "endpoint1_ablation_samples.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"对照图已保存至: {fig_path}")


if __name__ == "__main__":
    main()

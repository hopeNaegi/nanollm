from __future__ import annotations

import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optim import cross_entropy


def measure_one(
    seq_len: int,
    use_flash: bool,
    batch_size: int = 8,
    vocab_size: int = 4096,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    d_ff: int = 768,
    warmup: int = 3,
    iters: int = 10,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """测量给定序列长度 S 下，前向+反向传播的单步耗时 (ms) 与 GPU 峰值显存 (MB)。"""
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        use_flash=use_flash,
        device=device,
        dtype=dtype,
    )
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    try:
        # 预热：触发 CUDA 核编译与显存池稳定
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            loss = cross_entropy(model(x), y)
            loss.backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            loss = cross_entropy(model(x), y)
            loss.backward()
        torch.cuda.synchronize()

        avg_ms = (time.perf_counter() - t0) * 1000.0 / iters
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        return {"oom": False, "avg_ms": round(avg_ms, 2), "peak_mb": round(peak_mb, 1)}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True, "avg_ms": None, "peak_mb": None}
    finally:
        del model, x, y
        torch.cuda.empty_cache()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = [128, 256, 512, 1024, 2048]
    batch_size = 8
    num_layers = 4
    num_heads = 4
    elem_bytes = 2  # bfloat16 每个数占 2 字节

    print(f"=== 步骤 3.1：标准 Attention vs FlashAttention 实测 (RTX 5060, bf16, B={batch_size}) ===")
    print(
        f"{'Seq S':>6} | {'理论 S^2 矩阵(MB)':>16} | {'标准显存(MB)':>12} | "
        f"{'Flash显存(MB)':>13} | {'标准耗时(ms)':>12} | {'Flash耗时(ms)':>13}"
    )
    print("-" * 88)

    rows: list[dict] = []
    for s in seq_lens:
        # 理论估算：仅标准 Attention 每层存 scores + attn_weights 两个 (B, H, S, S) 张量（不含其它中间量）
        theo_s2_mb = (2 * num_layers * batch_size * num_heads * s * s * elem_bytes) / (1024**2)
        std_res = measure_one(seq_len=s, use_flash=False, batch_size=batch_size)
        fls_res = measure_one(seq_len=s, use_flash=True, batch_size=batch_size)

        std_mem_str = "OOM" if std_res["oom"] else f"{std_res['peak_mb']:.1f}"
        fls_mem_str = "OOM" if fls_res["oom"] else f"{fls_res['peak_mb']:.1f}"
        std_ms_str = "OOM" if std_res["oom"] else f"{std_res['avg_ms']:.2f}"
        fls_ms_str = "OOM" if fls_res["oom"] else f"{fls_res['avg_ms']:.2f}"

        print(
            f"{s:6d} | {theo_s2_mb:16.1f} | {std_mem_str:>12} | "
            f"{fls_mem_str:>13} | {std_ms_str:>12} | {fls_ms_str:>13}",
            flush=True,
        )
        rows.append(
            {
                "seq_len": s,
                "theo_s2_mb": round(theo_s2_mb, 1),
                "standard": std_res,
                "flash": fls_res,
            }
        )

    (out_dir / "endpoint3_attn_benchmark.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    # 画左右两张对比图：左 = 峰值显存 vs S，右 = 单步耗时 vs S
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    s_std = [r["seq_len"] for r in rows if not r["standard"]["oom"]]
    m_std = [r["standard"]["peak_mb"] for r in rows if not r["standard"]["oom"]]
    t_std = [r["standard"]["avg_ms"] for r in rows if not r["standard"]["oom"]]

    s_fls = [r["seq_len"] for r in rows if not r["flash"]["oom"]]
    m_fls = [r["flash"]["peak_mb"] for r in rows if not r["flash"]["oom"]]
    t_fls = [r["flash"]["avg_ms"] for r in rows if not r["flash"]["oom"]]

    axes[0].plot(s_std, m_std, marker="o", label="Standard Attention (O(S^2))", color="tab:red")
    axes[0].plot(s_fls, m_fls, marker="s", label="FlashAttention (O(S))", color="tab:blue")
    axes[0].set_title("Peak GPU Memory vs Context Length")
    axes[0].set_xlabel("Context Length S")
    axes[0].set_ylabel("Peak Memory (MB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(s_std, t_std, marker="o", label="Standard Attention", color="tab:red")
    axes[1].plot(s_fls, t_fls, marker="s", label="FlashAttention", color="tab:blue")
    axes[1].set_title("Forward + Backward Step Time vs Context Length")
    axes[1].set_xlabel("Context Length S")
    axes[1].set_ylabel("Step Time (ms)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(f"Endpoint 3: Standard vs FlashAttention (B={batch_size}, bf16, RTX 5060)")
    plt.tight_layout()
    fig_path = out_dir / "endpoint3_attn_benchmark.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存至: {fig_path}")


if __name__ == "__main__":
    main()

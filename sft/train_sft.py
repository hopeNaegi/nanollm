from __future__ import annotations

import json
import math
from pathlib import Path
import time

from datasets import load_dataset
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"


class GSM8KSFTDataset(Dataset):
    """
    SFT 核心数据构造：
      - input_ids = prompt_ids + answer_ids + [eos_id]
      - labels    = [-100] * len(prompt_ids) + answer_ids + [eos_id]
    只在回答 + <EOS> 位置计算交叉熵 Loss，题目部分用 -100 屏蔽。
    """

    def __init__(self, split_ds, tokenizer: AutoTokenizer, max_len: int = 512):
        self.samples: list[dict[str, list[int]]] = []
        eos_id = tokenizer.eos_token_id
        for row in split_ds:
            prompt = f"Question: {row['question'].strip()}\nAnswer:\n"
            answer = row["answer"].strip()

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [eos_id]

            input_ids = (prompt_ids + answer_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
            # 跳过极少数截断后没有有效回答 token 的样本
            if any(lbl != -100 for lbl in labels):
                self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


def make_collate_fn(pad_id: int):
    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_l = max(len(x["input_ids"]) for x in batch)
        input_ids, attn_mask, labels = [], [], []
        for x in batch:
            pad_n = max_l - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_n)
            attn_mask.append([1] * len(x["input_ids"]) + [0] * pad_n)
            labels.append(x["labels"] + [-100] * pad_n)  # padding 位同样标 -100 忽略
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def get_lr(step: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    warmup = max(1, total_steps // 10)
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def main() -> None:
    torch.manual_seed(42)
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs"
    save_dir = repo_root / "checkpoints" / "qwen05b_gsm8k_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载分词器与严格隔离的 GSM8K 训练集前 1500 题（绝不碰 test 集）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = load_dataset("openai/gsm8k", "main", split="train[:1500]")
    dataset = GSM8KSFTDataset(train_raw, tokenizer, max_len=512)
    print(f"SFT 训练集样本数: {len(dataset)} (仅来自 GSM8K train，与 test 零重叠)", flush=True)

    micro_bs = 4
    grad_accum = 4  # 等效 Batch Size = 4 * 4 = 16
    epochs = 2
    max_lr = 2e-4
    min_lr = 2e-5

    loader = DataLoader(
        dataset,
        batch_size=micro_bs,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )

    # 2. 加载基座模型并挂上 LoRA 小补丁（冻住原 4.94 亿参数，仅训练少量 LoRA 参数）
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    total_opt_steps = (len(loader) * epochs) // grad_accum
    print(f"开始 LoRA SFT | epochs={epochs} | 有效 BS={micro_bs * grad_accum} | 总更新步数={total_opt_steps}\n", flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    opt_step = 0
    accum_loss = 0.0
    micro_count = 0
    history = {"steps": [], "loss": []}
    optimizer.zero_grad()

    for epoch in range(1, epochs + 1):
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # 前向计算：内部自动右移 1 位并忽略 labels == -100 的位置
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            accum_loss += float(loss.item())
            micro_count += 1

            if micro_count % grad_accum == 0:
                lr = get_lr(opt_step, total_opt_steps, max_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                opt_step += 1
                history["steps"].append(opt_step)
                history["loss"].append(round(accum_loss, 4))
                if opt_step == 1 or opt_step % 15 == 0 or opt_step == total_opt_steps:
                    print(
                        f"Epoch {epoch} | Step {opt_step:3d}/{total_opt_steps} | "
                        f"lr={lr:.6f} | sft_loss={accum_loss:.4f}",
                        flush=True,
                    )
                accum_loss = 0.0

    total_sec = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # 3. 仅保存轻量的 LoRA 补丁权重（约 35 MB）
    model.save_pretrained(save_dir)
    print(f"\nSFT 完成！耗时 {total_sec:.1f}s | 峰值显存 {peak_mb:.1f} MB | LoRA 权重已存至: {save_dir}")

    meta = {
        "model_id": MODEL_ID,
        "train_samples": len(dataset),
        "epochs": epochs,
        "effective_batch_size": micro_bs * grad_accum,
        "total_opt_steps": opt_step,
        "total_sec": round(total_sec, 1),
        "peak_gpu_mb": round(peak_mb, 1),
        "history": history,
    }
    (out_dir / "endpoint4_sft_train.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 4. 绘制 SFT 训练 Loss 曲线
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(history["steps"], history["loss"], color="tab:purple", alpha=0.85, label="SFT Answer-Only Loss")
    plt.xlabel("Optimizer Step")
    plt.ylabel("Cross-Entropy Loss (Prompt Masked = -100)")
    plt.title("Endpoint 4: Qwen2.5-0.5B LoRA SFT on GSM8K (1500 train samples)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "endpoint4_sft_loss.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"SFT 曲线已保存至: {fig_path}")


if __name__ == "__main__":
    main()

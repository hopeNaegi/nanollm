from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from datasets import load_dataset
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"

GENERAL_PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: Explain what photosynthesis is in one sentence.\nAnswer:",
    "Question: Name three primary colors.\nAnswer:",
    "Question: Who wrote the play Romeo and Juliet?\nAnswer:",
    "Question: Why does ice float on water? Answer in one sentence.\nAnswer:",
    "Question: List three continents on Earth.\nAnswer:",
]


def format_math_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:\n"


def normalize_number(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = raw.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.6g}"
    except ValueError:
        return s


def extract_gt_answer(answer_text: str) -> str | None:
    m = re.search(r"####\s*([-+]?[\d,\.]+)", answer_text)
    return normalize_number(m.group(1)) if m else None


def extract_pred_answer(completion: str) -> tuple[str | None, bool]:
    """
    先截断在可能生成的下一个 'Question:' 之前；
    优先提取 GSM8K 标准格式 '#### <数字>'，否则兜底提取回答中最后一个数字。
    返回: (标准化数字字符串, 是否遵守了 #### 格式)
    """
    first_ans = completion.split("\nQuestion:")[0]
    m = re.search(r"####\s*([-+]?\$?[\d,\.]+)", first_ans)
    if m:
        return normalize_number(m.group(1)), True

    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", first_ans)
    if nums:
        return normalize_number(nums[-1]), False
    return None, False


def inspect_model_architecture(model: AutoModelForCausalLM) -> None:
    """对照查看真实开源现代模型 Qwen2.5-0.5B 的关键结构与 GQA 维度。"""
    cfg = model.config
    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    print("=== 现代模型真实结构速览 (Qwen2.5-0.5B) ===", flush=True)
    print(
        f"  Layers={cfg.num_hidden_layers} | d_model={cfg.hidden_size} | d_ff={cfg.intermediate_size} | "
        f"Vocab={cfg.vocab_size} | Params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    print(
        f"  GQA 配置: Q heads={cfg.num_attention_heads} vs KV heads={cfg.num_key_value_heads} "
        f"(每 {cfg.num_attention_heads // cfg.num_key_value_heads} 个 Q 头共享 1 组 KV，推理 KV Cache 显存降为 1/{cfg.num_attention_heads // cfg.num_key_value_heads})",
        flush=True,
    )
    print(
        f"  Layer 0 投影权重: q_proj={tuple(attn.q_proj.weight.shape)}, "
        f"k_proj={tuple(attn.k_proj.weight.shape)}, v_proj={tuple(attn.v_proj.weight.shape)}",
        flush=True,
    )
    print("  归一化与前馈: RMSNorm + RoPE + SwiGLU (与手写 TransformerLM 架构同源，额外引入 GQA 与 KV Cache)\n", flush=True)


@torch.no_grad()
def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int = 16,
    max_new_tokens: int = 256,
) -> list[str]:
    outputs_text: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs_text.extend(decoded)
        print(f"  已完成 {min(i + batch_size, len(prompts))}/{len(prompts)} 题...", flush=True)
    return outputs_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="base", help="评估标签，例如 base 或 sft")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 权重目录（不传则评估原始 Base 模型）")
    parser.add_argument("--num_samples", type=int, default=100, help="评测独立 test 集题数")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    if args.lora_path:
        print(f"加载 LoRA 适配器: {args.lora_path}", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        inspect_model_architecture(model)
    model.eval()

    # 1. 主任务独立评估：严格使用 GSM8K test 集前 num_samples 题
    test_ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.num_samples}]")
    prompts = [format_math_prompt(row["question"]) for row in test_ds]
    print(f"=== 开始评估 [{args.tag}]：独立 GSM8K test 集 ({len(prompts)} 题，贪心解码) ===", flush=True)
    completions = batch_generate(model, tokenizer, prompts, batch_size=args.batch_size, max_new_tokens=256)

    correct = 0
    fmt_ok = 0
    records = []
    for idx, (row, comp) in enumerate(zip(test_ds, completions)):
        gt = extract_gt_answer(row["answer"])
        pred, has_fmt = extract_pred_answer(comp)
        is_right = (pred is not None) and (pred == gt)
        correct += int(is_right)
        fmt_ok += int(has_fmt)
        records.append(
            {
                "idx": idx,
                "question": row["question"],
                "gt": gt,
                "pred": pred,
                "has_format": has_fmt,
                "correct": is_right,
                "completion": comp.split("\nQuestion:")[0].strip(),
            }
        )

    acc = correct / len(records) * 100.0
    fmt_rate = fmt_ok / len(records) * 100.0
    print(
        f"\n>>> [{args.tag}] GSM8K 独立测试集结果: "
        f"Exact Match = {correct}/{len(records)} ({acc:.1f}%) | "
        f"#### 格式遵守率 = {fmt_ok}/{len(records)} ({fmt_rate:.1f}%)\n",
        flush=True,
    )

    # 2. 副作用 / 能力退化检查：6 道非数学通用常识题
    print(f"=== 通用常识退化检查 [{args.tag}] (6 题) ===", flush=True)
    gen_outputs = batch_generate(model, tokenizer, GENERAL_PROMPTS, batch_size=6, max_new_tokens=64)
    general_records = []
    for q, ans in zip(GENERAL_PROMPTS, gen_outputs):
        clean_ans = ans.split("\nQuestion:")[0].strip()
        general_records.append({"prompt": q, "response": clean_ans})
        print(f"  Q: {q.splitlines()[0]}\n  A: {clean_ans}\n", flush=True)

    payload = {
        "tag": args.tag,
        "lora_path": args.lora_path,
        "num_samples": len(records),
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "format_count": fmt_ok,
        "format_rate_pct": round(fmt_rate, 2),
        "general_checks": general_records,
        "math_records": records,
    }
    json_path = out_dir / f"endpoint4_eval_{args.tag}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 导出前 10 个错题方便人工做 Bad Case 归因
    wrong = [r for r in records if not r["correct"]][:10]
    lines = [
        f"Eval Tag: {args.tag} | Exact Match: {correct}/{len(records)} ({acc:.1f}%) | Format Rate: {fmt_rate:.1f}%\n"
    ]
    for w in wrong:
        lines.append(
            f"--- Wrong #{w['idx']} (GT={w['gt']} | Pred={w['pred']} | Format={w['has_format']}) ---\n"
            f"Q: {w['question']}\nModel Output:\n{w['completion']}\n"
        )
    txt_path = out_dir / f"endpoint4_wrong_{args.tag}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"评估详情已保存至: {json_path}\n错题样例已保存至: {txt_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn import softmax
from cs336_basics.tokenizer import Tokenizer


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    eot_token: str = "<|endoftext|>",
    device: str = "cuda",
) -> str:
    """
    自回归文本生成：每次只取最后一个位置的 logits，按温度缩放采样 1 个新词并拼回输入。
    """
    model.eval()
    eot_ids = tokenizer.encode(eot_token)
    eot_id = eot_ids[0] if len(eot_ids) == 1 else None

    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # 超过上下文窗口时，只保留最近的 context_length 个词
        idx_cond = idx[:, -model.context_length :]
        # 前向得到 (1, t, V)，只取最后一个位置预测下一个词 -> (1, V)
        logits = model(idx_cond)[:, -1, :]

        if temperature <= 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if eot_id is not None and int(next_id.item()) == eot_id:
            break

    return tokenizer.decode(idx[0].tolist())


def export_stage_samples(
    ckpt_paths: list[tuple[str, Path]],
    tokenizer: Tokenizer,
    out_file: Path,
    prompt: str = "Once upon a time",
    temperature: float = 0.8,
    device: str = "cuda",
) -> None:
    """加载不同训练阶段的 Checkpoint，用同一开头生成文本并写入对比文件。"""
    from cs336_basics.train import build_model

    blocks: list[str] = [f"Prompt: {prompt!r} | Temperature: {temperature}\n"]
    for label, ckpt_path in ckpt_paths:
        model = build_model(device=device)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        step = payload["iteration"]

        torch.manual_seed(42)
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=120,
            temperature=temperature,
            device=device,
        )
        block = f"=== {label} (Step {step}) ===\n{text}\n"
        print(block)
        blocks.append(block)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(blocks), encoding="utf-8")
    print(f"多阶段生成对比已保存至: {out_file}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    ckpt_dir = repo_root / "checkpoints"
    out_dir = repo_root / "outputs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(data_dir / "vocab.json"),
        merges_filepath=str(data_dir / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )
    export_stage_samples(
        ckpt_paths=[
            ("Untrained", ckpt_dir / "endpoint1_step0.pt"),
            ("Mid-Training", ckpt_dir / "endpoint1_mid.pt"),
            ("Final", ckpt_dir / "endpoint1_final.pt"),
        ],
        tokenizer=tokenizer,
        out_file=out_dir / "endpoint1_samples.txt",
        device=device,
    )

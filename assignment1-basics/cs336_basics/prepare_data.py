from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"


def split_stories(
    input_path: str | Path,
    train_ratio: float = 0.9,
) -> tuple[str, str]:
    """
    第 1 块：按故事分隔符 <|endoftext|> 严格切分训练集与验证集。
    - 输入：原始文本文件路径
    - 输出：(train_text, val_text)，保证同一篇故事不会同时出现在两边
    """
    text = Path(input_path).read_text(encoding="utf-8")

    # 按 <|endoftext|> 拆成独立故事，并去掉首尾空白项
    stories = [s.strip() for s in text.split(EOT) if s.strip()]
    split_idx = int(len(stories) * train_ratio)

    train_stories = stories[:split_idx]
    val_stories = stories[split_idx:]

    # 每篇故事末尾补回 <|endoftext|>，让模型学到“故事结束”的边界信号
    sep = f"\n{EOT}\n"
    train_text = sep.join(train_stories) + f"\n{EOT}\n"
    val_text = sep.join(val_stories) + f"\n{EOT}\n"

    print(
        f"总故事数: {len(stories)} | "
        f"训练集: {len(train_stories)} 篇 | 验证集: {len(val_stories)} 篇"
    )
    return train_text, val_text


def save_tokenizer_files(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    out_dir: Path,
) -> None:
    """按 Tokenizer.from_files() 的格式保存词表与合并规则。"""
    raw_vocab = {str(k): v.decode("latin1") for k, v in vocab.items()}
    (out_dir / "vocab.json").write_text(
        json.dumps(raw_vocab, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
        for p1, p2 in merges:
            f.write(json.dumps([p1.decode("latin1"), p2.decode("latin1")]) + "\n")


def prepare_dataset(
    input_path: str | Path,
    out_dir: str | Path,
    vocab_size: int = 4096,
    train_ratio: float = 0.9,
) -> None:
    """
    第 2 块：仅在训练集上训练 BPE 词表，再用同一套词表分别编码 train/val 并保存为 .npy。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. 严格切分训练集和验证集
    train_text, val_text = split_stories(input_path, train_ratio=train_ratio)
    train_raw_path = out_path / "train_raw.txt"
    train_raw_path.write_text(train_text, encoding="utf-8")

    # 2. 只用训练集文本训练 BPE 词表（防止验证集信息泄漏进词表）
    print(f"正在训练集上训练 BPE 词表 (vocab_size={vocab_size})...")
    vocab, merges = train_bpe(
        input_path=train_raw_path,
        vocab_size=vocab_size,
        special_tokens=[EOT],
    )
    save_tokenizer_files(vocab, merges, out_path)

    # 3. 用同一套分词器分别编码 train 和 val，存成一维 uint16 数组
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT])
    train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.uint16)

    np.save(out_path / "train.npy", train_ids)
    np.save(out_path / "val.npy", val_ids)
    print(
        f"编码完成！train.npy: {len(train_ids):,} tokens | "
        f"val.npy: {len(val_ids):,} tokens | 保存目录: {out_path}"
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = (
        repo_root / "assignment1-basics" / "tests" / "fixtures" / "tinystories_sample_5M.txt"
    )
    prepare_dataset(
        input_path=corpus_path,
        out_dir=repo_root / "data",
        vocab_size=4096,
        train_ratio=0.9,
    )

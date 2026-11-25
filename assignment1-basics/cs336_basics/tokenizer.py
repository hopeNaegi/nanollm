from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import regex as re

from cs336_basics.bpe import GPT2_PAT, merge_word


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """【已写好】初始化词表、反向查询表、合并优先级表与特殊词元正则。"""
        # 1. 复制一份词表，避免修改外部传入的字典
        self.vocab: dict[int, bytes] = dict(vocab)
        # 2. 如果用户给了额外的 special_tokens 且词表里还没有，就追加进去
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []
        existing_bytes = set(self.vocab.values())
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in existing_bytes:
                self.vocab[len(self.vocab)] = tok_bytes
                existing_bytes.add(tok_bytes)

        # 3. 反向字典：bytes -> int ID（encode 时用来把字节块查成编号）
        self.byte_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # 4. 合并优先级表：pair -> 训练时的先后顺序号（越小表示越早合并、优先级越高）
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        # 5. 用于切分特殊词元的正则（按长度从长到短排，防止短的先截胡长的）
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")"
            self.special_pat = re.compile(pattern)
        else:
            self.special_pat = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """【已写好】从磁盘文件加载词表和合并规则。"""
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(k): v.encode("latin1") for k, v in raw_vocab.items()}
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                p1, p2 = json.loads(line)
                merges.append((p1.encode("latin1"), p2.encode("latin1")))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def decode(self, ids: list[int]) -> str:
        """
        TODO 1: 把整数词元 ID 列表解码回字符串。
        步骤（只要 2 行！）：
        1. 用 self.vocab[i] 把 ids 里的每个整数查成 bytes，再用 b"".join(...) 拼成一整串字节 raw_bytes
        2. 返回 raw_bytes.decode("utf-8", errors="replace")
        """
        # === 请在这里填写你的代码（2 行）===
        raw_bytes=b"".join([self.vocab[i]for i in ids])

        return raw_bytes.decode("utf-8",errors="replace")

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        """
        TODO 2: 把单个预词元的字节串（如 b'the'）按 self.merge_ranks 反复合并，并转成整数 ID 列表。
        步骤：
        1. 先把 word_bytes 拆成单字节列表：
           tokens = [bytes([b]) for b in word_bytes]
        2. 只要 len(tokens) >= 2，就循环：
           - 找出 tokens 里所有相邻对 (tokens[i], tokens[i + 1]) 中，
             在 self.merge_ranks 里存在、且 rank（顺序号）最小的那一对 best_pair
           - 如果没有任何一对在 self.merge_ranks 里，就 break 结束循环
           - 否则调用你刚才写的 merge_word(tokens, best_pair) 更新 tokens
        3. 最后把 tokens 里的每个字节块查 self.byte_to_id[t]，返回整数列表
        """
        # === 请在这里填写你的代码 ===
        tokens=[bytes([i]) for i in word_bytes]
        while len(tokens)>=2:
            best_pair=None
            best_rank=None
            for i in range(len(tokens)-1):
                pair=(tokens[i],tokens[i+1])
               
                if pair in self.merge_ranks:
                    rank=self.merge_ranks[pair]
                    if best_rank is None or rank<best_rank:
                        best_rank=rank
                        best_pair=pair
            if best_rank is None:
                break
            tokens=merge_word(tokens,best_pair)
        return  [self.byte_to_id[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        """【已写好】把整段文本编码为整数 ID 列表（会正确保留特殊词元）。"""
        if not text:
            return []

        # 先按特殊词元切分（带括号的 re.split 会把特殊词元本身也保留在列表里）
        if self.special_pat is not None:
            parts = self.special_pat.split(text)
        else:
            parts = [text]

        special_set = set(self.special_tokens)
        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in special_set:
                ids.append(self.byte_to_id[part.encode("utf-8")])
            else:
                for match in GPT2_PAT.finditer(part):
                    word_bytes = match.group(0).encode("utf-8")
                    ids.extend(self._encode_word(word_bytes))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        TODO 3: 接收一个字符串可迭代对象（如文件句柄），用生成器逐个产出词元 ID。
        提示（只要 2 行！）：
        for text in iterable:
            yield from self.encode(text)
        """
        # === 请在这里填写你的代码（2 行）===
        for text in iterable:
            yield from self.encode(text)

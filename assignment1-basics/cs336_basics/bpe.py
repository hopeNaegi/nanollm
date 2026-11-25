from __future__ import annotations

from collections import Counter, defaultdict
import os
import regex as re

# GPT-2 官方预分词正则表达式（讲义第 5 页）
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    TODO 1: 初始化词表 vocab: dict[int, bytes]
    规则：
    1. 先放入 0 ~ 255 共 256 个单字节，例如：0 -> b'\\x00', 65 -> b'A'
       （提示：用 bytes([i]) 可以把整数 i 转成单字节 bytes 对象）
    2. 再把 special_tokens 里的每个字符串用 .encode("utf-8") 转成 bytes，
       按顺序接着编号放进 vocab（例如从 256 开始）
    3. 返回 vocab
    """
    vocab: dict[int, bytes] = {}
    # === 请在这里填写你的代码（大约 5 行）===
    for i in range(256):
        vocab[i]=bytes([i])
    for token in special_tokens:
        vocab[len(vocab)]=token.encode("utf-8")

    
    return vocab


def merge_word(word: list[bytes], best_pair: tuple[bytes, bytes]) -> list[bytes]:
    """
    TODO 2: 把一个词里所有相邻出现的 best_pair 合并成一个新词元。
    举例：
        word = [b'n', b'e', b'w', b'e', b's', b't']
        best_pair = (b's', b't')
        返回 -> [b'n', b'e', b'w', b'e', b'st']

    提示（用 while 循环从左往右扫）：
    - 准备一个空列表 new_word = [] 和下标 i = 0
    - 当 i < len(word) 时：
        - 如果 i + 1 < len(word) 且 (word[i], word[i + 1]) == best_pair：
            把 word[i] + word[i + 1] 加进 new_word，然后 i += 2（跳过已合并的两个）
        - 否则：
            把 word[i] 加进 new_word，然后 i += 1
    - 返回 new_word
    """
    # === 请在这里填写你的代码（大约 10 行）===
    new_word=[]
    i=0
    while(i<len(word)):
        if i+1<len(word)and(word[i],word[i+1])==best_pair:
            new_word.append(word[i]+word[i+1])
            i+=2
        else:
            new_word.append(word[i])
            i+=1
    return new_word


def find_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    """
    TODO 3: 从 pair_counts 中选出出现次数最多的相邻字节对。
    规则（讲义第 6 页）：
    - 优先选出现次数（count）最大的；
    - 如果次数并列相同，选字典序更大的那一对（pair 本身在 Python 里可以直接比大小）。

    提示：
    Python 的 max() 可以传入 key=lambda item: (次数, 字节对)
    pair_counts.items() 里的每一项 item 是 (pair, count)。
    """
    # === 请在这里填写你的代码（1~3 行即可）===
    best_pair=max(pair_counts.items(),key=lambda item:(item[1],item[0]))
    return best_pair[0]
    


def pretokenize(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """
    【已写好】先按 special_tokens 切分文本，再用 GPT-2 正则预分词，
    最后统计每个预词元（拆成单字节元组）出现的次数。
    """
    if special_tokens:
        # 用 re.escape 转义特殊字符（如 <|endoftext|> 里的 |）
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in chunks:
        if not chunk:
            continue
        for match in GPT2_PAT.finditer(chunk):
            token_bytes = match.group(0).encode("utf-8")
            # 把 b'low' 变成 (b'l', b'o', b'w')
            word_tuple = tuple(bytes([b]) for b in token_bytes)
            word_counts[word_tuple] += 1
    return word_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    【主流程框架已写好】训练字节级 BPE 分词器，返回 (vocab, merges)。
    """
    # 1. 初始化词表（调用你写的 TODO 1）
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 2. 读取文件并预分词，得到每个词和它的频次
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    word_counts = pretokenize(text, special_tokens)

    # 把词拆成列表方便原地更新：words[i] 是第 i 个词的符号列表，word_freqs[i] 是它出现次数
    words: list[list[bytes]] = [list(w) for w in word_counts.keys()]
    word_freqs: list[int] = list(word_counts.values())

    # 3. 建立初始的相邻对计数（pair_counts）和倒排索引（pair_to_words，记录哪些词包含该字节对）
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for wid, (word, freq) in enumerate(zip(words, word_freqs)):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(wid)

    # 4. 循环合并，直到词表达到目标大小 vocab_size
    while len(vocab) < vocab_size and pair_counts:
        # 选出最优字节对（调用你写的 TODO 3）
        best_pair = find_best_pair(pair_counts)
        new_token = best_pair[0] + best_pair[1]

        # 记录新词元与合并历史
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        # 只更新包含 best_pair 的那些词（增量更新，速度很快）
        affected_wids = list(pair_to_words.pop(best_pair, ()))
        for wid in affected_wids:
            old_word = words[wid]
            # 对该词执行合并（调用你写的 TODO 2）
            new_word = merge_word(old_word, best_pair)
            if new_word == old_word:
                continue

            freq = word_freqs[wid]
            # 先减去旧词里的相邻对计数
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i + 1])
                if p in pair_counts:
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]

            # 再加上新词里的相邻对计数和倒排索引
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(wid)

            words[wid] = new_word

    return vocab, merges

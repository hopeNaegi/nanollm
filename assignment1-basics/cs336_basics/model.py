from __future__ import annotations

import torch
import torch.nn as nn

from cs336_basics.nn import Embedding, Linear, MultiHeadSelfAttention, RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    """
    A1 §3.5：预归一化（Pre-Norm）Transformer 块
    数据流：
      x -> ln1 -> attn -> (+ x 残差) -> ln2 -> ffn -> (+ 残差) -> 输出
    输入与输出形状完全相同：(batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 子层 1：先归一化，再做带 RoPE 的因果多头自注意力，最后加回残差
        x = x + self.attn(self.ln1(x))
        # 子层 2：先归一化，再做 SwiGLU 前馈网络，最后加回残差
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    A1 §3.6：完整的现代 Decoder-Only Transformer 语言模型
    输入：token_ids，形状 (batch_size, seq_len)
    输出：logits，形状 (batch_size, seq_len, vocab_size)（未归一化的下词预测分数）
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length

        # 1. 词元嵌入层：(B, S) -> (B, S, d_model)
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )

        # 2. 堆叠 num_layers 个结构相同的 TransformerBlock：(B, S, d_model) -> (B, S, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. 最终 RMSNorm（Pre-Norm 架构必须在最后补一次归一化）
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # 4. 语言模型头（LM Head）：(B, S, d_model) -> (B, S, vocab_size)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

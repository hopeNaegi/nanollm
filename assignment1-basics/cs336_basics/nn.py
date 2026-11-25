from __future__ import annotations

import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    """无偏置线性层：(..., d_in) -> (..., d_out)"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """词元嵌入层：(...,) -> (..., d_model)"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """均方根层归一化：(..., d_model) -> (..., d_model)"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU 激活函数：x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU 门控前馈网络：(..., d_model) -> (..., d_model)"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置嵌入（RoPE）：(..., seq_len, d_k) -> (..., seq_len, d_k)"""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        )
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """数值稳定的 Softmax"""
    max_val = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """缩放点积注意力：(..., seq_q, d_k) -> (..., seq_q, d_v)"""
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ V


class MultiHeadSelfAttention(nn.Module):
    """因果多头自注意力（可选配 RoPE）：(..., S, d_model) -> (..., S, d_model)"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_causal_mask: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_causal_mask = use_causal_mask

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        *batch_dims, S, _ = x.shape

        # 1. 线性投影得到 Q, K, V: (*batch_dims, S, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 拆分成 num_heads 个头，并把头维度移到句长 S 前面: (*batch_dims, num_heads, S, d_k)
        Q = Q.view(*batch_dims, S, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_dims, S, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_dims, S, self.num_heads, self.d_k).transpose(-3, -2)

        # 3. 对 Q 和 K 施加 RoPE 位置旋转（V 不旋转）
        if self.rope is not None:
            if token_positions is None:
                pos = torch.arange(S, device=x.device)
            else:
                pos = token_positions.unsqueeze(-2)
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        # 4. 构造 (S, S) 下三角因果掩码，阻止第 i 个词关注未来的第 j > i 个词
        if self.use_causal_mask:
            causal_mask = torch.tril(torch.ones((S, S), device=x.device, dtype=torch.bool))
        else:
            causal_mask = None

        # 5. 并行计算所有头的缩放点积注意力: (*batch_dims, num_heads, S, d_k)
        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # 6. 把各头拼回 (*batch_dims, S, d_model)，再过输出投影层融合各头信息
        merged = attn_out.transpose(-3, -2).contiguous().view(*batch_dims, S, self.d_model)
        return self.output_proj(merged)

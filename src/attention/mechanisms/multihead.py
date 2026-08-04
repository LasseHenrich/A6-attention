"""Multi-head attention (Task E).

Wraps the single-head op from Task D into ``num_heads`` parallel heads: per-
head Q/K/V projections, ``scaled_dot_product_attention`` per head (batched
over the head dimension), concatenation, and an output projection.  One class
serves both self- and cross-attention — the caller chooses by what it passes
as ``q_in``/``k_in``/``v_in``.

The student writes ``__init__`` and ``forward``.  The per-head attention must
go through the student's **own** ``scaled_dot_product_attention``;
``nn.MultiheadAttention`` / ``F.multi_head_attention_forward`` /
``F.scaled_dot_product_attention`` are forbidden.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from attention.mechanisms.scaled_dot_product import (
    scaled_dot_product_attention,
)


class MultiHeadAttention(nn.Module):
    """Multi-head attention returning ``[B, L, d_model]``."""

    def __init__(self, d_model: int, num_heads: int, *, bias: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.last_attn: Tensor | None = None
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            if bias:
                nn.init.zeros_(proj.bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        """``[B, L, d_model] -> [B, num_heads, L, head_dim]``."""
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q_in: Tensor,
        k_in: Tensor,
        v_in: Tensor,
        *,
        attn_mask: Tensor | None = None,
        additive_bias: Tensor | None = None,
    ) -> Tensor:
        """Self-attention: ``q_in = k_in = v_in``.  Cross-attention: distinct."""
        q = self.q_proj(q_in)
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)
        
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        attn_out, attn_weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask=attn_mask, additive_bias=additive_bias)
        
        b, _, l_q, _ = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l_q, self.d_model)
        
        self.last_attn = attn_weights
        
        return self.out_proj(attn_out)

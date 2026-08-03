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
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

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
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

"""Linear (kernel) attention (Task I).

Replaces the ``n x n`` softmax score matrix with a feature map ``φ`` and an
associativity reorder, turning attention's O(n^2) cost into O(n).  The feature
map is ``φ(x) = elu(x) + 1`` (positive-valued, so the normalizer is
well-defined).

Non-causal (parallel) form:
    KV = Σ_j φ(k_j) v_jᵀ ,  Z = Σ_j φ(k_j)
    out_t = φ(q_t)ᵀ KV / (φ(q_t)ᵀ Z)

Causal (prefix-sum) form: the same, with running sums over j <= t — a linear
RNN over the reordered state.

``F.scaled_dot_product_attention`` / ``nn.MultiheadAttention`` are forbidden —
linear attention must be the reorder, not softmax in disguise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _phi(x: Tensor) -> Tensor:
    """Positive feature map ``elu(x) + 1``."""
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    """Linear/kernel attention with a fixed-size running state."""

    def __init__(self, d_model: int, num_heads: int, *, feature_map: str = "elu") -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.feature_map = feature_map
        self.last_attn: Tensor | None = None  # no explicit matrix: entropy N/A
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

    def _split(self, x: Tensor) -> Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q_in: Tensor,
        k_in: Tensor,
        v_in: Tensor,
        *,
        attn_mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        """``attn_mask`` is accepted for interface compatibility but causality
        is expressed by ``causal`` (the prefix-sum form), not an additive mask.
        """
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

"""Additive (Bahdanau) attention (Task C).

A small scoring network over the encoder states that lets the RNN decoder
look back at *every* source position instead of squeezing the source through
one fixed context vector.  The student writes ``__init__`` (parameter
registration) and ``forward``.

Scoring:  ``e_ij = vᵀ tanh(W_q s_i + W_k h_j)``
Weights:  ``a_i = softmax_j(e_ij)``   (over the source axis)
Context:  ``c_i = Σ_j a_ij · h_j``

Delegating to ``nn.MultiheadAttention`` / ``F.scaled_dot_product_attention``
is forbidden — this is additive attention, written by hand.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention returning ``(context, weights)``."""

    def __init__(self, dec_dim: int, enc_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim
        self.attn_dim = attn_dim
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

    def forward(
        self,
        dec_state: Tensor,
        enc_states: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score ``dec_state`` [B, dec_dim] against ``enc_states`` [B, S, enc_dim].

        ``mask`` (if given) is ``[B, S]`` with ``True`` on real source tokens;
        ``PAD`` positions are removed from the softmax.  Returns
        ``(context [B, enc_dim], weights [B, S])``.
        """
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

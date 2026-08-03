"""Learned absolute positional encoding — the provided length-gen foil.

This is the ablation the Task-F length-generalization demo compares against:
an ``nn.Embedding`` over absolute positions.  It works in-distribution but
has no code for positions beyond the trained maximum, so it *cliffs* under
the length-generalization split — the empirical fault the student contrasts
with sinusoidal PE and ALiBi.  Students toggle against it; they do not
implement it.

Provided scaffold — students never modify this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LearnedAbsolutePositionalEncoding(nn.Module):
    """Adds a learned per-position vector to the token embeddings."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """``x`` is ``[B, L, d_model]``; adds the position codes for ``0..L``.

        Positions at or beyond ``max_len`` have no learned code; they are
        clamped to the last trained position, which is exactly why this
        scheme fails to extrapolate.
        """
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device).clamp_max(self.max_len - 1)
        return x + self.embed(pos).unsqueeze(0)

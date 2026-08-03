"""Attention masks consumed by the transformer (Tasks G, H).

Provided helper the student *calls* from the transformer.  It is subtly
wrong: the causal mask lets each position peek at a future token, so the
model trains to near-perfect **teacher-forced** accuracy but collapses under
**free-running** generation (it was reading the answer).  Find the bug,
record its line number in ``answers.py`` (before you edit — your edits shift
the line numbers), and fix it in place.
"""

from __future__ import annotations

import torch
from torch import Tensor


def causal_mask(seq_len: int) -> Tensor:
    """Return an additive ``[seq_len, seq_len]`` causal mask.

    Entry ``[i, j]`` should be ``0`` where query ``i`` may attend to key ``j``
    (``j <= i``) and ``-inf`` where it may not (``j > i``), so position ``i``
    never sees a future token.
    """
    full = torch.full((seq_len, seq_len), float("-inf"))
    return torch.triu(full, diagonal=2)

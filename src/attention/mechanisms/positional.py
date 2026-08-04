"""Positional encodings (Task F): sinusoidal PE and ALiBi.

Two hand-written constructions that give order-blind attention a sense of
position:

- ``sinusoidal_encoding`` — a fixed (non-learned) absolute code added to the
  token embeddings.
- ``alibi_slopes`` / ``alibi_bias`` — a per-head relative-distance penalty fed
  to attention through the ``additive_bias`` hook.

Delegating to an external positional-encoding library is forbidden; the
constructions are the exercise.  Standard ``torch`` math (``sin``, ``cos``,
``arange``, ``exp``) is allowed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_encoding(seq_len: int, d_model: int) -> Tensor:
    """Return the ``[seq_len, d_model]`` sinusoidal positional encoding.

    Even dimensions use ``sin(pos / 10000^(2i/d))``, odd dimensions the
    matching ``cos``, interleaved.  Each position gets a unique, smoothly
    varying code and relative offsets are linear in the encoding.
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1)
    
    # 1 / (10_000^(2i/d))
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

def alibi_slopes(num_heads: int) -> Tensor:
    """Return the ``[num_heads]`` geometric slope sequence for ALiBi.

    ``m_h = ratio^h`` for ``h = 1..num_heads`` with ``ratio = 2^(-8/num_heads)``
    (so 8 heads give slopes ``2^-1 .. 2^-8``).
    """
    h = torch.arange(1, num_heads + 1)
    return 2 ** (-8 * h / num_heads)


def alibi_bias(seq_len: int, num_heads: int) -> Tensor:
    """Return the ``[num_heads, seq_len, seq_len]`` causal ALiBi bias.

    ``bias_h[i, j] = -m_h * (i - j)`` for ``j <= i`` (a linear penalty growing
    with distance) and ``-inf`` for ``j > i`` (causal — no attending ahead).
    Added to attention scores before the softmax via ``additive_bias``.
    """
    slopes = alibi_slopes(num_heads).view(num_heads, 1, 1)
    
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    
    distance = i - j
    bias = -slopes * distance
    causal_mask = j > i
    bias = bias.masked_fill(causal_mask, -math.inf)
    
    return bias
    

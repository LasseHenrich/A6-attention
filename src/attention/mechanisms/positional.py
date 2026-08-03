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
    # ------ WRITE YOUR CODE HERE ------
    raise NotImplementedError("implement this for the task")


def alibi_slopes(num_heads: int) -> Tensor:
    """Return the ``[num_heads]`` geometric slope sequence for ALiBi.

    ``m_h = ratio^h`` for ``h = 1..num_heads`` with ``ratio = 2^(-8/num_heads)``
    (so 8 heads give slopes ``2^-1 .. 2^-8``).
    """
    # ------ WRITE YOUR CODE HERE ------
    raise NotImplementedError("implement this for the task")


def alibi_bias(seq_len: int, num_heads: int) -> Tensor:
    """Return the ``[num_heads, seq_len, seq_len]`` causal ALiBi bias.

    ``bias_h[i, j] = -m_h * (i - j)`` for ``j <= i`` (a linear penalty growing
    with distance) and ``-inf`` for ``j > i`` (causal — no attending ahead).
    Added to attention scores before the softmax via ``additive_bias``.
    """
    # ------ WRITE YOUR CODE HERE ------
    raise NotImplementedError("implement this for the task")

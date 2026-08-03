"""Scaled dot-product self-attention (Task D).

The core attention operation, reused everywhere: per-head inside
``MultiHeadAttention`` (E), inside the transformer (G, H), and mirrored by
``LinearAttention`` (I).  It is a pure function — the caller stashes the
returned weights for the entropy metric, since a free function has no
``self`` to write ``last_attn``.

Delegating to ``F.scaled_dot_product_attention`` / ``nn.MultiheadAttention``
is forbidden — the op is the exercise.  ``matmul``, ``softmax``, and masking
are allowed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None = None,
    additive_bias: Tensor | None = None,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute ``softmax(QKᵀ · scale + mask + bias) V``.

    ``q``/``k``/``v`` are ``[..., L, d]`` (any leading batch/head dims).
    ``scale=None`` uses ``1/sqrt(d_k)`` (the ablation arm passes ``scale=1.0``).
    ``mask`` is either an additive float mask (added to the scores) or a
    boolean mask (``True`` = attend, ``False`` = ``-inf``).  ``additive_bias``
    (e.g. ALiBi) is added to the scores before the softmax.  Returns
    ``(output [..., L, d], weights [..., Lq, Lk])``.
    """
    # ------ WRITE YOUR CODE HERE ------
    raise NotImplementedError("implement this for the task")

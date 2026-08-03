"""Transformer assembly (Tasks G, H).

Task G: a pre-norm decoder-only transformer — ``TransformerBlock`` (multi-head
self-attention + FFN, each wrapped in a residual with LayerNorm) stacked with
a causal mask, plus the causal-mask bug beat.  Task H adds the encoder-decoder
model.  This is genuinely *wiring already-built pieces*: ``MultiHeadAttention``
(E), the positional encodings (F), and the ``causal_mask`` helper.

Delegating to ``nn.Transformer*`` is forbidden — assemble it from your own
parts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from attention.config import ExperimentConfig
from attention.data import Batch
from attention.masking import causal_mask
from attention.mechanisms.multihead import MultiHeadAttention
from attention.mechanisms.positional import sinusoidal_encoding
from attention.models.base import SeqModel


class TransformerBlock(nn.Module):
    """Pre-norm residual block: self-attention + position-wise FFN."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        additive_bias: Tensor | None = None,
    ) -> Tensor:
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")


class DecoderOnlyTransformer(SeqModel):
    """embed -> +PE -> N causal blocks -> final LayerNorm -> unembed."""

    framing = "single_stream"

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

    def forward(self, batch: Batch) -> Tensor:
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")


class EncoderDecoderTransformer(SeqModel):
    """Bidirectional encoder + cross-attending decoder (Task H)."""

    framing = "encoder_decoder"

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

    def encode(self, source: Tensor) -> Tensor:
        """Run the **bidirectional** (unmasked) encoder over the source.

        ``source`` is ``[B, S]``; returns the encoder memory ``[B, S, d_model]``
        where every source position has attended to every other.
        """
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

    def forward(self, batch: Batch) -> Tensor:
        # ------ WRITE YOUR CODE HERE ------
        raise NotImplementedError("implement this for the task")

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
import math

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
        
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.num_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        
        lin1 = nn.Linear(cfg.d_model, cfg.d_ff)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.zeros_(lin1.bias)

        lin2 = nn.Linear(cfg.d_ff, cfg.d_model)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.zeros_(lin2.bias)
        
        scale = 1 / math.sqrt(2.0 / cfg.num_layers)
        with torch.no_grad():
            lin2.weight.mul_(scale)
            self.attn.out_proj.weight.mul_(scale)
        
        self.ffn = nn.Sequential(
            lin1,
            nn.GELU(),
            lin2
        )
        
        self.dropout = nn.Dropout(cfg.dropout)
        
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        additive_bias: Tensor | None = None,
    ) -> Tensor:
        norm_x = self.ln1(x)
        x = x + self.dropout(
            self.attn(
                norm_x, norm_x, norm_x,
                attn_mask=attn_mask,
                additive_bias=additive_bias
            )
        )
        
        norm_x2 = self.ln2(x)
        x = x + self.dropout(
            self.ffn(norm_x2)
        )
        
        return x


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

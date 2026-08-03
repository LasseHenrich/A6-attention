"""``SingleLayerModel`` — make a bare mechanism battery-runnable.

Wraps any attention mechanism into an embed -> (+PE) -> one-attention-layer
-> unembed model the harness can train, so a bare mechanism (Tasks D-F, I)
produces a scoreboard row before the full transformer (G) exists.

The mechanism may be the bare ``scaled_dot_product_attention`` *function*
(Task D; the wrapper supplies single-head q/k/v/out projections around it)
or a self-contained *module* — ``MultiHeadAttention`` (E) or
``LinearAttention`` (I).  The call shape is resolved by inspecting the
mechanism's signature, so one wrapper serves all three.

Provided scaffold — students never modify this module.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

from attention.config import ExperimentConfig
from attention.data import Batch
from attention.mechanisms.learned_positional import (
    LearnedAbsolutePositionalEncoding,
)
from attention.models.base import SeqModel


def _causal_additive_mask(seq_len: int, device: torch.device) -> Tensor:
    """``[L, L]`` additive mask: 0 where a query may attend, ``-inf`` above."""
    full = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(full, diagonal=1)


class SingleLayerModel(SeqModel):
    """Embed -> (+positional) -> one attention layer -> vocab projection."""

    framing = "single_stream"

    def __init__(
        self,
        mechanism: Callable[..., Any] | nn.Module,
        cfg: ExperimentConfig,
        *,
        causal: bool = True,
        positional: str | None = "sinusoidal",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.causal = causal
        self.positional = positional
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._is_module = isinstance(mechanism, nn.Module)
        if self._is_module:
            self.mechanism = mechanism
            self._fn = None
            params = inspect.signature(mechanism.forward).parameters
        else:
            # Bare SDPA function: supply our own single-head projections.
            self._fn = mechanism
            self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
            self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
            self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
            self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)
            params = inspect.signature(mechanism).parameters
        self._accepts_attn_mask = "attn_mask" in params or "mask" in params
        self._mask_kw = "attn_mask" if "attn_mask" in params else "mask"
        self._accepts_causal = "causal" in params
        self._accepts_bias = "additive_bias" in params

        if positional == "learned":
            self.pos = LearnedAbsolutePositionalEncoding(cfg.max_len, cfg.d_model)

    # -- positional handling -------------------------------------------------

    def _apply_embedding_position(self, x: Tensor) -> Tensor:
        if self.positional in (None, "none", "alibi"):
            return x
        if self.positional == "learned":
            return self.pos(x)
        if self.positional == "sinusoidal":
            from attention.mechanisms.positional import sinusoidal_encoding

            pe = sinusoidal_encoding(x.shape[1], self.cfg.d_model)
            return x + pe.to(x.device).unsqueeze(0)
        raise ValueError(f"unknown positional {self.positional!r}")

    def _alibi_bias(self, seq_len: int, num_heads: int, device) -> Tensor | None:
        if self.positional != "alibi":
            return None
        from attention.mechanisms.positional import alibi_bias

        return alibi_bias(seq_len, num_heads).to(device)

    # -- forward -------------------------------------------------------------

    def forward(self, batch: Batch) -> Tensor:
        x = self.embed(batch.input_ids)  # [B, L, d_model]
        x = self._apply_embedding_position(x)
        seq_len = x.shape[1]
        device = x.device

        mask = _causal_additive_mask(seq_len, device) if self.causal else None
        num_heads = self.cfg.num_heads if self._is_module else 1
        bias = self._alibi_bias(seq_len, num_heads, device)

        if self._is_module:
            out = self._call_module(x, mask, bias)
        else:
            out = self._call_function(x, mask, bias)
        return self.unembed(out)

    def _call_module(self, x: Tensor, mask: Tensor | None, bias: Tensor | None) -> Tensor:
        kwargs: dict[str, Any] = {}
        if self._accepts_attn_mask and mask is not None:
            kwargs[self._mask_kw] = mask
        if self._accepts_causal:
            kwargs["causal"] = self.causal
        if self._accepts_bias and bias is not None:
            kwargs["additive_bias"] = bias
        out = self.mechanism(x, x, x, **kwargs)
        self.last_attn = getattr(self.mechanism, "last_attn", None)
        return out

    def _call_function(self, x: Tensor, mask: Tensor | None, bias: Tensor | None) -> Tensor:
        assert self._fn is not None  # only called for the bare-function path
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out, weights = self._fn(q, k, v, mask=mask, additive_bias=bias, scale=None)
        self.last_attn = weights
        return self.o_proj(out)

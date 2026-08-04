"""RNN encoder-decoder (Tasks A and C).

Task A: the student implements a vanilla-tanh (Elman) recurrent cell and
assembles it into an encoder-decoder sequence-to-sequence model, with a
clean attention hook that stays inert until Task C fills it.

The student writes ``RNNCell``, ``Encoder``, ``Decoder``, and the
``Seq2SeqRNN`` wiring in full — ``__init__`` and ``forward`` alike.  The
``generate`` decode method is provided plumbing (used only for
observational free-running accuracy).  Calling ``nn.RNN``/``GRU``/``LSTM``
or their ``*Cell`` variants is forbidden — the recurrence is the exercise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from attention.config import ExperimentConfig
from attention.data import Batch
from attention.models.base import SeqModel
from attention.vocab import BOS, EOS


class RNNCell(nn.Module):
    """One vanilla-tanh (Elman) recurrence step.

    ``h_t = tanh(W_ih x_t + W_hh h_prev + b)`` — the same weights applied at
    every time step (weight sharing over time).
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.orthogonal_(self.W_hh.weight)
        
        self.W_ih = nn.Linear(input_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.W_ih.weight)
        torch.nn.init.zeros_(self.W_ih.bias)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """``x_t`` is ``[B, input_size]``, ``h_prev`` ``[B, hidden_size]``."""
        return torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))


class Encoder(nn.Module):
    """Embed the source and unroll the cell over time.

    Returns every hidden state ``[B, S, H]`` (Task C's attention needs them
    all) and the final state ``[B, H]``.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.cell = RNNCell(cfg.hidden_size, cfg.hidden_size)

    def forward(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """``src`` is ``[B, S]`` token ids."""
        batch_size, seq_len = src.shape
        
        embedded = self.embed(src)
        
        h_t = torch.zeros(batch_size, self.cfg.hidden_size, device=src.device, dtype=embedded.dtype)
        
        hidden_states = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h_t = self.cell(x_t, h_t)
            hidden_states.append(h_t)
            
        all_hidden = torch.stack(hidden_states, dim=1)
        final_state = h_t
        
        return all_hidden, final_state


class Decoder(nn.Module):
    """Unroll the cell over the teacher-forced target, with the attention hook.

    Task A: ``attention is None`` — the decoder sees the source only through
    ``enc_final`` (the fixed-context bottleneck), and ``enc_states`` is
    threaded through but not consulted.  Task C fills the hook.
    """

    def __init__(self, cfg: ExperimentConfig, *, attention: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.attention = attention
        self.last_attn: Tensor | None = None
    
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.cell = RNNCell(cfg.hidden_size, cfg.hidden_size)
        self.out_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

    def forward(
        self,
        tgt: Tensor,
        enc_states: Tensor,
        enc_final: Tensor,
        *,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """``tgt`` is the teacher-forced decoder input ``[B, T]``.

        Returns logits ``[B, T, VOCAB]``; ``logits[:, t]`` predicts the
        target at step ``t`` from decoder inputs ``0 .. t``.
        """
        batch_size, seq_len = tgt.shape
        
        embedded = self.embed(tgt)
        
        h_t = enc_final
        
        logits_list = []
        attn_weights_list = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h_t = self.cell(x_t, h_t)
            if self.attention is not None:
                _, attn_w = self.attention(h_t, enc_states, mask=src_mask)
                attn_weights_list.append(attn_w)
            step_logits = self.out_head(h_t)
            logits_list.append(step_logits)
            
        if attn_weights_list:
            self.last_attn = torch.stack(attn_weights_list, dim=1)
        
        logits = torch.stack(logits_list, dim=1)
        return logits


class Seq2SeqRNN(SeqModel):
    """Wire ``Encoder`` -> ``Decoder`` into an encoder-decoder model."""

    framing = "encoder_decoder"

    def __init__(self, cfg: ExperimentConfig, *, attention: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg, attention=attention)

    def forward(self, batch: Batch) -> Tensor:
        enc_states, enc_final = self.encoder(batch.source)
        logits = self.decoder(tgt=batch.target_in, enc_states=enc_states, enc_final=enc_final, src_mask=batch.source_padding_mask)
        return logits

    @torch.no_grad()
    def generate(self, batch: Batch, max_len: int) -> Tensor:
        """Free-running autoregressive decode (provided plumbing, not graded).

        Greedily decodes from ``BOS`` up to ``max_len`` steps, stopping the
        per-sequence contribution once ``EOS`` is produced.  Returns the
        predicted ids over the generated region ``[B, max_len]``.
        """
        self.eval()
        enc_states, enc_final = self.encoder(batch.source)
        device = batch.source.device
        b = batch.source.shape[0]
        prev = torch.full((b, 1), BOS, dtype=torch.long, device=device)
        h = enc_final
        outputs: list[Tensor] = []
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len):
            emb = self.decoder.embed(prev[:, -1])
            if self.decoder.attention is not None:
                context, _ = self.decoder.attention(h, enc_states, mask=batch.source_padding_mask)
                step_in = torch.cat([emb, context], dim=-1)
            else:
                step_in = emb
            h = self.decoder.cell(step_in, h)
            nxt = self.decoder.out(h).argmax(dim=-1)  # [B]
            nxt = torch.where(finished, torch.full_like(nxt, EOS), nxt)
            outputs.append(nxt)
            finished = finished | (nxt == EOS)
            prev = torch.cat([prev, nxt.unsqueeze(1)], dim=1)
        return torch.stack(outputs, dim=1)

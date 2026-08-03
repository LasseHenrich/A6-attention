"""The shared model interface every mechanism plugs into.

Every model is an ``nn.Module`` that exposes a ``framing`` class attribute
(so the harness knows how to build its batches) and a teacher-forced
``forward(batch) -> [B, T, VOCAB]`` (the graded artifact for model-building
tasks).  ``generate`` is provided plumbing used only for observational
free-running accuracy; it is never graded.

Provided scaffold — students never modify this module.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from attention.data import Batch


class SeqModel(nn.Module):
    """Base class documenting the harness contract.

    Subclasses set ``framing`` to ``"single_stream"`` or
    ``"encoder_decoder"`` and implement ``forward``.  ``last_attn`` holds the
    final layer's attention weights (set during ``forward``) so
    ``metrics.attention_entropy`` can read them without a re-run; it stays
    ``None`` for models without an explicit attention matrix.
    """

    framing: str = "single_stream"

    def __init__(self) -> None:
        super().__init__()
        self.last_attn: Tensor | None = None

    def forward(self, batch: Batch) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

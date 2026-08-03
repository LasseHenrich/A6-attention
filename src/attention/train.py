"""The reusable step-based training loop.

This is the **shared spine** every model runs through: draw a batch ->
forward -> loss -> backward -> optimizer step, with periodic evaluation.
The loop below *looks* reasonable but is subtly broken in a handful of
places.  Your job (Task B) is to find the bugs, record their line numbers in
``answers.py``, and repair them in place so the loop trains correctly.

There are exactly **5** bugs, all generic training-loop mistakes (nothing
transformer-specific). Look for: the order of gradient bookkeeping around the
step, what the loss counts, train/eval mode, detaching scalars, and where
gradient clipping happens. Record the line numbers **before** you start
editing — your edits will shift them.

The learning-rate schedule (warmup + optional cosine decay) is provided and
correct — it is **not** one of the bugs; leave it as it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from attention.config import ExperimentConfig
from attention.data import Batch, BatchGenerator
from attention.metrics import token_accuracy


def masked_cross_entropy(
    logits: Tensor, targets: Tensor, loss_mask: Tensor
) -> Tensor:
    """Mean cross-entropy over the output region only.

    ``loss_mask`` is 1 on the ``y1 .. EOS`` positions and 0 on ``PAD`` and
    prompt positions, so ``PAD``/prompt must never contribute.  ``logits`` is
    ``[B, T, V]``, ``targets`` ``[B, T]``, ``loss_mask`` ``[B, T]``.
    """
    b, t, v = logits.shape
    ce = F.cross_entropy(
        logits.reshape(b * t, v), targets.reshape(b * t), reduction="mean"
    )
    return ce


def train_step(
    model: nn.Module,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip: float | None,
) -> float:
    """One optimizer step; returns the (detached) scalar loss."""
    logits = model(batch)
    loss = masked_cross_entropy(logits, batch.targets, batch.loss_mask)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    if grad_clip is not None:
        clip_grad_norm_(model.parameters(), grad_clip)
    return loss


@torch.no_grad()
def evaluate(model: nn.Module, eval_sets: dict[str, Batch]) -> dict[str, float]:
    """Teacher-forced token accuracy per eval set, under ``no_grad`` + eval."""
    model.eval()
    device = next(model.parameters()).device
    out: dict[str, float] = {}
    for name, batch in eval_sets.items():
        b = batch.to(device)
        logits = model(b)
        out[name] = token_accuracy(logits, b.targets, b.loss_mask)
    return out


@dataclass
class History:
    """Accumulates ``(step, train_loss, eval_metrics)`` for the curves."""

    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    eval: dict[str, list[float]] = field(default_factory=dict)

    def record(
        self, step: int, loss: float, eval_metrics: dict[str, float]
    ) -> None:
        self.steps.append(step)
        self.train_loss.append(loss)
        for name, value in eval_metrics.items():
            self.eval.setdefault(name, []).append(value)

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "train_loss": self.train_loss,
            "eval": self.eval,
        }


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    cosine: bool = True,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for *warmup_steps*, then hold flat or cosine-decay.

    The multiplier ramps ``0 -> 1`` over the warmup, so the optimizer starts
    from a near-zero step and cannot lock onto a bad pattern before Adam's
    gradient statistics have settled.  Afterwards it either holds at the peak
    (``cosine=False``) or eases down to ``min_lr_ratio`` of the peak
    (``cosine=True``) so the late steps fine-tune rather than thrash.  Returns a
    plain ``LambdaLR`` — call ``.step()`` once per optimizer step.  (Provided
    and correct — not one of the bugs.)
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if not cosine:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: ExperimentConfig
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """The optional LR schedule for the loop: warmup and/or cosine decay.

    **Warmup** (``cfg.warmup_steps``) ramps the LR up from ~0 so the optimizer
    does not take a large, poorly-conditioned step before Adam's gradient
    statistics have settled — the standard cure for early training instability
    in transformers.  **Cosine decay** (``cfg.lr_schedule == "cosine"``) then
    eases the LR down so late steps fine-tune instead of thrash.  With the
    config defaults (no warmup, constant schedule) this returns ``None`` and the
    LR is held fixed.  (Provided and correct — not one of the bugs.)
    """
    if cfg.warmup_steps <= 0 and cfg.lr_schedule != "cosine":
        return None
    return warmup_lr_scheduler(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.steps,
        cosine=cfg.lr_schedule == "cosine",
        min_lr_ratio=cfg.min_lr_ratio,
    )


def train(
    model: nn.Module,
    cfg: ExperimentConfig,
    train_gen: BatchGenerator,
    eval_sets: dict[str, Batch],
) -> History:
    """Run ``cfg.steps`` of step-based training with periodic evaluation."""
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = build_scheduler(optimizer, cfg)
    history = History()
    for step in range(cfg.steps):
        model.eval()
        batch = train_gen.next_batch(cfg.batch_size).to(cfg.device)
        loss = train_step(model, batch, optimizer, grad_clip=cfg.grad_clip)
        if scheduler is not None:
            scheduler.step()
        if step % cfg.eval_every == 0:
            history.record(step, loss, evaluate(model, eval_sets))
    return history

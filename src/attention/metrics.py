"""The metrics panel logged for every run.

Six columns: token accuracy, wall-clock, peak memory, attention entropy,
gradient norm, parameter count.  Wall-clock and peak-memory helpers live in
``utils.py``; the rest are here.  Only ``param_count`` is deterministic and
machine-independent enough to (optionally) back a hashed verdict — the rest
are observational.

All provided — students never modify this module.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


def token_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction of correct argmax predictions over the masked region.

    Parameters
    ----------
    logits: ``[B, T, V]`` model output.
    targets: ``[B, T]`` label ids.
    mask: ``[B, T]`` 1 where the position counts, else 0.
    """
    preds = logits.argmax(dim=-1)
    m = mask.bool()
    total = int(m.sum().item())
    if total == 0:
        return 0.0
    correct = int(((preds == targets) & m).sum().item())
    return correct / total


def attention_entropy(attn: Tensor) -> Tensor:
    """Mean Shannon entropy (nats) of attention weights over the key axis.

    ``attn`` is a softmax distribution over keys with shape ``[..., Lq, Lk]``
    (the last axis sums to 1).  Returns a scalar when there is no head axis,
    or a per-head vector ``[H]`` when ``attn`` is ``[B, H, Lq, Lk]``.

    Linear attention has no explicit weight matrix, so entropy is undefined
    there (a Task-I teaching point) — callers simply do not invoke this.
    """
    eps = 1e-12
    ent = -(attn * (attn + eps).log()).sum(dim=-1)  # [..., Lq]
    if attn.dim() == 4:  # [B, H, Lq, Lk] -> per-head mean over B, Lq
        return ent.mean(dim=(0, 2))
    return ent.mean()


def self_attention_maps(model: nn.Module) -> list[Tensor]:
    """Every ``MultiHeadAttention`` weight tensor stashed by the last forward.

    Walks ``model.modules()`` (registration order, so index ``i`` is block
    ``i``) rather than reaching for an attribute like ``model.blocks`` — the
    student owns their module names, and only the mechanism's ``last_attn``
    contract is pinned.  Cross-attention maps are rectangular and are included
    too; callers that need square self-attention filter on shape.
    """
    from attention.mechanisms.multihead import MultiHeadAttention

    return [m.last_attn for m in model.modules() if isinstance(m, MultiHeadAttention) and m.last_attn is not None]


def cross_attention_maps(model: nn.Module) -> list[Tensor]:
    """The rectangular attention maps — decoder queries against encoder keys.

    Self-attention is square (``Lq == Lk``); cross-attention is not, because the
    target and source lengths differ.  Selecting on shape identifies the cross
    maps without knowing what the student named their decoder layers.
    """
    return [a for a in self_attention_maps(model) if a.shape[-2] != a.shape[-1]]


def input_dependency(model: nn.Module, forward: Callable[[], Tensor], embedding: nn.Embedding) -> Tensor:
    """``[P, S]`` matrix of ``‖∂ output_i / ∂ embed_j‖`` — who can see whom.

    A *structural* readout, not a learned one: it answers "can position ``i``'s
    representation depend on position ``j`` at all?", which is fixed by the
    architecture and holds at any weights, trained or not.  A causal stack
    yields **exact zeros** above the diagonal; a bidirectional encoder yields a
    dense matrix.

    ``forward`` must run *model* and return the tensor to differentiate,
    ``[1, P, d]``; ``embedding`` is the input embedding whose output the
    gradient is taken against.
    """
    captured: dict[str, Tensor] = {}

    def _hook(_module: nn.Module, _inputs: object, output: Tensor) -> None:
        output.retain_grad()
        captured["emb"] = output

    handle = embedding.register_forward_hook(_hook)
    try:
        out = forward()
        emb = captured["emb"]
        rows = []
        for i in range(out.shape[1]):
            (grad,) = torch.autograd.grad(out[0, i].sum(), emb, retain_graph=True)
            rows.append(grad[0].norm(dim=-1))  # [S]
    finally:
        handle.remove()
    return torch.stack(rows)


def first_embedding(model: nn.Module) -> nn.Embedding:
    """The first ``nn.Embedding`` registered — the source embedding by convention."""
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return module
    raise ValueError("model has no nn.Embedding")


def future_attention_mass(attn: Tensor) -> float:
    """Mean fraction of a query's attention that lands on a **future** key.

    ``attn`` is ``[..., Lq, Lk]`` with ``Lq == Lk`` (self-attention).  For each
    query ``i`` this sums the weight on every key ``j > i`` and averages over
    queries, heads, and batch.  The last query has no future key, so it is
    excluded rather than diluting the average with a forced zero.

    Under a correct causal mask the result is **exactly 0.0** — those logits
    were ``-inf`` before the softmax.  Any positive value means the model can
    see a token it is supposed to predict (Task G).
    """
    lq, lk = attn.shape[-2], attn.shape[-1]
    if lq != lk:
        raise ValueError(f"future_attention_mass needs a square self-attention map, got [{lq}, {lk}]")
    if lq < 2:
        return 0.0
    future = torch.triu(torch.ones(lq, lk, dtype=torch.bool, device=attn.device), diagonal=1)
    per_query = (attn * future).sum(dim=-1)  # [..., Lq]
    return float(per_query[..., :-1].mean().item())


def mean_attention_map(attn: Tensor, *, sample: int = 0) -> Tensor:
    """Reduce ``[B, H, Lq, Lk]`` (or ``[B, Lq, Lk]``) to one ``[Lq, Lk]`` map.

    Picks a single *sample* — so the token labels on the axes describe the
    sequence actually shown — and averages over heads, which is the right
    summary when several heads carry the same alignment.
    """
    one = attn[sample]
    return one.mean(dim=0) if one.dim() == 3 else one


def grad_norm(model: nn.Module) -> float:
    """Global L2 norm of all parameter gradients (a training-health read)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total**0.5


def param_count(model: nn.Module) -> int:
    """Total number of parameters — architecture-determined integer."""
    return sum(p.numel() for p in model.parameters())

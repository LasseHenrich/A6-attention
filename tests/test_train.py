"""Tests for the repaired training loop (Task B)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.data import make_batch
from attention.train import evaluate, masked_cross_entropy, train, train_step
from attention.utils import seed_everything
from attention.config import ExperimentConfig
from attention.data import BatchGenerator


class _LoopModel(nn.Module):
    """A tiny single-stream model used to exercise the loop in isolation."""

    framing = "single_stream"

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(22, 16)
        self.out = nn.Linear(16, 22)

    def forward(self, batch):
        return self.out(self.embed(batch.input_ids))


def _ref_masked_ce(logits, targets, loss_mask):
    b, t, v = logits.shape
    ce = F.cross_entropy(logits.reshape(b * t, v), targets.reshape(b * t), reduction="none").reshape(b, t)
    mask = loss_mask.to(ce.dtype)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# masked_cross_entropy
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_masked_cross_entropy_smoke():
    loss = masked_cross_entropy(torch.randn(2, 3, 22), torch.randint(0, 22, (2, 3)), torch.ones(2, 3))
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0


# ----- Correctness tests -----


def test_masked_cross_entropy_matches_reference():
    pass


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_train_step_smoke():
    model = _LoopModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = make_batch("copy", 8, seed=0, framing="single_stream", n=5)
    assert isinstance(train_step(model, batch, opt, grad_clip=1.0), float)


# ----- Correctness tests -----


def test_train_step_matches_reference_step():
    pass


def test_train_step_returns_python_float():
    pass


# ---------------------------------------------------------------------------
# evaluate / train
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_train_smoke():
    cfg = ExperimentConfig(steps=3, eval_every=1, batch_size=8, d_model=16, hidden_size=16, max_len=40)
    model = _LoopModel()
    gen = BatchGenerator("copy", seed=1234, framing="single_stream", n=5)
    ev = {"copy": make_batch("copy", 16, seed=900000, framing="single_stream", n=5)}
    hist = train(model, cfg, gen, ev)
    assert len(hist.train_loss) >= 1


# ----- Correctness tests -----


def test_evaluate_does_not_update_parameters():
    pass

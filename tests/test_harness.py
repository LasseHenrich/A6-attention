"""Tests for the training spine and battery scoreboard (provided infra)."""

import torch

from attention.config import ExperimentConfig
from attention.data import make_batch
from attention.harness import run_battery
from attention.metrics import (
    attention_entropy,
    param_count,
    token_accuracy,
)
from attention.models.rnn import Seq2SeqRNN
from attention.train import evaluate, masked_cross_entropy, train_step

CFG = ExperimentConfig(d_model=32, hidden_size=32, max_len=40, steps=4, eval_every=2, batch_size=8)


# ---------------------------------------------------------------------------
# masked_cross_entropy
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_masked_cross_entropy_smoke():
    logits = torch.randn(2, 3, 22)
    targets = torch.randint(0, 22, (2, 3))
    mask = torch.ones(2, 3)
    loss = masked_cross_entropy(logits, targets, mask)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0


# ----- Correctness tests -----


def test_masked_cross_entropy_ignores_masked_positions():
    pass


# ---------------------------------------------------------------------------
# train_step / evaluate
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_train_step_smoke():
    model = Seq2SeqRNN(CFG)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = make_batch("copy", 8, seed=0, framing="encoder_decoder", n=5)
    loss = train_step(model, batch, opt, grad_clip=1.0)
    assert isinstance(loss, float)


# ----- Correctness tests -----


def test_train_step_updates_parameters():
    pass


def test_evaluate_does_not_update_parameters():
    pass


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


# ----- Correctness tests -----


def test_token_accuracy_perfect_and_masked():
    pass


def test_attention_entropy_uniform_is_max():
    pass


def test_attention_entropy_per_head_shape():
    pass


# ---------------------------------------------------------------------------
# run_battery
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_run_battery_smoke():
    sb = run_battery(lambda c: Seq2SeqRNN(c), CFG)
    assert sb["framing"] == "encoder_decoder"
    assert set(sb["rows"]) == {"copy", "reverse", "sort", "recall", "selective_copy"}


# ----- Correctness tests -----


def test_run_battery_rows_have_metrics():
    pass

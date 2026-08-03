"""Tests for linear (kernel) attention (Task I)."""

import torch

from attention.mechanisms.linear import LinearAttention
from attention.utils import seed_everything


# ---------------------------------------------------------------------------
# LinearAttention
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_linear_smoke():
    la = LinearAttention(32, 4)
    x = torch.randn(3, 6, 32)
    assert la(x, x, x).shape == (3, 6, 32)


def test_linear_causal_runs():
    pass


def test_linear_long_sequence_runs():
    pass


# ----- Correctness tests -----


def test_linear_normalizer_makes_equal_values_position_invariant():
    pass


def test_linear_causal_matches_noncausal_at_final_position():
    pass


def test_linear_causal_ignores_future():
    pass


def test_linear_gradients_flow():
    pass

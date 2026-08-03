"""Tests for positional encodings: sinusoidal PE and ALiBi (Task F)."""

import torch

from attention.mechanisms.positional import (
    alibi_bias,
    alibi_slopes,
    sinusoidal_encoding,
)


# ---------------------------------------------------------------------------
# sinusoidal_encoding
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_sinusoidal_smoke():
    pe = sinusoidal_encoding(16, 64)
    assert isinstance(pe, torch.Tensor) and pe.shape == (16, 64)


# ----- Correctness tests -----


def test_sinusoidal_known_entries():
    pass


def test_sinusoidal_values_bounded():
    pass


# ---------------------------------------------------------------------------
# alibi_slopes
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_alibi_slopes_smoke():
    slopes = alibi_slopes(8)
    assert isinstance(slopes, torch.Tensor) and slopes.shape == (8,)


# ----- Correctness tests -----


def test_alibi_slopes_geometric_for_eight_heads():
    pass


# ---------------------------------------------------------------------------
# alibi_bias
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_alibi_bias_smoke():
    bias = alibi_bias(16, 8)
    assert isinstance(bias, torch.Tensor) and bias.shape == (8, 16, 16)


# ----- Correctness tests -----


def test_alibi_bias_causal_future_is_neg_inf():
    pass


def test_alibi_bias_grows_linearly_with_distance():
    pass


def test_alibi_bias_diagonal_is_zero():
    pass

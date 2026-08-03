"""Tests for scaled dot-product attention (Task D)."""

import math

import torch

from attention.mechanisms.scaled_dot_product import (
    scaled_dot_product_attention as sdpa,
)
from attention.utils import seed_everything


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_sdpa_smoke():
    q = torch.randn(2, 5, 8)
    out, weights = sdpa(q, q, q)
    assert isinstance(out, torch.Tensor) and isinstance(weights, torch.Tensor)
    assert out.shape == (2, 5, 8) and weights.shape == (2, 5, 5)


def test_sdpa_runs_with_mask_and_bias():
    pass


# ----- Correctness tests -----


def test_sdpa_weights_sum_to_one():
    pass


def test_sdpa_causal_mask_zeroes_future():
    pass


def test_sdpa_permutation_equivariant():
    pass


def test_sdpa_scale_default_is_inverse_sqrt_dk():
    pass

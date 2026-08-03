"""Tests for multi-head attention (Task E)."""

import torch

from attention.mechanisms.multihead import MultiHeadAttention
from attention.utils import seed_everything


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_mha_self_smoke():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(3, 6, 32)
    out = mha(x, x, x)
    assert isinstance(out, torch.Tensor) and out.shape == (3, 6, 32)


def test_mha_cross_smoke():
    mha = MultiHeadAttention(32, 4)
    q = torch.randn(3, 4, 32)
    kv = torch.randn(3, 6, 32)
    assert mha(q, kv, kv).shape == (3, 4, 32)


def test_mha_causal_mask_runs():
    pass


# ----- Correctness tests -----


def test_mha_shapes_across_num_heads():
    pass


def test_mha_permutation_equivariant():
    pass


def test_mha_mask_respected():
    pass


def test_mha_cross_depends_on_kv():
    pass


def test_mha_gradients_flow():
    pass

"""Tests for the decoder-only transformer + causal mask (Task G)."""

import torch

from attention.config import ExperimentConfig
from attention.data import make_batch
from attention.masking import causal_mask
from attention.models.transformer import (
    DecoderOnlyTransformer,
    TransformerBlock,
)
from attention.utils import seed_everything

CFG = ExperimentConfig(d_model=32, num_heads=4, num_layers=2, d_ff=64, max_len=40)


# ---------------------------------------------------------------------------
# causal_mask
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_causal_mask_smoke():
    m = causal_mask(5)
    assert isinstance(m, torch.Tensor) and m.shape == (5, 5)


# ----- Correctness tests -----


def test_causal_mask_blocks_the_future_only():
    pass


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_block_smoke():
    block = TransformerBlock(CFG)
    x = torch.randn(3, 6, CFG.d_model)
    assert block(x).shape == (3, 6, CFG.d_model)


# ----- Correctness tests -----


def test_block_has_residual_passthrough():
    pass


def test_block_respects_causal_mask():
    pass


# ---------------------------------------------------------------------------
# DecoderOnlyTransformer
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_decoder_only_smoke():
    model = DecoderOnlyTransformer(CFG)
    batch = make_batch("copy", 3, seed=0, framing="single_stream", n=6)
    logits = model(batch)
    assert logits.shape[0] == 3 and logits.shape[2] == CFG.vocab_size


# ----- Correctness tests -----


def test_decoder_only_is_causal_with_correct_mask():
    pass

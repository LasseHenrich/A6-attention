"""Tests for the encoder-decoder transformer + cross-attention (Task H)."""

import torch

from attention.config import ExperimentConfig
from attention.data import make_batch
from attention.models.transformer import EncoderDecoderTransformer
from attention.utils import seed_everything

CFG = ExperimentConfig(d_model=32, num_heads=4, num_layers=2, d_ff=64, max_len=40)


# ---------------------------------------------------------------------------
# EncoderDecoderTransformer
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_encdec_smoke():
    model = EncoderDecoderTransformer(CFG)
    batch = make_batch("copy", 3, seed=0, framing="encoder_decoder", n=6)
    logits = model(batch)
    assert logits.shape[0] == 3 and logits.shape[2] == CFG.vocab_size


def test_encode_returns_memory():
    pass


# ----- Correctness tests -----


def test_decoder_side_is_causal():
    pass


def test_cross_attention_depends_on_source():
    pass


def test_encoder_is_bidirectional():
    pass

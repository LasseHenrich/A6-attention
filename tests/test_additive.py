"""Tests for additive (Bahdanau) attention (Task C)."""

import torch

from attention.config import ExperimentConfig
from attention.data import make_batch
from attention.mechanisms.additive import AdditiveAttention
from attention.models.rnn import Decoder, Seq2SeqRNN
from attention.utils import seed_everything

CFG = ExperimentConfig(d_model=32, hidden_size=32, attn_dim=24, max_len=40)


# ---------------------------------------------------------------------------
# AdditiveAttention
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_additive_smoke():
    attn = AdditiveAttention(32, 32, 24)
    context, weights = attn(torch.randn(4, 32), torch.randn(4, 6, 32))
    assert isinstance(context, torch.Tensor) and isinstance(weights, torch.Tensor)
    assert context.shape == (4, 32) and weights.shape == (4, 6)


# ----- Correctness tests -----


def test_additive_weights_are_a_distribution():
    pass


def test_additive_context_is_weighted_encoder_states():
    pass


def test_additive_masks_pad_positions():
    pass


def test_additive_gradients_flow():
    pass


# ---------------------------------------------------------------------------
# Decoder hook + Seq2SeqRNN factory
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_seq2seq_with_attention_smoke():
    attn = AdditiveAttention(CFG.hidden_size, CFG.hidden_size, CFG.attn_dim)
    model = Seq2SeqRNN(CFG, attention=attn)
    batch = make_batch("copy", 4, seed=0, framing="encoder_decoder", n=5)
    logits = model(batch)
    assert logits.shape[0] == 4 and logits.shape[2] == CFG.vocab_size


# ----- Correctness tests -----


def test_decoder_now_depends_on_enc_states():
    pass


def test_decoder_with_attention_still_causal():
    pass

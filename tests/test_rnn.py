"""Tests for the RNN encoder-decoder (Task A; hook re-checked in Task C)."""

import torch

from attention.config import ExperimentConfig
from attention.data import make_batch
from attention.models.rnn import Decoder, Encoder, RNNCell, Seq2SeqRNN
from attention.utils import seed_everything

CFG = ExperimentConfig(d_model=32, hidden_size=32, max_len=40)


# ---------------------------------------------------------------------------
# RNNCell
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_rnncell_smoke():
    cell = RNNCell(4, 6)
    out = cell(torch.randn(3, 4), torch.zeros(3, 6))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 6)


# ----- Correctness tests -----


def test_rnncell_output_in_tanh_range():
    pass


def test_rnncell_depends_on_both_inputs():
    pass


def test_rnncell_gradients_flow():
    pass


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_encoder_smoke():
    enc = Encoder(CFG)
    all_h, final = enc(torch.randint(6, 22, (2, 5)))
    assert isinstance(all_h, torch.Tensor) and isinstance(final, torch.Tensor)
    assert all_h.shape == (2, 5, CFG.hidden_size)
    assert final.shape == (2, CFG.hidden_size)


# ----- Correctness tests -----


def test_encoder_final_state_is_last_hidden():
    pass


def test_encoder_recurrence_is_causal():
    pass


# ---------------------------------------------------------------------------
# Decoder (attention None)
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_decoder_smoke():
    dec = Decoder(CFG, attention=None)
    logits = dec(
        torch.randint(6, 22, (2, 4)),
        torch.randn(2, 5, CFG.hidden_size),
        torch.randn(2, CFG.hidden_size),
    )
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 4, CFG.vocab_size)


# ----- Correctness tests -----


def test_decoder_invariant_to_enc_states():
    pass


def test_decoder_teacher_forcing_is_causal():
    pass


# ---------------------------------------------------------------------------
# Seq2SeqRNN
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_seq2seq_smoke():
    model = Seq2SeqRNN(CFG)
    batch = make_batch("copy", 4, seed=0, framing="encoder_decoder", n=5)
    logits = model(batch)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == 4 and logits.shape[2] == CFG.vocab_size


def test_seq2seq_is_differentiable():
    pass


# ----- Correctness tests -----


def test_seq2seq_depends_on_source():
    pass

"""Tests for the synthetic battery generators and framing (provided infra)."""

import random

import torch

from attention.data import (
    BatchGenerator,
    copy_task,
    make_batch,
    recall_task,
    reverse_task,
    selective_copy_task,
    sort_task,
)
from attention.vocab import BOS, CONTENT, EOS, QUERY, SEP


# ---------------------------------------------------------------------------
# Per-sample generators
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_generators_smoke():
    rng = random.Random(0)
    for fn in (copy_task, reverse_task, sort_task):
        x, y = fn(5, rng=rng)
        assert isinstance(x, list) and isinstance(y, list)
    x, y = recall_task(3, rng=rng)
    assert isinstance(x, list) and isinstance(y, list)
    x, y = selective_copy_task(8, 2, rng=rng)
    assert isinstance(x, list) and isinstance(y, list)


# ----- Correctness tests -----


def test_copy_is_identity():
    pass


def test_reverse_reverses():
    pass


def test_sort_sorts():
    pass


def test_recall_returns_queried_value():
    pass


def test_recall_keys_are_distinct():
    pass


def test_selective_copy_targets_follow_delims():
    pass


# ---------------------------------------------------------------------------
# Framing + BatchGenerator
# ---------------------------------------------------------------------------


# ----- Smoke tests -----


def test_make_batch_single_stream_smoke():
    b = make_batch("copy", 4, seed=0, framing="single_stream", n=5)
    assert b.input_ids is not None
    assert b.input_ids.shape[0] == 4
    assert b.targets.shape == b.input_ids.shape


def test_make_batch_encoder_decoder_smoke():
    b = make_batch("copy", 4, seed=0, framing="encoder_decoder", n=5)
    assert b.source is not None and b.target_in is not None
    assert b.source_padding_mask is not None


# ----- Correctness tests -----


def test_single_stream_loss_region_is_output_only():
    pass


def test_batch_stream_is_deterministic_per_seed():
    pass


def test_framing_is_data_consistent_across_models():
    # Same seed -> same underlying (x, y) regardless of framing choice.
    pass


def test_batch_to_moves_tensors():
    pass

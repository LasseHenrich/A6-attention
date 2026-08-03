"""Task A demo: a forward / shape / parameter walk of the RNN.

There is no training in Task A — this prints the shapes flowing through the
encoder-decoder (source -> enc_states / enc_final -> decoder -> logits) and
the model's parameter count, so you can fill in ``a_observations`` and
``a_param_count`` in ``answers.py``.

Provided — students run this; they do not edit it.
"""

from __future__ import annotations

import argparse

import torch

from attention.config import ExperimentConfig
from attention.data import make_batch
from attention.metrics import param_count
from attention.models import Seq2SeqRNN


def walk(cfg: ExperimentConfig) -> None:
    model = Seq2SeqRNN(cfg)
    model.eval()
    batch = make_batch("copy", 4, seed=0, framing="encoder_decoder", n=6)

    print("=== Task A — RNN forward / shape / parameter walk ===")
    print(f"config: d_model={cfg.d_model} hidden_size={cfg.hidden_size} vocab_size={cfg.vocab_size}")
    print()
    print(f"source           : {tuple(batch.source.shape)}  [B, S]")
    with torch.no_grad():
        enc_states, enc_final = model.encoder(batch.source)
        print(f"enc_states       : {tuple(enc_states.shape)}  [B, S, H]")
        print(f"enc_final        : {tuple(enc_final.shape)}  [B, H]  (the whole source squeezed into one vector — the bottleneck)")
        logits = model(batch)
        print(f"target_in        : {tuple(batch.target_in.shape)}  [B, T]")
        print(f"logits           : {tuple(logits.shape)}  [B, T, VOCAB]")
    print()
    print(f"parameter count  : {param_count(model)}")
    print()
    print("Note: the model is UNTRAINED, so any accuracy is at chance.")
    print("Record a_observations and a_param_count in answers.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task A RNN shape/param walk")
    parser.add_argument("--part", default="all", help="unused; for uniformity")
    parser.parse_args()
    walk(ExperimentConfig())


if __name__ == "__main__":
    main()

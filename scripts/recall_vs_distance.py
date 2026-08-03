"""Task C demo: additive attention vs the plain RNN.

Three parts (``--part``):
  recall   — associative-recall accuracy vs needle distance, RNN vs +Bahdanau
  serial   — forward wall-clock vs sequence length (still serial)
  gradient — gradient norm reaching each source position (still vanishing)

Writes the series to the run directory for the plotter.  Timing is
machine-relative (single-thread).  Provided — students run it; do not edit.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from attention.config import ExperimentConfig
from attention.data import BatchGenerator, make_batch
from attention.harness import _EVAL_BATCH
from attention.mechanisms.additive import AdditiveAttention
from attention.metrics import token_accuracy
from attention.models import Seq2SeqRNN
from attention.run import init_run, write_json
from attention.train import train
from attention.utils import seed_everything, timeit, use_single_thread

_RESULTS = Path(__file__).resolve().parents[1] / "results"


def _make(cfg: ExperimentConfig, use_attention: bool) -> Seq2SeqRNN:
    if use_attention:
        attn = AdditiveAttention(cfg.hidden_size, cfg.hidden_size, cfg.attn_dim)
        return Seq2SeqRNN(cfg, attention=attn)
    return Seq2SeqRNN(cfg)


def recall_sweep(cfg: ExperimentConfig, num_pairs: int, steps: int) -> dict[str, list[float]]:
    distances = list(range(num_pairs))
    out: dict[str, list[float]] = {"needle_distance": [float(d) for d in distances]}
    for use_attention, label in ((False, "rnn"), (True, "rnn_bahdanau")):
        seed_everything(cfg.init_seed)
        model = _make(cfg, use_attention)
        gen = BatchGenerator(
            "recall",
            seed=cfg.data_seed,
            framing="encoder_decoder",
            num_pairs=num_pairs,
        )
        train(model, dataclasses.replace(cfg, steps=steps), gen, {})
        accs: list[float] = []
        for d in distances:
            ev = make_batch(
                "recall",
                _EVAL_BATCH,
                seed=800_000 + d,
                framing="encoder_decoder",
                num_pairs=num_pairs,
                needle_distance=d,
            )
            model.eval()
            with torch.no_grad():
                accs.append(token_accuracy(model(ev), ev.targets, ev.loss_mask))
        out[label] = accs
    return out


def serial_timing(cfg: ExperimentConfig, lengths: list[int]) -> dict[str, list[float]]:
    use_single_thread()
    model = _make(cfg, True)
    model.eval()
    times: list[float] = []
    for n in lengths:
        batch = make_batch("copy", 8, seed=0, framing="encoder_decoder", n=n)
        with torch.no_grad():
            t = timeit(lambda: model(batch), warmup=2, n=5)
        times.append(t.median)
    return {"length": [float(n) for n in lengths], "forward_seconds": times}


def gradient_probe(cfg: ExperimentConfig, n: int) -> dict[str, list[float]]:
    """Grad norm of the loss w.r.t. each source embedding position (BPTT)."""
    from attention.train import masked_cross_entropy

    seed_everything(cfg.init_seed)
    model = _make(cfg, True)
    batch = make_batch("copy", 16, seed=0, framing="encoder_decoder", n=n)
    emb = model.encoder.embed(batch.source).detach().requires_grad_(True)

    # Re-run the encoder from the captured embeddings so we can read grads.
    h = emb.new_zeros(emb.shape[0], cfg.hidden_size)
    states = []
    for t in range(emb.shape[1]):
        h = model.encoder.cell(emb[:, t], h)
        states.append(h)
    enc_states = torch.stack(states, dim=1)
    logits = model.decoder(batch.target_in, enc_states, h, src_mask=batch.source_padding_mask)
    masked_cross_entropy(logits, batch.targets, batch.loss_mask).backward()
    per_pos = emb.grad.norm(dim=-1).mean(dim=0)  # [S]
    return {
        "source_position": [float(i) for i in range(emb.shape[1])],
        "grad_norm": per_pos.tolist(),
    }


def run(part: str, cfg: ExperimentConfig, steps: int) -> None:
    run_dir = init_run(cfg, _RESULTS, tag="bahdanau")
    if part in ("recall", "all"):
        data = recall_sweep(cfg, num_pairs=16, steps=steps)
        write_json(data, run_dir / "recall_vs_distance.json")
        print("recall vs distance:", data)
    if part in ("serial", "all"):
        data = serial_timing(cfg, [8, 16, 32, 48, 64])
        write_json(data, run_dir / "serial_timing.json")
        print("serial timing:", data)
    if part in ("gradient", "all"):
        data = gradient_probe(cfg, n=32)
        write_json(data, run_dir / "gradient_probe.json")
        print("gradient probe:", data)
    print(f"\nWritten to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task C Bahdanau demos")
    parser.add_argument(
        "--part",
        default="all",
        choices=["all", "recall", "serial", "gradient"],
    )
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    run(args.part, ExperimentConfig(), args.steps)


if __name__ == "__main__":
    main()

"""Task I demo: the efficiency frontier of linear attention.

Three parts (``--part``):
  cost   — forward wall-clock vs sequence length, softmax vs linear (forward
           only, up to ~2048).  Softmax grows quadratically; linear linearly.
  snr    — training-free retrieval signal-to-noise of the reordered state as
           the number of stored pairs grows.  Explains why recall degrades.
  recall — associative-recall accuracy vs stored pairs, softmax vs linear.

Timing is machine-relative (single thread, warmup discarded).  Provided —
students run it; do not edit.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from attention.config import ExperimentConfig
from attention.data import BatchGenerator, make_batch
from attention.harness import _EVAL_BATCH
from attention.mechanisms.linear import LinearAttention, _phi
from attention.mechanisms.scaled_dot_product import (
    scaled_dot_product_attention as sdpa,
)
from attention.metrics import token_accuracy
from attention.models.wrappers import SingleLayerModel
from attention.run import init_run, write_json
from attention.train import train
from attention.utils import seed_everything, timeit, use_single_thread

_RESULTS = Path(__file__).resolve().parents[1] / "results"


def cost_vs_length(lengths: list[int]) -> dict:
    use_single_thread()
    la = LinearAttention(64, 4).eval()
    softmax_t, linear_t = [], []
    for n in lengths:
        x = torch.randn(1, n, 64)

        def _soft() -> None:
            with torch.no_grad():
                sdpa(x, x, x)

        def _lin() -> None:
            with torch.no_grad():
                la(x, x, x)

        softmax_t.append(timeit(_soft, warmup=1, n=3).median)
        linear_t.append(timeit(_lin, warmup=1, n=3).median)
    return {
        "length": [float(n) for n in lengths],
        "softmax_seconds": softmax_t,
        "linear_seconds": linear_t,
    }


def retrieval_snr(num_pairs_grid: list[int], d: int = 64) -> dict:
    """Training-free SNR of retrieving stored values from the linear state."""
    snr = []
    for p in num_pairs_grid:
        seed_everything(p)
        keys = torch.randn(p, d)
        values = torch.randn(p, d)
        fk = _phi(keys)  # [P, d]
        state = fk.t() @ values  # Σ φ(k) vᵀ  [d, d]
        z = fk.sum(dim=0)  # [d]
        ratios = []
        for i in range(p):
            num = fk[i] @ state
            den = fk[i] @ z + 1e-6
            v_hat = num / den
            signal = values[i].norm()
            noise = (v_hat - values[i]).norm() + 1e-6
            ratios.append(float(signal / noise))
        snr.append(sum(ratios) / len(ratios))
    return {"num_pairs": [float(p) for p in num_pairs_grid], "snr": snr}


def recall_vs_load(cfg: ExperimentConfig, loads: list[int], steps: int) -> dict:
    out: dict[str, object] = {"num_pairs": [float(p) for p in loads]}
    for label, mech_fn in (
        ("softmax", lambda c: SingleLayerModel(sdpa, c, positional="learned")),
        ("linear", lambda c: SingleLayerModel(LinearAttention(c.d_model, c.num_heads), c, positional="learned")),
    ):
        accs = []
        for p in loads:
            seed_everything(cfg.init_seed)
            model = mech_fn(cfg)
            gen = BatchGenerator("recall", seed=cfg.data_seed, framing="single_stream", num_pairs=p)
            train(model, dataclasses.replace(cfg, steps=steps), gen, {})
            ev = make_batch("recall", _EVAL_BATCH, seed=930_000 + p, framing="single_stream", num_pairs=p)
            model.eval()
            with torch.no_grad():
                accs.append(token_accuracy(model(ev), ev.targets, ev.loss_mask))
        out[label] = accs
    return out


def run(part: str, cfg: ExperimentConfig, steps: int) -> None:
    run_dir = init_run(cfg, _RESULTS, tag="linear")
    data: dict[str, object] = {}
    if part in ("all", "cost"):
        data["cost"] = cost_vs_length(list(cfg.bench_lengths))
        print("cost vs length:", data["cost"])
    if part in ("all", "snr"):
        data["snr"] = retrieval_snr([2, 4, 8, 12, 16])
        print("retrieval SNR:", data["snr"])
    if part in ("all", "recall"):
        data["recall"] = recall_vs_load(cfg, [2, 4, 8, 12], steps)
        print("recall vs load:", data["recall"])
    write_json(data, run_dir / "linear_benchmark.json")
    print(f"\nWritten to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task I linear benchmark")
    parser.add_argument("--part", default="all", choices=["all", "cost", "snr", "recall"])
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    run(args.part, ExperimentConfig(), args.steps)


if __name__ == "__main__":
    main()

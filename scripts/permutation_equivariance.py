"""Task D demo: permutation equivariance (the provable order-blindness).

Two views:
  1. Raw-op property — permuting the input rows of a position-free, unmasked
     self-attention permutes the output rows identically (equivariance).
  2. Position-free probe — a ``SingleLayerModel`` with ``causal=False,
     positional=None`` trained on Reverse/Sort sits near chance, because a set
     operation cannot recover order.

Provided — students run it; do not edit.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from attention.config import ExperimentConfig
from attention.data import BatchGenerator, make_batch
from attention.harness import _EVAL_BATCH
from attention.mechanisms.scaled_dot_product import (
    scaled_dot_product_attention as sdpa,
)
from attention.metrics import token_accuracy
from attention.models.wrappers import SingleLayerModel
from attention.run import init_run, write_json
from attention.train import train
from attention.utils import seed_everything

_RESULTS = Path(__file__).resolve().parents[1] / "results"


def equivariance_gap() -> float:
    """Max abs difference between attn(Px) and P·attn(x) (should be ~0)."""
    seed_everything(0)
    x = torch.randn(4, 10, 16)
    perm = torch.randperm(10)
    base, _ = sdpa(x, x, x)
    permd, _ = sdpa(x[:, perm], x[:, perm], x[:, perm])
    return float((permd - base[:, perm]).abs().max().item())


def position_free_probe(cfg: ExperimentConfig, steps: int) -> dict[str, float]:
    accs: dict[str, float] = {}
    for task in ("reverse", "sort"):
        seed_everything(cfg.init_seed)
        model = SingleLayerModel(sdpa, cfg, causal=False, positional=None)
        gen = BatchGenerator(task, seed=cfg.data_seed, framing="single_stream", n_min=4, n_max=8)
        train(model, dataclasses.replace(cfg, steps=steps), gen, {})
        ev = make_batch(task, _EVAL_BATCH, seed=700_000, framing="single_stream", n_min=4, n_max=8)
        model.eval()
        with torch.no_grad():
            accs[task] = token_accuracy(model(ev), ev.targets, ev.loss_mask)
    return accs


def run(part: str, cfg: ExperimentConfig, steps: int) -> None:
    run_dir = init_run(cfg, _RESULTS, tag="equivariance")
    data: dict[str, object] = {"equivariance_gap": equivariance_gap()}
    print(f"equivariance gap (||attn(Px) - P·attn(x)||_max): {data['equivariance_gap']:.2e}  (≈ 0 ⇒ equivariant)")
    if part in ("all", "probe"):
        data["position_free_accuracy"] = position_free_probe(cfg, steps)
        print(f"position-free probe accuracy: {data['position_free_accuracy']}")
    write_json(data, run_dir / "permutation_equivariance.json")
    print(f"\nWritten to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task D equivariance demo")
    parser.add_argument("--part", default="all", choices=["all", "op", "probe"])
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    run(args.part, ExperimentConfig(), args.steps)


if __name__ == "__main__":
    main()

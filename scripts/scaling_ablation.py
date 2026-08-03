"""Task D demo: the 1/sqrt(d_k) scaling ablation.

Sweeps the key dimension d_k and reports, for the scaled (``scale=None``) vs
unscaled (``scale=1.0``) attention op: the variance of the pre-softmax
logits, the softmax max-probability (saturation), and the gradient norm
reaching the queries.  Unscaled scores grow with d_k, saturate the softmax,
and collapse the gradient — the empirical scaling fault.  Forward + one
backward only; no training.

Provided — students run it; do not edit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from attention.config import ExperimentConfig
from attention.mechanisms.scaled_dot_product import (
    scaled_dot_product_attention as sdpa,
)
from attention.run import init_run, write_json
from attention.utils import seed_everything

_RESULTS = Path(__file__).resolve().parents[1] / "results"


def _scores(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    return (q @ k.transpose(-2, -1)) * scale


def ablation(d_ks: list[int], batch: int = 64, length: int = 16) -> dict:
    import math

    out: dict[str, list[float]] = {"d_k": [float(d) for d in d_ks]}
    for arm in ("scaled", "unscaled"):
        logit_var, sm_max, grad_norm = [], [], []
        for d in d_ks:
            seed_everything(d)
            q = torch.randn(batch, length, d, requires_grad=True)
            k = torch.randn(batch, length, d)
            v = torch.randn(batch, length, d)
            scale = (1.0 / math.sqrt(d)) if arm == "scaled" else 1.0
            output, weights = sdpa(q, k, v, scale=scale)
            logit_var.append(float(_scores(q, k, scale).var().item()))
            sm_max.append(float(weights.max(dim=-1).values.mean().item()))
            output.sum().backward()
            grad_norm.append(float(q.grad.norm().item()))
        out[f"logit_var_{arm}"] = logit_var
        out[f"softmax_max_{arm}"] = sm_max
        out[f"grad_norm_{arm}"] = grad_norm
    return out


def run(cfg: ExperimentConfig) -> None:
    run_dir = init_run(cfg, _RESULTS, tag="scaling")
    data = ablation([8, 16, 32, 64, 128, 256])
    write_json(data, run_dir / "scaling_ablation.json")
    print("scaling ablation:")
    for i, d in enumerate(data["d_k"]):
        print(
            f"  d_k={int(d):4d}  unscaled softmax_max={data['softmax_max_unscaled'][i]:.3f}  scaled softmax_max={data['softmax_max_scaled'][i]:.3f}"
        )
    print(f"\nWritten to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task D scaling ablation")
    parser.add_argument("--part", default="all", help="unused; uniformity")
    parser.parse_args()
    run(ExperimentConfig())


if __name__ == "__main__":
    main()

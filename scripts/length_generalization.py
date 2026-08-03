"""Task F demo: length generalization and ALiBi's locality trade-off.

Two parts (``--part``):
  length   — train Copy at n<=32 under each of {none, learned, sinusoidal,
             alibi}, evaluate at longer lengths.  Learned absolute PE cliffs;
             relative schemes extrapolate.
  locality — mean attention weight by query-key distance, ALiBi vs sinusoidal.
             ALiBi's penalty concentrates weight nearby and starves distant
             positions (a steeper decay) — the locality prior itself.

Provided — students run it; do not edit.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from attention.config import ExperimentConfig
from attention.data import BatchGenerator, make_batch
from attention.mechanisms.multihead import MultiHeadAttention
from attention.mechanisms.scaled_dot_product import (
    scaled_dot_product_attention as sdpa,
)
from attention.metrics import token_accuracy
from attention.models.wrappers import SingleLayerModel
from attention.run import init_run, write_json
from attention.train import train
from attention.utils import seed_everything

_RESULTS = Path(__file__).resolve().parents[1] / "results"
_SCHEMES = ["none", "learned", "sinusoidal", "alibi"]


def _probe(cfg: ExperimentConfig, scheme: str) -> SingleLayerModel:
    pos = None if scheme == "none" else scheme
    return SingleLayerModel(sdpa, cfg, causal=True, positional=pos)


def _mh_probe(cfg: ExperimentConfig, scheme: str) -> SingleLayerModel:
    """Multi-head probe — ALiBi's per-head slopes are only meaningful with
    several heads (a single head's slope is ``2**-8``, i.e. no locality)."""
    pos = None if scheme == "none" else scheme
    attn = MultiHeadAttention(cfg.d_model, cfg.num_heads)
    return SingleLayerModel(attn, cfg, causal=True, positional=pos)


def length_split(cfg: ExperimentConfig, steps: int) -> dict:
    eval_lengths = [16, 32, 64, 128]
    out: dict[str, object] = {"eval_length": [float(n) for n in eval_lengths]}
    for scheme in _SCHEMES:
        seed_everything(cfg.init_seed)
        model = _probe(cfg, scheme)
        gen = BatchGenerator("copy", seed=cfg.data_seed, framing="single_stream", n_min=4, n_max=32)
        train(model, dataclasses.replace(cfg, steps=steps), gen, {})
        accs = []
        for n in eval_lengths:
            ev = make_batch("copy", 64, seed=600_000 + n, framing="single_stream", n=n)
            model.eval()
            with torch.no_grad():
                accs.append(token_accuracy(model(ev), ev.targets, ev.loss_mask))
        out[scheme] = accs
    return out


def locality(cfg: ExperimentConfig, num_pairs: int, steps: int) -> dict:
    """ALiBi's locality prior, read straight off the attention weights.

    Both schemes are trained on the same recall distribution; we then average
    each model's attention weight by query-key distance ``|i - j|``.  ALiBi's
    linear penalty concentrates weight on nearby positions and starves distant
    ones — a far steeper decay than sinusoidal — and that locality is what
    makes long-range lookups (distant recall) a liability.

    We measure the weights directly rather than end-task recall accuracy: with
    ALiBi's geometric per-head slopes the model can route retrieval through its
    near-zero-slope heads, so trained accuracy barely moves while the prior is
    plainly visible in the attention distribution.
    """
    ev = make_batch("recall", 128, seed=650_000, framing="single_stream", num_pairs=num_pairs)
    seq_len = ev.input_ids.shape[1]
    max_d = min(seq_len - 1, 24)
    out: dict[str, object] = {"distance": [float(d) for d in range(max_d + 1)]}
    for scheme in ("sinusoidal", "alibi"):
        seed_everything(cfg.init_seed)
        model = _mh_probe(cfg, scheme)
        gen = BatchGenerator("recall", seed=cfg.data_seed, framing="single_stream", num_pairs=num_pairs)
        train(model, dataclasses.replace(cfg, steps=steps), gen, {})
        model.eval()
        with torch.no_grad():
            model(ev)
        weights = model.last_attn
        assert weights is not None, "multi-head probe must stash per-head weights"
        by_pair = weights.mean(dim=(0, 1))  # [L, L]: mean weight over batch and heads
        total = torch.zeros(max_d + 1)
        count = torch.zeros(max_d + 1)
        for i in range(seq_len):
            for j in range(i + 1):
                d = i - j
                if d <= max_d:
                    total[d] += by_pair[i, j]
                    count[d] += 1
        out[scheme] = (total / count.clamp(min=1)).tolist()
    return out


def run(part: str, cfg: ExperimentConfig, steps: int) -> None:
    run_dir = init_run(cfg, _RESULTS, tag="lengthgen")
    if part in ("all", "length"):
        data = length_split(cfg, steps)
        write_json(data, run_dir / "length_generalization.json")
        print("length generalization (accuracy vs eval length):")
        for s in _SCHEMES:
            print(f"  {s:11s} {data[s]}")
    if part in ("all", "locality"):
        data = locality(cfg, num_pairs=16, steps=steps)
        write_json(data, run_dir / "alibi_locality.json")
        print("ALiBi-locality (mean attention weight vs query-key distance):")
        for scheme in ("sinusoidal", "alibi"):
            print(f"  {scheme:11s} {data[scheme]}")
    print(f"\nWritten to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task F length-gen demo")
    parser.add_argument("--part", default="all", choices=["all", "length", "locality"])
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    run(args.part, ExperimentConfig(), args.steps)


if __name__ == "__main__":
    main()

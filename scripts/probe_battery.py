"""Tasks D/E demo: self-attention / multi-head battery rows.

Runs the shared battery through the provided ``SingleLayerModel`` probe at the
comparable config (``causal=True, positional=None``), producing scoreboard
**row #3** (scaled dot-product self-attention, Task D) and **row #4**
(multi-head self-attention, Task E) — directly comparable to the RNN (B) and
RNN+Bahdanau (C) rows, so each successor visibly earns its keep row-over-row.

``--mechanism self-attn`` scores bare SDPA (the wrapper supplies single-head
projections); ``--mechanism multihead`` scores ``MultiHeadAttention``.

Provided — students run this; they do not edit it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from attention.config import ExperimentConfig
from attention.harness import run_battery
from attention.mechanisms.multihead import MultiHeadAttention
from attention.mechanisms.scaled_dot_product import (
    scaled_dot_product_attention as sdpa,
)
from attention.models.wrappers import SingleLayerModel
from attention.run import init_run, write_metrics, write_scoreboard
from attention.utils import use_single_thread

_RESULTS = Path(__file__).resolve().parents[1] / "results"

# tag -> human label (tag also names the run dir and the heatmap row).
_MECHANISMS = {
    "self-attn": "scaled dot-product self-attention",
    "multihead": "multi-head self-attention",
}


def build_factory(mechanism: str):
    def make(cfg: ExperimentConfig) -> SingleLayerModel:
        if mechanism == "multihead":
            attn = MultiHeadAttention(cfg.d_model, cfg.num_heads)
            return SingleLayerModel(attn, cfg, causal=True, positional=None)
        return SingleLayerModel(sdpa, cfg, causal=True, positional=None)

    return make


def run(cfg: ExperimentConfig, mechanism: str) -> dict:
    # Pin to one thread: these batch-64, ~20-token matmuls are far smaller than
    # the multi-threaded pool's dispatch overhead, so one thread runs faster and
    # more reproducibly (same reason sort_gap.py / two_relation.py pin here).
    use_single_thread()
    scoreboard = run_battery(build_factory(mechanism), cfg)
    run_dir = init_run(cfg, _RESULTS, tag=mechanism)
    write_metrics(run_dir, scoreboard.pop("histories"))
    write_scoreboard(run_dir, scoreboard)

    print(f"=== {_MECHANISMS[mechanism]} — battery scoreboard ===")
    for task, row in scoreboard["rows"].items():
        print(f"  {task:16s} acc={row['accuracy']:.3f}  params={row['param_count']}")
    print(f"\nScoreboard written to {run_dir}")
    return scoreboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-attention / multi-head battery row")
    parser.add_argument(
        "--mechanism",
        choices=sorted(_MECHANISMS),
        default="self-attn",
        help="which probe mechanism to score on the battery",
    )
    parser.add_argument("--steps", type=int, default=None, help="override cfg.steps")
    parser.add_argument("--part", default="all", help="unused; uniformity")
    args = parser.parse_args()

    overrides = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    cfg = ExperimentConfig(**overrides)
    run(cfg, args.mechanism)


if __name__ == "__main__":
    main()

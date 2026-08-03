"""Task B/C demo: train the RNN across the battery (scoreboard rows #1, #2).

With the repaired training loop this trains a fresh ``Seq2SeqRNN`` on each
battery task and prints the token-accuracy scoreboard — the RNN baseline
(battery row #1).  Pass ``--attention`` (Task C) to add Bahdanau attention to
the decoder and produce row #2.

Provided — students run this; they do not edit it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from attention.config import ExperimentConfig
from attention.harness import run_battery
from attention.models import Seq2SeqRNN
from attention.run import init_run, write_metrics, write_scoreboard
from attention.utils import use_single_thread

_RESULTS = Path(__file__).resolve().parents[1] / "results"


def build_factory(use_attention: bool):
    def make(cfg: ExperimentConfig) -> Seq2SeqRNN:
        if use_attention:
            from attention.mechanisms.additive import AdditiveAttention

            attn = AdditiveAttention(cfg.hidden_size, cfg.hidden_size, cfg.attn_dim)
            return Seq2SeqRNN(cfg, attention=attn)
        return Seq2SeqRNN(cfg)

    return make


def run(cfg: ExperimentConfig, use_attention: bool) -> dict:
    # Pin to one thread: these batch-64, ~20-token matmuls are far smaller than
    # the multi-threaded pool's dispatch overhead, so one thread runs faster and
    # more reproducibly (same reason sort_gap.py / two_relation.py pin here).
    use_single_thread()
    factory = build_factory(use_attention)
    scoreboard = run_battery(factory, cfg)
    tag = "rnn-attn" if use_attention else "rnn"
    run_dir = init_run(cfg, _RESULTS, tag=tag)
    write_metrics(run_dir, scoreboard.pop("histories"))
    write_scoreboard(run_dir, scoreboard)

    label = "RNN + Bahdanau" if use_attention else "RNN"
    print(f"=== {label} — battery scoreboard ===")
    for task, row in scoreboard["rows"].items():
        print(f"  {task:16s} acc={row['accuracy']:.3f}  params={row['param_count']}")
    print(f"\nScoreboard written to {run_dir}")
    return scoreboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RNN battery")
    parser.add_argument("--attention", action="store_true", help="add Bahdanau attention (Task C)")
    parser.add_argument("--steps", type=int, default=None, help="override cfg.steps")
    parser.add_argument("--part", default="all", help="unused; uniformity")
    args = parser.parse_args()

    overrides = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    cfg = ExperimentConfig(**overrides)
    run(cfg, args.attention)


if __name__ == "__main__":
    main()

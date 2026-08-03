"""The battery scoreboard — ``run_battery``, the global eval spine.

One call trains and scores a mechanism across the shared battery, so any two
mechanisms are directly comparable row-for-row.  The batch *framing* is read
from ``model.framing`` and the data seed is shared, so the RNN
(encoder-decoder) and a transformer (single-stream) are scored on the same
underlying ``(x, y)`` data — the cross-method consistency that lets a
student read "Reverse: chance" in one row and "Reverse: solved" in the next.

All provided — students never modify this module.
"""

from __future__ import annotations

from typing import Callable

import torch

from attention.config import ExperimentConfig
from attention.data import BatchGenerator, make_batch
from attention.metrics import param_count, token_accuracy
from attention.models.base import SeqModel
from attention.train import train

# The shared battery: (task name, generation knobs).  Length-generalization
# is a Task-F *protocol* layered on Copy/Recall, not a standalone row, so it
# is not part of the comparable scoreboard.
BATTERY: list[tuple[str, dict[str, object]]] = [
    ("copy", {"n_min": 4, "n_max": 12}),
    ("reverse", {"n_min": 4, "n_max": 12}),
    ("sort", {"n_min": 4, "n_max": 12}),
    ("recall", {"num_pairs": 4}),
    ("selective_copy", {"n": 16, "num_marked": 3}),
]

# Reserved eval seeds (distinct from the training data_seed) so each fixed
# eval batch is independently reproducible.
_EVAL_SEED_BASE = 900_000
_EVAL_BATCH = 128


def _eval_seed(task_index: int) -> int:
    return _EVAL_SEED_BASE + task_index


def run_battery(
    make_model: Callable[[ExperimentConfig], SeqModel],
    cfg: ExperimentConfig,
    *,
    tasks: list[tuple[str, dict[str, object]]] | None = None,
) -> dict[str, object]:
    """Train a fresh model per battery task; return a scoreboard dict.

    ``make_model`` is a *factory* so a fresh model is built per task/seed.
    The returned dict is JSON-serializable and is what
    ``run.write_scoreboard`` persists.  Besides the ``rows``, it carries a
    ``histories`` key (per-task training ``History`` dicts) that the demo
    scripts pop and persist via ``run.write_metrics`` for the plotter.
    """
    battery = tasks if tasks is not None else BATTERY
    probe = make_model(cfg)
    framing = probe.framing
    del probe

    rows: dict[str, dict[str, float]] = {}
    histories: dict[str, dict[str, object]] = {}
    for idx, (task, knobs) in enumerate(battery):
        train_gen = BatchGenerator(task, seed=cfg.data_seed, framing=framing, **knobs)
        eval_batch = make_batch(
            task,
            _EVAL_BATCH,
            seed=_eval_seed(idx),
            framing=framing,
            **knobs,
        )
        model = make_model(cfg)
        history = train(model, cfg, train_gen, {task: eval_batch})

        model.eval()
        with torch.no_grad():
            b = eval_batch.to(cfg.device)
            logits = model(b)
            acc = token_accuracy(logits, b.targets, b.loss_mask)
        rows[task] = {
            "accuracy": acc,
            "param_count": param_count(model),
            "final_train_loss": (history.train_loss[-1] if history.train_loss else float("nan")),
        }
        histories[task] = history.to_dict()

    return {"framing": framing, "rows": rows, "histories": histories}

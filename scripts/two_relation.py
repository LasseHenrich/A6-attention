"""Task E demo: single-head vs multi-head on a two-relation task.

Constructs a task whose target needs **two** alignments at once — "the token
after the DELIM" AND "the value bound to the queried key" — and measures each
relation *separately* across head counts and seeds.

The overall token accuracy averages three output positions, two of which every
model solves perfectly, so it compresses the entire result into a shrug.  The
demo therefore also reports:

- **per-relation accuracy** at each output position, swept over ``num_heads``
  and repeated across seeds (the seed spread is the point — a single run of the
  one-head arm is bimodal);
- two reference levels the reader needs to judge the numbers: uniform **chance**
  over the content alphabet, and the **value-set prior** — the accuracy of
  ignoring the key entirely and guessing among the stored values;
- **per-head attention rows** at the two output queries, which show what each
  head actually attends to rather than inferring it from an entropy scalar.

Provided — students run it; do not edit.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from pathlib import Path

import torch
import torch.nn as nn

from attention.config import ExperimentConfig
from attention.data import Batch, _pad, _single_stream_rows
from attention.mechanisms.multihead import MultiHeadAttention
from attention.metrics import attention_entropy, token_accuracy
from attention.models.wrappers import SingleLayerModel
from attention.run import init_run, write_json
from attention.train import train_step, warmup_lr_scheduler
from attention.utils import seed_everything, use_single_thread
from attention.vocab import CONTENT, DELIM, QUERY, decode_token

_RESULTS = Path(__file__).resolve().parents[1] / "results"
_C = list(CONTENT)

_NUM_PAIRS = 3

# Training-stability knobs for this probe.  A single head is prone to locking
# onto a dead attention pattern (e.g. always attending SEP) before it sees any
# signal, which collapses relation B to chance on some seeds.  A gentler
# schedule and a smaller query/key init widen the basin so the outcome depends
# on the mechanism rather than the draw:
#   - lower peak LR than the battery default (cfg.lr = 3e-3), with warmup;
#   - shrink the Q/K projection weights so initial attention starts near-uniform
#     and early gradients reach every source position, not just the lucky one.
_LR = 1e-3
_WARMUP_FRAC = 0.1
_QK_INIT_SCALE = 0.5

# The stream is ``BOS DELIM marked k0 v0 k1 v1 k2 v2 QUERY kq SEP marked value EOS``
# (15 tokens, fixed).  A query at position ``t`` predicts token ``t + 1``, so
# the three supervised positions are:
_RELATION_POSITIONS: dict[str, int] = {
    "A: token after DELIM (position lookup)": 11,
    "B: value bound to the key (content lookup)": 12,
    "EOS": 13,
}

# Source positions the two relations *should* attend to: the marked token sits
# at index 2, and the value of pair ``i`` at index ``4 + 2 * i``.
_MARKED_INDEX = 2

# Head counts to sweep.  d_model = 64, so head_dim runs 64 -> 8.
_HEAD_COUNTS = (1, 2, 4, 8)


def _sample(rng: random.Random, num_pairs: int = _NUM_PAIRS, *, query_index: int | None = None) -> tuple[list[int], list[int]]:
    marked = rng.choice(_C)
    keys = rng.sample(_C, num_pairs)
    values = [rng.choice(_C) for _ in range(num_pairs)]
    qi = rng.randrange(num_pairs) if query_index is None else query_index
    x = [DELIM, marked]
    for k, val in zip(keys, values):
        x += [k, val]
    x += [QUERY, keys[qi]]
    return x, [marked, values[qi]]


def two_relation_batch(batch_size: int, seed: int, *, query_index: int | None = None) -> Batch:
    """A batch of two-relation samples.

    ``query_index`` pins *which* stored pair is queried, so every row of the
    batch shares one correct source position and a batch-averaged attention row
    stays meaningful.  ``None`` (the default) draws it uniformly.
    """
    rng = random.Random(seed)
    rows = [_single_stream_rows(*_sample(rng, query_index=query_index)) for _ in range(batch_size)]
    inp = [r[0] for r in rows]
    tgt = [r[1] for r in rows]
    msk = [r[2] for r in rows]
    length = max(len(r) for r in inp)
    return Batch(
        framing="single_stream",
        input_ids=torch.tensor(_pad(inp, length), dtype=torch.long),
        targets=torch.tensor(_pad(tgt, length), dtype=torch.long),
        loss_mask=torch.tensor(_pad(msk, length, 0), dtype=torch.float),
    )


_SPECIAL_ROLES = frozenset({"BOS", "DELIM", "QUERY", "SEP", "EOS"})


def _role_names(num_pairs: int) -> list[str]:
    """The structural role of each stream position (constant across the batch).

    Unlike the sampled token identities, these do not change from sample to
    sample, so they are the honest axis label for a batch-averaged attention
    row.  ``kq`` is the queried key; ``→marked`` / ``→value`` are the two
    supervised output positions.
    """
    roles = ["BOS", "DELIM", "marked"]
    for i in range(num_pairs):
        roles += [f"k{i}", f"v{i}"]
    return roles + ["QUERY", "kq", "SEP", "→marked", "→value", "EOS"]


def _key_labels(roles: list[str], tokens: list[str]) -> list[str]:
    """``"role: char"`` for content slots; the bare role for special tokens."""
    return [role if role in _SPECIAL_ROLES else f"{role}: {tok}" for role, tok in zip(roles, tokens)]


def relation_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Token accuracy at each supervised position, kept separate.

    Averaging these is what hides the result: the two easy positions carry the
    mean while the one hard relation moves underneath it.
    """
    preds = logits.argmax(dim=-1)
    return {name: float((preds[:, t] == targets[:, t]).float().mean().item()) for name, t in _RELATION_POSITIONS.items()}


class _TwoRelationGen:
    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._i = 0

    def next_batch(self, batch_size: int) -> Batch:
        self._i += 1
        return two_relation_batch(batch_size, self._seed + self._i)


def _scale_qk_init(mech: nn.Module, scale: float) -> None:
    """Shrink the query/key projection weights so attention starts near-uniform.

    Reaches for the reference projection names; a differently-named student
    implementation simply keeps its own init (the probe still runs, just
    without this stabilizer).
    """
    with torch.no_grad():
        for name in ("q_proj", "k_proj"):
            proj = getattr(mech, name, None)
            if isinstance(proj, nn.Linear):
                proj.weight.mul_(scale)


def _train_scheduled(model: nn.Module, cfg: ExperimentConfig, gen: _TwoRelationGen, *, lr: float, warmup_frac: float) -> None:
    """Train with a lowered peak LR and a warmup→cosine schedule.

    Reuses the shared ``train_step`` (so grad clipping and the zero-grad idiom
    are identical to the battery loop) and wraps it in a scheduler, rather than
    touching the frozen ``train.py`` that Task B repairs.
    """
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = warmup_lr_scheduler(optimizer, warmup_steps=int(warmup_frac * cfg.steps), total_steps=cfg.steps)
    for _ in range(cfg.steps):
        model.train()
        batch = gen.next_batch(cfg.batch_size).to(cfg.device)
        train_step(model, batch, optimizer, grad_clip=cfg.grad_clip)
        scheduler.step()


def train_probe(
    cfg: ExperimentConfig,
    num_heads: int,
    steps: int,
    *,
    init_seed: int | None = None,
    lr: float = _LR,
    warmup_frac: float = _WARMUP_FRAC,
    qk_init_scale: float = _QK_INIT_SCALE,
):
    seed = cfg.init_seed if init_seed is None else init_seed
    seed_everything(seed)
    c = dataclasses.replace(cfg, num_heads=num_heads, steps=steps, init_seed=seed)
    mech = MultiHeadAttention(c.d_model, num_heads)
    _scale_qk_init(mech, qk_init_scale)
    # Use the provided learned absolute PE (available from Task A); sinusoidal
    # PE is not implemented until Task F, and this demo is about head count,
    # not the positional scheme.
    model = SingleLayerModel(mech, c, causal=True, positional="learned")
    _train_scheduled(model, c, _TwoRelationGen(c.data_seed), lr=lr, warmup_frac=warmup_frac)
    ev = two_relation_batch(256, seed=700_000)
    model.eval()
    with torch.no_grad():
        logits = model(ev)
        acc = token_accuracy(logits, ev.targets, ev.loss_mask)
        per_relation = relation_accuracy(logits, ev.targets)
        entropy = None
        if model.last_attn is not None:
            entropy = attention_entropy(model.last_attn).tolist()
    return model, acc, per_relation, entropy


def head_attention_rows(model: SingleLayerModel, *, query_index: int = 0, seed: int = 700_001) -> dict[str, object]:
    """Per-head attention at the two output queries, on a fixed-``qi`` batch.

    Returns one ``[2, L]`` matrix per head: row 0 is the query that must emit
    the marked token, row 1 the query that must emit the bound value.  With the
    queried pair pinned, the correct source column is the same in every row of
    the batch, so the batch mean is a real distribution and not a smear.
    """
    ev = two_relation_batch(128, seed=seed, query_index=query_index)
    model.eval()
    with torch.no_grad():
        model(ev)
    attn = model.last_attn  # [B, H, L, L]
    if attn is None:
        return {}
    queries = [_RELATION_POSITIONS["A: token after DELIM (position lookup)"], _RELATION_POSITIONS["B: value bound to the key (content lookup)"]]
    rows = attn[:, :, queries, :].mean(dim=0)  # [H, 2, L]
    tokens = [decode_token(t) for t in ev.input_ids[0].tolist()]
    return {
        "rows": rows.tolist(),
        "query_labels": ["query A: →marked", "query B: →value"],
        "tokens": tokens,
        "key_labels": _key_labels(_role_names(_NUM_PAIRS), tokens),
        "correct_columns": [_MARKED_INDEX, 4 + 2 * query_index],
    }


def run(cfg: ExperimentConfig, steps: int, seeds: int, *, lr: float, warmup_frac: float, qk_init_scale: float) -> None:
    use_single_thread()  # these 15-token matmuls thrash a multi-threaded pool
    run_dir = init_run(cfg, _RESULTS, tag="two-relation")
    data: dict[str, object] = {"accuracy": {}, "per_head_entropy": {}}
    sweep: dict[str, dict[str, list[float]]] = {}
    head_rows: dict[str, object] = {}
    print(f"training: lr={lr} warmup_frac={warmup_frac} qk_init_scale={qk_init_scale}")

    for h in _HEAD_COUNTS:
        per_seed: dict[str, list[float]] = {name: [] for name in _RELATION_POSITIONS}
        per_seed["overall"] = []
        for s in range(seeds):
            model, acc, per_relation, entropy = train_probe(
                cfg, h, steps, init_seed=cfg.init_seed + s, lr=lr, warmup_frac=warmup_frac, qk_init_scale=qk_init_scale
            )
            per_seed["overall"].append(acc)
            for name, value in per_relation.items():
                per_seed[name].append(value)
            if s == 0:
                # Seed 0 reproduces the original two-arm headline exactly.
                if h in (1, 4):
                    data["accuracy"][f"heads_{h}"] = acc
                    if entropy is not None:
                        data["per_head_entropy"][f"heads_{h}"] = entropy
                if h == cfg.num_heads:
                    head_rows = head_attention_rows(model)
            hard = per_relation["B: value bound to the key (content lookup)"]
            print(f"num_heads={h} seed={cfg.init_seed + s}: overall={acc:.3f} relation_B={hard:.3f}")
        sweep[f"heads_{h}"] = per_seed

    data["sweep"] = {"num_heads": list(_HEAD_COUNTS), "per_seed": sweep, "seeds": seeds}
    data["chance"] = 1.0 / len(_C)
    data["value_set_prior"] = 1.0 / _NUM_PAIRS
    data["head_attention"] = head_rows
    write_json(data, run_dir / "two_relation.json")
    print(f"\nchance={data['chance']:.3f}  value-set prior={data['value_set_prior']:.3f}")
    print(f"Written to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task E two-relation demo")
    parser.add_argument("--part", default="all", help="unused; uniformity")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=3, help="init seeds per head count (the spread is the point)")
    parser.add_argument("--lr", type=float, default=_LR, help="peak learning rate (battery default is 3e-3)")
    parser.add_argument("--warmup-frac", type=float, default=_WARMUP_FRAC, help="fraction of steps spent warming up")
    parser.add_argument("--qk-init-scale", type=float, default=_QK_INIT_SCALE, help="multiplier on the Q/K projection init")
    args = parser.parse_args()
    run(ExperimentConfig(), args.steps, args.seeds, lr=args.lr, warmup_frac=args.warmup_frac, qk_init_scale=args.qk_init_scale)


if __name__ == "__main__":
    main()

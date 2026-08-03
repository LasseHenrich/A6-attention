"""Task G demo: train the decoder-only transformer; expose the mask bug.

Trains the transformer on the battery (the capstone scoreboard row) and, on
Copy, reports **teacher-forced** vs **free-running** token accuracy.  A
correct causal mask keeps the two close; the causal-mask bug drives them
apart — near-perfect teacher-forced (the model peeks at the answer) but a
collapse under free-running generation.

Three readouts, in rising order of how much they show:

- the two summary accuracies (``train_transformer.json``);
- **where the leak is**: *every* self-attention head of the trained model,
  drawn as a grid with each panel titled by the fraction of that head's
  attention landing on a *future* key.  The leak is soft and does not survive a
  head-mean, so the demo shows all heads and selects none: under a correct mask
  every head reads exactly 0% future mass, while under the bug a few heads park
  their mass on the key one step ahead of each query.  Run the demo before the
  fix and again after — each run is saved separately, and the two grids
  (``g-heads-leaky.png`` / ``g-heads-correct.png``) contrast the mask states;
- **what the leak costs**: accuracy at each generated position, teacher-forced
  vs free-running.

Every readout is measured on **the student's own current model**, so the demo
never has to name the buggy mask, and every figure is informative both before
and after the fix.

Provided — students run it; do not edit.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from attention.config import ExperimentConfig
from attention.data import Batch, BatchGenerator, make_batch
from attention.harness import run_battery
from attention.masking import causal_mask
from attention.metrics import future_attention_mass, mean_attention_map, self_attention_maps, token_accuracy
from attention.models.transformer import DecoderOnlyTransformer
from attention.run import init_run, write_json, write_metrics, write_scoreboard
from attention.utils import seed_everything, use_single_thread
from attention.vocab import SEP, decode_token

_RESULTS = Path(__file__).resolve().parents[1] / "results"

# Copy length for the position-resolved readout.  Kept inside the training
# range: past it, sinusoidal PE stops extrapolating and a *correct* model
# collapses too, which would confound the mask leak with a positional limit.
_PER_POSITION_N = 12


def _logits_of(model: DecoderOnlyTransformer, ids: torch.Tensor) -> torch.Tensor:
    dummy = torch.zeros_like(ids)
    batch = Batch(
        framing="single_stream",
        input_ids=ids,
        targets=dummy,
        loss_mask=dummy.float(),
    )
    return model(batch)


def free_running_accuracy(model: DecoderOnlyTransformer, n: int, seed: int) -> float:
    """Greedy autoregressive accuracy on Copy at fixed length ``n``."""
    ev = make_batch("copy", 128, seed=seed, framing="single_stream", n=n)
    ids = ev.input_ids
    total_len = ids.shape[1]
    prompt_len = n + 2  # BOS + n content + SEP
    assert (ids[:, prompt_len - 1] == SEP).all()
    cur = ids[:, :prompt_len].clone()
    model.eval()
    with torch.no_grad():
        for _ in range(total_len - prompt_len):
            nxt = _logits_of(model, cur)[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
    gen = cur[:, prompt_len:]
    truth = ids[:, prompt_len:]
    mask = ev.loss_mask[:, prompt_len - 1 : total_len - 1].bool()
    correct = ((gen == truth) & mask).sum().item()
    return correct / max(int(mask.sum().item()), 1)


def teacher_forced_accuracy(model: DecoderOnlyTransformer, n: int, seed: int) -> float:
    ev = make_batch("copy", 128, seed=seed, framing="single_stream", n=n)
    model.eval()
    with torch.no_grad():
        return token_accuracy(model(ev), ev.targets, ev.loss_mask)


def attention_readout(model: DecoderOnlyTransformer, n: int, seed: int) -> dict[str, object]:
    """The last layer's attention map, and how much of it looks at the future.

    ``future_mass`` is the fraction of a query's attention spent on keys it is
    not allowed to see.  A correct causal mask makes it exactly ``0.0``; a mask
    that leaks makes it positive, and the map shows *where* the leak sits.
    """
    ev = make_batch("copy", 128, seed=seed, framing="single_stream", n=n)
    model.eval()
    with torch.no_grad():
        model(ev)
        maps = self_attention_maps(model)
        per_layer = [future_attention_mass(a) for a in maps]
        attn_map = mean_attention_map(maps[-1]).tolist()
    return {
        "map": attn_map,
        "tokens": [decode_token(t) for t in ev.input_ids[0].tolist()],
        "future_mass_per_layer": per_layer,
        "future_mass": max(per_layer) if per_layer else 0.0,
    }


def mask_is_causal(seq_len: int = 8) -> bool:
    """Does ``causal_mask`` honor its own contract?

    The helper documents that entry ``[i, j]`` is ``-inf`` wherever ``j > i``,
    so *no* entry strictly above the diagonal may be finite.  This checks that
    promise; it does not know what a correct mask looks like.  Its value keys
    the run as leaky or correct so the two head-grids stay separate on disk.
    """
    rows, cols = torch.triu_indices(seq_len, seq_len, offset=1)
    return not bool(torch.isfinite(causal_mask(seq_len)[rows, cols]).any())


def head_maps(model: DecoderOnlyTransformer, n: int, seed: int) -> tuple[list[dict[str, object]], list[str]]:
    """*Every* head of *every* self-attention layer on one Copy example.

    Returns one entry per (layer, head) — the ``[L, L]`` map and its
    future-attention-mass — rather than a head-mean.  The mean dilutes a
    single-head leak by ``num_heads`` and hides it; showing all heads lets the
    reader see for themselves that most are diffuse and only a few peek at the
    next token.  No head is selected — that would be circular (we would be
    sorting on the very quantity the figure claims to reveal).
    """
    ev = make_batch("copy", 1, seed=seed, framing="single_stream", n=n)
    model.eval()
    with torch.no_grad():
        model(ev)
    tokens = [decode_token(int(t)) for t in ev.input_ids[0]]
    entries: list[dict[str, object]] = []
    for layer, attn in enumerate(self_attention_maps(model)):  # each [B, H, L, L]
        heads = attn[0]  # [H, L, L]
        for head in range(heads.shape[0]):
            single = heads[head]  # [L, L]
            entries.append(
                {
                    "layer": layer,
                    "head": head,
                    "future_mass": future_attention_mass(single),
                    "map": single.tolist(),
                }
            )
    return entries, tokens


def accuracy_by_position(model: DecoderOnlyTransformer, n: int, seed: int) -> dict[str, list[float]]:
    """Teacher-forced and free-running accuracy at each generated token.

    Position ``k`` is the k-th output token ``y_k``.  The trailing ``EOS`` is
    dropped: it is predictable from the sequence length alone, so it says
    nothing about whether the model can copy.
    """
    ev = make_batch("copy", 128, seed=seed, framing="single_stream", n=n)
    ids = ev.input_ids
    total_len = ids.shape[1]
    prompt_len = n + 2  # BOS + n content + SEP
    assert (ids[:, prompt_len - 1] == SEP).all()
    n_gen = total_len - prompt_len - 1  # y1..yn, excluding EOS

    model.eval()
    with torch.no_grad():
        preds = model(ev).argmax(dim=-1)
        teacher = [float((preds[:, prompt_len - 1 + k] == ev.targets[:, prompt_len - 1 + k]).float().mean()) for k in range(n_gen)]

        cur = ids[:, :prompt_len].clone()
        for _ in range(n_gen):
            nxt = _logits_of(model, cur)[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
        gen = cur[:, prompt_len:]
        truth = ids[:, prompt_len:]
        free = [float((gen[:, k] == truth[:, k]).float().mean()) for k in range(n_gen)]

    return {"position": list(range(1, n_gen + 1)), "teacher_forced": teacher, "free_running": free}


def run(cfg: ExperimentConfig, steps: int) -> None:
    from attention.train import train

    # Pin to one thread: these batch-64, ~20-token matmuls are far smaller than
    # the multi-threaded pool's dispatch overhead, so one thread runs faster and
    # more reproducibly (same reason sort_gap.py / two_relation.py pin here).
    use_single_thread()
    run_dir = init_run(cfg, _RESULTS, tag="transformer")

    # Capstone battery row (teacher-forced accuracy).
    scoreboard = run_battery(DecoderOnlyTransformer, dataclasses.replace(cfg, steps=steps))
    write_metrics(run_dir, scoreboard.pop("histories"))
    write_scoreboard(run_dir, scoreboard)
    print("decoder-only transformer battery:")
    for task, row in scoreboard["rows"].items():
        print(f"  {task:16s} acc={row['accuracy']:.3f}")

    # Teacher-forced vs free-running on Copy, at the fixed length n=8.
    seed_everything(cfg.init_seed)
    model = DecoderOnlyTransformer(cfg)
    gen = BatchGenerator("copy", seed=cfg.data_seed, framing="single_stream", n=8)
    train(model, dataclasses.replace(cfg, steps=steps), gen, {})
    tf = teacher_forced_accuracy(model, n=8, seed=910_000)
    fr = free_running_accuracy(model, n=8, seed=910_000)
    heads, tokens = head_maps(model, n=8, seed=910_000)
    data: dict[str, object] = {"teacher_forced": tf, "free_running": fr}
    data["attention"] = attention_readout(model, n=8, seed=910_000)
    data["mask_ok"] = mask_is_causal()
    data["head_maps"] = heads
    data["tokens"] = tokens
    print(f"\nCopy: teacher_forced={tf:.3f}  free_running={fr:.3f}")
    print("(A large gap ⇒ the model is peeking — check the causal mask.)")

    mass = data["attention"]["future_mass"]  # type: ignore[index]
    print(f"attention mass on future keys: {mass:.4f}  (a correct causal mask makes this exactly 0)")

    # Position-resolved readout, on a model trained over the whole length
    # range so evaluating at n=12 stays in-distribution.
    seed_everything(cfg.init_seed)
    var_model = DecoderOnlyTransformer(cfg)
    var_gen = BatchGenerator("copy", seed=cfg.data_seed, framing="single_stream", n_min=4, n_max=12)
    train(var_model, dataclasses.replace(cfg, steps=steps), var_gen, {})
    by_pos = accuracy_by_position(var_model, n=_PER_POSITION_N, seed=910_001)
    data["by_position"] = by_pos
    data["by_position_length"] = _PER_POSITION_N
    print(f"\nCopy n={_PER_POSITION_N}, free-running accuracy per output token:")
    print("  " + " ".join(f"{a:.2f}" for a in by_pos["free_running"]))

    write_json(data, run_dir / "train_transformer.json")
    print(f"\nWritten to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task G transformer demo")
    parser.add_argument("--part", default="all", help="unused; uniformity")
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    run(ExperimentConfig(), args.steps)


if __name__ == "__main__":
    main()

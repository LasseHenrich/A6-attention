"""Task H demo: the encoder-decoder transformer, and what the encoder buys.

Trains a decoder-only transformer (G) and an encoder-decoder transformer (H)
on Sort.  **Both solve it** — the accuracies are printed side by side and they
are near-identical, because the single-stream framing ``BOS x .. SEP y ..``
puts the whole source in the decoder's *left context*: a causal decoder has
already seen every input token before it emits the first output token.  There
is no Sort gap.

What the encoder actually changes, the demo shows in three figures:

- ``h-alignment.png`` — the decoder-only model's self-attention (output
  positions → source) beside the encoder-decoder's cross-attention (decoder
  steps → memory), with the true argsort permutation overlaid.  **Both** trace
  it; cross-attention just makes the alignment explicit and peakier.
- ``h-cross-attention-alignment.png`` — cross-attention *is* the alignment,
  read across tasks: the diagonal on Copy, the anti-diagonal on Reverse, the
  argsort permutation on Sort.
- ``h-source-dependency.png`` — a training-free finite-difference probe of
  which source positions each source representation can see.  The decoder-only
  stack is causal over the source (upper triangle exactly zero); the encoder is
  bidirectional (dense).  That difference is structural, not learned.

Also runs the full battery for the encoder-decoder row.

Provided — students run it; do not edit.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from torch import Tensor

from attention.config import ExperimentConfig
from attention.data import Batch, BatchGenerator, make_batch
from attention.harness import _EVAL_BATCH, run_battery
from attention.metrics import cross_attention_maps, token_accuracy
from attention.models.transformer import (
    DecoderOnlyTransformer,
    EncoderDecoderTransformer,
)
from attention.run import init_run, write_json, write_metrics, write_scoreboard
from attention.train import train
from attention.utils import seed_everything, use_single_thread
from attention.vocab import N_CONTENT, N_SPECIAL, SEP, decode_token

_RESULTS = Path(__file__).resolve().parents[1] / "results"

# Fixed length for the alignment figures, so every row of the batch shares one
# source geometry and the decoder rows line up with the source columns.  The
# battery below still scores Sort at varying n.
_ALIGN_N = 8
_ALIGN_TASKS = ("copy", "reverse", "sort")


def _train(model_cls, cfg: ExperimentConfig, steps: int, task: str, n: int):
    """Train *model_cls* on *task* at fixed length *n*.

    Seeded from ``cfg.init_seed``, so two calls with the same arguments return
    identical models — the encoder-decoder Sort model behind the two-panel
    figure is the same one behind the three-panel figure's Sort cell.
    """
    seed_everything(cfg.init_seed)
    model = model_cls(cfg)
    gen = BatchGenerator(task, seed=cfg.data_seed, framing=model.framing, n=n)
    train(model, dataclasses.replace(cfg, steps=steps), gen, {})
    return model


def _sort_accuracy(model, n: int) -> float:
    ev = make_batch("sort", _EVAL_BATCH, seed=920_000, framing=model.framing, n=n)
    model.eval()
    with torch.no_grad():
        return token_accuracy(model(ev), ev.targets, ev.loss_mask)


def _peakiness(alignment: Tensor) -> float:
    """Mean of each output step's largest attention weight — 1.0 is a hard alignment."""
    return float(alignment.max(dim=-1).values.mean())


def true_alignment(source: list[int], task: str) -> list[int]:
    """Source index each output position *should* come from."""
    n = len(source)
    if task == "copy":
        return list(range(n))
    if task == "reverse":
        return list(range(n - 1, -1, -1))
    if task == "sort":
        return sorted(range(n), key=lambda i: (source[i], i))  # stable: ties keep input order
    raise ValueError(f"no alignment defined for task {task!r}")


def cross_alignment_readout(model: EncoderDecoderTransformer, task: str, n: int) -> dict[str, object]:
    """Read the encoder-decoder's cross-attention for the three-panel figure.

    ``alignment_accuracy`` scores the argmax source position of each decoder
    row by the **token it holds**, not by its index: the content alphabet
    repeats, so two source positions can both be the right answer.
    """
    ev = make_batch(task, _EVAL_BATCH, seed=930_000, framing="encoder_decoder", n=n)
    model.eval()
    with torch.no_grad():
        model(ev)
        attn = cross_attention_maps(model)[0].mean(dim=1)  # first cross layer, head-mean -> [B, T, S]

    rows = attn[:, :n, :]  # decoder row t predicts y_t; drop the trailing EOS row
    picked = rows.argmax(dim=-1)  # [B, n]
    src = ev.source
    tgt = ev.targets[:, :n]
    hits = torch.gather(src, 1, picked) == tgt
    accuracy = float(hits.float().mean().item())

    sample_src = src[0].tolist()
    return {
        "matrix": rows[0].tolist(),
        "source_tokens": [decode_token(t) for t in sample_src],
        "target_tokens": [decode_token(t) for t in tgt[0].tolist()],
        "true_alignment": true_alignment(sample_src, task),
        "alignment_accuracy": accuracy,
    }


def _distinct_example(n: int, seed: int, batch: int = 64) -> int:
    """Index of the first example whose source tokens are all distinct.

    Sort draws symbols *with replacement*, so most examples repeat a symbol.  A
    repeated symbol makes "which source position should step ``t`` attend to?"
    ambiguous — either copy of the symbol is a correct answer — which would make
    the argsort overlay in the figure a lie.  Pick an example without ties.
    """
    ev = make_batch("sort", batch, seed=seed, framing="encoder_decoder", n=n)
    for i in range(batch):
        row = ev.source[i][:n].tolist()
        if len(set(row)) == n:
            return i
    raise RuntimeError(f"no duplicate-free sort example in {batch} draws at n={n}")


def decoder_only_alignment(model: DecoderOnlyTransformer, n: int, seed: int, index: int) -> tuple[Tensor, list[int]]:
    """Self-attention from the output positions back onto the source ``[n, n]``."""
    ev = make_batch("sort", 64, seed=seed, framing="single_stream", n=n)
    model.eval()
    with torch.no_grad():
        model(ev)
    assert model.last_attn is not None
    ids = ev.input_ids[index]
    sep = int((ids == SEP).nonzero()[0])
    weights = model.last_attn.mean(dim=1)[index]  # [L, L], head-mean
    # Rows: the positions that predict y1..yn (starting at SEP).  Cols: source x1..xn.
    return weights[sep : sep + n, 1:sep], ids[1:sep].tolist()


def encoder_decoder_alignment(model: EncoderDecoderTransformer, n: int, seed: int, index: int) -> tuple[Tensor, list[int]]:
    """Cross-attention from decoder steps into the encoder memory ``[n, n]``.

    Reads the **last** cross-attention map via ``cross_attention_maps`` — which
    selects the rectangular (decoder × source) maps by shape, so it needs no
    knowledge of how the student named their decoder layers.
    """
    ev = make_batch("sort", 64, seed=seed, framing="encoder_decoder", n=n)
    model.eval()
    with torch.no_grad():
        model(ev)
    cross = cross_attention_maps(model)[-1]  # last cross layer [B, H, T, S]
    weights = cross.mean(dim=1)[index]  # [T, S], head-mean
    return weights[:n, :n], ev.source[index][:n].tolist()


def _dependency_matrix(represent, source: list[int], n: int) -> list[list[float]]:
    """How much does the representation at source position ``i`` move when token ``j`` changes?

    A finite-difference probe over the model's *public* forward pass — it needs
    no knowledge of how the model names its submodules.  Entry ``[i, j]`` is the
    L2 change at position ``i`` caused by swapping the token at position ``j``.
    Zero means position ``i`` cannot see position ``j`` at all.

    Untrained: this measures what the **architecture** permits, not what a
    trained model chooses to attend to.

    ``represent`` returns ``[rows, d]``; the result is ``[rows, n]``.
    """
    base = represent(source)
    rows = base.shape[0]
    matrix = [[0.0] * n for _ in range(rows)]
    for j in range(n):
        perturbed = list(source)
        # Any other content token; the specific choice does not matter.
        perturbed[j] = N_SPECIAL + (source[j] - N_SPECIAL + 1) % N_CONTENT
        delta = (represent(perturbed) - base).norm(dim=-1)  # [rows]
        for i in range(rows):
            matrix[i][j] = float(delta[i])
    return matrix


def source_dependency_panels(cfg: ExperimentConfig, n: int) -> tuple[dict[str, list[list[float]]], int]:
    """Which source positions can influence each source representation, per architecture.

    Also returns how many source positions the decoder-only model's **first
    output position** depends on.  The panels show the *source-to-source*
    structure (causal vs bidirectional); this count answers the different
    question of what the decoder has seen by the time it emits ``y1`` — the
    whole source, because in ``BOS x .. SEP y ..`` every source token is to its
    left.
    """
    ev = make_batch("sort", 1, seed=940_000, framing="single_stream", n=n)
    ids = ev.input_ids[0].tolist()
    sep = ids.index(SEP)
    source = ids[1:sep]

    seed_everything(cfg.init_seed)
    dec = DecoderOnlyTransformer(cfg)
    dec.eval()

    def dec_logits(src: list[int]) -> Tensor:
        stream = torch.tensor([ids[:1] + src + ids[sep:]])
        batch = Batch(
            framing="single_stream",
            input_ids=stream,
            targets=torch.zeros_like(stream),
            loss_mask=torch.zeros_like(stream).float(),
        )
        with torch.no_grad():
            return dec(batch)[0]

    def dec_represent(src: list[int]) -> Tensor:
        return dec_logits(src)[1:sep]  # logits over the source region

    def dec_first_output(src: list[int]) -> Tensor:
        return dec_logits(src)[sep : sep + 1]  # the position that predicts y1

    seed_everything(cfg.init_seed)
    enc = EncoderDecoderTransformer(cfg)
    enc.eval()

    def enc_represent(src: list[int]) -> Tensor:
        with torch.no_grad():
            return enc.encode(torch.tensor([src]))[0]  # memory [n, d_model]

    output_row = _dependency_matrix(dec_first_output, source, n)[0]
    panels = {
        "decoder-only: causal over the source": _dependency_matrix(dec_represent, source, n),
        "encoder: bidirectional": _dependency_matrix(enc_represent, source, n),
    }
    return panels, sum(1 for v in output_row if v > 0.0)


def run(cfg: ExperimentConfig, steps: int) -> None:
    use_single_thread()
    run_dir = init_run(cfg, _RESULTS, tag="sort-gap")

    # Trained at a single fixed length: the alignment a variable-length model
    # learns is the same shape but blurred across lengths, and the figures are
    # the point here.  The battery row below still scores Sort at varying n.
    enc_models = {task: _train(EncoderDecoderTransformer, cfg, steps, task, _ALIGN_N) for task in _ALIGN_TASKS}
    dec_model = _train(DecoderOnlyTransformer, cfg, steps, "sort", _ALIGN_N)
    enc_sort = enc_models["sort"]

    dec_acc = _sort_accuracy(dec_model, _ALIGN_N)
    enc_acc = _sort_accuracy(enc_sort, _ALIGN_N)

    data: dict[str, object] = {"decoder_only": dec_acc, "encoder_decoder": enc_acc, "n": _ALIGN_N}

    # Three-panel: cross-attention *is* the alignment, read task by task.
    alignments = {task: cross_alignment_readout(enc_models[task], task, _ALIGN_N) for task in _ALIGN_TASKS}
    data["alignment"] = alignments

    # Two-panel: decoder-only self-attention vs encoder-decoder cross-attention
    # on one duplicate-free Sort example, so a single argsort overlays both.
    example = _distinct_example(_ALIGN_N, seed=930_000)
    dec_align, dec_src = decoder_only_alignment(dec_model, _ALIGN_N, seed=930_000, index=example)
    enc_align, enc_src = encoder_decoder_alignment(enc_sort, _ALIGN_N, seed=930_000, index=example)
    assert dec_src == enc_src, "both framings must draw the same example for a shared overlay"
    data.update(
        {
            "decoder_only_alignment": dec_align.tolist(),
            "encoder_decoder_alignment": enc_align.tolist(),
            "decoder_only_source": [decode_token(t) for t in dec_src],
            "encoder_decoder_source": [decode_token(t) for t in enc_src],
            "decoder_only_argsort": torch.tensor(dec_src).argsort(stable=True).tolist(),
            "encoder_decoder_argsort": torch.tensor(enc_src).argsort(stable=True).tolist(),
            "decoder_only_peakiness": _peakiness(dec_align),
            "encoder_decoder_peakiness": _peakiness(enc_align),
        }
    )

    # Source dependency (untrained, structural): causal over the source vs bidirectional.
    panels, output_sees = source_dependency_panels(cfg, _ALIGN_N)
    data["source_dependency"] = panels
    data["decoder_only_output_sees_source_positions"] = output_sees

    write_json(data, run_dir / "sort_gap.json")

    dec_hits = int((dec_align.argmax(dim=-1) == torch.tensor(data["decoder_only_argsort"])).sum())
    enc_hits = int((enc_align.argmax(dim=-1) == torch.tensor(data["encoder_decoder_argsort"])).sum())
    print("Sort — both architectures solve it (no gap):\n")
    print(f"{'':24s} {'accuracy':>9} {'alignment peakiness':>21} {'argmax = argsort':>18}")
    print(f"{'decoder-only':24s} {dec_acc:>9.3f} {data['decoder_only_peakiness']:>21.3f} {f'{dec_hits}/{_ALIGN_N}':>18}")
    print(f"{'encoder-decoder':24s} {enc_acc:>9.3f} {data['encoder_decoder_peakiness']:>21.3f} {f'{enc_hits}/{_ALIGN_N}':>18}")
    print(f"\nBefore it emits y1, the decoder-only model already depends on {output_sees}/{_ALIGN_N} source tokens.")
    print("(The encoder reveals no input the decoder-only model lacked — the whole")
    print(" source is already its left context.  What it buys is a bidirectional")
    print(" source encoding, and a sharper, explicit alignment.)")
    print("\ncross-attention alignment accuracy (argmax source token vs target token):")
    for task, a in alignments.items():
        print(f"  {task:8s} {a['alignment_accuracy']:.3f}")

    scoreboard = run_battery(EncoderDecoderTransformer, dataclasses.replace(cfg, steps=steps))
    write_metrics(run_dir, scoreboard.pop("histories"))
    write_scoreboard(run_dir, scoreboard)
    print("\nencoder-decoder battery:")
    for task, row in scoreboard["rows"].items():
        print(f"  {task:16s} acc={row['accuracy']:.3f}")
    print(f"\nWritten to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task H encoder-decoder demo")
    parser.add_argument("--part", default="all", help="unused; uniformity")
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    run(ExperimentConfig(), args.steps)


if __name__ == "__main__":
    main()

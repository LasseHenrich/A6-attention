"""Synthetic task battery: generators, framing adapters, and batching.

Every battery task is defined abstractly as a map from an input token list
to a target token list, ``(x, y)``.  Per-sample generators draw from a
passed-in ``random.Random`` (they never create one), so whoever owns the
RNG owns the whole stream.  A thin framing adapter turns each ``(x, y)``
into the shape the model under test consumes; there are exactly two
framings (single-stream and encoder-decoder), chosen by the model type.

All provided — students never modify this module.

Framings
--------
Single-stream (decoder-only)::

    BOS  x1 .. xn  SEP  y1 .. ym  EOS

    Teacher-forced causal LM: ``targets`` is the stream shifted left by one
    (each position predicts the next token) and ``loss_mask`` is 1 only on
    the ``y1 .. EOS`` region.

Encoder-decoder::

    source:  x1 .. xn
    target:  BOS  y1 .. ym  EOS   (decoder input teacher-forced;
                                   loss on y1 .. EOS)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from attention.vocab import BOS, CONTENT, DELIM, EOS, PAD, QUERY, SEP

_CONTENT = list(CONTENT)


# ---------------------------------------------------------------------------
# Per-sample generators (pure; draw from a passed-in RNG)
# ---------------------------------------------------------------------------


def copy_task(n: int, *, rng: random.Random) -> tuple[list[int], list[int]]:
    """``y = x``: reproduce the input.  Order-preserving identity."""
    x = [rng.choice(_CONTENT) for _ in range(n)]
    return x, list(x)


def reverse_task(n: int, *, rng: random.Random) -> tuple[list[int], list[int]]:
    """``y = reversed(x)``: needs order."""
    x = [rng.choice(_CONTENT) for _ in range(n)]
    return x, list(reversed(x))


def sort_task(n: int, *, rng: random.Random) -> tuple[list[int], list[int]]:
    """``y = sorted(x)`` ascending by id: needs order and comparison."""
    x = [rng.choice(_CONTENT) for _ in range(n)]
    return x, sorted(x)


def recall_task(
    num_pairs: int,
    *,
    needle_distance: int | None = None,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Associative recall: bind ``num_pairs`` (key, value) pairs, look one up.

    The input is ``k0 v0 k1 v1 .. QUERY kq`` and the target is the value
    bound to ``kq``.  Keys are sampled without replacement from the content
    alphabet so a lookup is unambiguous; values are sampled with
    replacement.

    ``needle_distance`` selects *how far back* the queried pair sits,
    measured in pairs from the end: ``0`` queries the most-recently-stored
    pair (shortest distance), ``num_pairs - 1`` the first-stored pair
    (longest).  ``None`` picks a uniformly random pair.
    """
    if num_pairs > len(_CONTENT):
        raise ValueError(f"num_pairs={num_pairs} exceeds distinct keys available ({len(_CONTENT)})")
    keys = rng.sample(_CONTENT, num_pairs)
    values = [rng.choice(_CONTENT) for _ in range(num_pairs)]

    if needle_distance is None:
        q_index = rng.randrange(num_pairs)
    else:
        if not 0 <= needle_distance < num_pairs:
            raise ValueError(f"needle_distance={needle_distance} out of range [0, {num_pairs})")
        q_index = num_pairs - 1 - needle_distance

    x: list[int] = []
    for k, v in zip(keys, values):
        x.extend((k, v))
    x.extend((QUERY, keys[q_index]))
    return x, [values[q_index]]


def selective_copy_task(
    n: int,
    num_marked: int,
    *,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Copy only the content token immediately after each ``DELIM``.

    The input interleaves ``num_marked`` ``(DELIM, value)`` units with
    ``n - 2*num_marked`` distractor content tokens, shuffled at unit
    granularity so each ``DELIM`` keeps its following value.  The target is
    the marked values in order of appearance.
    """
    if n < 2 * num_marked:
        raise ValueError(f"n={n} too small for num_marked={num_marked} (need n >= 2*num_marked)")
    marked = [rng.choice(_CONTENT) for _ in range(num_marked)]
    n_distract = n - 2 * num_marked
    units: list[list[int]] = [[DELIM, v] for v in marked]
    units += [[rng.choice(_CONTENT)] for _ in range(n_distract)]
    rng.shuffle(units)

    x: list[int] = []
    y: list[int] = []
    for unit in units:
        x.extend(unit)
        if unit[0] == DELIM:
            y.append(unit[1])
    return x, y


_GENERATORS = {
    "copy": copy_task,
    "reverse": reverse_task,
    "sort": sort_task,
    "recall": recall_task,
    "selective_copy": selective_copy_task,
}


# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    """A framed, padded, tensorized batch.

    Fields used depend on ``framing``.  Single-stream uses ``input_ids``;
    encoder-decoder uses ``source`` / ``target_in`` / ``source_padding_mask``.
    ``targets`` and ``loss_mask`` are aligned to the model's output logits
    in both framings, so ``masked_cross_entropy`` treats them uniformly.
    """

    framing: str
    targets: Tensor  # [B, T] label at each output position
    loss_mask: Tensor  # [B, T] 1.0 where the label counts, else 0.0
    input_ids: Tensor | None = None  # single-stream model input [B, L]
    source: Tensor | None = None  # encoder-decoder source [B, S]
    target_in: Tensor | None = None  # encoder-decoder decoder input [B, T]
    source_padding_mask: Tensor | None = None  # [B, S] True on real tokens

    def to(self, device: str | torch.device) -> Batch:
        """Return a copy with every tensor moved to *device*."""

        def _mv(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.to(device)

        return Batch(
            framing=self.framing,
            targets=_mv(self.targets),
            loss_mask=_mv(self.loss_mask),
            input_ids=_mv(self.input_ids),
            source=_mv(self.source),
            target_in=_mv(self.target_in),
            source_padding_mask=_mv(self.source_padding_mask),
        )

    @property
    def batch_size(self) -> int:
        return self.targets.shape[0]


# ---------------------------------------------------------------------------
# Framing (one abstract (x, y) -> per-model rows)
# ---------------------------------------------------------------------------


def _single_stream_rows(x: list[int], y: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Return (input_ids, targets, loss_mask) rows for one sample.

    ``targets`` is ``input_ids`` shifted left by one (next-token); the last
    position has no next token, so its target is ``PAD`` and its mask 0.
    ``loss_mask`` is 1 exactly where the predicted (next) token lies in the
    ``y1 .. EOS`` output region.
    """
    stream = [BOS, *x, SEP, *y, EOS]
    input_ids = stream
    targets = stream[1:] + [PAD]
    # Output region in the stream: the y tokens and the EOS, i.e. the tail
    # of length len(y) + 1.  A position t predicts stream[t+1]; mask it when
    # that predicted token is in the output region.
    out_len = len(y) + 1  # y1..ym plus EOS
    loss_mask = [0] * len(stream)
    first_out = len(stream) - out_len  # index of y1 in the stream
    for t in range(first_out - 1, len(stream) - 1):
        loss_mask[t] = 1
    return input_ids, targets, loss_mask


def _encoder_decoder_rows(x: list[int], y: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Return (source, target_in, target_out) rows for one sample."""
    source = list(x)
    target_in = [BOS, *y]
    target_out = [*y, EOS]
    return source, target_in, target_out


def _pad(rows: list[list[int]], length: int, value: int = PAD) -> list[list[int]]:
    return [r + [value] * (length - len(r)) for r in rows]


# ---------------------------------------------------------------------------
# BatchGenerator + make_batch
# ---------------------------------------------------------------------------


class BatchGenerator:
    """Owns one RNG, seeded once at construction.  Same seed -> same stream.

    Knobs are task-specific and passed as keyword args:

    - copy / reverse / sort: ``n`` (fixed) or ``n_min`` / ``n_max`` (drawn).
    - recall: ``num_pairs`` (or ``num_pairs_min`` / ``num_pairs_max``),
      optional ``needle_distance``.
    - selective_copy: ``n`` / ``n_min`` / ``n_max`` and ``num_marked``.
    """

    def __init__(self, task: str, *, seed: int, framing: str, **knobs: object) -> None:
        if task not in _GENERATORS:
            raise ValueError(f"unknown task {task!r}")
        if framing not in ("single_stream", "encoder_decoder"):
            raise ValueError(f"unknown framing {framing!r}")
        self.task = task
        self.framing = framing
        self.knobs = knobs
        self._rng = random.Random(seed)

    # -- sampling one raw (x, y) --------------------------------------------

    def _draw_int(self, fixed: str, lo: str, hi: str, default: int) -> int:
        if self.knobs.get(lo) is not None and self.knobs.get(hi) is not None:
            return self._rng.randint(int(self.knobs[lo]), int(self.knobs[hi]))
        return int(self.knobs.get(fixed, default))

    def _sample(self) -> tuple[list[int], list[int]]:
        if self.task == "recall":
            num_pairs = self._draw_int("num_pairs", "num_pairs_min", "num_pairs_max", 4)
            nd = self.knobs.get("needle_distance")
            return recall_task(
                num_pairs,
                needle_distance=None if nd is None else int(nd),  # type: ignore[arg-type]
                rng=self._rng,
            )
        if self.task == "selective_copy":
            n = self._draw_int("n", "n_min", "n_max", 16)
            num_marked = int(self.knobs.get("num_marked", 3))
            return selective_copy_task(n, num_marked, rng=self._rng)
        n = self._draw_int("n", "n_min", "n_max", 12)
        return _GENERATORS[self.task](n, rng=self._rng)  # type: ignore[operator]

    # -- one framed, padded, tensorized batch -------------------------------

    def next_batch(self, batch_size: int) -> Batch:
        samples = [self._sample() for _ in range(batch_size)]
        if self.framing == "single_stream":
            return self._frame_single_stream(samples)
        return self._frame_encoder_decoder(samples)

    def _frame_single_stream(self, samples: list[tuple[list[int], list[int]]]) -> Batch:
        rows = [_single_stream_rows(x, y) for x, y in samples]
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

    def _frame_encoder_decoder(self, samples: list[tuple[list[int], list[int]]]) -> Batch:
        rows = [_encoder_decoder_rows(x, y) for x, y in samples]
        src = [r[0] for r in rows]
        tin = [r[1] for r in rows]
        tout = [r[2] for r in rows]
        s_len = max(len(r) for r in src)
        t_len = max(len(r) for r in tin)
        src_pad = _pad(src, s_len)
        src_mask = [[1] * len(r) + [0] * (s_len - len(r)) for r in src]
        loss_mask = [[1] * len(r) + [0] * (t_len - len(r)) for r in tout]
        return Batch(
            framing="encoder_decoder",
            source=torch.tensor(src_pad, dtype=torch.long),
            source_padding_mask=torch.tensor(src_mask, dtype=torch.bool),
            target_in=torch.tensor(_pad(tin, t_len), dtype=torch.long),
            targets=torch.tensor(_pad(tout, t_len), dtype=torch.long),
            loss_mask=torch.tensor(loss_mask, dtype=torch.float),
        )


def make_batch(task: str, batch_size: int, *, seed: int, framing: str, **knobs: object) -> Batch:
    """One fixed batch = ``BatchGenerator(...).next_batch(batch_size)``."""
    return BatchGenerator(task, seed=seed, framing=framing, **knobs).next_batch(batch_size)

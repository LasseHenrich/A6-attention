# Task F — Positional Encoding & ALiBi

## Background

Tasks D and E showed that self-attention is **order-blind**: with no
positional signal it treats the input as a set, so Reverse and Sort are
unsolvable. This task supplies position — and shows that *how* you encode
position has real consequences for generalizing to longer sequences.

You implement two schemes and compare four (the fourth, learned absolute PE,
is provided):

### Sinusoidal positional encoding

A fixed (non-learned) code added to the token embeddings. Dimension `2i` of
position `pos` uses `sin(pos / 10000^(2i/d))` and dimension `2i+1` the
matching `cos`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

Different frequencies across dimensions give every position a unique,
smoothly varying code, and relative offsets are linear functions of the code.
Because it is a smooth function of position (not a lookup table), it produces
*some* code for positions beyond those seen in training.

### Learned absolute positional encoding (provided)

An embedding table with one learned vector per absolute position. It works
in-distribution but has **no vector for positions beyond the trained
maximum** — so it *cliffs* when evaluated on longer sequences. This is the
provided foil (`LearnedAbsolutePositionalEncoding`); you compare against it,
you do not implement it.

### ALiBi — attention with linear biases

Instead of adding anything to the embeddings, ALiBi adds a **relative-distance
penalty to the attention scores**, before the softmax. Each head `h` gets a
fixed slope `m_h`, and the penalty grows linearly with distance:

```
bias_h[i, j] = -m_h · (i - j)     for j <= i   (attend to the past)
bias_h[i, j] = -inf               for j >  i   (causal)
```

The slopes are a geometric sequence, `m_h = ratio^h` with
`ratio = 2^(-8/num_heads)` (so 8 heads give slopes `2^-1 … 2^-8`). ALiBi feeds
through the `additive_bias` hook your attention op already accepts — no change
to the attention computation itself.

Because ALiBi encodes *relative* distance (which recurs at every length), it
**extrapolates strongly** to longer sequences.

### The two faults (both empirical)

- **Absolute-PE cliff.** Learned absolute PE assigns each absolute position its
  own vector, so positions past the trained maximum have no meaningful code
  and accuracy collapses beyond the training length. Sinusoidal PE, being
  smooth, extrapolates somewhat; ALiBi, being relative, extrapolates best.
- **ALiBi's locality prior.** The linear distance penalty biases attention
  toward *nearby* positions — `make demo-f` shows this directly: ALiBi's mean
  attention weight falls off with query–key distance far more sharply than
  sinusoidal's, concentrating mass locally and starving distant keys. That
  helps local structure and length extrapolation, but is a liability for tasks
  that need distant lookups (long-range associative recall): a useful bias in
  one regime hurts in another.

Both are **empirical / inductive-bias** faults — about how each prior
interacts with the data, not hard impossibilities (unlike D/E's *provable*
order-blindness, which positions here *remove*).

## What to Implement

Fill in three functions in `src/attention/mechanisms/positional.py`:

- `sinusoidal_encoding(seq_len, d_model) -> [seq_len, d_model]` — the sin/cos
  construction above (interleaved even/odd).
- `alibi_slopes(num_heads) -> [num_heads]` — the geometric slope sequence.
- `alibi_bias(seq_len, num_heads) -> [num_heads, seq_len, seq_len]` — the
  causal linear-distance penalty above.

### Constraints

- Write the constructions by hand. Importing or delegating to an **external
  positional-encoding helper** (a library ALiBi/RoPE) is **forbidden** (checked
  automatically). Standard `torch` math (`sin`, `cos`, `arange`, `exp`) is
  allowed.

## What these tests do NOT check

- The **length-generalization accuracy** and ALiBi-locality curves —
  observational.
- That you added the sinusoidal PE to the embeddings correctly *inside a full
  model* — the encoding function is graded in isolation; integration is
  exercised by the probe/transformer tests.

Grading is a guide, not a proof of correctness.

## Deliverables

- Implement `sinusoidal_encoding`, `alibi_slopes`, and `alibi_bias` in
  `src/attention/mechanisms/positional.py`.
- Run `make demo-f` (length generalization + ALiBi locality) and record
  `f_length_observations`, `f_alibi_observations`, `f_abs_pe_status`, and
  `f_alibi_status` in `answers.py`.
- Run `make test-f` and confirm the tests pass.
- Run `make submit-f` to generate `submission.json`.

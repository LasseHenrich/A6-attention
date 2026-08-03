# Task H — Encoder + Cross-Attention

## Background

The decoder-only transformer (Task G) reads a single stream left-to-right.
Some tasks are hard for it precisely *because* it only ever sees the tokens to
the left at each step. **Sort** is the clearest example: to emit the smallest
element first, you need to have seen the *whole* input. This task adds a
**bidirectional encoder** and **cross-attention** to build the full
encoder-decoder transformer, and closes the loop back to Bahdanau attention
(Task C).

### The encoder-decoder architecture

```
source:  x1 … xn   →  [ bidirectional encoder ]  →  memory
target:  BOS y1 …   →  [ causal decoder,          →  logits
                         cross-attending to memory ]
```

- **Memory is the encoder's output:** the sequence of `[B, S, d_model]`
  vectors the encoder produces, one per source position. It is the same thing
  Task C called the encoder states (`enc_states`) — the fixed source-side
  representation that attention reads from — now built by a stack of
  transformer blocks instead of an RNN. The decoder never re-runs the encoder;
  it attends into this memory.
- **The encoder is bidirectional (unmasked):** every source position attends
  to *every* other source position, so each memory vector is a
  fully-contextualized summary of the whole input. This is the crucial
  difference from the causal decoder.
- **The decoder is still causal** in its *self*-attention (it may not see
  future target tokens), but each decoder layer adds a **cross-attention**
  sublayer that attends into the encoder memory.

### Cross-attention = Bahdanau grown up

Cross-attention is **not a new mechanism**. It is your `MultiHeadAttention`
called with **decoder queries** and **encoder keys/values**:

```
cross = MultiHeadAttention(q_in = decoder_hidden,
                           k_in = v_in = encoder_memory)
```

This is exactly Task C's additive attention — the decoder querying the source
— now multi-head and dot-product.

Build the decoder layer from pieces you already have. It does three things, in
order: attend **causally** over the target produced so far, **cross-attend**
into the memory, then apply a position-wise **FFN**.

- **Causal self-attention** — you already have a block that runs self-attention
  and then an FFN: your `TransformerBlock`. Reuse it here, run with the causal
  mask. It brings its own FFN along; that extra position-wise transform is
  harmless, and grading checks how the layers connect, not the exact sublayer
  count.
- **Cross-attention, then FFN** — add these two sublayers yourself, each
  wrapped the way every sublayer in Task G was: pre-norm, the sublayer, output
  dropout, residual add.

```
x = causal_self_attention_block(x, causal_mask)        # your reused TransformerBlock
x = x + dropout(cross_attention(LayerNorm(x), memory, memory))
x = x + dropout(FFN(LayerNorm(x)))
```

Because you reuse `TransformerBlock` and `MultiHeadAttention`, the
initialization and dropout practices from Tasks E and G carry over. The one
thing to remember: the cross-attention and the FFN you add by hand also write
into the residual stream, so give their output projections the same output
dropout and the same `1 / sqrt(2 · num_layers)` residual scaling that Task G's
block applied (again an in-place `mul_` under `torch.no_grad()`), for the same
reason.

**A note on source padding.** In general, cross-attention should mask out `PAD` positions in the source so the decoder never attends to padding — the same masked softmax as Task C. This assignment's batches are fixed-length with no padding, so that mask is inert and omitted here (the batch still carries a `source_padding_mask` field — all-ones in this assignment, so you can ignore it); with variable-length sources you would add it.

### What the encoder actually buys (and what it does not)

Producing a sorted output requires a **global view** of the input: you cannot
name the smallest element until you have seen them all. It is tempting to
conclude that a causal decoder-only model must therefore struggle — it only
ever attends leftward.

**Measure it before you believe it.** `make demo-h` trains both architectures
on Sort and prints their accuracy. They tie, at ≈0.997. There is no gap.

The reason is the single-stream framing from Task G:
BOS x1 … xn SEP y1 … ym EOS
Every source token lies to the **left** of every output position. When the
decoder-only model emits `y1`, the causal mask has already let it attend to
all of `x1..xn`. It has the global view. "causal" restricts *what a position
may look at*, not *how much of the input the model has seen*.

So what does the encoder change? Two things, with different epistemic status.

1. **Bidirectional source encoding — structural.** The encoder runs its blocks
   *unmasked*, so source position `i` attends to `j > i`. In the decoder-only
   stack it cannot. This follows from the masks alone. `h-source-dependency.png`
   shows it: the decoder-only panel's upper triangle is *exactly* zero; the
   encoder's is not.

2. **A sharper, explicit alignment — empirical.** `h-alignment.png` puts the
   decoder-only self-attention (output → source) beside the encoder-decoder
   cross-attention (decoder → memory), true argsort overlaid. **Both** trace it;
   cross-attention just makes the alignment explicit and peakier.

And you can watch that cross-attention *be* the alignment across tasks:
`h-cross-attention-alignment.png` reads the encoder-decoder's cross-attention
on Copy, Reverse, and Sort — the diagonal, the anti-diagonal, and the argsort
permutation, each traced by the learned weights. That is exactly Task C's
additive alignment, now multi-head and dot-product.

Be precise about claim 2: a measurement at *this* scale, not a proof a
decoder-only model cannot align as sharply. Contrast Task D/E's *provable*
order-blindness, and point 1, which is provable. Do not over-claim.

The lesson: an architectural story that sounds compelling ("ranking needs
global context, so the encoder must win") can be simply false, and you find
out by running the experiment.

## What to Implement

In `src/attention/models/transformer.py`, fill in `EncoderDecoderTransformer`
(`__init__`, the `encode` method, and `forward`):

- **`__init__`** — register the source/target embeddings, the encoder blocks
  (reuse `TransformerBlock`), and, per decoder layer, a causal
  self-attention block (also a `TransformerBlock`), a cross-attention
  `MultiHeadAttention`, a FFN, and the LayerNorms for the cross-attention and
  FFN sublayers, plus the final norm and unembed. Initialize the weights in
  `__init__`: small-normal embeddings and a Xavier unembed; the encoder and
  decoder self-attention blocks initialize themselves, but the cross-attention
  and FFN you build here must be Xavier-inited and residual-scaled explicitly
  (they write into the residual stream too), as described above.
- **`encode(source) -> memory`** — embed the source, add positional encoding
  (your `sinusoidal_encoding` from Task F, added to the embeddings exactly as
  in Task G — including the dropout applied to the embedding + positional sum),
  and run the encoder blocks **unmasked** (bidirectional). Returns the memory
  `[B, S, d_model]`.
- **`forward(batch)`** — the encoder-decoder batch carries `batch.source`
  `[B, S]` and `batch.target_in` `[B, T]` (the single-stream `batch.input_ids`
  is `None` in this framing). Call `self.encode(batch.source)` for the memory,
  embed `batch.target_in` and add positional encoding the same way, run the
  causal, cross-attending decoder over it, and unembed to logits
  `[B, T, VOCAB]`.

### Constraints

- Reuse your own `TransformerBlock` and `MultiHeadAttention`. `nn.Transformer`
  and its variants are **forbidden** (checked automatically).
- The encoder must be **unmasked** (bidirectional); the decoder's
  *self*-attention must stay **causal**.

## What these tests do NOT check

- The **Sort-gap accuracy** — observational; only the seeded forward
  properties and the wiring verdicts are graded.
- **Exact forward math against a reference** — any assembly satisfying the
  causality, cross-dependency, and bidirectionality verdicts passes; the
  Sort-gap demo is the behavioral backstop.
- That the model *learns* to use cross-attention meaningfully — grading checks
  connectivity, not what training discovers.

Grading is a guide, not a proof of correctness.

## Deliverables

- Implement `EncoderDecoderTransformer` (`__init__`, `encode`, `forward`) in
  `src/attention/models/transformer.py`.
- Run `make demo-h` (the Sort comparison, the alignment and source-dependency
  figures, and the encoder-decoder battery row) and record
  `h_sort_observations`, `h_crossattn_observations`,
  `h_bidirectionality_status`, and `h_alignment_status` in `answers.py`.
  Answer from the figures and the printed table, not from what you expected to
  happen.
- Run `make test-h` and confirm the tests pass.
- Run `make submit-h` to generate `submission.json`.

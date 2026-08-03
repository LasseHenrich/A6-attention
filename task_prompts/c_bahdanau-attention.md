# Task C — Additive (Bahdanau) Attention

## Background

Task A's encoder-decoder RNN has a bottleneck: the decoder sees the whole
source only through a single fixed vector, `enc_final`. Long or distant
information gets squeezed out. **Attention** fixes this by letting the
decoder look back at *every* encoder state and build a fresh, query-dependent
summary at each output step.

### The idea

Instead of one fixed context vector, the decoder computes, at each step, a
**weighted average of all encoder states** — where the weights depend on what
the decoder is currently trying to produce. If the decoder's current state
"asks about" a particular source position, that position gets a high weight.

### Additive (Bahdanau) scoring

Additive attention scores the decoder's current state `s` against each
encoder state `h_j` with a small one-hidden-layer network:

```
e_j = vᵀ · tanh(W_q · s + W_k · h_j)
```

- `W_q` maps the decoder state into an attention space; `W_k` maps each
  encoder state into the same space; they are added and squashed by `tanh`.
- `v` projects the result to a single scalar score `e_j` per source position.

The scores become **attention weights** with a softmax over the source
positions:

```
a_j = softmax_j(e_j)          (a_1 … a_S sum to 1)
```

and the **context vector** is the weighted average of the encoder states:

```
c = Σ_j a_j · h_j
```

The weights `a_j` are a *learned, interpretable alignment* between the current
output step and the input positions — you can read off which source token the
decoder is attending to. The context `c` is fed into the decoder step
(concatenated with the step input, then projected as usual).

**Masking.** Some source positions are `PAD` (padding). They must be removed
from the softmax (given weight ~0), so the context never mixes in padding.
The decoder passes a source mask for exactly this.

### What attention fixes — and what it does not

Attention **relieves the bottleneck**: the decoder can consult every encoder
state, so long-range recall improves. But the model is *still an RNN
underneath*:

- The encoder and decoder are still **unrolled step by step**, so inference
  wall-clock still grows with sequence length — attention does not make the
  recurrence parallel.
- Gradients still flow back **through the recurrence** (repeated `W_hh`
  through `tanh`), so the gradient reaching early positions still shrinks —
  attention adds a shortcut but does not cure the vanishing-gradient path.

These residual limits are **empirical / inductive-bias** properties of the
*recurrence*, not of attention. The next family (self-attention, Task D)
removes recurrence entirely.

## What to Implement

Fill in `AdditiveAttention` in `src/attention/mechanisms/additive.py`. Write
both `__init__` (register the scoring parameters — `W_q`, `W_k`, and `v` as
linear layers work well) and `forward`:

```
forward(dec_state, enc_states, *, mask=None) -> (context, weights)
```

- `dec_state` is `[B, dec_dim]`, `enc_states` is `[B, S, enc_dim]`.
- Compute the additive scores, apply the `mask` (if given; `True` marks real
  source tokens, `PAD` positions are removed from the softmax), softmax over
  the source axis to get `weights [B, S]`, and return
  `(context [B, enc_dim], weights [B, S])` where `context` is the
  weight-averaged encoder states.

The decoder hook is already in place: constructing
`Seq2SeqRNN(cfg, attention=AdditiveAttention(...))` routes each decoder step
through your module. You do not rewrite the decoder — you provide the
attention module it calls.

### Constraints

- Write additive attention by hand. Delegating to `nn.MultiheadAttention` or
  `torch.nn.functional.scaled_dot_product_attention` is **forbidden** (checked
  automatically). `nn.Linear`, `torch.tanh`, and `softmax` are allowed.

## What these tests do NOT check

- The **trained recall / gradient / timing curves** — those are observational;
  only seeded forward properties are graded.
- **Exact scoring math against a reference** — a valid masked softmax over the
  encoder states that is not the additive form could still pass; the
  anti-cheat scan and the trained demos are the backstop.
- Your **context-combination choice** — any wiring that makes the decoder
  logits depend on the encoder states passes.

Grading is a guide, not a proof of correctness.

## Deliverables

- Implement `AdditiveAttention` in `src/attention/mechanisms/additive.py`.
- Run `make demo-c` (recall-vs-distance, gradient-norm, serial-timing) and
  record `c_recall_observations`, `c_gradient_observations`,
  `c_serial_observations`, and `c_fault_status` in `answers.py`.
- Run `make test-c` and confirm the tests pass.
- Run `make submit-c` to generate `submission.json`.

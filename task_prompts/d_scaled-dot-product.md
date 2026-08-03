# Task D — Scaled Dot-Product Self-Attention

## Background

Bahdanau attention (Task C) scored the decoder state against each encoder
state with a small MLP. **Scaled dot-product attention** replaces that MLP
with a plain dot product, which is cheaper and fully parallel — and it drops
the recurrence entirely. This is the single operation reused by every model
for the rest of the assignment.

### Queries, keys, values

Each position produces three vectors: a **query** `q`, a **key** `k`, and a
**value** `v`. Position `i` attends to position `j` by the similarity of its
query to that key, `qᵢ · kⱼ`. In *self*-attention every position is all
three: `q = k = v = x`. The full operation is:

```
scores  = Q Kᵀ / √dₖ          # [L, L] similarity of every query to every key
weights = softmax(scores)      # each row a distribution over keys
output  = weights V            # each position: a weighted average of values
```

`dₖ` is the key/query dimension. Compared to additive scoring there is no
per-pair MLP and no time loop — one matmul, softmax, another matmul.

### Why divide by √dₖ

The dot product of two `dₖ`-dimensional random vectors has variance
proportional to `dₖ`. So as `dₖ` grows, the raw scores grow too, the softmax
is pushed toward a near one-hot distribution (it **saturates**), and its
gradient flattens — training stalls. Dividing by `√dₖ` normalizes the score
variance back to ~1, keeping the softmax in a useful range.

This is an **empirical / optimization** fault: the unscaled model is not
*incapable* of representing the right function; it just trains badly. A bigger
network would not fix it, but nothing about capacity changes.

### Masking and biases

- A **mask** blocks some query→key pairs. A *causal* mask stops position `t`
  from attending to positions `> t` (needed for autoregressive models); a
  *padding* mask blocks `PAD`. Masking is done by adding `-inf` to the blocked
  scores before the softmax (or, for a boolean mask, filling them with `-inf`).
- An **additive bias** is added to the scores before the softmax. Task F's
  ALiBi uses this hook; Task D just threads it through.

A subtle point used later: a **causal mask leaks position information** — a
position can count how many tokens are to its left. So a *masked* model is not
truly order-blind, which matters for the next property.

### Permutation equivariance (order-blindness)

With **no positional signal and no mask**, self-attention treats its input as
a *set*: permute the input rows and the output rows permute identically —
`attn(Px) = P · attn(x)`. This is a *provable structural* property. A
consequence: order-dependent tasks (Reverse, Sort) are **unsolvable** by bare
self-attention, no matter how long you train — there is no information in the
operation about position. Task F adds positions to fix this.

Note the contrast in *epistemic status*: the scaling fault is **empirical**
(bad optimization), while order-blindness is **provable** (a structural
impossibility). This distinction — capability vs optimization — recurs
throughout the assignment.

### One relation at a time

One softmax produces one weighted average — the layer attends to **one
relation** at a time. Tasks that need two alignments at once want more; that
motivates multi-head attention (Task E).

## What to Implement

Fill in `scaled_dot_product_attention` in
`src/attention/mechanisms/scaled_dot_product.py`:

```
scaled_dot_product_attention(q, k, v, *, mask=None,
                             additive_bias=None, scale=None) -> (output, weights)
```

- Compute `scores = q kᵀ · scale`; `scale=None` means `1/√dₖ`.
- Add `additive_bias` if given; apply `mask` if given (a float mask is added,
  a boolean mask marks positions to keep).
- Softmax over the **key** axis, then multiply by `v`.
- Return `(output [..., L, d], weights [..., Lq, Lk])`. The caller stashes the
  weights (a free function cannot record them itself).

Your function must accept leading batch/head dimensions (heads are batched in
by Task E), so operate on the last two axes.

### Constraints

- Write the op yourself. `torch.nn.functional.scaled_dot_product_attention`,
  `nn.MultiheadAttention`, and `F.multi_head_attention_forward` are
  **forbidden** (checked automatically). `matmul`, `softmax`, and masking are
  allowed.

## What these tests do NOT check

- The **trained battery accuracy** or the ablation curves — observational only.
- **Head-batched inputs beyond the tested shapes** — the contract is fixed but
  only the graded shapes are verified (Task E exercises the head dimension).

Grading is a guide, not a proof of correctness.

## Deliverables

- Implement `scaled_dot_product_attention` in
  `src/attention/mechanisms/scaled_dot_product.py`.
- Run `make demo-d` (scaling ablation + permutation equivariance) and record
  `d_scaling_observations`, `d_equivariance_observations`, `d_scaling_status`,
  and `d_equivariance_status` in `answers.py`.
- Run `make test-d` and confirm the tests pass.
- Run `make submit-d` to generate `submission.json`.

# Task E — Multi-Head Self-Attention

## Background

A single scaled dot-product attention layer (Task D) produces one softmax —
**one weighted average, one relation at a time**. Many tasks need to attend to
several things at once: "the token after the delimiter" *and* "the value bound
to the queried key". **Multi-head attention** runs several attention
operations in parallel, each in its own learned subspace, so the layer can
capture several relations in a single step.

### The head decomposition

Multi-head attention projects the input to `d_model` dimensions, then splits
that width into `h` **heads** of size `d_head = d_model / h`. Each head runs
its own scaled dot-product attention over its slice, the head outputs are
concatenated back to `d_model`, and a final linear layer mixes them:

```
[B, L, d_model]                          input
  → Q,K,V projections → [B, L, d_model]
  → reshape to heads   → [B, h, L, d_head]
  → scaled_dot_product_attention per head (batched over h)
  → concat heads       → [B, L, d_model]
  → output projection  → [B, L, d_model]
```

Each head is exactly Task D applied to a slice — you **reuse your own
`scaled_dot_product_attention`**, batched over the head dimension. The split
is a reshape, not new math.

### Why heads help — and what they don't fix

- **Several relations at once (empirical win).** With `h` heads the layer
  computes `h` weighted averages in `h` subspaces, so it can attend to several
  things simultaneously. On a task that needs two alignments at once, a single
  head plateaus while two or more heads solve it. This is a
  capacity/inductive-bias win — *empirical*.
- **Per-head specialization.** Different heads learn to attend to different
  positions. **Attention entropy** measures how concentrated vs. spread out a
  head's softmax weights are over the source positions — a sharp, low-entropy
  head focuses on one position (a precise lookup); a diffuse, high-entropy head
  averages over many. It matters because it is a training-free readout of *what
  each head is doing*: heads settling at different entropies is the evidence
  they have specialized rather than all collapsing onto the same relation.
- **Still order-blind (provable).** A concatenation of permutation-equivariant
  heads is still permutation-equivariant — more heads add **no** sense of
  position. Order tasks remain unsolvable without a positional signal (Task
  F). This is the same *provable* structural limit from Task D.

### Initializing the projections

The Q, K, V, and output projections are this layer's only weights, and how they start matters for training. Initialize each with **Xavier/Glorot** (`torch.nn.init.xavier_uniform_`) and a zero bias. Xavier scales a matrix by both its fan-in and fan-out so the pre-softmax scores keep variance near 1: too large and the softmax saturates (its gradient nearly vanishes), too small and every position gets near-equal weight and the head learns slowly. This is the original Transformer's choice, and the same fan-in/fan-out reasoning you applied to the RNN's input matrix in Task A.

### Self- vs cross-attention

The same module serves both. **Self-attention** passes the same tensor as
query, key, and value (`q_in = k_in = v_in = x`). **Cross-attention** (Task H)
passes decoder queries with encoder keys/values (`q_in` differs from
`k_in = v_in`). One class, chosen by the caller — no separate cross-attention
module.

## What to Implement

Fill in `MultiHeadAttention` in `src/attention/mechanisms/multihead.py` —
both `__init__` (register the Q/K/V projections and the output projection;
assert `d_model % num_heads == 0`; initialize the projections with Xavier/Glorot and a zero bias, per *Initializing the projections* above) and `forward`:

```
forward(q_in, k_in, v_in, *, attn_mask=None, additive_bias=None) -> [B, L, d_model]
```

- Project `q_in`/`k_in`/`v_in` to `d_model`, reshape into `num_heads` heads,
  call **your own `scaled_dot_product_attention`** batched over the head
  dimension (threading `attn_mask` and `additive_bias` straight through),
  concatenate the heads, and apply the output projection.
- Stash the per-head attention weights so the entropy metric can read them.

### Constraints

- Per-head attention must go through **your own** `scaled_dot_product_attention`
  from Task D. `nn.MultiheadAttention`, `F.multi_head_attention_forward`, and
  `F.scaled_dot_product_attention` are **forbidden** (checked automatically);
  the grader also confirms you call your own SDPA.

## What these tests do NOT check

- The **two-relation training result** and **entropy specialization** —
  observational.
- That heads learn *distinct* relations in practice — grading checks forward
  properties, not what training discovers.
- **Exact forward math against a reference** — e.g. that `num_heads=1`
  reproduces plain single-head attention is not a graded numerical match; the
  anti-cheat scan and the two-relation demo are the backstop.

Grading is a guide, not a proof of correctness.

## Deliverables

- Implement `MultiHeadAttention` in `src/attention/mechanisms/multihead.py`.
- Run `make demo-e` (single-vs-multi accuracy + per-head entropy) and record
  `e_head_observations`, `e_order_observations`, `e_two_relation_status`, and
  `e_order_status` in `answers.py`.
- Run `make test-e` and confirm the tests pass.
- Run `make submit-e` to generate `submission.json`.

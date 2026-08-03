# Task I — Linear (Kernel) Attention

## Background

Softmax attention forms an `n × n` score matrix — its cost and memory grow
**quadratically** with sequence length. Linear (kernel) attention removes that
matrix with a feature map and a reordering trick, computing attention in **O(n)**
instead of **O(n²)**. This final task builds it and measures the trade it
makes: a clear **cost-vs-length win** paid for with a **recall-vs-load loss**.

### The kernel trick and the reorder

Softmax attention computes `softmax(qᵀk)`. If we approximate the softmax kernel
by a dot product of feature maps, `softmax(qᵀk) ≈ φ(q)ᵀφ(k)`, then attention
becomes:

```
out = (φ(Q) φ(K)ᵀ) V
```

By **associativity**, we can multiply the other way:

```
out = φ(Q) (φ(K)ᵀ V)
```

Computing `φ(K)ᵀ V` **first** produces a small `d × d` matrix and costs
**O(n·d²)** — *linear* in sequence length. That reorder is the whole idea, and
the O(n) scaling is a **provable** complexity fact. The feature map must be
positive so the normalizer is well-defined; this task uses `φ(x) = elu(x) + 1`.

### The running-state / linear-RNN view

Define the state:

```
KV = Σ_j φ(k_j) v_jᵀ        (a fixed d × d summary of all keys/values)
Z  = Σ_j φ(k_j)             (the normalizer)
out_t = φ(q_t)ᵀ KV / (φ(q_t)ᵀ Z)
```

`KV` is a fixed-size summary of the whole sequence — an RNN-like state. The
**causal** form is the prefix sum of it: `S_t = Σ_{j≤t} φ(k_j) v_jᵀ`, so
position `t` depends only on `j ≤ t`. Linear attention is "a linear RNN over
the reordered state".

### The cost-vs-length win (provable)

The forward-only cost benchmark (up to `n ≈ 2048`) shows softmax's time growing
quadratically while linear attention's grows linearly. This is a complexity
fact — it follows from the reorder, not from any experiment.

### The recall-vs-load loss (empirical)

The state `KV` is **finite-rank** (`d × d`). Storing many (key, value) pairs
superimposes them in that fixed-size state, so retrieving any one value picks
up interference from the others — the **retrieval signal-to-noise ratio falls
as the number of stored pairs grows**. Softmax attention, which keeps every
pair individually addressable through the full score matrix, does not pay this.
So linear attention's associative-recall accuracy **degrades under load** where
softmax holds up. This is an **empirical** capacity trade, not a hard zero.

### No attention matrix ⇒ no entropy

Linear attention never materializes per-key weights, so the **attention
entropy** metric is **undefined** (N/A) for it — a genuine consequence of the
reorder, and part of the interpretability you trade away.

## What to Implement

Fill in `LinearAttention` in `src/attention/mechanisms/linear.py` — both
`__init__` (register the Q/K/V and output projections, initialized with
Xavier/Glorot and a zero bias as in Task E) and `forward`:

```
forward(q_in, k_in, v_in, *, attn_mask=None, causal=False) -> [B, L, d_model]
```

- Project and split into heads, apply the feature map `φ = elu + 1`.
- **Non-causal (parallel)** form: build `KV` and `Z` once, then
  `out_t = φ(q_t)ᵀ KV / (φ(q_t)ᵀ Z)`. This is the benchmark path — it must
  **not** build an `n × n` matrix.
- **Causal (prefix-sum)** form: the same with running (cumulative) sums over
  `j ≤ t`, so position `t` never sees a future key/value.
- Concatenate heads and apply the output projection.

Causality is expressed by the `causal` flag (the prefix-sum form), not by an
additive mask.

### Constraints

- Linear attention must be the **reorder**, not softmax in disguise.
  `F.scaled_dot_product_attention` and `nn.MultiheadAttention` are
  **forbidden** (checked automatically). `matmul`, `cumsum`, and `elu` are
  allowed. The grader also checks that your op scales **sub-quadratically**.

## What these tests do NOT check

- The **benchmark curves, SNR, and recall accuracies** — observational.
- **Exact forward math against a reference** — the specific feature map is not
  verified numerically; the causality, self-consistency, and sub-quadratic
  verdicts plus the anti-cheat scan are the backstop.
- That linear attention *approximates* softmax well — it deliberately does not.
- **Absolute timings** — only the machine-relative scaling verdict is graded.

Grading is a guide, not a proof of correctness.

## Deliverables

- Implement `LinearAttention` in `src/attention/mechanisms/linear.py`.
- Run `make demo-i` (cost-vs-length, recall-vs-load, SNR) and record
  `i_cost_observations`, `i_recall_observations`, `i_entropy_na`,
  `i_cost_status`, and `i_recall_status` in `answers.py`.
- Run `make test-i` and confirm the tests pass.
- Run `make submit-i` to generate `submission.json`.

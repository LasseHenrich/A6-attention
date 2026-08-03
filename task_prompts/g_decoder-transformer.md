# Task G — Decoder-Only Transformer (and the Causal-Mask Bug)

## Background

You have built every piece of a transformer: multi-head attention (E), the
positional encodings (F), and the scaled dot-product op underneath (D). This
task **assembles** them into a decoder-only transformer — the capstone
architecture — and asks you to find one subtle bug that lets the model cheat.

### The transformer block

A transformer block has two sublayers: multi-head **self-attention** and a
position-wise **feed-forward network** (FFN). Each is wrapped in a **residual
connection** with **LayerNorm**. This assignment uses **pre-norm** ordering
(LayerNorm *before* each sublayer, residual around it):

```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

- **LayerNorm** normalizes each position's feature vector (zero mean, unit
  variance) so activations stay well-scaled as depth grows.
- The **FFN** is `Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)`, with
  `d_ff = 4·d_model` — a per-position nonlinearity.
- The **residual** (`x + sublayer(...)`) lets information and gradients bypass
  each sublayer, which is what makes stacking many blocks trainable.
- **Dropout** is applied to each sublayer's output before it is added back to the residual (and to the embedding + positional-encoding sum). Randomly zeroing a fraction of activations during training keeps the model from leaning on any single unit — a regularizer. It is automatically disabled at evaluation, and defaults to `0.0` here, but this is where a transformer places dropout (Vaswani et al. §5.4).

(Pre-norm is more stable for small CPU training than the original *post-norm*
transformer; GELU is used where the original used ReLU. These are the
intended design, though the graded properties do not pin them.)

### The decoder-only stack

```
input_ids → embed → + positional encoding
          → N causal transformer blocks
          → final LayerNorm → unembed → logits [B, L, VOCAB]
```

The model is an autoregressive language model over the single-stream framing
`BOS x … SEP y … EOS`: at each position it predicts the next token. Note that
the positional encodings are added to the embeddings. Compute the sinusoidal code for the *actual* sequence length on each forward pass rather than caching a fixed-size table: recomputing it lets the model accept sequences longer than `max_len`, which is exactly what the Task F length-extrapolation study depends on.

### Initializing the stack

Two initialization practices keep a deep stack trainable:

- **Xavier/Glorot on every linear layer, a small normal on the embeddings.** As in Tasks A and E, Xavier balances each layer's fan-in and fan-out so activations hold a stable scale through the depth of the stack; the embedding table is drawn from a small normal (standard deviation ~0.02) so the token signal does not overpower the positional code it is added to.
- **Scale the residual-writing projections by `1 / sqrt(2 · num_layers)`.** Every block *adds* its two sublayer outputs onto the residual stream, so without care the stream's variance grows with depth. Shrinking the two projections that write into it — the attention output projection and the FFN's second linear — by this factor holds that variance roughly constant as blocks stack. This is GPT-2's trick for training deeper transformers stably. Because these are *already-initialized* parameters rather than freshly-created ones, you scale them in place; PyTorch rejects an in-place edit to a parameter that requires grad unless you wrap it in a `with torch.no_grad():` block (e.g. `with torch.no_grad(): proj.weight.mul_(scale)`).

### Causal masking — and the bug

An autoregressive model must predict token `t+1` using only tokens `≤ t`. If
position `t` can attend to positions `> t`, it **sees the answer**. The
**causal mask** enforces this: it blocks every query→key pair where the key is
in the future. You *apply* this mask in your stack (pass it to each block's
attention); the mask itself is built by the provided `causal_mask` helper in
`src/attention/masking.py`.

**That helper is subtly wrong** — the distributed `causal_mask` lets each
position peek at a future token. Rather than being told where, you meet this
bug the way you would in real work: **observe → diagnose → fix → verify**.
`make demo-g` measures everything on *your own current model*, so every readout
is informative both before and after the fix — **run it before you touch
`masking.py`**, then again after, and read the two runs against each other.

The demo gives you two readouts:

- **The symptom — `g-accuracy-by-position.png`.** Teacher-forced (predict each
  token given the *ground-truth* prefix) and free-running (feed the model its
  *own* predictions) accuracy at each output token, with **one panel per mask
  state** once you have run the demo buggy and then fixed. In the **buggy** panel
  the two curves split: teacher-forced stays near-perfect — the model reads the
  answer through the hole — while free-running falls toward chance, because at
  generation time the future token is not there. In the **fixed** panel
  free-running has climbed back to meet teacher-forced and the two sit together
  near the top. That split-under-the-bug, agree-once-fixed is the tell, and it is
  the un-gameable one: no amount of soft, well-behaved attention closes a
  free-running gap that a real leak opens.
- **Where the leak sits — `g-heads-leaky.png` / `g-heads-correct.png`.**
  *Every* self-attention head of the trained model, drawn as a grid, each panel
  titled with the fraction of that head's attention landing on a key it should
  not see (its *future-attention-mass*). The leak is soft and **does not survive
  a head-mean** — averaging over heads would smear a sharp, single-head leak
  into a faint haze — so the demo shows every head and cherry-picks none. Scan
  them: on the leaky grid a few heads park their mass on the marked next-token
  cell while the rest stay diffuse; on the correct grid every head reads exactly
  **0%** future mass. You run the demo twice — buggy, then fixed — so both grids
  land on disk to contrast.

Observe the symptom first, then go find its cause — that is the order in which
you will meet this bug in real work.

This leak is a **correctness bug**, categorically different from the *faults*
of Tasks D–F. Those faults were inductive-bias limitations (a model that
*can't* do something, or does it poorly for a structural reason). A model that
peeks isn't exhibiting a limitation — it's simply **wrong**.

## What to Implement

In `src/attention/models/transformer.py`, fill in:

- **`TransformerBlock`** — the pre-norm block above, reusing your
  `MultiHeadAttention`. Write `__init__` (register the two LayerNorms, the
  attention, the FFN, and a `nn.Dropout(cfg.dropout)`) and `forward` (the two
  residual sublayers, each sublayer output passed through the dropout before it
  re-enters the residual). Initialize the block's own weights **in `__init__`**:
  Xavier on the FFN linears, and — since this block writes two sublayer outputs
  into the residual — scale its attention-output and FFN-output projections by
  `1 / sqrt(2 · num_layers)` (see *Initializing the stack*).
- **`DecoderOnlyTransformer`** — embed → add positional encoding → run `N`
  blocks with the causal mask applied → final LayerNorm → unembed. Write
  `__init__` and `forward`. In `__init__`, initialize the weights this model
  owns directly — a small-normal embedding table and a Xavier unembed (the
  blocks initialize themselves) — and apply dropout to the embedding +
  positional sum in `forward`.

Then, in `src/attention/masking.py`:

- **Find the causal-mask bug**, record its line number as `g_bug_line` in
  `answers.py` (referencing the file as distributed, *before* you edit), and
  **fix it in place**.

### Constraints

- Assemble the transformer from your own parts. `nn.Transformer`,
  `nn.TransformerEncoder`/`Decoder`, and their `*Layer` variants are
  **forbidden** (checked automatically); your `MultiHeadAttention` must be
  used.

## What these tests do NOT check

- **Training convergence / capstone battery accuracy** — observational.
- **Exact forward math or pre-norm placement against a reference** — a
  post-norm block or a different FFN can pass the property checks; the prompt
  still pins pre-norm/GELU as the intended design, and the battery row is the
  behavioral backstop.
- That **free-running generation is correct end-to-end** — `generate` is
  provided; only forward properties and the mask are graded.

Grading is a guide, not a proof of correctness.

## Deliverables

In this order — **observe, diagnose, fix, verify**:

1. Implement `TransformerBlock` and `DecoderOnlyTransformer` in
   `src/attention/models/transformer.py`.
2. Run `make demo-g` **with the mask still buggy**. Read the teacher-forced vs
   free-running gap in `g-accuracy-by-position.png` — note that teacher-forced
   alone looks flawless, and that the free-running collapse is not perfectly
   uniform across positions (watch the token nearest the boundary). Then scan
   `g-heads-leaky.png` and count *how many* heads actually park their mass on
   the marked next-token cell versus how many stay diffuse.
3. Find the causal-mask bug in `src/attention/masking.py` and record
   `g_bug_line` in `answers.py` (the line number in the file *as distributed*,
   before you edit it). Then fix it in place.
4. Run `make demo-g` again: `g-accuracy-by-position.png` now shows both panels
   side by side — free-running has recovered to meet teacher-forced in the fixed
   panel — and `g-heads-correct.png` shows every head at exactly `0%` future
   mass while keeping its own distinct pattern (sharp or diffuse — heads
   specialize, they do not all align). Record `g_mask_observations` and
   `g_leak_status` from what changed between the two runs.
5. Run `make test-g` and confirm the tests pass.
6. Run `make submit-g` to generate `submission.json`.

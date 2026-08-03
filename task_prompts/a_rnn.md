# Task A — The RNN Module

## Background

This assignment builds a small family of sequence models and uses them to
study *what each model can and cannot do*. Every model reads a sequence of
tokens (small integers) and produces another sequence of tokens. The first
model — and the baseline the rest of the assignment improves on — is a
**recurrent neural network (RNN)**.

You have trained models that see their whole input at once (an MLP on a
feature vector, a CNN on an image). A sequence model is different: it reads
the input **one token at a time**, carrying information forward in a *hidden
state*.

### Recurrent networks and the Elman cell

A recurrent network keeps a hidden state vector `h` and updates it at every
time step from the current input and the previous state. The update rule is
the same at every step — the network **shares one set of weights across
time**. This assignment uses the classic **vanilla (Elman) cell**:

```
h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)
```

- `x_t` is the input at step `t` (here, a token embedding).
- `h_{t-1}` is the previous hidden state; `h_0` starts at zeros.
- `W_ih` maps the input into the state; `W_hh` mixes the previous state
  forward; `b` is a bias.
- `tanh` squashes every component into `(-1, 1)`, keeping the state bounded.

Because the same `W_ih`, `W_hh`, `b` are used at every step, the cell is a
single small module applied in a loop.

```
x_1      x_2      x_3
 │        │        │
 ▼        ▼        ▼
[cell]→[cell]→[cell]→ ...      (same weights at every step)
 │        │        │
 h_1      h_2      h_3
```

### Sequence-to-sequence: encoder and decoder

To map an input sequence to an output sequence we use two RNNs:

- The **encoder** reads the source tokens and unrolls the cell over them,
  producing a hidden state after each token. Its **final state** is a single
  vector meant to summarize the entire source.
- The **decoder** is a second RNN. It starts from the encoder's final state
  and generates the target sequence one token at a time, projecting each
  hidden state to a distribution over the vocabulary (the **output head**).

```
source:  x1 x2 x3          target:  BOS  y1 y2  EOS
         └──encoder──┐              └────decoder────┐
                  enc_final ──────► h_0 of decoder
```

**The fixed-context bottleneck.** In this task the decoder sees the source
**only** through that single `enc_final` vector. A fixed-size vector must
summarize an arbitrarily long source, so information about early or distant
tokens gets squeezed out. This bottleneck is the central weakness of the
plain encoder-decoder RNN — and it is exactly what *attention* (Task C)
relieves. Task A does not fix it; it builds the baseline that motivates
everything after.

### Teacher forcing

During training the decoder is fed the **ground-truth** previous target
token at each step, not its own previous prediction. This is *teacher
forcing*: it gives a stable, parallel-friendly training signal. At step `t`
the decoder reads target-input token `t` and predicts target token `t`
(`logits[:, t]` is the prediction for position `t`). Because each step only
consumes inputs up to `t`, the prediction at step `t` never depends on
*future* target tokens. (Free-running *generation*, where the model feeds
its own predictions back in, is provided for you and is not part of this
task.)

### Embeddings, logits, and autograd

Token ids are turned into vectors by an **embedding** table (`nn.Embedding`),
and the decoder's hidden state is turned into vocabulary scores (**logits**)
by a linear **output head** (`nn.Linear`). You build the forward pass out of
differentiable PyTorch operations; PyTorch's autograd then supplies the
backward pass for free. For a recurrence this backward pass is
*backpropagation through time* — gradients flow back through every unrolled
step — but you never write it: a differentiable forward is all that is
needed.

**There is no training in this task.** You build a correct model and it is
graded on its forward pass. Task B supplies the training loop and trains it.

### Initializing the recurrent weights

A layer's weights need starting values before training. PyTorch picks defaults for you, but a recurrent cell is the one place where those defaults are worth improving on, because the *same* `W_hh` multiplies the hidden state at every step and its scale compounds over the sequence. Two small, standard choices make the vanilla RNN train far more reliably:

- **Recurrent matrix `W_hh` — orthogonal.** An orthogonal matrix has every singular value equal to exactly 1, so multiplying by it preserves length. That is precisely what you want when the same matrix is applied over and over: the hidden state neither shrinks toward zero nor blows up across the sequence, and — because backpropagation through time runs that multiplication in reverse — neither does the gradient. This directly counters the vanishing/exploding-gradient problem that makes plain RNNs hard to train. Use `torch.nn.init.orthogonal_(self.h2h.weight)`.
- **Input matrix `W_ih` — Xavier/Glorot.** This map is an ordinary feed-forward layer into a `tanh`. Two terms describe a weight matrix's shape: the *fan-in* is the number of inputs feeding each output unit (here `input_size`), and the *fan-out* is the number of units each input feeds (here `hidden_size`) — for an `nn.Linear` with weight shape `[out, in]` they are simply `in` and `out`. Xavier draws each weight with variance `2 / (fan_in + fan_out)`; the uniform form used here samples from `[-a, a]` with `a = sqrt(6 / (fan_in + fan_out))`, which has exactly that variance. Balancing the two fans this way keeps the pre-activation variance near 1, so `tanh` stays in its useful, roughly-linear middle range instead of saturated near ±1 (where its gradient is almost zero). Use `torch.nn.init.xavier_uniform_(self.x2h.weight)`.
- **Bias — zero.** There is no reason to start the bias anywhere else: `torch.nn.init.zeros_(self.x2h.bias)`.

These are conventions, not laws — the model will still pass this task's tests under PyTorch's defaults — but orthogonal recurrent weights plus a Xavier input map are the usual first recipe for a tanh RNN, and they matter once Task B actually trains it.

### The model interface

Every model in this assignment exposes a class attribute `framing` (the RNN
uses `"encoder_decoder"`) and a `forward(batch)` returning logits of shape
`[B, T, VOCAB]`. The harness reads `framing` to build the right kind of
batch. A `Batch` for the encoder-decoder framing carries `source`,
`target_in` (the teacher-forced decoder input, `BOS y1 … ym`), and the
`targets`/`loss_mask` used later for training.

## What to Implement

Fill in the four classes in `src/attention/models/rnn.py`. You write the
`__init__` (which parameters to register) **and** the `forward` for each —
the stubs give you the signatures, shapes, and the recurrence formula.

1. **`RNNCell`** — one Elman step. Register the recurrence parameters (two `nn.Linear` layers work well) and implement `h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`. Input `x_t` is `[B, input_size]`, `h_prev` is `[B, hidden_size]`, output is `[B, hidden_size]`. Initialize the weights as described in *Initializing the recurrent weights* above: orthogonal `W_hh`, Xavier `W_ih`, zero bias.

2. **`Encoder`** — embed the source, then **unroll the cell over time in an
   explicit Python loop**, collecting every hidden state. Return
   `(all_hidden [B, S, H], final_state [B, H])`. (Task C needs every hidden
   state, which is why you return them all.)

3. **`Decoder`** — embed the teacher-forced target, initialize the hidden
   state from `enc_final`, and unroll the cell over the target, projecting
   each step to vocabulary logits `[B, T, VOCAB]`. In Task A the decoder must
   see the source **only** through `enc_final`: the `enc_states` argument is
   threaded through the signature (Task C will use it) but must **not** be
   consulted while `self.attention is None`. Leave that attention branch
   present but inert.

4. **`Seq2SeqRNN`** — wire `Encoder` → `Decoder`: encode `batch.source`, then
   decode `batch.target_in`, returning logits over the output region.

### Constraints

- Build the forward pass from **differentiable torch operations only**, so
  autograd supplies the backward pass.
- **Write the encoder and decoder loops by hand.** Calling `nn.RNN`,
  `nn.GRU`, `nn.LSTM`, or their `*Cell` variants is **forbidden** — building
  the recurrence yourself is the exercise. This is checked automatically.
  `nn.Linear`, `nn.Embedding`, `torch.tanh`, and an explicit time loop are
  all allowed and expected.
- The decoder's logits at step `t` must not depend on target inputs at
  positions after `t` (a consequence of the left-to-right recurrence).

## What these tests do NOT check

Passing every test does **not** prove your model is fully correct. In
particular the tests do **not** check:

- **Any training, convergence, or accuracy** — Task A grades forward
  properties only; the model is untrained here.
- **The exact recurrence math against a reference** — a forward pass that
  satisfies every checked property but computes a subtly different recurrence
  can still pass. Task B's trained battery row is the behavioral backstop.
- **Initialization quality** — every property holds under any reasonable init, including PyTorch's defaults. The orthogonal/Xavier recipe above is recommended for training reliability in Task B, not a requirement checked here.

Grading is a guide, not a proof of correctness.

## Deliverables

- Implement `RNNCell`, `Encoder`, `Decoder`, and `Seq2SeqRNN` in
  `src/attention/models/rnn.py`.
- Run `make demo-a` to print the shape/parameter walk, then record
  `a_observations` and `a_param_count` in `answers.py`.
- Run `make test-a` and confirm the tests pass.
- Run `make submit-a` to generate `submission.json`.

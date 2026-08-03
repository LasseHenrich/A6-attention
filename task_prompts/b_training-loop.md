# Task B — Fix the Training Loop

## Background

Task A built a correct RNN but never trained it. Training needs a loop that
repeatedly draws a batch, runs a forward pass, computes a loss, and updates
the parameters. You are given such a loop in `src/attention/train.py` — but
it is **subtly broken**. Every function is present and looks plausible; the
bugs are small mistakes in *how* the pieces are wired together. Your job is
to find them, record where they are, and repair them.

This is a **debugging** task. The skill is reading code against its *intended*
behaviour and localizing the defects — different from writing a loop from
scratch. The repaired loop is the **shared spine reused by every later task**
(C–I), so getting it right here makes every later result trustworthy.

### Anatomy of a step-based training loop

The data here is an *infinite* stream of freshly generated batches, so we
train for a fixed number of **steps** rather than epochs. One step is:

```
1. draw a batch
2. optimizer.zero_grad()      # clear gradients left over from last step
3. logits = model(batch)      # forward pass
4. loss = loss_fn(logits, …)  # how wrong are we?
5. loss.backward()            # compute gradients of the loss
6. optimizer.step()           # nudge parameters down the gradient
```

**Order matters.** Each piece has to be in the right place:

- `zero_grad` must happen **before** `backward`, because PyTorch *accumulates*
  gradients — if you never clear them (or clear them at the wrong time), the
  update is computed from stale or wiped-out gradients.
- `optimizer.step()` applies the gradients that `backward` produced. If
  gradients are cleared *after* `backward` but *before* `step`, the step has
  nothing to apply and the model never learns.
- Gradient clipping (`clip_grad_norm_`) rescales the gradients to a maximum
  norm; it must run **after** `backward` and **before** `step`, or it does
  nothing.

### Masked cross-entropy

The loss is cross-entropy, but only over the **output region**. Each batch
carries a `loss_mask` that is 1 on the tokens the model should predict
(`y1 … EOS`) and 0 on prompt tokens and `PAD`. If the loss ignores the mask
and averages over *all* positions, it counts padding and prompt tokens — the
training signal is diluted and the model optimizes the wrong thing. Correct
masked cross-entropy averages the per-position loss over the masked
positions only.

### train() / eval() mode and detaching

- A model has a **train mode** and an **eval mode** (`model.train()` /
  `model.eval()`); layers like dropout behave differently in each. After you
  evaluate (which switches to eval mode), you must switch back to train mode
  before the next training step, or the model trains in the wrong mode.
- Evaluation must not update parameters and should run under
  `torch.no_grad()` so it builds no gradient graph.
- When you record the loss for logging, take `loss.item()` (a plain Python
  float). Keeping the loss **tensor** around holds the whole autograd graph
  alive across steps — a memory leak that matters especially on a GPU.

### Learning-rate schedule and gradient clipping

Two pieces of the loop are about *stability* rather than correctness — they
keep training from blowing up, and both are standard practice for transformers
and recurrent networks alike. Neither is one of the bugs.

- **Gradient clipping** (`clip_grad_norm_`) caps the total norm of the
  gradients just before the optimizer step. A single bad batch — or the
  exploding gradients that deep and recurrent networks are especially prone to
  — can produce an enormous gradient whose step destroys everything learned so
  far. Clipping rescales any over-large gradient back to a fixed norm, so no
  one step can move the parameters too far. It only helps if it runs **after**
  `backward` (the gradients must exist) and **before** `step` (so the step uses
  the clipped values) — which is one of the ordering bugs to find.

- **Learning-rate warmup and decay.** The loop builds an optional schedule in
  `build_scheduler`. **Warmup** raises the LR linearly from ~0 over the first
  `warmup_steps` steps: early in training Adam's estimate of the gradient
  variance is based on almost no data and is unreliable, so a full-size step
  can throw the model into a bad region and destabilize the whole run — warmup
  lets those statistics settle before the steps grow. This is the schedule the
  original Transformer used, and it is near-universal in transformer training
  today. **Cosine decay** (`lr_schedule="cosine"`) then eases the LR down near
  the end so the final steps fine-tune rather than bounce around the minimum.
  The schedule is **provided and correct** — it is *not* a bug, and the config
  defaults leave it off (a flat LR) unless you enable it.

### Reading symptoms

Each bug has an observable symptom. Learning to map symptom → cause is the
method:

- **Loss flat / not decreasing** ⇒ suspect the optimizer step or gradient
  bookkeeping (nothing is actually updating the parameters).
- **Loss decreases but the model ignores obvious structure** ⇒ suspect the
  loss mask or reduction (it is counting the wrong positions).
- **Memory grows over training** ⇒ suspect an un-detached loss tensor.
- **Predictions shift by one, or eval behaves oddly** ⇒ suspect target
  alignment or train/eval mode.

## What to Implement

`src/attention/train.py` contains **exactly 5 bugs**. They span:

- what the loss counts (masking / reduction),
- gradient bookkeeping around the optimizer step,
- where gradient clipping happens,
- detaching the logged loss,
- train vs eval mode in the loop.

Do two things:

1. **Record the bug line numbers in `answers.py`** as `b_bug_lines` — a list
   of 5 line numbers **in the file as you received it**. Do this *before* you
   edit, because editing shifts the line numbers away from the distributed
   baseline.

2. **Repair the loop in place** so it trains correctly. The fixes are
   surgical (a moved line, a changed argument), not a rewrite. Your repaired
   loop must be *behaviourally* correct — matching the intended behaviour —
   not textually identical to any particular solution.

Then run `make demo-b` to train the RNN across the battery with your repaired
loop (the first time anything trains), and record `b_symptom_observations`
and `b_rnn_observations` in `answers.py`.

## What these tests do NOT check

- **Full training convergence or final battery accuracy** — only single
  seeded sub-steps and the reported bug lines are graded; the scoreboard row
  is observational.
- **Correct use of every GPU practice at runtime** — CPU grading checks
  behavioural proxies (the step returns a float; eval leaves parameters
  unchanged), not real cross-device placement.
- **That your edit matches a reference text** — only behaviour is graded, so
  any correct fix passes.

Grading is a guide, not a proof of correctness.

## Deliverables

- Record `b_bug_lines` in `answers.py` (5 line numbers, before editing).
- Repair the 5 bugs in `src/attention/train.py`.
- Run `make demo-b` and record `b_symptom_observations` and
  `b_rnn_observations` in `answers.py`.
- Run `make test-b` and confirm the tests pass.
- Run `make submit-b` to generate `submission.json`.

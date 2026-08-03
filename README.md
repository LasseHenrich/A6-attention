# A6 — Implementing & Interrogating Attention Mechanisms

This assignment builds a small library of sequence models — a recurrent
network, several attention mechanisms, and two transformer architectures —
and uses them to interrogate *what each mechanism can and cannot do*. You
implement one component per task, run it on a shared battery of synthetic
algorithmic tasks, and record what you observe.

## Getting started

```bash
make install        # git init + uv sync
make test-a         # run the tests for a task
make submit-a       # write submission.json for a task
make demo-a         # run a task's demo and refresh its figures
```

Upload `submission.json` to the course page after running `make submit`.

## Tasks

| # | Topic | You implement |
|---|-------|---------------|
| a | RNN module | a recurrent cell + encoder/decoder shell |
| b | Fix the training loop | repair a broken trainer; report the bug lines |
| c | Additive (Bahdanau) attention | additive scoring + context vector |
| d | Scaled dot-product attention | `softmax(QKᵀ/√dₖ)V` |
| e | Multi-head attention | per-head projections, concat, output proj |
| f | Positional encoding & ALiBi | sinusoidal PE + ALiBi distance bias |
| g | Decoder-only transformer | assemble the causal stack; fix the mask bug |
| h | Encoder + cross-attention | bidirectional encoder + cross-attention |
| i | Linear (kernel) attention | feature map φ + reordered running state |

See `task_prompts/` for the full description of each task.

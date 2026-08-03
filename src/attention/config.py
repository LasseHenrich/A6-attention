"""Experiment configuration for A6-attention.

``ExperimentConfig`` is a single flat, immutable dataclass.  It is
**provided complete from Task A with every field any task will ever use**
and sensible defaults, so students never edit it.  A stable config means a
stable ``config_hash`` across the whole assignment (see ``run.py``): adding
nothing and editing nothing keeps every run's identity fixed.

The "first used in" comments are a *reading guide* — which task first
exercises a field — not an implementation schedule.  Every field exists
from Task A.

Provided scaffold — students never modify this module.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    # --- reproducibility (a) --------------------------------------------
    # Two independent seeds: weight init vs the shared data stream.  Keeping
    # them separate lets the data be identical across mechanisms while the
    # weights differ per architecture (see harness / run docs).
    init_seed: int = 0
    data_seed: int = 1234

    # --- vocabulary / sequence (a) --------------------------------------
    vocab_size: int = 22
    max_len: int = 64

    # --- model dimensions (a; num_heads e; layers/ff/dropout g) ---------
    d_model: int = 64
    hidden_size: int = 64  # RNN hidden state width
    attn_dim: int = 64  # additive-attention hidden width (c)
    num_heads: int = 4  # (e)
    num_layers: int = 2  # transformer depth (g)
    d_ff: int = 256  # transformer FFN width, ~4 * d_model (g)
    dropout: float = 0.0  # (g); 0.0 keeps forward deterministic

    # --- positional scheme (f) ------------------------------------------
    # one of "none" | "sinusoidal" | "learned" | "alibi"
    positional: str = "sinusoidal"

    # --- optimization (a; grad_clip / eval_every / schedule referenced by b) --
    batch_size: int = 64
    steps: int = 2000
    lr: float = 3e-3
    grad_clip: float | None = 1.0
    eval_every: int = 100
    # LR schedule (b).  Defaults are a no-op — a flat LR, exactly as before —
    # so adding these fields leaves every existing run's ``config_hash`` and
    # training dynamics unchanged.  ``warmup_steps > 0`` ramps the LR up from
    # ~0 over that many steps; ``lr_schedule="cosine"`` decays it afterwards.
    warmup_steps: int = 0
    lr_schedule: str = "constant"  # "constant" | "cosine"
    min_lr_ratio: float = 0.0  # cosine floor, as a fraction of peak lr

    # --- linear attention (i) -------------------------------------------
    feature_map: str = "elu"  # feature map phi for LinearAttention
    bench_lengths: tuple[int, ...] = field(default=(128, 256, 512, 1024, 2048))

    # --- device ---------------------------------------------------------
    # Grading runs CPU-only; device never enters a hashed artifact.
    device: str = "cpu"

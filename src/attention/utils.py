"""Cross-cutting utilities: seeding, threading, timing, and memory.

All provided — students never modify this module.
"""

import gc
import random
import time
import tracemalloc
from typing import Callable, NamedTuple

import numpy as np
import torch


class TimingResult(NamedTuple):
    """Wall-clock timing summary in seconds (see :func:`timeit`)."""

    median: float
    mean: float
    std: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch global RNGs from *seed*.

    Call this once before building a model so weight initialization is
    reproducible.  The data stream uses a *local* ``random.Random`` seeded
    from ``cfg.data_seed`` (see ``attention.data``), so shuffling never
    couples to this global state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def use_single_thread() -> None:
    """Pin PyTorch inter-op and intra-op parallelism to one thread.

    Call before any timing experiment so wall-clock measurements are
    reproducible and comparable across machines.
    """
    torch.set_num_threads(1)
    # Inter-op parallelism can only be set before any parallel work has
    # started in the process; if that has already happened, the intra-op
    # setting above still gives single-threaded, reproducible timing.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def timeit(
    fn: Callable[[], None],
    *,
    warmup: int = 3,
    n: int = 10,
) -> TimingResult:
    """Time *fn* with warm-up, discarding the first *warmup* calls.

    Reports both median and mean.  The median is robust to the occasional
    slow pass from OS scheduling, so it is the more reproducible run-to-run
    summary.  Garbage collection is disabled during the timed loop so
    collection pauses do not land inside a measured pass.
    """
    for _ in range(warmup):
        fn()
    times: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()
    arr = np.array(times)
    return TimingResult(float(np.median(arr)), float(arr.mean()), float(arr.std()))


def peak_memory(fn: Callable[[], None]) -> int:
    """Return the peak traced Python allocation of *fn*, in bytes.

    Uses ``tracemalloc`` so the figure is process-local and does not depend
    on OS accounting.  Purely observational — never hashed.
    """
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)

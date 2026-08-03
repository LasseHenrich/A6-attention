"""Shared plotting style: rc params, a colorblind-safe palette, helpers.

Centralizes figure size/DPI, the palette, axis conventions, and the
machine-relative normalization helper so no timing plot ever shows an
absolute wall-clock.  Matplotlib uses the Agg backend (no display).

Provided scaffold — students never modify this subpackage.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colorblind-safe qualitative palette (Wong 2011).
PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

FIG_SIZE = (7.0, 4.5)
DPI = 130

# Marker color for reference cells drawn over a heatmap (the argsort target,
# the leaked diagonal).  Chosen to stay legible on both viridis and magma.
OVERLAY_COLOR = "#FF3B30"


def new_figure(title: str = "", xlabel: str = "", ylabel: str = "") -> tuple[plt.Figure, plt.Axes]:
    """Return a styled ``(fig, ax)`` with the shared size/DPI and labels."""
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def new_panels(count: int, title: str = "") -> tuple[plt.Figure, list[plt.Axes]]:
    """Return ``(fig, axes)`` for *count* side-by-side matrix panels.

    Wider than ``new_figure`` and grid-free: these axes hold ``imshow`` panels,
    where a grid would sit on top of the cells.
    """
    fig, axes = plt.subplots(1, count, figsize=(5.2 * count, 4.6), dpi=DPI, squeeze=False)
    if title:
        fig.suptitle(title)
    return fig, list(axes[0])


def color(index: int) -> str:
    """Stable palette color for series *index*."""
    return PALETTE[index % len(PALETTE)]


def relative_to_baseline(values: list[float], baseline: float) -> list[float]:
    """Normalize timing/memory *values* to a per-machine *baseline* ratio.

    Timing and peak memory are machine-dependent and are never shown as
    absolute numbers — only as a ratio to a baseline the student measures on
    their own computer.
    """
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return [v / baseline for v in values]


def save(fig: plt.Figure, path: str) -> None:
    """Save *fig* to *path* (tight layout) and close it."""
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

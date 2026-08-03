"""Matrix figures: attention maps, cross-attention alignments, dependency maps.

An attention row is a probability distribution over keys, so a weight is a
**magnitude** — it gets a single-hue sequential ramp (pale = no mass, dark =
all mass), never a categorical or rainbow map.  Ground truth is drawn *on top*
of the ramp as an accent-colored open marker, so the reader compares "where the
model looked" against "where it should have looked" in one glance.

Provided scaffold — students never modify this subpackage.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from attention.plotting import style

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm
from matplotlib.image import AxesImage

# Magnitude in [0, 1] -> one hue, light to dark.  Matches palette slot 0.
SEQUENTIAL_CMAP = "Blues"

# The ground-truth overlay is not a data series; it is an annotation, so it
# takes the accent hue (palette slot 1) and never a sequential step.
OVERLAY_COLOR = style.PALETTE[1]

# Cell coordinates as ``(row, col)`` pairs.
Overlay = Sequence[tuple[int, int]]


def _apply_ticks(
    ax: Axes,
    matrix: np.ndarray,
    xticklabels: Sequence[str] | None,
    yticklabels: Sequence[str] | None,
    *,
    xtick_rotation: float = 90,
    ytick_rotation: float = 0,
) -> None:
    """Label every cell when labels are supplied, else fall back to indices.

    A rotation strictly between 0 and 90 is anchored at its right end so a long
    diagonal label points at its own tick rather than drifting off it.
    """
    n_rows, n_cols = matrix.shape
    if xticklabels is not None:
        ax.set_xticks(range(n_cols))
        ha = "right" if 0 < xtick_rotation < 90 else "center"
        ax.set_xticklabels(xticklabels, rotation=xtick_rotation, ha=ha, rotation_mode="anchor", fontsize=7)
    else:
        ax.set_xticks(range(0, n_cols, max(1, n_cols // 12)))
    if yticklabels is not None:
        ax.set_yticks(range(n_rows))
        va = "bottom" if 0 < ytick_rotation < 90 else "center"
        ax.set_yticklabels(yticklabels, rotation=ytick_rotation, ha="right", va=va, rotation_mode="anchor", fontsize=7)
    else:
        ax.set_yticks(range(0, n_rows, max(1, n_rows // 12)))


def _draw_overlay(ax: Axes, overlay: Overlay, label: str | None) -> None:
    """Mark the ground-truth cell of each row with an open accent square."""
    if not overlay:
        return
    rows = [r for r, _ in overlay]
    cols = [c for _, c in overlay]
    ax.scatter(
        cols,
        rows,
        s=60,
        facecolors="none",
        edgecolors=OVERLAY_COLOR,
        linewidths=1.6,
        label=label,
    )
    if label is not None:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)


def _imshow(
    ax: Axes,
    matrix: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float | None,
    annotate: bool,
    gamma: float | None = None,
) -> AxesImage:
    # A trained attention row is nearly one-hot: one cell near 1.0 and a long
    # tail near 0.  On a linear ramp the tail is invisible, so ``gamma < 1``
    # expands the low end without ever reordering magnitudes.
    if gamma is not None:
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=1.0 if vmax is None else vmax)
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    else:
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.grid(False)  # a grid over cells is noise; the cell edges carry the structure
    if annotate:
        hi = float(np.nanmax(matrix)) if matrix.size else 1.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                # Ink stays readable against its own cell, never colored by series.
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value > 0.6 * hi else "black")
    return im


def heatmap(
    matrix: np.ndarray,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    cbar_label: str = "attention weight",
    out: str,
    cmap: str = SEQUENTIAL_CMAP,
    vmin: float = 0.0,
    vmax: float | None = None,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    overlay: Overlay | None = None,
    overlay_label: str | None = None,
    annotate: bool = False,
    gamma: float | None = None,
    xtick_rotation: float = 90,
    ytick_rotation: float = 0,
) -> None:
    """Render one ``[rows, cols]`` matrix as a sequential heatmap.

    ``overlay`` marks ``(row, col)`` cells with an open accent square — used to
    draw the *correct* source position on top of a learned alignment.
    """
    fig, ax = style.new_figure(title, xlabel, ylabel)
    im = _imshow(ax, matrix, cmap=cmap, vmin=vmin, vmax=vmax, annotate=annotate, gamma=gamma)
    _apply_ticks(ax, matrix, xticklabels, yticklabels, xtick_rotation=xtick_rotation, ytick_rotation=ytick_rotation)
    if overlay is not None:
        _draw_overlay(ax, overlay, overlay_label)
    fig.colorbar(im, ax=ax, label=cbar_label)
    style.save(fig, out)


def heatmap_grid(
    panels: Sequence[tuple[str, np.ndarray]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    cbar_label: str = "attention weight",
    out: str,
    cmap: str = SEQUENTIAL_CMAP,
    vmin: float = 0.0,
    vmax: float | None = None,
    ncols: int | None = None,
    overlays: Sequence[Overlay | None] | None = None,
    overlay_label: str | None = None,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    annotate: bool = False,
    gamma: float | None = None,
    xtick_rotation: float = 90,
    ytick_rotation: float = 0,
) -> None:
    """Small multiples of matrices under **one shared color scale**.

    Panels are drawn in the caller's order (deterministic by construction).  A
    shared ``vmax`` is computed across every panel unless one is given, so a
    dark cell means the same thing in every panel — per-panel autoscaling would
    make weak and strong alignments look identical.
    """
    if not panels:
        raise ValueError("heatmap_grid needs at least one panel")
    if vmax is None:
        vmax = max(float(np.nanmax(m)) for _, m in panels)

    n = len(panels)
    cols = ncols if ncols is not None else min(n, 3)
    rows = (n + cols - 1) // cols
    width, height = style.FIG_SIZE
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols / 2.0, height * rows / 1.6), dpi=style.DPI, squeeze=False, layout="constrained")
    if title:
        fig.suptitle(title)

    im: AxesImage | None = None
    for idx, (label, matrix) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        im = _imshow(ax, matrix, cmap=cmap, vmin=vmin, vmax=vmax, annotate=annotate, gamma=gamma)
        ax.set_title(label, fontsize=9)
        # y-tick labels repeat down a shared axis, so only the leftmost column
        # carries them; inner panels would just stack duplicates.
        first_col = idx % cols == 0
        panel_yticklabels = yticklabels if first_col else None
        _apply_ticks(ax, matrix, xticklabels, panel_yticklabels, xtick_rotation=xtick_rotation, ytick_rotation=ytick_rotation)
        if not first_col and yticklabels is not None:
            ax.set_yticks([])  # no index fallback on inner panels when named labels exist
        if overlays is not None and overlays[idx] is not None:
            _draw_overlay(ax, overlays[idx], overlay_label if idx == 0 else None)
        if idx % cols == 0:
            ax.set_ylabel(ylabel)
        if idx // cols == rows - 1:
            ax.set_xlabel(xlabel)

    for idx in range(n, rows * cols):  # blank the unused cells of a ragged grid
        axes[idx // cols][idx % cols].axis("off")

    assert im is not None
    fig.colorbar(im, ax=axes.ravel().tolist(), label=cbar_label, fraction=0.025)
    fig.savefig(out)
    plt.close(fig)

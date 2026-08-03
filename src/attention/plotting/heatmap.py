"""Attention-map and dependency-structure heatmaps (g, h).

Where ``scoreboard.py`` draws one annotated accuracy table, this module draws
side-by-side matrix panels on a shared color scale — the shape a comparison of
two attention maps needs.  ``overlay`` marks reference cells (the leaked
super-diagonal, the argsort target) so the reader can see whether the mass
lands where it should.

Provided scaffold — students never modify this subpackage.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from attention.plotting import style


def attention_heatmap(
    panels: dict[str, np.ndarray],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    out: str,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    overlay: Sequence[tuple[int, int]] | None = None,
    overlay_label: str = "",
    cmap: str = "viridis",
    vmin: float = 0.0,
    cbar_label: str = "",
) -> None:
    """Draw one matrix panel per entry of *panels*, sharing a color scale.

    ``panels`` maps ``panel_label -> 2-D array``.  Panels are drawn in
    insertion order (callers pass an ordered dict; the order is meaningful —
    "before" then "after").  ``overlay`` is a sequence of ``(row, col)`` cells
    marked with open squares on every panel.
    """
    if not panels:
        raise ValueError("attention_heatmap needs at least one panel")
    vmax = max(float(m.max()) for m in panels.values())

    fig, axes = style.new_panels(len(panels), title)
    for ax, (label, matrix) in zip(axes, panels.items()):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(xlabel)
        if xticklabels is not None:
            ax.set_xticks(range(len(xticklabels)))
            ax.set_xticklabels(xticklabels, fontsize=7, rotation=90)
        if yticklabels is not None:
            ax.set_yticks(range(len(yticklabels)))
            ax.set_yticklabels(yticklabels, fontsize=7)
        if overlay:
            rows = [r for r, _ in overlay]
            cols = [c for _, c in overlay]
            ax.scatter(
                cols,
                rows,
                marker="s",
                s=64,
                facecolors="none",
                edgecolors=style.OVERLAY_COLOR,
                linewidths=1.3,
                label=overlay_label or None,
            )
            if overlay_label:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        bar = fig.colorbar(im, ax=ax, fraction=0.046)
        if cbar_label:
            bar.set_label(cbar_label, fontsize=8)
    axes[0].set_ylabel(ylabel)
    style.save(fig, out)


def diagonal_overlay(n: int, offset: int) -> list[tuple[int, int]]:
    """Cells ``(i, i + offset)`` inside an ``n x n`` grid — a marked diagonal."""
    return [(i, i + offset) for i in range(n) if 0 <= i + offset < n]

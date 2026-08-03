"""The battery scoreboard table / heatmap.

Provided scaffold — students never modify this subpackage.
"""

from __future__ import annotations

import numpy as np

from attention.plotting import style


def scoreboard_heatmap(
    scoreboards: dict[str, dict[str, float]],
    *,
    title: str = "",
    out: str,
) -> None:
    """Heatmap of per-task accuracy across mechanisms.

    ``scoreboards`` maps ``mechanism_label -> {task -> accuracy}``.  Rows are
    mechanisms, columns are battery tasks, sorted for deterministic output.
    """
    mechs = sorted(scoreboards)
    tasks = sorted({t for m in scoreboards.values() for t in m})
    data = np.array([[scoreboards[m].get(t, float("nan")) for t in tasks] for m in mechs])

    fig, ax = style.new_figure(title)
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(len(mechs)))
    ax.set_yticklabels(mechs)
    for i in range(len(mechs)):
        for j in range(len(tasks)):
            ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
            )
    fig.colorbar(im, ax=ax, label="token accuracy")
    style.save(fig, out)

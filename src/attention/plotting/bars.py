"""Bar / grouped-bar charts (per-head entropy, single-vs-multi, TF-vs-free).

Provided scaffold — students never modify this subpackage.
"""

from __future__ import annotations

import numpy as np

from attention.plotting import style


def grouped_bars(
    groups: dict[str, dict[str, float]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    out: str,
) -> None:
    """Plot grouped bars: ``group_label -> {series_label -> value}``.

    Categories (x) are the group labels; within each group one bar per
    series.  Sorted order throughout for deterministic output.
    """
    group_labels = sorted(groups)
    series_labels = sorted({s for g in groups.values() for s in g})
    n_series = max(len(series_labels), 1)
    width = 0.8 / n_series
    x = np.arange(len(group_labels))

    fig, ax = style.new_figure(title, xlabel, ylabel)
    for i, series in enumerate(series_labels):
        heights = [groups[g].get(series, 0.0) for g in group_labels]
        ax.bar(x + i * width, heights, width, color=style.color(i), label=series)
    ax.set_xticks(x + width * (n_series - 1) / 2)
    ax.set_xticklabels(group_labels)
    ax.legend()
    style.save(fig, out)

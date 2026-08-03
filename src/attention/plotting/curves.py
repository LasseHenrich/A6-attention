"""Line plots vs an x-axis (training curves, ablation sweeps, cost-vs-length).

Three forms live here.  ``line_plot`` draws named series on one axis, with an
optional across-seed spread band and an optional reference line (chance, a
threshold).  ``line_grid`` facets the same series into small multiples, which
is what a comparison across more than a handful of tasks needs — past ~6
series a single axis stops being readable and more hues do not fix it.

Provided scaffold — students never modify this subpackage.
"""

from __future__ import annotations

from typing import Sequence

from attention.plotting import style

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator

# ``label -> (xs, ys)``; with a band, ``label -> (xs, lo, hi)``.
Series = dict[str, tuple[list[float], list[float]]]
Bands = dict[str, tuple[list[float], list[float]]]


def _reference_lines(ax: plt.Axes, levels: Sequence[tuple[float, str]], *, labelled: bool = True) -> None:
    """Recessive solid hairlines — never dashed, never a series color.

    Reference levels (chance, a prior, a threshold) are context, not data: they
    stay in muted ink so they read behind the series rather than beside them.
    """
    for level, label in levels:
        ax.axhline(level, color="0.45", linewidth=1.0, zorder=0)
        if labelled and label:
            ax.text(0.995, level, f" {label}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="0.35")


def _as_levels(hline: float | None, hline_label: str | None, hlines: Sequence[tuple[float, str]] | None) -> list[tuple[float, str]]:
    if hlines is not None:
        return list(hlines)
    if hline is None:
        return []
    return [(hline, hline_label or "")]


def _plot_series(ax: plt.Axes, series: Series, bands: Bands | None, *, marker: str | None) -> None:
    for i, label in enumerate(sorted(series)):
        xs, ys = series[label]
        ax.plot(xs, ys, marker=marker, color=style.color(i), label=label, linewidth=2.0)
        if bands is not None and label in bands:
            lo, hi = bands[label]
            ax.fill_between(xs, lo, hi, color=style.color(i), alpha=0.18, linewidth=0)


def line_plot(
    series: Series,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    out: str,
    logx: bool = False,
    logy: bool = False,
    bands: Bands | None = None,
    hline: float | None = None,
    hline_label: str | None = None,
    hlines: Sequence[tuple[float, str]] | None = None,
    marker: str | None = "o",
    xticks: Sequence[float] | None = None,
) -> None:
    """Plot named ``label -> (xs, ys)`` series and save to *out*.

    Series are drawn in sorted label order for deterministic output.  ``bands``
    maps a label to ``(lo, hi)`` envelopes (e.g. min/max across seeds) shaded
    under that series' own hue, so a claimed effect can be read against its
    seed noise instead of asserted from one run.  ``hline``/``hlines`` draw
    reference levels (chance accuracy, a prior, a threshold).
    """
    fig, ax = style.new_figure(title, xlabel, ylabel)
    _reference_lines(ax, _as_levels(hline, hline_label, hlines))
    _plot_series(ax, series, bands, marker=marker)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xticks is not None:
        # Explicit ticks are the categories being compared; a log axis would
        # otherwise sprinkle "3 x 10^0" minor labels between them.
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(list(xticks))
        ax.set_xticklabels([f"{x:g}" for x in xticks])
    ax.legend()
    style.save(fig, out)


def line_grid(
    panels: Sequence[tuple[str, Series]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    out: str,
    ncols: int | None = None,
    bands: Sequence[Bands | None] | None = None,
    hline: float | None = None,
    hline_label: str | None = None,
    hlines: Sequence[tuple[float, str]] | None = None,
    sharey: bool = True,
    marker: str | None = None,
    xinteger: bool = False,
    label_refs_each: bool = False,
) -> None:
    """Small multiples: one panel per ``(panel_title, series)``, shared axes.

    Panels keep the caller's order.  Every panel uses the same hue for the same
    series label, so a series' color follows the entity across the grid; the
    legend is drawn once, on the first panel.  ``xinteger`` forces integer x
    ticks (for count/index axes).  ``label_refs_each`` labels the reference
    lines on every panel rather than only the first — useful when a shared
    reference (e.g. chance) sits far from the data in some panels.
    """
    if not panels:
        raise ValueError("line_grid needs at least one panel")
    n = len(panels)
    cols = ncols if ncols is not None else min(n, 3)
    rows = (n + cols - 1) // cols
    width, height = style.FIG_SIZE
    fig, axes = plt.subplots(
        rows, cols, figsize=(width * cols / 2.2, height * rows / 1.5), dpi=style.DPI, squeeze=False, sharey=sharey, sharex=True, layout="constrained"
    )
    if title:
        fig.suptitle(title)

    for idx, (panel_title, series) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        ax.grid(True, alpha=0.3)
        ax.set_title(panel_title, fontsize=9)
        _reference_lines(ax, _as_levels(hline, hline_label, hlines), labelled=label_refs_each or idx == 0)
        _plot_series(ax, series, None if bands is None else bands[idx], marker=marker)
        if xinteger:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if idx % cols == 0:
            ax.set_ylabel(ylabel)
        if idx // cols == rows - 1:
            ax.set_xlabel(xlabel)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.savefig(out)
    plt.close(fig)

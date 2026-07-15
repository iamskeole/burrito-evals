"""
research_plots.py — consistent, publication-ready charts for exploratory research.

Call `set_theme()` once at the top of your notebook, then use the chart helpers below.
Every helper returns the matplotlib Axes so you can keep tweaking it, and
`savefig()` writes both a PNG (for quick viewing) and a vector PDF/SVG (for the paper)
from the same figure so your notebook plot and paper figure never drift apart.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Okabe-Ito colorblind-safe palette — distinguishable in print and by colorblind readers.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def set_theme(context: str = "notebook", palette: Sequence[str] = PALETTE) -> None:
    """One call, consistent look everywhere: no gridline clutter, no default-blue
    matplotlib ugliness, colorblind-safe categorical palette, crisp export DPI."""
    sns.set_theme(context=context, style="ticks", palette=palette)
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "pdf.fonttype": 42,   # embed real fonts (not Type 3) so paper PDFs stay editable
        "ps.fonttype": 42,
    })


def savefig(fig: plt.Figure, name: str, folder: str = "figures",
            formats: Sequence[str] = ("pdf", "png")) -> None:
    """Save `fig` under `folder/name.<ext>` for every format in `formats`.
    Keep 'pdf' or 'svg' in there for LaTeX \\includegraphics; 'png' for quick sharing."""
    os.makedirs(folder, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(folder, f"{name}.{fmt}"), bbox_inches="tight")


def _finish(ax, title, xlabel, ylabel):
    if title:
        ax.set_title(title, pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def bar(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
        order: Optional[Sequence] = None, hue_order: Optional[Sequence] = None,
        title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
        ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """Grouped bar chart with bootstrap error bars — e.g. mean latency by backend x effort."""
    ax = ax or plt.gca()
    sns.barplot(data=df, x=x, y=y, hue=hue, order=order, hue_order=hue_order,
                errorbar=("ci", 95), capsize=0.08, err_kws={"linewidth": 1.2}, ax=ax, **kwargs)
    if hue:
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    return _finish(ax, title, xlabel, ylabel or y)


def violin(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
           order: Optional[Sequence] = None, split: bool = False,
           title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
           ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """Distribution shape per group — e.g. latency spread per config, catches bimodality
    that a mean±std bar chart hides."""
    ax = ax or plt.gca()
    sns.violinplot(data=df, x=x, y=y, hue=hue, order=order, split=split,
                    inner="quartile", cut=0, linewidth=1, density_norm="width", ax=ax, **kwargs)
    if hue:
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    return _finish(ax, title, xlabel, ylabel or y)


def line(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
         order: Optional[Sequence] = None,
         title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
         ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """Trend across an ordered category — e.g. accuracy vs reasoning_effort per model.
    Pass `order=['low','medium','high']` to control the x-axis ordering explicitly,
    since pandas won't know it's ordinal on its own."""
    ax = ax or plt.gca()
    d = df.copy()
    if order is not None:
        d[x] = pd.Categorical(d[x], categories=order, ordered=True)
        d = d.sort_values(x)
    sns.lineplot(data=d, x=x, y=y, hue=hue, marker="o", errorbar=("ci", 95), ax=ax, **kwargs)
    if hue:
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    return _finish(ax, title, xlabel, ylabel or y)


def _place_labels_no_overlap(ax, xy, labels, font_size=8.5, color="#333333",
                              base_radius=10, max_radius=70, n_radii=7, n_directions=16):
    """Radial-search label placement: for each point, try candidate offsets at
    increasing radius/angle and keep the first one that doesn't collide with an
    already-placed label or another point marker. Draws a thin leader line when
    the label ends up more than a few points away from its marker.

    This is a dependency-free stand-in for the `adjustText` package (not always
    installable in locked-down environments) — good enough for the dozens-of-points
    scale a magic quadrant plot usually has.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # treat each marker as a small box in display (pixel) space, so labels avoid points too
    marker_r = 7
    point_boxes = []
    for (px, py) in xy:
        dx, dy = ax.transData.transform((px, py))
        point_boxes.append((dx - marker_r, dy - marker_r, dx + marker_r, dy + marker_r))

    def overlaps(b1, b2, pad=2.0):
        return not (b1[2] + pad < b2[0] or b2[2] + pad < b1[0]
                    or b1[3] + pad < b2[1] or b2[3] + pad < b1[1])

    placed_boxes = []
    radii = np.linspace(base_radius, max_radius, n_radii)
    angles = np.linspace(0, 360, n_directions, endpoint=False)

    for (px, py), lab in zip(xy, labels):
        best = None
        for radius in radii:
            for ang in angles:
                dx = radius * np.cos(np.radians(ang))
                dy = radius * np.sin(np.radians(ang))
                ha = "left" if dx >= 0 else "right"
                va = "bottom" if dy >= 0 else "top"
                t = ax.annotate(lab, xy=(px, py), xytext=(dx, dy), textcoords="offset points",
                                 fontsize=font_size, color=color, ha=ha, va=va, zorder=4)
                fig.canvas.draw()
                bb = t.get_window_extent(renderer=renderer)
                box = (bb.x0, bb.y0, bb.x1, bb.y1)
                conflict = any(overlaps(box, pb) for pb in point_boxes) or \
                           any(overlaps(box, pb) for pb in placed_boxes)
                if not conflict:
                    best = (t, box, radius, dx, dy)
                    break
                t.remove()
            if best:
                break
        if best is None:
            # nothing collision-free found within search range — place at max radius anyway
            dx, dy = max_radius, max_radius
            t = ax.annotate(lab, xy=(px, py), xytext=(dx, dy), textcoords="offset points",
                             fontsize=font_size, color=color, ha="left", va="bottom", zorder=4)
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=renderer)
            best = (t, (bb.x0, bb.y0, bb.x1, bb.y1), max_radius, dx, dy)

        t, box, radius, dx, dy = best
        placed_boxes.append(box)
        if radius > base_radius + 5:  # far enough from the point that a leader line helps
            ax.annotate("", xy=(px, py), xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6,
                                         shrinkA=0, shrinkB=6), zorder=2)


def magic_quadrant(df: pd.DataFrame, x: str, y: str, label: str,
                    x_thresh: Optional[float] = None, y_thresh: Optional[float] = None,
                    quadrant_labels: Optional[Sequence[str]] = None,
                    title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                    annotate: bool = True, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """Scatter split into four quadrants by median (or explicit thresholds) — e.g.
    accuracy (y) vs latency (x) tradeoffs across configs, one point per config.

    `quadrant_labels` (optional): 4 short strings for [top-left, top-right, bottom-left,
    bottom-right], e.g. ("slow but accurate", "best of both", "avoid", "fast but weak").

    Point labels use collision-avoidance so nearby points don't render overlapping text.
    """
    ax = ax or plt.gca()
    xt = x_thresh if x_thresh is not None else df[x].median()
    yt = y_thresh if y_thresh is not None else df[y].median()

    sns.scatterplot(data=df, x=x, y=y, s=90, ax=ax, zorder=3, **kwargs)
    ax.axvline(xt, color="#999999", linestyle="--", linewidth=1, zorder=1)
    ax.axhline(yt, color="#999999", linestyle="--", linewidth=1, zorder=1)

    if annotate:
        xy = list(zip(df[x], df[y]))
        labels = [str(v) for v in df[label]]
        _place_labels_no_overlap(ax, xy, labels)

    if quadrant_labels:
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        positions = [(xlo, yhi, "left", "top"), (xhi, yhi, "right", "top"),
                     (xlo, ylo, "left", "bottom"), (xhi, ylo, "right", "bottom")]
        for (px, py, ha, va), lab in zip(positions, quadrant_labels):
            ax.text(px, py, lab, ha=ha, va=va, fontsize=8.5, color="#999999", style="italic")

    return _finish(ax, title, xlabel or x, ylabel or y)
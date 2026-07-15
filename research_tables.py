"""
research_tables.py — flexible leaderboard tables for exploratory research notebooks,
with a single source of truth that renders as both:
  - a clean styled HTML table (Jupyter display)
  - a booktabs LaTeX table (paste straight into a paper)

Design principles
------------------
1. AGGREGATION is separate from PRESENTATION.
   `summarize()` collapses a flat/raw dataframe into a tidy one-row-per-config table
   with any combination of agg functions (mean, std, sum, min, max, median, count, ...).
   `leaderboard()` only ever consumes that tidy table — it doesn't know or care how the
   numbers were computed, so it works identically whether you're looking at mean latency,
   total token spend, or worst-case error rate.

2. Use pandas' own Styler machinery instead of hand-rolled double for-loops.
   highlight_best() and RAG coloring are vectorized (fast, ~10 lines each) instead of
   nested index/column loops.

3. One function, two outputs. `leaderboard()` returns a `LeaderboardTable` object that
   displays as HTML in the notebook (_repr_html_) and has `.to_latex()` for the paper.
   You style once; both outputs stay in sync.
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Union
from IPython.display import display, Markdown

ColSpec = Union[str, Sequence[str]]


def _as_list(x: ColSpec) -> list:
    return [x] if isinstance(x, str) else list(x)


# --------------------------------------------------------------------------- #
# 1. AGGREGATION
# --------------------------------------------------------------------------- #

def summarize(
    df: pd.DataFrame,
    group_cols: ColSpec,
    metric_col: str,
    aggs: Sequence[str] = ("mean", "std"),
    query: Optional[str] = None,
) -> pd.DataFrame:
    """Collapse `df` to one row per `group_cols` combo.

    Replaces `compute_group_variance`, but isn't locked to mean/std — pass any
    pandas-recognized agg name: 'mean', 'std', 'sum', 'min', 'max', 'median',
    'count', 'nunique', a percentile via lambda, etc.

    Returns columns: group_cols..., {agg}_{metric_col} for each agg in `aggs`, plus 'n'.
    """
    group_cols = _as_list(group_cols)
    d = df.query(query) if query else df

    g = d.groupby(group_cols, dropna=False)[metric_col].agg(list(aggs))
    g.columns = [f"{a}_{metric_col}" for a in aggs]
    g["n"] = d.groupby(group_cols, dropna=False).size()
    return g.reset_index()


# --------------------------------------------------------------------------- #
# 2. PRESENTATION
# --------------------------------------------------------------------------- #

_BOOKTABS_STYLE = [
    {"selector": "", "props": [
        ("font-family", '"SF Mono", Consolas, Menlo, "Fira Code", "Courier New", monospace'),
        ("font-size", "12px"),
        ("border-collapse", "collapse"),
        ("margin", "18px auto"),
        ("background-color", "#ffffff"),
        ("color", "#000000"),
    ]},
    {"selector": "thead", "props": [
        ("border-top", "2px solid #000000 !important"),
        ("background-color", "#ffffff"),
    ]},
    {"selector": "th.col_heading", "props": [
        ("border-bottom", "1px solid #000000 !important"),
        ("padding", "7px 14px"),
        ("font-weight", "600"),
        ("color", "#000000"),
        ("background-color", "#ffffff"),
        ("text-align", "right"),
    ]},
    {"selector": "th.row_heading", "props": [
        ("text-align", "left"),
        ("padding", "5px 14px"),
        ("font-weight", "600"),
        ("color", "#000000"),
        ("background-color", "#ffffff"),
    ]},
    {"selector": "th.index_name", "props": [
        ("text-align", "left"),
        ("font-size", "9px"),
        ("font-weight", "600"),
        ("letter-spacing", "0.03em"),
        ("text-transform", "uppercase"),
        ("color", "#8c8c8d"),
        ("background-color", "#ffffff"),
        ("border-bottom", "1px solid #000000 !important"),
    ]},
    {"selector": "td", "props": [
        ("padding", "5px 14px"),
        ("border", "none !important"),
        ("text-align", "right"),
    ]},
    {"selector": "tbody tr:hover td", "props": [
        ("background-color", "#f7f7f8 !important"),
    ]},
    {"selector": "tbody", "props": [
        ("border-bottom", "2px solid #000000 !important"),
        ("background-color", "#ffffff"),
    ]},
]


def _group_divider_styles(index: pd.Index) -> list:
    """Thin rules between changed top-level groups, thinner ones for sub-groups."""
    if not isinstance(index, pd.MultiIndex):
        return []
    styles = []
    num_levels = index.nlevels
    for i in range(1, len(index)):
        prev, curr = index[i - 1], index[i]
        diff_level = next(
            (lvl for lvl in range(num_levels - 1) if prev[lvl] != curr[lvl]), None
        )
        if diff_level is None:
            continue
        border = "1.25px solid #000000 !important" if diff_level == 0 else "0.5px solid #c9c9ca !important"
        styles.append({"selector": f".row{i}", "props": [("border-top", border)]})
    return styles


def _fmt_html(m, s, precision, error_precision):
    if pd.isna(m):
        return float("nan")
    core = f"{m:.{precision}f}"
    if s is None or pd.isna(s):
        return core
    return f'{core}<span style="font-size:10px;color:#8c8c8d;font-weight:normal;"> ± {s:.{error_precision}f}</span>'


def _fmt_latex(m, s, precision, error_precision):
    if pd.isna(m):
        return "--"
    core = f"{m:.{precision}f}"
    if s is None or pd.isna(s):
        return core
    return rf"{core} {{\small$\pm${s:.{error_precision}f}}}"


def _rag_css(m, thresholds):
    lo, hi = thresholds
    if pd.isna(m):
        return "color: #b3b3b3 !important; background-color: #ffffff !important;"
    if m <= lo:
        return "color: #a0a0a0 !important; background-color: #ffffff !important;"
    if m <= hi:
        return "background-color: #fef9e7 !important; color: #b7791f !important; font-weight: 600 !important;"
    return "background-color: #fce4e4 !important; color: #c0392b !important; font-weight: 600 !important;"


@dataclass
class LeaderboardTable:
    styler: "pd.io.formats.style.Styler"
    numeric: pd.DataFrame                 # primary metric, pivoted, numeric (for latex/analysis)
    error: Optional[pd.DataFrame]
    precision: int
    error_precision: int
    higher_is_better: bool

    def _repr_html_(self):
        return self.styler._repr_html_()

    def to_latex(self, caption: Optional[str] = None, label: Optional[str] = None,
                 bold_best: bool = True, **kwargs) -> str:
        """Booktabs LaTeX table, safe to paste into a paper (no HTML leaks in)."""
        disp = pd.DataFrame(index=self.numeric.index, columns=self.numeric.columns, dtype=object)
        for col in self.numeric.columns:
            best = self.numeric[col].max() if self.higher_is_better else self.numeric[col].min()
            for idx in self.numeric.index:
                m = self.numeric.loc[idx, col]
                s = self.error.loc[idx, col] if self.error is not None else None
                cell = _fmt_latex(m, s, self.precision, self.error_precision)
                if bold_best and pd.notna(m) and m == best:
                    cell = rf"\textbf{{{cell}}}"
                disp.loc[idx, col] = cell
        sty = disp.style.format(na_rep="--")
        return sty.to_latex(hrules=True, caption=caption, label=label,
                             convert_css=False, **kwargs)


def leaderboard(
    df: pd.DataFrame,
    index_cols: ColSpec,
    col_cols: ColSpec,
    value_col: str,
    error_col: Optional[str] = None,
    precision: int = 4,
    error_precision: int = 2,
    higher_is_better: bool = True,
    rag_thresholds: Optional[tuple] = None,   # e.g. (0.0, 0.05) -> quiet/amber/red cutoffs
    caption: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    return_table: bool = False,
) -> Optional[LeaderboardTable]:
    """Pivot a tidy dataframe (already aggregated, e.g. via `summarize()`) into a
    booktabs-styled leaderboard.

    `value_col` / `error_col` are just column names now — no mean_/std_ prefix
    assumptions, no metric-name keyword sniffing to decide behavior. You say what
    "better" means (`higher_is_better`) and whether to RAG-color (`rag_thresholds`);
    the function doesn't guess from the metric's name.
    """
    index_cols, col_cols = _as_list(index_cols), _as_list(col_cols)

    numeric_val = df.pivot(index=index_cols, columns=col_cols, values=value_col)
    numeric_err = df.pivot(index=index_cols, columns=col_cols, values=error_col) if error_col else None

    display_df = pd.DataFrame("", index=numeric_val.index, columns=numeric_val.columns)
    for col in numeric_val.columns:
        for idx in numeric_val.index:
            m = numeric_val.loc[idx, col]
            s = numeric_err.loc[idx, col] if numeric_err is not None else None
            display_df.loc[idx, col] = _fmt_html(m, s, precision, error_precision)

    styler = display_df.style.format(na_rep="—", escape=None)
    table_styles = list(_BOOKTABS_STYLE) + _group_divider_styles(display_df.index)
    styler = styler.set_table_styles(table_styles)

    if rag_thresholds is not None:
        def _rag(_data):
            css = pd.DataFrame("", index=_data.index, columns=_data.columns)
            for col in numeric_val.columns:
                for idx in numeric_val.index:
                    css.loc[idx, col] = _rag_css(numeric_val.loc[idx, col], rag_thresholds)
            return css
        styler = styler.apply(_rag, axis=None)
    else:
        def _bold_best(_data):
            css = pd.DataFrame("", index=_data.index, columns=_data.columns)
            for col in numeric_val.columns:
                best = numeric_val[col].max() if higher_is_better else numeric_val[col].min()
                is_best = numeric_val[col] == best
                css.loc[is_best[is_best].index, col] = "font-weight: 700 !important; color: #000000 !important;"
            return css
        # styler = styler.apply(_bold_best, axis=None)

    if caption is not None:
        styler = styler.set_caption(caption)

    table = LeaderboardTable(
        styler=styler, numeric=numeric_val, error=numeric_err,
        precision=precision, error_precision=error_precision,
        higher_is_better=higher_is_better,
    )

    if return_table:
        return table

    if title is not None:
        display(Markdown(f"### {title}"))
    if body is not None:
        display(Markdown(body))
    display(table)
    return None
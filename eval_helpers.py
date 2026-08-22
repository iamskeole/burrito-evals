"""
eval_helpers.py
Unified evaluation analysis, pivoting, and publication-ready visualization suite.

Plot methods return ``(fig, axes, data)`` so charts and the underlying
aggregates can be handed off together. Layout, encoding, and export polish
iterated with Grok (xAI).
"""

from __future__ import annotations

import re
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from pandas.api.types import CategoricalDtype, is_object_dtype, is_string_dtype
from typing import List, Optional, Tuple, Union, Dict

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


# ===================================================================
# 0. VISUAL STYLE SYSTEM — "Airy Modern Tech" (publication grade)
# ===================================================================
# Design principles:
#   • Quiet neutral canvas, one deliberate color vocabulary tied to
#     backend identity (stable across every chart and every kernel).
#   • Zero chart-junk: no heavy spines, no tick marks, soft grids.
#   • Typography sized like conference figures / model cards.
#   • Consistent margins, footers, and legend treatment.

INK = '#0f172a'       # headline / strongest ink
SLATE = '#334155'     # body / axis labels
MUTED = '#64748b'     # secondary labels
FAINT = '#94a3b8'     # captions, footnotes
GRID = '#e2e8f0'      # gridlines
SPINE = '#cbd5e1'     # axis spines
CANVAS = '#ffffff'
BAND = '#f8fafc'      # subtle band fills

FONT_STACK = [
    'Inter', 'SF Pro Display', 'Helvetica Neue', 'Arial',
    'Liberation Sans', 'DejaVu Sans',
]

# Backend identity → color. Most-specific pattern first.
# Tailwind-inspired backend palette: saturated accents for the interesting
# stacks, soft slate/gray for the vanilla baselines.
BACKEND_COLOR_RULES: List[Tuple[str, str]] = [
    # Same-hue pairs: base = -500, pt = -700 (darker, not washed out)
    # llama family → amber; vllm family → violet
    ('burrito-pt@llamacpp',    '#b45309'),  # amber-700  (pt · llama)
    ('burrito@llamacpp',       '#f59e0b'),  # amber-500  (base · llama)
    ('burrito-pt@vllm',        '#6d28d9'),  # violet-700 (pt · vllm)
    ('burrito@vllm',           '#8b5cf6'),  # violet-500 (base · vllm)
    ('llamacpp@fixed-jinja',   '#f43f5e'),  # rose-500
    ('llamacpp@default-jinja', '#a8b0bc'),  # cool light gray (vanilla)
    ('vllm',                   '#c4c4c4'),  # warm light gray (vanilla)
]

FALLBACK_PALETTE = [
    '#6366f1',  # indigo-500
    '#ec4899',  # pink-500
    '#22c55e',  # green-500
    '#a855f7',  # purple-500
    '#f97316',  # orange-500
    '#06b6d4',  # cyan-500
    '#84cc16',  # lime-500
    '#e879f9',  # fuchsia-400
    '#2dd4bf',  # teal-400
    '#fb7185',  # rose-400
]

DEFAULT_DPI = 300

# Left→right (and top→bottom) ideal backend order: custom stacks first,
# fixed jinja, then vanilla baselines at the end.
BACKEND_DISPLAY_ORDER: List[str] = [
    'burrito-pt@llamacpp',
    'burrito-pt@vllm',
    'burrito@llamacpp',
    'burrito@vllm',
    'llamacpp@fixed-jinja',
    'llamacpp@default-jinja',
    'vllm',
]


def _configure_matplotlib_style() -> None:
    """Applies the shared visual language once, at import time."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': FONT_STACK,
        'font.size': 9,
        'text.color': SLATE,
        'axes.titlesize': 12,
        'axes.titleweight': 700,
        'axes.titlelocation': 'left',
        'axes.titlepad': 14,
        'axes.labelsize': 9,
        'axes.labelweight': 700,
        'axes.labelcolor': SLATE,
        'axes.edgecolor': SPINE,
        'axes.linewidth': 0.7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': SLATE,
        'ytick.color': SLATE,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        'legend.fontsize': 7.5,
        'legend.frameon': False,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.5,
        'legend.labelspacing': 0.45,
        'legend.columnspacing': 1.2,
        'figure.facecolor': CANVAS,
        'axes.facecolor': CANVAS,
        'savefig.facecolor': CANVAS,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.18,
        'figure.dpi': 120,
        'axes.grid': False,
        'grid.color': GRID,
        'grid.linewidth': 0.65,
        'grid.alpha': 0.85,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'patch.force_edgecolor': True,
    })


_configure_matplotlib_style()


def save_figure(fig, path: str, dpi: int = 300, transparent: bool = False) -> str:
    """Saves a figure with consistent, publication-ready export settings."""
    fig.savefig(
        path, dpi=dpi, bbox_inches='tight', pad_inches=0.18,
        transparent=transparent, facecolor=fig.get_facecolor(),
    )
    return path


# ===================================================================
# 1. DATA PREPARATION & FILTERING UTILITIES
# ===================================================================

DEFAULT_CATEGORY_ORDERS = {
    'reasoning_effort': ['none', 'low', 'medium', 'high'],
    'reasoning_level': ['none', 'low', 'medium', 'high'],
    'reasoning': ['none', 'low', 'medium', 'high'],
    'model_size': ['1b', '3b', '7b', '8b', '14b', '20b', '32b', '70b', '405b'],
    'quantization': ['fp16', 'q8_0', 'q6_k', 'q4_k_m', 'q3_k_m', 'q2_k'],
    'fc_model': [0, 1],
    'browser_enabled': [0, 1],
    'python_enabled': [0, 1],
    'test_type': ['multi_turn', 'live', 'non_live', 'GPT-OSS'],
    'wire_api': ['chat', 'responses'],
    'backend': list(BACKEND_DISPLAY_ORDER),
}

AUTO_ORDER_MAX_CATEGORIES = 40


def apply_sql_filter(df: pd.DataFrame, filter_query: Optional[str]) -> pd.DataFrame:
    """Translates SQL-like syntax (LIKE, AND, OR, =) or Python expressions and filters DF."""
    if not filter_query or not str(filter_query).strip():
        return df

    q = str(filter_query).strip()
    q = re.sub(r"(\w+)\s+LIKE\s+'([^%]+)%'", r"\1.str.startswith('\2')", q, flags=re.IGNORECASE)
    q = re.sub(r"(\w+)\s+LIKE\s+'%([^%]+)'", r"\1.str.endswith('\2')", q, flags=re.IGNORECASE)
    q = re.sub(r"(\w+)\s+LIKE\s+'%([^%]+)%'", r"\1.str.contains('\2')", q, flags=re.IGNORECASE)
    q = re.sub(r"(?<![!=<>])=(?![=])", "==", q)
    q = re.sub(r"\bAND\b", "and", q, flags=re.IGNORECASE)
    q = re.sub(r"\bOR\b", "or", q, flags=re.IGNORECASE)
    q = re.sub(r"\bNOT\b", "not", q, flags=re.IGNORECASE)

    try:
        return df.query(q, engine='python')
    except Exception as e:
        raise ValueError(
            f"Failed to apply filter query '{filter_query}' (parsed as '{q}'): {e}"
        )


def apply_categorical_orders(
    df: pd.DataFrame,
    custom_orders: dict = None,
    auto_order_low_cardinality: bool = True,
    max_auto_categories: int = AUTO_ORDER_MAX_CATEGORIES,
) -> pd.DataFrame:
    """
    Enforces logical categorical ordering on the dataframe.

    Two passes:
    1. Columns named in DEFAULT_CATEGORY_ORDERS / custom_orders get the
       explicit order given (e.g. reasoning_effort: none < low < medium < high).
    2. Any other low-cardinality text column (<= max_auto_categories unique
       values) is stamped as Categorical in first-seen order. This keeps
       things like `test_name` or `backend` from being silently re-sorted
       alphabetically by groupby/pivot/unstack.
    """
    df_out = df.copy()
    orders = DEFAULT_CATEGORY_ORDERS.copy()
    if custom_orders:
        orders.update(custom_orders)

    handled_cols = set()
    for col, categories in orders.items():
        if col in df_out.columns:
            existing = [c for c in categories if c in df_out[col].unique()]
            leftovers = [
                c for c in df_out[col].unique()
                if c not in existing and pd.notna(c)
            ]
            full_order = existing + leftovers
            cat_type = CategoricalDtype(categories=full_order, ordered=True)
            df_out[col] = df_out[col].astype(cat_type)
            handled_cols.add(col)

    if auto_order_low_cardinality:
        for col in df_out.columns:
            if col in handled_cols:
                continue
            series = df_out[col]
            if not (is_object_dtype(series) or is_string_dtype(series)):
                continue
            nunique = series.nunique(dropna=True)
            if 1 < nunique <= max_auto_categories:
                order = list(pd.unique(series.dropna()))
                df_out[col] = series.astype(
                    CategoricalDtype(categories=order, ordered=True)
                )

    return df_out


def ensure_seed_col(
    df: pd.DataFrame, seed_col: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """Ensures a valid seed column exists, extracting from run_name if needed."""
    df_out = df.copy()
    if seed_col and seed_col in df_out.columns:
        return df_out, seed_col
    if '_seed_id' in df_out.columns:
        return df_out, '_seed_id'
    if 'seed' in df_out.columns:
        return df_out, 'seed'
    if 'run_name' in df_out.columns:
        extracted = df_out['run_name'].str.extract(r'_s-(\d+)', expand=False)
        df_out['_seed_id'] = extracted.fillna(df_out['run_name'])
        return df_out, '_seed_id'
    raise KeyError(
        "Could not find or extract a seed column. Please specify `seed_col`."
    )


def shorten_label(name: str) -> str:
    """Shortens long model/backend names for clean chart labels.

    Preserve-thinking stacks share the same stem as their base backend
    with a ``bp-`` prefix so pairs stay comparable:
      burrito@llamacpp      → burrito-llama
      burrito-pt@llamacpp   → burrito-pt-llama
      burrito@vllm          → burrito-vllm
      burrito-pt@vllm       → burrito-pt-vllm
    """
    s = str(name).strip()
    # Longer patterns first so pt@… is not partially rewritten by base@…
    replacements = [
        ("llamacpp@default-jinja", "llama-default"),
        ("llamacpp@fixed-jinja", "llama-fixed"),
        ("burrito-pt@llamacpp", "burrito-pt-llama"),
        ("burrito-pt@vllm", "burrito-pt-vllm"),
        ("burrito@llamacpp", "burrito-llama"),
        ("burrito@vllm", "burrito-vllm"),
        ("gpt-oss-20b-FC", "20b-FC"),
        ("gpt-oss-20b", "20b"),
        ("fc_model=", "fc="),
        ("fc_model=0", "fc=0"),
        ("fc_model=1", "fc=1"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


# Marker shapes for reasoning-effort levels (shared by all plots).
EFFORT_MARKERS: Dict[str, str] = {
    'none': 'x',
    'low': 'o',      # circle
    'medium': 'D',   # diamond
    'high': 's',     # square
}


def get_effort_marker(name_or_effort) -> str:
    """Return matplotlib marker for a reasoning-effort level or label containing one."""
    s = str(name_or_effort).lower()
    for key, mk in EFFORT_MARKERS.items():
        if key in s:
            return mk
    return 'o'


def strip_fc_effort_label(name: str) -> str:
    """Strip fc / effort tokens from a combined group label for clean backend legends.

    Examples
    --------
    'b-vllm·fc0·lo' → 'b-vllm'
    'burrito@vllm · 0' → 'b-vllm'
    'burrito-llama (fc_model=1, reasoning_effort=high)' → 'burrito-llama'
    """
    s = shorten_label(str(name))
    # Collapse common ultra-short stems first (same as _bar_tick_label)
    s = (
        s.replace('burrito-pt-llama', 'bp-llama')
         .replace('burrito-pt-vllm', 'bp-vllm')
         .replace('burrito-llama', 'b-llama')
         .replace('burrito-vllm', 'b-vllm')
         .replace('llama-default', 'l-def')
         .replace('llama-fixed', 'l-fix')
    )
    # Parenthetical factors → ·tokens then strip
    s = re.sub(r'wire_api\s*=\s*responses', 're', s, flags=re.I)
    s = re.sub(r'wire_api\s*=\s*chat', 'ch', s, flags=re.I)
    s = re.sub(r'fc_model\s*=\s*([01])', r'fc\1', s, flags=re.I)
    s = re.sub(r'fc\s*=\s*([01])', r'fc\1', s, flags=re.I)
    s = re.sub(
        r'reasoning[_\s-]?(?:effort|level)?\s*=\s*(none|low|medium|high)',
        r'\1', s, flags=re.I,
    )
    def _paren_to_dots(m):
        inner = m.group(1)
        parts = re.split(r'[,\s]+', inner.strip())
        parts = [p for p in parts if p]
        return '·' + '·'.join(parts) if parts else ''
    s = re.sub(r'\(([^)]*)\)', _paren_to_dots, s)
    # Normalize spaces around · and bare fc digits from multi-col joins
    s = re.sub(r'\s*·\s*', '·', s)
    s = re.sub(r'\s+', '·', s.strip())
    # Strip fc / effort tokens (named and bare trailing 0/1)
    s = re.sub(r'[·\s]*fc[_=]?[01]\b', '', s, flags=re.I)
    s = re.sub(r'[·\s]*[01](?=[·\s]|$)', '', s)
    s = re.sub(
        r'[·\s]*(?:none|low|lo|medium|md|high|hi)\b',
        '', s, flags=re.I,
    )
    s = re.sub(r'[·\s]+', '·', s).strip('· \t')
    return s or shorten_label(name)


def get_semantic_style(name) -> Tuple[str, str, str]:
    """
    Maps a group label to (color, linestyle, marker).

    Contract (aligned across the module):
      • color     = backend family identity (stable palette)
      • linestyle = fc_model when present: dotted for fc=0, solid for fc=1
                    (no fc factor → solid)
      • marker    = default 'o' (effort-aware plots override via get_effort_marker)

    Effort markers (○ low / ◇ medium / □ high) and fc fill (hollow / solid)
    are handled by individual plot methods so legends stay non-duplicative:
    backend legend shows family only (e.g. b-vllm), with separate entries
    for effort shapes and fc0/fc1.
    """
    s = str(name).lower()

    color = None
    for pattern, c in BACKEND_COLOR_RULES:
        if pattern in s:
            color = c
            break
    if color is None:
        if 'default' in s:
            color = '#94a3b8'
        elif 'fixed' in s:
            color = '#f43f5e'
        elif 'burrito' in s:
            color = '#8b5cf6'
        else:
            digest = hashlib.md5(s.encode('utf-8')).hexdigest()
            color = FALLBACK_PALETTE[int(digest, 16) % len(FALLBACK_PALETTE)]

    # Detect fc factor in many label forms:
    #   "fc_model=1", "fc=0", "fc1", "· 1", trailing bare 0/1 from combined
    #   group labels like "burrito@vllm · 0" (group_col=['backend','fc_model'])
    is_fc1 = bool(
        re.search(r'fc[_=]?model\s*=\s*1|(?<![\d.])fc\s*=\s*1|(?<![\w])fc1\b', s)
    ) or s in ('1', '1.0') or 'tool' in s
    is_fc0 = bool(
        re.search(r'fc[_=]?model\s*=\s*0|(?<![\d.])fc\s*=\s*0|(?<![\w])fc0\b', s)
    ) or s in ('0', '0.0')
    # Combined multi-factor labels from _prepare_data / plot helpers:
    # "backend · 0" or "backend · 1" (last token is the fc value)
    if not is_fc1 and not is_fc0:
        m_bare = re.search(r'[·,\s]([01])(?:\s*$|\s*[·,])', s)
        if m_bare:
            if m_bare.group(1) == '1':
                is_fc1 = True
            else:
                is_fc0 = True
        elif re.search(r'(?:^|[·\s])1(?:\s*$)', s) and not re.search(
            r'reasoning|effort|level|wire|chat|resp', s
        ):
            # lone trailing 1 that isn't an effort/wire token
            is_fc1 = True
        elif re.search(r'(?:^|[·\s])0(?:\s*$)', s) and not re.search(
            r'reasoning|effort|level|wire|chat|resp', s
        ):
            is_fc0 = True

    has_fc = is_fc1 or is_fc0 or bool(
        re.search(r'fc[_=]?model\s*=|\bfc\s*=|(?<![\w])fc[01]\b', s)
    )
    if has_fc:
        linestyle = '-' if is_fc1 else 'dotted'
    else:
        linestyle = '-'
    marker = 'o'  # default; effort plots use get_effort_marker

    return color, linestyle, marker


def get_bar_style(name) -> tuple:
    """
    Visual encoding for multi-factor bar identity (depth, not clutter).

    Factors are applied only when present in the series label:
      • color  → always (backend family)
      • alpha  → only if fc_model appears  (fc=1 ~0.90 · fc=0 faded)
      • dots   → only if wire_api appears  (responses dotted · chat solid)
      • edge   → full-strength backend color border on every bar

    When the hue is backend-only (no fc / no wire_api), every bar is solid
    at full opacity — no encoding chrome.
    """
    color, ls, _marker = get_semantic_style(name)
    s = str(name).lower()

    has_fc = bool(re.search(r'fc_model\s*=|\bfc\s*=', s))
    has_wire = bool(re.search(r'wire[_\s-]?api\s*=', s))

    if has_fc:
        is_fc1 = (ls == '--') or bool(
            re.search(r'fc_model\s*=\s*1|\bfc\s*=\s*1', s)
        )
        alpha = 0.90 if is_fc1 else 0.42
    else:
        alpha = 1.0

    if has_wire:
        if re.search(r'wire[_\s-]?api\s*=\s*chat', s):
            is_chat = True
        elif re.search(r'wire[_\s-]?api\s*=\s*responses', s):
            is_chat = False
        else:
            is_chat = bool(re.search(r'\bchat\b', s)) and not bool(
                re.search(r'\bresponses\b', s)
            )
        # chat = dotted; responses = clean solid
        hatch = '...' if is_chat else None
    else:
        hatch = None

    # Light vanilla fills need a slightly stronger border to stay defined
    edge = _darken(color, 0.18) if alpha >= 0.85 else color
    # Always use a readable border for very light fills
    if color.lower() in ('#c4c4c4', '#a8b0bc', '#94a3b8', '#cbd5e1', '#e2e8f0'):
        edge = _darken(color, 0.28)
    lw = 0.85
    return color, hatch, edge, alpha, lw


def fmt_metric(v: float, force_decimals: Optional[int] = None) -> str:
    """Format accuracy / rate for bar labels.

    Adaptive precision so near-zero values aren't rounded away to "0":
      ≥ 10  → 0 decimals   (88)
      ≥ 1   → 1 decimal    (3.2)
      < 1   → 2 decimals   (0.17)
    """
    if v is None or (isinstance(v, float) and (v != v)):  # NaN
        return "—"
    v = float(v)
    if force_decimals is not None:
        return f"{v:.{force_decimals}f}"
    av = abs(v)
    if av == 0:
        return "0"
    if av < 1:
        return f"{v:.2f}"
    if av < 10:
        return f"{v:.1f}"
    return f"{v:.0f}"



def is_percent_metric(col: Optional[str] = None, label: Optional[str] = None) -> bool:
    """Heuristic: accuracy / success / pct metrics should lock a 0–100 axis."""
    s = f"{col or ''} {label or ''}".lower()
    keys = (
        'correct', 'accuracy', 'acc', 'pct', 'percent', 'is_error',
        'success_rate', 'success_pct', 'num_turns_success',
    )
    return any(k in s for k in keys)


def percent_axis_limit(data_max: float, headroom: float = 8.0) -> float:
    """Ceiling for a percent axis: always at least 100, plus label headroom."""
    base = max(100.0, float(data_max) if data_max is not None else 0.0)
    return min(130.0, base + headroom)

def detect_encoding_factors(labels) -> tuple:
    """Return (has_fc, has_wire) across a set of series labels.

    Recognises explicit tokens (``fc_model=0``, ``fc=1``, ``fc0``) and the
    bare trailing ``· 0`` / ``· 1`` form produced when ``group_col`` is a
    multi-column list that includes ``fc_model``.
    """
    has_fc = False
    has_wire = False
    for name in labels:
        s = str(name).lower()
        if (
            re.search(r'fc_model\s*=|\bfc\s*=|(?<![\w])fc[01]\b', s)
            or re.search(r'[·,\s][01](?:\s*$|\s*[·,])', s)
            or s.strip() in ('0', '1', '0.0', '1.0')
        ):
            has_fc = True
        if re.search(r'wire[_\s-]?api\s*=|\bchat\b|\bresponses\b', s):
            has_wire = True
        if has_fc and has_wire:
            break
    return has_fc, has_wire


def _lighten(hex_color: str, factor: float = 0.35) -> str:
    """Blend a hex color toward white by `factor` (0=unchanged, 1=white)."""
    r, g, b = to_rgb(hex_color)
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def _darken(hex_color: str, factor: float = 0.35) -> str:
    """Blend a hex color toward black by `factor` (0=unchanged, 1=black)."""
    r, g, b = to_rgb(hex_color)
    r = r * (1 - factor)
    g = g * (1 - factor)
    b = b * (1 - factor)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# ===================================================================
# 2. PIVOT ENGINE & RESULT CONTAINER
# ===================================================================

class EvalPivotResult:
    """Container holding aggregated pivot results, formatters, and filtered DF."""

    def __init__(
        self,
        numeric_df: pd.DataFrame,
        formatted_df: pd.DataFrame,
        index_cols: List[str],
        col_cols: List[str],
        filtered_df: pd.DataFrame = None,
        col_label_map: Optional[Dict] = None,
        idx_label_map: Optional[Dict] = None,
    ):
        self.numeric_df = numeric_df
        self.formatted_df = formatted_df
        self.index_cols = index_cols
        self.col_cols = col_cols
        self.filtered_df = filtered_df
        self.col_label_map = col_label_map or {}
        self.idx_label_map = idx_label_map or {}

    def _get_highlighted_df(
        self,
        highlight_best: Optional[str] = 'max',
        highlight_axis: str = 'row',
        wrapper_fmt: str = "**{}**",
    ) -> pd.DataFrame:
        df_out = self.formatted_df.copy()
        if not highlight_best or self.numeric_df.empty:
            return df_out

        num_df = self.numeric_df
        # Map numeric MultiIndex cols → flat display headers
        def to_fmt_col(col):
            if col in df_out.columns:
                return col
            mapped = self.col_label_map.get(col)
            if mapped is not None and mapped in df_out.columns:
                return mapped
            return None

        def to_fmt_idx(idx):
            if idx in df_out.index:
                return idx
            mapped = self.idx_label_map.get(idx)
            if mapped is not None and mapped in df_out.index:
                return mapped
            # single-level backend shorten
            if isinstance(idx, str):
                s = shorten_label(idx)
                if s in df_out.index:
                    return s
            return None

        if highlight_axis == 'row':
            for idx in num_df.index:
                fidx = to_fmt_idx(idx)
                if fidx is None:
                    continue
                row_vals = num_df.loc[idx]
                valid_vals = row_vals.dropna()
                if valid_vals.empty:
                    continue
                target_val = (
                    valid_vals.max() if highlight_best == 'max' else valid_vals.min()
                )
                best_cols = row_vals[
                    np.isclose(row_vals.astype(float), target_val, atol=1e-6, equal_nan=False)
                ].index
                for col in best_cols:
                    fcol = to_fmt_col(col)
                    if fcol is None:
                        continue
                    val_str = str(df_out.loc[fidx, fcol])
                    if val_str not in ("N/A", "nan", "None", "—", "-"):
                        df_out.loc[fidx, fcol] = wrapper_fmt.format(val_str)

        elif highlight_axis == 'col':
            for col in num_df.columns:
                fcol = to_fmt_col(col)
                if fcol is None or fcol not in df_out.columns:
                    continue
                col_vals = num_df[col]
                valid_vals = col_vals.dropna()
                if valid_vals.empty:
                    continue
                target_val = (
                    valid_vals.max() if highlight_best == 'max' else valid_vals.min()
                )
                best_idxs = col_vals[
                    np.isclose(col_vals.astype(float), target_val, atol=1e-6, equal_nan=False)
                ].index
                for idx in best_idxs:
                    fidx = to_fmt_idx(idx)
                    if fidx is None:
                        continue
                    val_str = str(df_out.loc[fidx, fcol])
                    if val_str not in ("N/A", "nan", "None", "—", "-"):
                        df_out.loc[fidx, fcol] = wrapper_fmt.format(val_str)

        return df_out


    def with_column_style(self, style: str = 'compact') -> "EvalPivotResult":
        """Return a copy with column headers re-styled (compact | verbose)."""
        if not self.col_label_map:
            return self
        # Rebuild from original MultiIndex keys stored as map keys
        # formatted_df already has compact labels; rebuild from numeric cols
        ordered = list(self.numeric_df.columns)
        new_map = {
            c: _format_col_key(self.col_cols, c, style=style) for c in ordered
        }
        inv_old = {v: k for k, v in self.col_label_map.items()}
        # Current formatted columns are compact strings → map back to keys → new
        fmt = self.formatted_df.copy()
        rename = {}
        for col in fmt.columns:
            key = inv_old.get(col)
            if key is not None and key in new_map:
                rename[col] = new_map[key]
            elif col in new_map:
                rename[col] = new_map[col]
        fmt = fmt.rename(columns=rename)
        return EvalPivotResult(
            self.numeric_df, fmt, self.index_cols, self.col_cols,
            filtered_df=self.filtered_df, col_label_map=new_map,
        )

    def split_by(self, level: Union[str, int]) -> Dict[str, "EvalPivotResult"]:
        """Split a wide MultiIndex pivot into narrower tables by one column factor.

        Parameters
        ----------
        level : str or int
            Column level name (e.g. ``'wire_api'``) or position.

        Returns
        -------
        dict
            Mapping of level-value → EvalPivotResult with that slice of columns.
        """
        if not isinstance(self.numeric_df.columns, pd.MultiIndex):
            return {'all': self}
        if isinstance(level, str):
            if level not in self.col_cols:
                raise KeyError(f"{level!r} not in column factors {self.col_cols}")
            level_idx = self.col_cols.index(level)
        else:
            level_idx = int(level)

        remaining = [c for i, c in enumerate(self.col_cols) if i != level_idx]
        out = {}
        for val in self.numeric_df.columns.get_level_values(level_idx).unique():
            mask = self.numeric_df.columns.get_level_values(level_idx) == val
            num = self.numeric_df.loc[:, mask].copy()
            # drop the split level from MultiIndex
            if isinstance(num.columns, pd.MultiIndex):
                num.columns = num.columns.droplevel(level_idx)
            # rebuild formatted labels for remaining levels
            ordered = list(num.columns)
            # columns may now be tuples of remaining length or scalars
            label_map = {
                c: _format_col_key(
                    remaining,
                    c if isinstance(c, tuple) else (c,),
                    style='compact',
                )
                for c in ordered
            }
            # Align formatted via original map
            inv = {v: k for k, v in self.col_label_map.items()}
            fmt_cols = {}
            for c in ordered:
                # reconstruct original full key
                if isinstance(c, tuple):
                    full = list(c)
                    full.insert(level_idx, val)
                    full_key = tuple(full)
                else:
                    full = [c]
                    full.insert(level_idx, val)
                    full_key = tuple(full)
                compact = self.col_label_map.get(full_key)
                if compact is not None and compact in self.formatted_df.columns:
                    fmt_cols[label_map[c]] = self.formatted_df[compact]
            fmt = pd.DataFrame(fmt_cols, index=self.formatted_df.index)
            out[str(val)] = EvalPivotResult(
                num, fmt, self.index_cols, remaining,
                filtered_df=self.filtered_df, col_label_map=label_map,
            )
        return out

    def to_markdown(
        self,
        highlight_best: Optional[str] = 'max',
        highlight_axis: str = 'row',
    ) -> str:
        df = self._get_highlighted_df(
            highlight_best, highlight_axis, wrapper_fmt="**{}**"
        )
        # Prefer backend short names on the index for scanability
        if isinstance(df.index, pd.Index) and not isinstance(df.index, pd.MultiIndex):
            df = df.rename(index=lambda x: shorten_label(x) if isinstance(x, str) else x)
        elif isinstance(df.index, pd.MultiIndex):
            df.index = pd.MultiIndex.from_tuples([
                tuple(shorten_label(v) if isinstance(v, str) else v for v in key)
                for key in df.index
            ], names=df.index.names)
        try:
            return df.to_markdown()
        except ImportError:
            return df.to_string()

    def to_html(
        self,
        highlight_best: Optional[str] = 'max',
        highlight_axis: str = 'row',
    ) -> str:
        return self._get_highlighted_df(
            highlight_best, highlight_axis, wrapper_fmt="<b>{}</b>"
        ).to_html(escape=False)

    def to_latex(
        self,
        highlight_best: Optional[str] = 'max',
        highlight_axis: str = 'row',
        caption: str = "",
        label: str = "",
        resize_to_fit: bool = True,
    ) -> str:
        """Export a paper-ready LaTeX table (booktabs-friendly).

        Special characters in labels (``_``, ``·``, ``—``) are sanitized so the
        result compiles on Overleaf without manual cleanup.
        """
        def _tex_escape_label(s: str) -> str:
            s = str(s)
            s = s.replace('·', '--')          # middle-dot → --
            s = s.replace('—', '--')          # em dash
            s = s.replace('–', '-')
            s = s.replace('_', r'\_')
            s = s.replace('%', r'\%')
            s = s.replace('&', r'\&')
            s = s.replace('#', r'\#')
            s = s.replace('±', r'$\pm$')
            return s

        def _tex_cell(x) -> str:
            if x is None or (isinstance(x, float) and x != x):
                return '--'
            s = str(x)
            if s in ('—', '-', '–', 'nan', 'NaN', 'N/A', 'None', ''):
                return '--'
            # Already highlighted?
            bold = s.startswith('\\textbf{') and s.endswith('}')
            core = s[8:-1] if bold else s
            core = (
                core.replace(' ± ', r' $\pm$ ')
                    .replace('±', r'$\pm$')
                    .replace('—', '--')
                    .replace('·', '--')
            )
            # Don't escape $ \ already introduced for \pm
            return f'\\textbf{{{core}}}' if bold else core

        df_fmt = self._get_highlighted_df(
            highlight_best, highlight_axis, wrapper_fmt="\\textbf{{{}}}"
        ).copy()

        # Sanitize index / column labels
        if isinstance(df_fmt.index, pd.MultiIndex):
            df_fmt.index = pd.MultiIndex.from_tuples([
                tuple(_tex_escape_label(v) for v in key) for key in df_fmt.index
            ], names=[_tex_escape_label(n) if n else n for n in df_fmt.index.names])
        else:
            df_fmt.index = pd.Index(
                [_tex_escape_label(i) for i in df_fmt.index],
                name=_tex_escape_label(df_fmt.index.name)
                if df_fmt.index.name else None,
            )
        if isinstance(df_fmt.columns, pd.MultiIndex):
            df_fmt.columns = pd.MultiIndex.from_tuples([
                tuple(_tex_escape_label(v) for v in key) for key in df_fmt.columns
            ], names=[_tex_escape_label(n) if n else n for n in df_fmt.columns.names])
        else:
            df_fmt.columns = pd.Index(
                [_tex_escape_label(c) for c in df_fmt.columns],
                name=_tex_escape_label(df_fmt.columns.name)
                if df_fmt.columns.name else None,
            )

        map_fn = getattr(df_fmt, 'map', getattr(df_fmt, 'applymap', None))
        df_fmt = map_fn(_tex_cell)

        latex_str = df_fmt.to_latex(
            escape=False,
            index=True,
            bold_rows=False,
        )
        # pandas sometimes wraps with \begin{table}; strip so we control structure
        latex_str = latex_str.strip()
        if latex_str.startswith('\\begin{table}'):
            # keep only the tabular block
            import re as _re
            m = _re.search(
                r'\\begin\{tabular\}.*\\end\{tabular\}',
                latex_str, flags=_re.S,
            )
            if m:
                latex_str = m.group(0)

        if resize_to_fit:
            latex_str = (
                "\\resizebox{\\textwidth}{!}{%\n"
                + latex_str
                + "\n}"
            )

        if caption or label:
            cap = (caption or "").replace('±', r'$\pm$').replace('·', '--')
            lab = label or ""
            latex_str = (
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                f"{latex_str}\n"
                + (f"\\caption{{{cap}}}\n" if cap else "")
                + (f"\\label{{{lab}}}\n" if lab else "")
                + "\\end{table}"
            )
        return latex_str


    def to_terminal(
        self,
        highlight_best: Optional[str] = 'max',
        highlight_axis: str = 'row',
    ) -> str:
        return self._get_highlighted_df(
            highlight_best, highlight_axis, wrapper_fmt="\033[1m{}\033[0m"
        ).to_string()

    def display(
        self,
        highlight_best: Optional[str] = 'max',
        highlight_axis: str = 'row',
        format_type: str = 'markdown',
    ):
        if format_type == 'markdown':
            print(self.to_markdown(highlight_best, highlight_axis))
        elif format_type == 'latex':
            print(self.to_latex(highlight_best, highlight_axis))
        elif format_type == 'terminal':
            print(self.to_terminal(highlight_best, highlight_axis))



# ------------------------------------------------------------------
# Pivot column labels (compact, paper-friendly)
# ------------------------------------------------------------------
_COL_ABBREV = {
    'wire_api': {'chat': 'ch', 'responses': 're'},
    'fc_model': {0: 'fc0', 1: 'fc1', '0': 'fc0', '1': 'fc1', 0.0: 'fc0', 1.0: 'fc1'},
    'reasoning_effort': {
        'none': 'n', 'low': 'lo', 'medium': 'md', 'high': 'hi',
    },
    'reasoning_level': {
        'none': 'n', 'low': 'lo', 'medium': 'md', 'high': 'hi',
    },
    'python_enabled': {0: 'py0', 1: 'py1', '0': 'py0', '1': 'py1', 0.0: 'py0', 1.0: 'py1'},
    'browser_enabled': {0: 'br0', 1: 'br1', '0': 'br0', '1': 'br1', 0.0: 'br0', 1.0: 'br1'},
}

_COL_VERBOSE = {
    'wire_api': {'chat': 'chat', 'responses': 'resp'},
    'fc_model': {0: 'fc=0', 1: 'fc=1', '0': 'fc=0', '1': 'fc=1', 0.0: 'fc=0', 1.0: 'fc=1'},
    'reasoning_effort': {
        'none': 'none', 'low': 'low', 'medium': 'med', 'high': 'high',
    },
    'reasoning_level': {
        'none': 'none', 'low': 'low', 'medium': 'med', 'high': 'high',
    },
    'python_enabled': {
        0: 'py=0', 1: 'py=1', '0': 'py=0', '1': 'py=1', 0.0: 'py=0', 1.0: 'py=1',
    },
    'browser_enabled': {
        0: 'br=0', 1: 'br=1', '0': 'br=0', '1': 'br=1', 0.0: 'br=0', 1.0: 'br=1',
    },
}


def _format_col_key(col_cols: List[str], key, style: str = 'compact') -> str:
    """Turn a MultiIndex key into a scannable header string.

    Examples (compact)
    ------------------
    ('responses', 0, 'low', 0) → 're·fc0·lo·py0'
    ('chat', 1, 'medium', 1)   → 'ch·fc1·md·py1'
    """
    if not isinstance(key, tuple):
        key = (key,)
    lookup = _COL_ABBREV if style == 'compact' else _COL_VERBOSE
    parts = []
    for col_name, val in zip(col_cols, key):
        mapping = lookup.get(col_name, {})
        # try raw, then str, then lower str
        if val in mapping:
            parts.append(mapping[val])
        elif str(val) in mapping:
            parts.append(mapping[str(val)])
        elif isinstance(val, str) and val.lower() in mapping:
            parts.append(mapping[val.lower()])
        else:
            # backend-like names etc.
            s = str(val)
            s = (
                s.replace('llamacpp@default-jinja', 'l-def')
                .replace('llamacpp@fixed-jinja', 'l-fix')
                .replace('burrito-pt@llamacpp', 'bp-llama')
                .replace('burrito-pt@vllm', 'bp-vllm')
                .replace('burrito@llamacpp', 'b-llama')
                .replace('burrito@vllm', 'b-vllm')
            )
            parts.append(s)
    sep = '·' if style == 'compact' else ' · '
    return sep.join(parts)


def _sort_pivot_columns(col_cols: List[str], columns) -> list:
    """Order pivot columns consistently with chart hue ordering."""
    effort_rank = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}

    def rank_one(col_name, val):
        if col_name in ('reasoning_effort', 'reasoning_level', 'reasoning'):
            return effort_rank.get(str(val).lower(), 50)
        if col_name == 'fc_model':
            try:
                return int(val)
            except Exception:
                return 50
        if col_name == 'wire_api':
            s = str(val).lower()
            return 0 if s == 'chat' else (1 if s == 'responses' else 50)
        if col_name in ('python_enabled', 'browser_enabled'):
            try:
                return int(val)
            except Exception:
                return 50
        if col_name == 'backend':
            s = str(val).lower()
            hits = [
                (i, pat) for i, pat in enumerate(BACKEND_DISPLAY_ORDER)
                if pat.lower() in s
            ]
            if hits:
                return max(hits, key=lambda t: len(t[1]))[0]
            return 100
        return str(val)

    def key_fn(col_key):
        if not isinstance(col_key, tuple):
            col_key = (col_key,)
        return tuple(
            rank_one(name, val)
            for name, val in zip(col_cols, col_key)
        ) + (col_key,)

    return sorted(list(columns), key=key_fn)



def apply_consistency(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    seed_col: str,
    mode: str,
    unit_col: Optional[str] = None,
    pass_threshold: float = 1.0,
    within_seed_agg: str = 'mean',
    min_turns: Optional[int] = None,
    pass_k: Optional[int] = None,
) -> pd.DataFrame:
    """Collapse seed replicates into a pass^k / fail^k indicator per unit.

    Parameters
    ----------
    mode : {'all_pass', 'all_fail', 'any_pass'}
        all_pass — 1 iff every seed scores ≥ pass_threshold (pass^k)
        all_fail — 1 iff every seed scores ≤ 0 (fail^k)
        any_pass — 1 iff at least one seed scores ≥ pass_threshold
    unit_col : str, optional
        Column identifying the same example across seeds (default: ``test_id``
        if present, else each row is its own unit — weaker).
    min_turns : int, optional
        Soft threshold on turns: a seed "passes" if
        ``mt_num_turns_success >= min_turns`` (when that column exists).
    pass_k : int, optional
        If set (or mode is ``pass_at`` / ``pass@k``), use the combinatorial
        **pass@k** estimator per unit from available seeds:
        ``1 - C(n-c, k)/C(n, k)``.  ``pass_k=1`` equals per-task accuracy
        (aggregates to ordinary mean accuracy).  Distinct from ``all_pass``
        (pass^k = success on every seed).
    group_cols : list
        Pivot/group axes (e.g. backend, fc_model) excluding seed and unit.

    Returns
    -------
    DataFrame with columns ``group_cols + [value_col]`` where ``value_col`` is
    the mean consistency rate within each group (one row per group).
    Also attaches ``_n_units`` and ``_n_seeds`` for footers.
    """
    mode = str(mode).lower().replace('-', '_').replace('^', '_').replace('@', '_')
    aliases = {
        'pass_all': 'all_pass', 'pass_hat_k': 'all_pass', 'passk_all': 'all_pass',
        'fail_all': 'all_fail', 'fail_k': 'all_fail', 'failk': 'all_fail',
        'any': 'any_pass', 'pass_any': 'any_pass',
        'pass_at': 'pass_at', 'pass_at_k': 'pass_at', 'passk': 'pass_at',
    }
    mode = aliases.get(mode, mode)
    if pass_k is not None and mode in ('all_pass', 'any_pass', None, 'pass_at'):
        mode = 'pass_at'
    if mode not in ('all_pass', 'all_fail', 'any_pass', 'pass_at'):
        raise ValueError(
            f"consistency must be all_pass|all_fail|any_pass|pass_at, got {mode!r}"
        )
    if mode == 'pass_at' and pass_k is None:
        pass_k = 1

    if unit_col is None:
        unit_col = 'test_id' if 'test_id' in df.columns else None

    work = df.copy()
    keys_seed = list(group_cols)
    if unit_col:
        keys_seed = keys_seed + [unit_col]
    keys_seed = keys_seed + [seed_col]

    # Soft threshold: survived ≥ min_turns on that seed (pass^k@n)
    if min_turns is not None and 'mt_num_turns_success' in work.columns:
        work = work.copy()
        work['_pass_soft'] = (
            work['mt_num_turns_success'].fillna(0) >= int(min_turns)
        ).astype(float)
        score_col = '_pass_soft'
    else:
        score_col = value_col

    per_seed = (
        work.groupby(keys_seed, observed=True)[score_col]
        .agg(within_seed_agg)
        .reset_index()
        .rename(columns={score_col: value_col})
    )

    unit_keys = list(group_cols) + ([unit_col] if unit_col else [])
    if not unit_keys:
        # fallback: treat whole group as one unit
        unit_keys = list(group_cols)

    def _reduce(s: pd.Series) -> float:
        if s.empty:
            return float('nan')
        if mode == 'pass_at':
            n = int(s.shape[0])
            c = int((s >= pass_threshold).sum())
            return pass_at_k(n, c, int(pass_k))
        if mode == 'all_pass':
            return float((s >= pass_threshold).all())
        if mode == 'all_fail':
            return float((s <= 0).all())
        return float((s >= pass_threshold).any())

    if unit_col:
        per_unit = (
            per_seed.groupby(unit_keys, observed=True)[value_col]
            .agg(_reduce)
            .reset_index()
        )
        n_seeds = per_seed.groupby(unit_keys, observed=True)[seed_col].nunique()
        # mean rate of units that are consistent
        out = (
            per_unit.groupby(group_cols, observed=True)[value_col]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        out = out.rename(columns={'mean': value_col, 'count': '_n_units'})
        out['_n_seeds'] = int(per_seed[seed_col].nunique())
    else:
        # no unit id: consistency of the group-level seed scores themselves
        per_group = (
            per_seed.groupby(group_cols, observed=True)[value_col]
            .agg(_reduce)
            .reset_index()
        )
        per_group['_n_units'] = 1
        per_group['_n_seeds'] = int(per_seed[seed_col].nunique())
        per_group['std'] = 0.0
        out = per_group

    return out




def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples with c successes.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Special case k=1 → c/n (empirical per-task accuracy).
    """
    if n <= 0 or k <= 0:
        return float('nan')
    if k > n:
        k = n
    c = int(max(0, min(int(c), n)))
    if n - c < k:
        return 1.0
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    return float(1.0 - ratio)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k estimator: P(all k draws succeed).

    pass^k = C(c, k) / C(n, k)

    k=1 → c/n; k=n → 1 iff c=n else 0 (all_pass).
    """
    if n <= 0 or k <= 0:
        return float('nan')
    if k > n:
        return float('nan')
    c = int(max(0, min(int(c), n)))
    if c < k:
        return 0.0
    # C(c,k)/C(n,k) = prod_{i=0..k-1} (c-i)/(n-i)
    ratio = 1.0
    for i in range(k):
        ratio *= (c - i) / (n - i)
    return float(ratio)


def binomial_se(p: float, n: int) -> float:
    """Standard error of a proportion: sqrt(p(1-p)/n)."""
    if n is None or n <= 0 or p != p:
        return 0.0
    p = float(p)
    p = min(max(p, 0.0), 1.0)
    return float(np.sqrt(p * (1.0 - p) / n))


def expand_consistency_units(
    df: pd.DataFrame,
    value_col: str,
    seed_col: str,
    mode: str,
    unit_col: Optional[str] = None,
    pass_threshold: float = 1.0,
    within_seed_agg: str = 'mean',
    min_turns: Optional[int] = None,
    pass_k: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    """Rewrite *df* so each example is a single 0/1 consistency score.

    Seeds are collapsed; the unit id is written into ``seed_col`` so existing
    plot aggregations treat units like seeds and report ± std across units.
    Returns (new_df, seed_col).
    """
    mode = str(mode).lower().replace('-', '_').replace('^', '_')
    aliases = {
        'pass_all': 'all_pass', 'pass_k': 'all_pass', 'passk': 'all_pass',
        'fail_all': 'all_fail', 'fail_k': 'all_fail', 'failk': 'all_fail',
        'any': 'any_pass', 'pass_any': 'any_pass',
    }
    mode = aliases.get(mode, mode)

    if unit_col is None:
        unit_col = 'test_id' if 'test_id' in df.columns else None
    if unit_col is None or unit_col not in df.columns:
        raise ValueError(
            "consistency requires a unit_col (e.g. 'test_id') identifying "
            "the same example across seeds."
        )

    # Only experimental *factors* define a cell. Including per-run metrics
    # (token counts, error strings, latency, …) would split the same example
    # into singleton seed groups and collapse all_pass → 0 everywhere.
    FACTOR_COLS = (
        'backend', 'fc_model', 'wire_api', 'reasoning_effort', 'reasoning_level',
        'reasoning', 'python_enabled', 'browser_enabled', 'test_name', 'test_type',
        'temperature', 'model_name', 'model_size', 'quantization', 'batch_size',
    )
    group_cols = [c for c in FACTOR_COLS if c in df.columns and c not in (
        seed_col, value_col, unit_col,
    )]

    keys_seed = group_cols + [unit_col, seed_col]
    work = df
    if min_turns is not None and 'mt_num_turns_success' in df.columns:
        work = df.copy()
        work['_pass_soft'] = (
            work['mt_num_turns_success'].fillna(0) >= int(min_turns)
        ).astype(float)
        score_col = '_pass_soft'
    else:
        score_col = value_col
    per_seed = (
        work.groupby(keys_seed, observed=True)[score_col]
        .agg(within_seed_agg)
        .reset_index()
        .rename(columns={score_col: value_col})
    )

    mode = str(mode).lower().replace('-', '_').replace('^', '_').replace('@', '_')
    if mode in ('pass_at', 'pass_at_k', 'passk') or pass_k is not None:
        mode = 'pass_at'
        if pass_k is None:
            pass_k = 1
    def _reduce(s: pd.Series) -> float:
        if s.empty:
            return float('nan')
        if mode == 'pass_at':
            n = int(s.shape[0])
            c = int((s >= pass_threshold).sum())
            return pass_at_k(n, c, int(pass_k))
        if mode == 'all_pass':
            return float((s >= pass_threshold).all())
        if mode == 'all_fail':
            return float((s <= 0).all())
        return float((s >= pass_threshold).any())

    unit_keys = group_cols + [unit_col]
    per_unit = (
        per_seed.groupby(unit_keys, observed=True)[value_col]
        .agg(_reduce)
        .reset_index()
    )
    # Re-use seed_col slot so plot aggregations still group something;
    # mark frame so callers can switch to binomial SE on the rate.
    per_unit[seed_col] = per_unit[unit_col].astype(str)
    per_unit['_consistency_mode'] = mode
    per_unit['_n_units_total'] = per_unit.groupby(
        [c for c in group_cols if c in per_unit.columns], observed=True
    )[unit_col].transform('count') if group_cols else len(per_unit)
    return per_unit, seed_col





def compute_turn_survival(
    df: pd.DataFrame,
    group_cols: Optional[Union[str, List[str]]] = None,
    filter_query: Optional[str] = None,
    max_turn: Optional[int] = None,
    min_reached: int = 50,
) -> pd.DataFrame:
    """Per-turn survival rates (Tian-style step reliability).

    Requires ``mt_failed_turn_idx`` and ``mt_num_turns_total``.

    For each turn t (0-based):
      reached  = runs with mt_num_turns_total > t
      passed   = failed_turn_idx is NaN  OR  failed_turn_idx > t
      rate     = passed / reached   among reached

    Also returns cumulative product of conditional rates as ``cum_survival``,
    which estimates end-to-end pass probability under independent turns.

    min_reached : int
        Turns with fewer than this many trajectories are marked NaN for
        pass_rate / cum_survival (avoids noisy "recovery" spikes at the
        tail where only a handful of long trajectories remain).

    Returns
    -------
    DataFrame with columns:
      group cols (if any), turn, n_reached, n_passed, pass_rate, cum_survival, se
    """
    work = apply_sql_filter(df, filter_query)
    work = apply_categorical_orders(work)

    need = {'mt_failed_turn_idx', 'mt_num_turns_total'}
    missing = need - set(work.columns)
    if missing:
        raise ValueError(
            f"compute_turn_survival needs {need}; missing {missing}. "
            "Re-run eval_aggregator.py with the updated mapper."
        )

    if group_cols is None:
        group_cols = []
    elif isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    if max_turn is None:
        max_turn = int(work['mt_num_turns_total'].max()) if len(work) else 0
    turns = list(range(int(max_turn)))

    def _one_group(g: pd.DataFrame) -> pd.DataFrame:
        rows = []
        cum = 1.0
        cum_valid = True
        for t in turns:
            reached = g['mt_num_turns_total'] > t
            n_reached = int(reached.sum())
            if n_reached == 0:
                rows.append({
                    'turn': t, 'n_reached': 0, 'n_passed': 0,
                    'pass_rate': float('nan'), 'cum_survival': float('nan'),
                    'se': float('nan'),
                })
                continue
            passed = g['mt_failed_turn_idx'].isna() | (g['mt_failed_turn_idx'] > t)
            n_passed = int((reached & passed).sum())
            rate = n_passed / n_reached
            se = binomial_se(rate, n_reached)
            # Tail turns with tiny n are selection-biased + noisy — blank them
            if n_reached < min_reached:
                rows.append({
                    'turn': t, 'n_reached': n_reached, 'n_passed': n_passed,
                    'pass_rate': float('nan'), 'cum_survival': float('nan'),
                    'se': float('nan'),
                })
                cum_valid = False
                continue
            if cum_valid:
                cum = cum * rate
                cum_out = cum
            else:
                cum_out = float('nan')
            rows.append({
                'turn': t,
                'n_reached': n_reached,
                'n_passed': n_passed,
                'pass_rate': rate,
                'cum_survival': cum_out,
                'se': se,
            })
        return pd.DataFrame(rows)

    if not group_cols:
        return _one_group(work)

    parts = []
    for keys, g in work.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        block = _one_group(g)
        for c, v in zip(group_cols, keys):
            block[c] = v
        parts.append(block)
    if not parts:
        return pd.DataFrame(
            columns=group_cols + [
                'turn', 'n_reached', 'n_passed', 'pass_rate', 'cum_survival', 'se'
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    # stable column order
    cols = group_cols + [
        'turn', 'n_reached', 'n_passed', 'pass_rate', 'cum_survival', 'se'
    ]
    return out[cols]



def compute_pass_curves(
    df: pd.DataFrame,
    value_col: str = 'correct',
    group_cols: Optional[Union[str, List[str]]] = None,
    filter_query: Optional[str] = None,
    seed_col: Optional[str] = None,
    unit_col: Optional[str] = None,
    k_max: Optional[int] = None,
    pass_threshold: float = 1.0,
    min_turns: Optional[int] = None,
) -> pd.DataFrame:
    """Per-group pass@k and pass^k curves for k = 1..k_max.

    Returns long DataFrame:
      group cols, k, pass_at, pass_hat, n_units, n_seeds
    """
    work = apply_sql_filter(df, filter_query)
    work = apply_categorical_orders(work)
    work, seed_col = ensure_seed_col(work, seed_col)
    if unit_col is None:
        unit_col = 'test_id' if 'test_id' in work.columns else None
    if unit_col is None:
        raise ValueError("compute_pass_curves needs unit_col (e.g. test_id)")

    if group_cols is None:
        group_cols = []
    elif isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    # Soft turn threshold → binary pass per row
    if min_turns is not None and 'mt_num_turns_success' in work.columns:
        work = work.copy()
        work['_s'] = (work['mt_num_turns_success'].fillna(0) >= int(min_turns)).astype(float)
        score = '_s'
    else:
        score = value_col

    keys = group_cols + [unit_col, seed_col]
    per_seed = (
        work.groupby(keys, observed=True)[score]
        .mean()
        .reset_index()
        .rename(columns={score: '_v'})
    )
    # n, c per unit
    unit_keys = group_cols + [unit_col]
    stats = (
        per_seed.groupby(unit_keys, observed=True)['_v']
        .agg(n='count', c=lambda s: (s >= pass_threshold).sum())
        .reset_index()
    )

    if k_max is None:
        k_max = int(stats['n'].max()) if len(stats) else 1
    k_max = int(k_max)

    rows = []
    if not group_cols:
        groups = [((), stats)]
    else:
        groups = list(stats.groupby(group_cols, observed=True))

    for gkey, g in groups:
        if not isinstance(gkey, tuple):
            gkey = (gkey,)
        n_units = len(g)
        n_seeds_typ = int(g['n'].median()) if n_units else 0
        for k in range(1, k_max + 1):
            at_vals = [
                pass_at_k(int(r.n), int(r.c), k) for r in g.itertuples(index=False)
            ]
            hat_vals = [
                pass_hat_k(int(r.n), int(r.c), k) for r in g.itertuples(index=False)
            ]
            # drop nan (e.g. hat when k > n for that unit)
            at_clean = [v for v in at_vals if v == v]
            hat_clean = [v for v in hat_vals if v == v]
            row = {
                'k': k,
                'pass_at': float(np.mean(at_clean)) if at_clean else float('nan'),
                'pass_hat': float(np.mean(hat_clean)) if hat_clean else float('nan'),
                'n_units': n_units,
                'n_seeds': n_seeds_typ,
            }
            for c, v in zip(group_cols, gkey):
                row[c] = v
            rows.append(row)

    out = pd.DataFrame(rows)
    cols = group_cols + ['k', 'pass_at', 'pass_hat', 'n_units', 'n_seeds']
    return out[cols] if len(out) else pd.DataFrame(columns=cols)


def pivot_evals(
    df: pd.DataFrame,
    value_col: str,
    index: Union[str, List[str]],
    columns: Union[str, List[str]],
    filter_query: Optional[str] = None,
    seed_col: Optional[str] = None,
    within_seed_agg: str = 'mean',
    across_seed_agg: str = 'mean',
    show_variation: str = 'std',
    precision: int = 2,
    multiply_by: float = 1.0,
    consistency: Optional[str] = None,
    unit_col: Optional[str] = None,
    pass_threshold: float = 1.0,
    min_turns: Optional[int] = None,
    pass_k: Optional[int] = None,
) -> EvalPivotResult:
    """
    Pivots eval data with two-stage seed aggregation:
    (1) collapse to one value per (index, columns, seed) via within_seed_agg,
    (2) collapse across seeds via across_seed_agg, reporting variation.

    Note: does NOT auto-rescale fractional columns — set multiply_by=100
    explicitly for columns like `correct`.

    consistency : {None, 'all_pass', 'all_fail', 'any_pass'}
        If set, each example (``unit_col``, default ``test_id``) is scored for
        seed-level agreement before aggregation:
          all_pass — 1 only if every seed passes (pass^k)
          all_fail — 1 only if every seed fails (fail^k)
          any_pass — 1 if any seed passes
        The reported cell is then the mean rate of such units (± std across
        units when show_variation='std').
    """
    index_cols = [index] if isinstance(index, str) else list(index)
    col_cols = [columns] if isinstance(columns, str) else list(columns)

    filtered_df = apply_sql_filter(df, filter_query)
    filtered_df = apply_categorical_orders(filtered_df)
    filtered_df, seed_col = ensure_seed_col(filtered_df, seed_col)

    if consistency:
        # pass^k / fail^k path — consistency per unit across seeds, then mean
        cons = apply_consistency(
            filtered_df,
            value_col=value_col,
            group_cols=index_cols + col_cols,
            seed_col=seed_col,
            mode=consistency,
            unit_col=unit_col,
            pass_threshold=pass_threshold,
            within_seed_agg=within_seed_agg,
            min_turns=min_turns,
            pass_k=pass_k,
        )
        cons[value_col] = cons[value_col] * multiply_by
        # Binomial SE on the rate (not sample-std of 0/1 units)
        if '_n_units' in cons.columns:
            cons['std'] = cons.apply(
                lambda r: binomial_se(
                    (r[value_col] / multiply_by) if multiply_by else r[value_col],
                    int(r['_n_units']) if pd.notna(r['_n_units']) else 0,
                ) * (multiply_by if multiply_by else 1.0),
                axis=1,
            )
        else:
            cons['std'] = 0.0
        cons['min'] = cons[value_col]
        cons['max'] = cons[value_col]
        cons['count'] = cons.get('_n_units', 1)
        cons['mean'] = cons[value_col]
        seed_summary = cons
        across_seed_agg = 'mean'
    else:
        group_keys = index_cols + col_cols + [seed_col]
        seed_metrics = (
            filtered_df.groupby(group_keys, observed=True)[value_col]
            .agg(within_seed_agg)
            .reset_index()
        )
        seed_metrics[value_col] = seed_metrics[value_col] * multiply_by

        agg_funcs = ['mean', 'std', 'min', 'max', 'count']
        seed_summary = (
            seed_metrics.groupby(index_cols + col_cols, observed=True)[value_col]
            .agg(agg_funcs)
            .reset_index()
        )

    def format_cell(row):
        val = row[across_seed_agg] if across_seed_agg in row.index else row.get('mean')
        std = row['std'] if 'std' in row.index else 0.0
        if pd.isna(val):
            return "—"
        val_str = f"{val:.{precision}f}"
        if show_variation == 'std':
            std_str = f"{0.0 if pd.isna(std) else std:.{precision}f}"
            return f"{val_str} ± {std_str}"
        elif show_variation == 'minmax' and 'min' in row.index:
            return f"{val_str} [{row['min']:.{precision}f}-{row['max']:.{precision}f}]"
        return val_str

    seed_summary['formatted_cell'] = seed_summary.apply(format_cell, axis=1)

    numeric_pivot = seed_summary.pivot(
        index=index_cols, columns=col_cols, values=across_seed_agg
    )
    formatted_pivot = seed_summary.pivot(
        index=index_cols, columns=col_cols, values='formatted_cell'
    )

    # --- column order + flat labels ---
    ordered_cols = _sort_pivot_columns(col_cols, list(numeric_pivot.columns))
    numeric_pivot = numeric_pivot.reindex(columns=ordered_cols)
    formatted_pivot = formatted_pivot.reindex(columns=ordered_cols)
    col_label_map = {
        c: _format_col_key(col_cols, c, style='compact') for c in ordered_cols
    }

    # --- row order + flat labels (when index is multi-factor) ---
    ordered_idx = _sort_pivot_columns(index_cols, list(numeric_pivot.index))
    numeric_pivot = numeric_pivot.reindex(index=ordered_idx)
    formatted_pivot = formatted_pivot.reindex(index=ordered_idx)
    idx_label_map = {
        i: _format_col_key(index_cols, i, style='compact') for i in ordered_idx
    }

    formatted_flat = formatted_pivot.copy()
    # Flat column headers
    formatted_flat.columns = pd.Index(
        [col_label_map[c] for c in formatted_pivot.columns],
        name=(
            ' · '.join(col_cols) if len(col_cols) > 1
            else (col_cols[0] if col_cols else None)
        ),
    )
    # Flat row headers (and short backend names when single-level)
    if isinstance(formatted_flat.index, pd.MultiIndex) or len(index_cols) > 1:
        formatted_flat.index = pd.Index(
            [idx_label_map[i] for i in formatted_pivot.index],
            name=(
                ' · '.join(index_cols) if len(index_cols) > 1
                else (index_cols[0] if index_cols else None)
            ),
        )
    else:
        formatted_flat.index = pd.Index(
            [
                shorten_label(i) if isinstance(i, str) else idx_label_map.get(i, i)
                for i in formatted_pivot.index
            ],
            name=formatted_flat.index.name,
        )
        # Keep numeric index aligned for highlight mapping
        # (numeric keeps original keys; maps handle translation)

    # Drop all-empty columns that just add width
    keep = []
    for c in formatted_flat.columns:
        s = formatted_flat[c]
        if s.isna().all():
            continue
        as_str = s.astype(str)
        if (as_str == 'N/A').all() or (as_str == 'nan').all():
            continue
        keep.append(c)
    if keep:
        inv = {v: k for k, v in col_label_map.items()}
        numeric_pivot = numeric_pivot[[inv[c] for c in keep if c in inv]]
        formatted_flat = formatted_flat[keep]

    # Missing combinations → em dash (cleaner than nan / N/A in wide tables)
    formatted_flat = formatted_flat.fillna('—')
    formatted_flat = formatted_flat.replace(
        {pd.NA: '—', 'nan': '—', 'NaN': '—', 'N/A': '—', 'None': '—'}
    )
    # applymap for any remaining float nan stringification edge cases
    map_fn = getattr(formatted_flat, 'map', getattr(formatted_flat, 'applymap', None))
    if map_fn is not None:
        formatted_flat = map_fn(
            lambda x: '—' if (x is None or (isinstance(x, float) and x != x)
                              or str(x).lower() in ('nan', 'none', 'n/a', ''))
            else x
        )

    return EvalPivotResult(
        numeric_pivot, formatted_flat, index_cols, col_cols,
        filtered_df=filtered_df,
        col_label_map=col_label_map,
        idx_label_map=idx_label_map,
    )


# ===================================================================
# 3. RESEARCH VISUALIZER
# ===================================================================

class EvalPlotter:
    """Publication-grade visualizer: quiet grids, aligned grouped bars,
    stable color identity, honest axes.

    Return contract
    ---------------
    Every ``plot_*`` method returns ``(fig, ax_or_axes, data)`` where
    ``data`` is a pandas DataFrame of the exact aggregates that were drawn
    (means, stds, bin centers, survival rates, etc.). Use ``data`` for
    numeric follow-up or agent handoff; the figure is for human inspection.
    """

    # ---------------------------------------------------------------
    # shared internals
    # ---------------------------------------------------------------

    @classmethod
    def _prepare_data(
        cls,
        data: Union[pd.DataFrame, 'EvalPivotResult'],
        filter_query: Optional[str] = None,
        group_col: Optional[Union[str, List[str]]] = None,
        seed_col: Optional[str] = None,
        value_col: Optional[str] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, str, Optional[str]]:
        if isinstance(data, EvalPivotResult):
            df_proc = data.filtered_df.copy()
        elif isinstance(data, pd.DataFrame):
            df_proc = data.copy()
        else:
            raise TypeError(
                "`data` must be a pandas DataFrame or an EvalPivotResult object."
            )

        df_proc = apply_sql_filter(df_proc, filter_query)
        if df_proc.empty:
            raise ValueError("Dataframe is empty after applying filter query.")

        df_proc, seed_col = ensure_seed_col(df_proc, seed_col)

        if consistency:
            df_proc, seed_col = expand_consistency_units(
                df_proc,
                value_col=value_col,
                seed_col=seed_col,
                mode=consistency,
                unit_col=unit_col,
                pass_threshold=pass_threshold,
                min_turns=min_turns,
                pass_k=pass_k,
            )
            df_proc = apply_categorical_orders(df_proc)

        # Normalize value columns to a 0-100 percent scale.
        # Some suites (AIME25, GPQA) store the same metric as a 0/1 fraction
        # while BFCL multi-turn / live / non-live store 0-100. A mixed filter
        # used to leave the fraction rows unscaled (max>1 short-circuits the
        # old global check), so those bars rendered at ~1% instead of ~70%.
        if value_col and value_col in df_proc.columns:
            col = df_proc[value_col]
            max_val = col.dropna().max()
            if max_val is not None and 0 < max_val <= 1.0:
                df_proc[value_col] = col * 100.0
            elif max_val is not None and max_val > 1.0:
                frac = col.notna() & (col >= 0.0) & (col <= 1.0)
                if frac.any():
                    le1 = col[frac]
                    # Only lift when the <=1 subset is pure binary {0,1}
                    # (accuracy-as-fraction), never a genuine 0.5% rate.
                    if le1.dropna().isin([0.0, 1.0]).all():
                        df_proc.loc[frac, value_col] = col.loc[frac] * 100.0

        df_proc = apply_categorical_orders(df_proc)
        df_proc, seed_col = ensure_seed_col(df_proc, seed_col)

        final_group_col = None
        if group_col:
            if isinstance(group_col, list):
                if len(group_col) == 1:
                    final_group_col = group_col[0]
                else:
                    def make_label(row):
                        base = str(row[group_col[0]])
                        extras = [f"{col}={row[col]}" for col in group_col[1:]]
                        return f"{base} ({', '.join(extras)})"

                    df_proc['_combined_group'] = df_proc.apply(make_label, axis=1)
                    final_group_col = '_combined_group'
            else:
                final_group_col = group_col

        return df_proc, seed_col, final_group_col

    @classmethod
    def _ordered_levels(
        cls, df: pd.DataFrame, col: str, present_in: pd.Series
    ) -> list:
        present = set(present_in.unique())
        if isinstance(df[col].dtype, CategoricalDtype):
            return [c for c in df[col].cat.categories if c in present]
        return [c for c in pd.unique(df[col]) if c in present]

    @classmethod
    def _new_figure(cls, figsize, dpi: Optional[float] = None, **kwargs):
        """Create a figure at the module default DPI (300 unless overridden)."""
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi or DEFAULT_DPI, **kwargs)
        return fig, ax

    @classmethod
    def _apply_paper_style(cls, ax, grid_axis: str = 'y'):
        ax.set_axisbelow(True)
        ax.grid(
            True, axis=grid_axis, linestyle='-', linewidth=0.65,
            alpha=0.85, color=GRID, zorder=0,
        )
        other = 'x' if grid_axis == 'y' else 'y'
        ax.grid(False, axis=other)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(SPINE)
            ax.spines[spine].set_linewidth(0.7)
        ax.tick_params(length=0, labelsize=8, colors=SLATE, pad=3)

    @classmethod
    def _add_encoding_legend(
        cls, ax, y: float = 0.02,
        has_fc: bool = True, has_wire: bool = True,
        fig=None,
    ):
        """Encoding legend pinned to the bottom of the *figure*.

        Uses figure coordinates so under-bar labels never collide with it.
        """
        if not has_fc and not has_wire:
            return
        from matplotlib.patches import Patch
        ink = '#64748b'
        if has_fc and has_wire:
            handles = [
                Patch(facecolor=ink, edgecolor=ink, linewidth=0.9,
                      alpha=0.90, hatch='...', label='fc=1  ·  chat'),
                Patch(facecolor=ink, edgecolor=ink, linewidth=0.9,
                      alpha=0.90, label='fc=1  ·  responses'),
                Patch(facecolor=ink, edgecolor=ink, linewidth=0.9,
                      alpha=0.42, hatch='...', label='fc=0  ·  chat'),
                Patch(facecolor=ink, edgecolor=ink, linewidth=0.9,
                      alpha=0.42, label='fc=0  ·  responses'),
            ]
            ncol = 4
        elif has_fc:
            handles = [
                Patch(facecolor=ink, edgecolor=ink, linewidth=0.9,
                      alpha=0.90, label='fc=1'),
                Patch(facecolor=ink, edgecolor=ink, linewidth=0.9,
                      alpha=0.42, label='fc=0'),
            ]
            ncol = 2
        else:
            handles = [
                Patch(facecolor=ink, edgecolor=ink, linewidth=0.9,
                      alpha=1.0, hatch='...', label='chat'),
                Patch(facecolor=ink, edgecolor=ink, linewidth=0.9,
                      alpha=1.0, label='responses'),
            ]
            ncol = 2

        target = fig if fig is not None else ax.figure
        leg = target.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, y),
            ncol=ncol, frameon=False, fontsize=7.2,
            handlelength=2.0, handleheight=1.1,
            columnspacing=1.6, borderaxespad=0.0,
            title='encoding', title_fontsize=6.8,
        )
        if leg.get_title() is not None:
            leg.get_title().set_color(MUTED)
            leg.get_title().set_fontweight(700)
        for t in leg.get_texts():
            t.set_color(SLATE)

    @classmethod
    def _style_title_labels(
        cls, ax, title, xlabel, ylabel, subtitle=None, title_pad=None,
        subtitle_y: float = 1.012,
    ):
        pad = title_pad if title_pad is not None else (14 if subtitle else 10)
        ax.set_title(
            title, fontsize=12, fontweight='bold', pad=pad,
            loc='left', color=INK,
        )
        if subtitle:
            ax.text(
                0.0, subtitle_y, subtitle, transform=ax.transAxes,
                fontsize=7.8, color=FAINT, ha='left', va='bottom',
            )
        if xlabel is not None:
            ax.set_xlabel(
                xlabel, fontsize=9, fontweight=700,
                color=SLATE, labelpad=7,
            )
        if ylabel is not None:
            ax.set_ylabel(
                ylabel, fontsize=9, fontweight=700,
                color=SLATE, labelpad=7,
            )

    @classmethod
    def _add_footer(cls, fig, text: str):
        fig.text(
            0.995, 0.005, text, ha='right', va='bottom',
            fontsize=6.5, color=FAINT, style='italic',
        )

    @staticmethod
    def _dodge_offsets(levels: list, width: float, gap: float) -> Dict:
        n = len(levels)
        total_w = n * width + (n - 1) * gap
        start = -total_w / 2.0 + width / 2.0
        return {lvl: start + i * (width + gap) for i, lvl in enumerate(levels)}

    @classmethod
    def _legend_outside_right(cls, ax, title=None, fontsize=7.5):
        leg = ax.legend(
            bbox_to_anchor=(1.02, 1.0), loc='upper left',
            frameon=False, fontsize=fontsize,
            handlelength=1.6, labelspacing=0.55,
            title=title, title_fontsize=8,
        )
        if leg and leg.get_title():
            leg.get_title().set_fontweight(700)
            leg.get_title().set_color(SLATE)
        return leg

    @classmethod
    def _legend_top(cls, ax, ncol=None, fontsize=7.5, y=1.06):
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return None
        n = ncol if ncol is not None else min(len(labels), 8)
        return ax.legend(
            handles, labels,
            loc='lower center', bbox_to_anchor=(0.5, y),
            ncol=n, frameon=False, fontsize=fontsize,
            handlelength=1.4, columnspacing=1.4,
        )

    # ---------------------------------------------------------------
    # line chart
    # ---------------------------------------------------------------

    @classmethod

    @classmethod
    def _fix_consistency_std(cls, summary: pd.DataFrame, df_proc: pd.DataFrame,
                               value_col: str, group_cols: list) -> pd.DataFrame:
        """Replace sample-std of 0/1 units with binomial SE of the rate.

        Consistency indicators are binary; the naive ±std across units is
        ~sqrt(p(1-p)) and looks enormous. The right uncertainty on the
        reported rate is sqrt(p(1-p)/n).
        """
        if '_consistency_mode' not in df_proc.columns:
            return summary
        # n units per group from df_proc
        gcols = [c for c in group_cols if c is not None and c in df_proc.columns]
        if not gcols:
            n_map = {'_all': len(df_proc)}
            summary = summary.copy()
            summary['std'] = summary.apply(
                lambda r: binomial_se(
                    (r['mean'] / 100.0) if r['mean'] > 1 else r['mean'],
                    len(df_proc),
                ) * (100.0 if r['mean'] > 1 else 1.0),
                axis=1,
            )
            return summary

        counts = (
            df_proc.groupby(gcols, observed=True)
            .size()
            .rename('_n')
            .reset_index()
        )
        summary = summary.merge(counts, on=gcols, how='left')
        def _se(row):
            mu = float(row['mean']) if pd.notna(row['mean']) else 0.0
            n = int(row['_n']) if pd.notna(row.get('_n', None)) else 0
            # detect percent scale
            if mu > 1.0 or (summary['mean'].max() is not None and summary['mean'].max() > 1.0):
                p = mu / 100.0
                return binomial_se(p, n) * 100.0
            return binomial_se(mu, n)
        summary = summary.copy()
        summary['std'] = summary.apply(_se, axis=1)
        summary = summary.drop(columns=['_n'], errors='ignore')
        return summary

    @classmethod
    def plot_line_scaling(
        cls,
        data: Union[pd.DataFrame, EvalPivotResult],
        x_col: Optional[str],
        y_col: str,
        group_col: Optional[Union[str, List[str]]] = None,
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Performance Scaling across Reasoning Effort",
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Accuracy (%)",
        linewidth: float = 2.0,
        markersize: float = 6.0,
        band_alpha: float = 0.12,
        show_values: bool = False,
        figsize: Tuple[float, float] = (8.2, 4.8),
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None):
        """Line chart with honest 0–100% Y-axis and ±1 std band across seeds."""
        df_proc, seed_col, grp_col = cls._prepare_data(
            data, filter_query, group_col, seed_col, value_col=y_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold,
            min_turns=min_turns,
            pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _is_consistency = '_consistency_mode' in df_proc.columns
        _footer_n_label = 'units' if _is_consistency else 'seeds'

        if x_col is None:
            x_col = '__overall__'
            df_proc = df_proc.copy()
            df_proc[x_col] = 'overall'

        fig, ax = cls._new_figure(figsize, dpi=dpi)

        group_keys = (
            [x_col, grp_col, seed_col] if grp_col else [x_col, seed_col]
        )
        seed_scores = (
            df_proc.groupby(group_keys, observed=True)[y_col]
            .mean()
            .reset_index()
        )
        across_keys = [x_col, grp_col] if grp_col else [x_col]
        across_seeds = (
            seed_scores.groupby(across_keys, observed=True)[y_col]
            .agg(['mean', 'std'])
            .reset_index()
        )
        if isinstance(df_proc[x_col].dtype, CategoricalDtype):
            across_seeds[x_col] = pd.Categorical(
                across_seeds[x_col],
                categories=df_proc[x_col].cat.categories,
                ordered=True,
            )
        across_seeds = across_seeds.sort_values(by=x_col)

        cls._apply_paper_style(ax, grid_axis='y')
        max_top = 0.0

        # Detect whether x is an effort axis → per-point effort markers
        x_is_effort = bool(
            x_col and re.search(
                r'reasoning|effort|level', str(x_col), flags=re.I
            )
        )
        from matplotlib.lines import Line2D
        backend_proxies = []
        effort_proxies_done = set()
        effort_proxies = []
        fc_proxies = []
        seen_backend_labels = set()
        has_fc_factor = False

        if grp_col:
            groups = cls._ordered_levels(df_proc, grp_col, across_seeds[grp_col])
            groups = cls._sort_hue_pairs(groups)
            has_fc_factor, _ = detect_encoding_factors(groups)
            for grp in groups:
                sub = across_seeds[across_seeds[grp_col] == grp].dropna(
                    subset=['mean']
                )
                if sub.empty:
                    continue
                color, ls, _ = get_semantic_style(grp)
                # Backend legend label strips fc/effort when those are encoded
                # separately (avoids b-vllm-fc0 / b-vllm-fc1 duplicates).
                leg_label = (
                    strip_fc_effort_label(grp) if has_fc_factor
                    else shorten_label(grp)
                )
                # Line without markers; markers added per-point when x is effort
                ax.plot(
                    sub[x_col], sub['mean'],
                    color=color, linestyle=ls,
                    marker=None, linewidth=linewidth, zorder=3,
                )
                std_vals = sub['std'].fillna(0.0)
                ax.fill_between(
                    sub[x_col],
                    (sub['mean'] - std_vals).clip(lower=0),
                    sub['mean'] + std_vals,
                    color=color, alpha=band_alpha, edgecolor='none', zorder=2,
                )
                # Per-point markers
                for _, r in sub.iterrows():
                    mk = (
                        get_effort_marker(r[x_col]) if x_is_effort
                        else 'o'
                    )
                    is_fc0 = ls in (':', 'dotted')
                    if is_fc0 and has_fc_factor:
                        ax.scatter(
                            r[x_col], r['mean'], s=markersize ** 1.6,
                            facecolors='none', edgecolors=color,
                            marker=mk, linewidths=1.3, zorder=4,
                        )
                    else:
                        ax.scatter(
                            r[x_col], r['mean'], s=markersize ** 1.5,
                            color=color, marker=mk, edgecolor='white',
                            linewidth=1.0, zorder=4,
                        )
                    if x_is_effort:
                        ek = str(r[x_col]).lower()
                        if ek not in effort_proxies_done:
                            effort_proxies.append(Line2D(
                                [0], [0], color=SLATE, marker=get_effort_marker(ek),
                                linestyle='None', markersize=7,
                                markerfacecolor=SLATE, markeredgecolor='white',
                                label=str(r[x_col]),
                            ))
                            effort_proxies_done.add(ek)
                max_top = max(
                    max_top,
                    float((sub['mean'] + std_vals).max()),
                )
                if show_values:
                    for _, r in sub.iterrows():
                        ax.annotate(
                            fmt_metric(float(r['mean'])),
                            (r[x_col], r['mean']),
                            textcoords='offset points', xytext=(0, 6),
                            ha='center', va='bottom',
                            fontsize=6.0, fontweight='bold', color=INK,
                            zorder=5,
                        )
                if leg_label not in seen_backend_labels:
                    seen_backend_labels.add(leg_label)
                    backend_proxies.append(Line2D(
                        [0], [0], color=color, linestyle=ls, marker='o',
                        markersize=6, markeredgecolor='white', label=leg_label,
                    ))
            if has_fc_factor:
                fc_proxies = [
                    Line2D(
                        [0], [0], color=SLATE, marker='o', linestyle='None',
                        markersize=7, markerfacecolor='none',
                        markeredgecolor=SLATE, markeredgewidth=1.4, label='fc0',
                    ),
                    Line2D(
                        [0], [0], color=SLATE, marker='o', linestyle='None',
                        markersize=7, markerfacecolor=SLATE,
                        markeredgecolor='white', label='fc1',
                    ),
                ]
        else:
            sub = across_seeds.dropna(subset=['mean'])
            ax.plot(
                sub[x_col], sub['mean'], color='#4f46e5', marker=None,
                linewidth=linewidth, zorder=3,
            )
            for _, r in sub.iterrows():
                mk = get_effort_marker(r[x_col]) if x_is_effort else 'o'
                ax.scatter(
                    r[x_col], r['mean'], s=markersize ** 1.5,
                    color='#4f46e5', marker=mk, edgecolor='white',
                    linewidth=1.0, zorder=4,
                )
            std_vals = sub['std'].fillna(0.0)
            ax.fill_between(
                sub[x_col],
                (sub['mean'] - std_vals).clip(lower=0),
                sub['mean'] + std_vals,
                color='#4f46e5', alpha=band_alpha, zorder=2,
            )
            max_top = float((sub['mean'] + std_vals).max()) if len(sub) else 0

        if is_percent_metric(y_col, ylabel):
            ax.set_ylim(0, percent_axis_limit(max_top, headroom=8))
            ax.axhline(100, color=SPINE, linewidth=0.6, linestyle=':', zorder=1)
        else:
            ax.set_ylim(0, max(max_top * 1.12, 1.0))

        x_label_final = (
            None if x_col in ('overall', '__overall__')
            else (xlabel if xlabel else x_col.replace('_', ' ').title())
        )
        cls._style_title_labels(
            ax, title, x_label_final, ylabel, subtitle,
            title_pad=(12 if subtitle else 8), subtitle_y=1.004,
        )

        all_h = list(backend_proxies) + list(effort_proxies) + list(fc_proxies)
        if all_h:
            n = len(all_h)
            if n >= 6:
                fig.legend(
                    all_h, [h.get_label() for h in all_h],
                    loc='upper center', bbox_to_anchor=(0.5, -0.02),
                    ncol=min(n, 8), frameon=False, fontsize=6.5,
                    handlelength=1.6, columnspacing=1.1,
                )
                fig.tight_layout(rect=[0, 0.10, 1, 1])
            else:
                ax.legend(
                    all_h, [h.get_label() for h in all_h],
                    bbox_to_anchor=(1.02, 1.0), loc='upper left',
                    frameon=False, fontsize=7.0, handlelength=1.6,
                )
                fig.tight_layout(rect=[0, 0.02, 0.82, 1])
        else:
            fig.tight_layout(rect=[0, 0.02, 1, 1])

        notes = [f"shaded band = ±1 std across n={n_seeds} {_footer_n_label}"]
        if x_is_effort:
            notes.append("markers = effort (○ low · ◇ med · □ high)")
        if has_fc_factor:
            notes.append("dotted/hollow = fc0 · solid/filled = fc1")
        cls._add_footer(fig, "  ·  ".join(notes))
        return fig, ax, across_seeds

    # ---------------------------------------------------------------
    # grouped / clustered bar chart
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # grouped / clustered bar chart
    # ---------------------------------------------------------------

    @classmethod
    def plot_bar_comparison(
        cls,
        data: Union[pd.DataFrame, EvalPivotResult],
        x_col: str,
        y_col: str,
        hue_col: Optional[Union[str, List[str]]] = None,
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Benchmark Accuracy Comparison",
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Accuracy (%)",
        bar_width: Optional[float] = None,
        bar_gap: float = 0.03,
        show_values: bool = True,
        value_fontsize: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        label_bars: Optional[bool] = None,
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None):
        """
        Clustered bar chart with stable dodge slots.

        Dense multi-series charts (backend × fc_model, etc.):
          • fc_model=0 → hatched (shaded); fc_model=1 → solid fill
          • Rotated identity labels sit under each bar — no legend
          • Value labels on top when there is room
          • Figure width scales with #categories × #series

        Sparse charts keep a compact top/bottom legend.
        Override with label_bars=True/False.
        """
        df_proc, seed_col, final_hue = cls._prepare_data(
            data, filter_query, hue_col, seed_col, value_col=y_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold,
            min_turns=min_turns,
            pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _is_consistency = '_consistency_mode' in df_proc.columns
        _footer_n_label = 'units' if _is_consistency else 'seeds'

        # Pool everything into a single category when x_col is omitted
        if x_col is None:
            x_col = '__overall__'
            df_proc = df_proc.copy()
            df_proc[x_col] = 'overall'

        group_cols = (
            [x_col, seed_col] if final_hue is None
            else [x_col, final_hue, seed_col]
        )
        seed_agg = (
            df_proc.groupby(group_cols, observed=True)[y_col]
            .mean()
            .reset_index()
        )
        across_cols = [x_col, final_hue] if final_hue else [x_col]
        summary = (
            seed_agg.groupby(across_cols, observed=True)[y_col]
            .agg(['mean', 'std'])
            .reset_index()
        )
        summary = cls._fix_consistency_std(
            summary, df_proc, y_col,
            [c for c in across_cols if c != seed_col],
        )

        categories = cls._ordered_levels(df_proc, x_col, summary[x_col])
        n_cats = len(categories)
        hue_levels = (
            cls._ordered_levels(df_proc, final_hue, summary[final_hue])
            if final_hue else [None]
        )
        if final_hue:
            hue_levels = cls._sort_hue_pairs(hue_levels)
        n_hue = len(hue_levels)

        # Direct under-bar labels once the legend would be a cognitive tax
        use_bar_labels = (
            label_bars if label_bars is not None
            else bool(final_hue and n_hue >= 4)
        )

        # --- adaptive canvas ------------------------------------------------
        if figsize is None:
            # ~0.22" per bar keeps rotated under-labels readable
            bars_total = max(n_cats * max(n_hue, 1), 1)
            fig_w = max(9.0, min(0.26 * bars_total + 2.0, 20.0))
            fig_h = 5.4 if use_bar_labels else 5.2
            figsize = (fig_w, fig_h)

        fig, ax = cls._new_figure(figsize, dpi=dpi)

        # Pack bars tightly inside each category slot; fixed offsets so a
        # missing series leaves a visible gap instead of recentering (which
        # is what made bars look like they were sliding under neighbors).
        # Cluster must fit strictly inside one category slot (width ≤ 1.0)
        # so neighboring groups never bleed into each other. Always leave a
        # visible hairline gap between bars for legibility.
        max_cluster = 0.84 if n_hue <= 6 else (0.88 if n_hue <= 14 else 0.92)
        if bar_width is not None:
            bw = bar_width
            gap = bar_gap
        else:
            # Target gap ≈ 18–25% of bar width; solve for bw given max_cluster.
            # n*bw + (n-1)*(g_ratio*bw) ≤ max_cluster
            # bw ≤ max_cluster / (n + (n-1)*g_ratio)
            g_ratio = 0.22 if n_hue <= 8 else (0.18 if n_hue <= 16 else 0.14)
            denom = n_hue + (n_hue - 1) * g_ratio
            bw = max_cluster / max(denom, 1)
            bw = max(min(bw, 0.30), 0.016)
            gap = g_ratio * bw
        offsets = cls._dodge_offsets(hue_levels, bw, gap)

        cls._apply_paper_style(ax, grid_axis='y')

        if value_fontsize is None:
            value_fontsize = (
            6.2 if bw >= 0.10 else
            5.4 if bw >= 0.06 else
            4.6 if bw >= 0.04 else 3.8
        )
        annotate_vals = bool(show_values)  # always label, even on hairline bars

        max_top = 0.0
        # Collect under-bar label positions for a second pass
        bar_label_jobs = []  # (x_pos, short_label, color)

        for c_idx, cat in enumerate(categories):
            cat_sub = summary[summary[x_col] == cat]
            for h in hue_levels:
                row = (
                    cat_sub[cat_sub[final_hue] == h] if final_hue else cat_sub
                )
                if row.empty or pd.isna(row['mean'].iloc[0]):
                    continue
                val = float(row['mean'].iloc[0])
                std = row['std'].iloc[0]
                std = 0.0 if pd.isna(std) else float(std)

                x_pos = c_idx + offsets[h]
                grp_name = h if final_hue else cat
                color, hatch, edge, face_alpha, lw = get_bar_style(grp_name)

                ax.bar(
                    x_pos, val, width=bw * 0.92,
                    color=color, alpha=face_alpha,
                    edgecolor=edge, linewidth=lw, hatch=hatch,
                    zorder=3, label=shorten_label(grp_name),
                )
                if std > 0:
                    ax.errorbar(
                        x_pos, val, yerr=std, fmt='none', ecolor='#475569',
                        elinewidth=0.7, capsize=1.4, capthick=0.7, zorder=4,
                    )

                top = val + std
                max_top = max(max_top, top)
                if annotate_vals:
                    rot = 90 if bw < 0.12 else 0
                    mean_fs = value_fontsize if bw >= 0.06 else max(4.5, value_fontsize - 1.0)
                    std_fs = max(4.0, mean_fs - 1.2)
                    mean_str = fmt_metric(val)
                    std_str = f" (±{fmt_metric(std)})"
                    # Mean in ink / bold
                    ax.annotate(
                        mean_str,
                        xy=(x_pos, top), xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', rotation=rot,
                        fontsize=mean_fs, fontweight='bold',
                        color=INK, zorder=5, annotation_clip=False,
                    )
                    # Std in faint / smaller — offset past the mean glyphs
                    # so the two never collide (adapts to 0 vs 0.16 vs 88)
                    if rot:
                        # Rotated 90°: text grows along +y in display points
                        extra = len(mean_str) * mean_fs * 0.62 + 1.5
                        ax.annotate(
                            std_str,
                            xy=(x_pos, top),
                            xytext=(0, 3 + extra),
                            textcoords='offset points',
                            ha='center', va='bottom', rotation=90,
                            fontsize=std_fs, fontweight='regular',
                            color=FAINT, zorder=5, annotation_clip=False,
                        )
                    else:
                        ax.annotate(
                            std_str,
                            xy=(x_pos, top),
                            xytext=(0, 3 + mean_fs + 2),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=std_fs, fontweight='regular',
                            color=FAINT, zorder=5, annotation_clip=False,
                        )

                if use_bar_labels and final_hue:
                    bar_label_jobs.append(
                        (x_pos, cls._bar_tick_label(grp_name), color)
                    )

        headroom = 22 if annotate_vals else 8
        if is_percent_metric(y_col, ylabel):
            y_top = percent_axis_limit(max_top, headroom=headroom * 0.45)
        else:
            y_top = max(max_top + headroom, 50)
        ax.set_ylim(0, y_top)
        if is_percent_metric(y_col, ylabel):
            ax.axhline(100, color=SPINE, linewidth=0.55, linestyle=':', zorder=1)

        # X ticks = category centers only
        ax.set_xticks(range(n_cats))
        cat_labels = [
            '' if c in ('overall', '__overall__') else shorten_label(c)
            for c in categories
        ]
        if use_bar_labels:
            # Series labels occupy the near-axis band; category names sit
            # further down so the two never collide.
            ax.set_xticklabels(
                cat_labels, rotation=0, ha='center', fontsize=8.5,
                fontweight=700, color=SLATE,
            )
            # Scale pad with label length (3-factor labels need more room)
            sample_len = max((len(t) for _, t, _ in bar_label_jobs), default=8)
            # Just enough room for rotated under-bar labels — avoid a large empty band
            ax.tick_params(axis='x', pad=max(28, 3.6 * sample_len))
            x_label_final = None
        else:
            max_len = max((len(str(l)) for l in cat_labels), default=0)
            rot = 0 if max_len <= 12 else (18 if max_len <= 18 else 28)
            ax.set_xticklabels(
                cat_labels, rotation=rot,
                ha=('right' if rot else 'center'), fontsize=8,
            )
            x_label_final = xlabel if xlabel else x_col.replace('_', ' ').title()

        # Under-bar series labels (rotated)
        if use_bar_labels and bar_label_jobs:
            # Data-x + axes-fraction-y keeps labels glued under the spine
            for x_pos, lab, color in bar_label_jobs:
                ax.annotate(
                    lab,
                    xy=(x_pos, 0.0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -4),
                    textcoords='offset points',
                    ha='center', va='top',
                    rotation=90,
                    fontsize=5.0 if len(lab) > 10 else 5.5,
                    fontweight=400,
                    color=color,
                    clip_on=False,
                    zorder=6,
                )

        ax.set_xlim(-0.55, n_cats - 0.45)

        # Subtle vertical guides between category clusters
        if n_cats > 1 and n_hue >= 3:
            for i in range(n_cats - 1):
                ax.axvline(
                    i + 0.5, color=GRID, linewidth=0.7,
                    linestyle='-', zorder=0, alpha=0.9,
                )

        # Encode key in subtitle when using hatch language
        cls._style_title_labels(
            ax, title, x_label_final, ylabel, subtitle,
            title_pad=(12 if subtitle else 8),
            subtitle_y=1.004,
        )

        if final_hue and use_bar_labels:
            has_fc, has_wire = detect_encoding_factors(hue_levels)
            # Sit below the axis / under-bar labels
            cls._add_encoding_legend(
                ax, y=0.01, fig=fig,
                has_fc=has_fc, has_wire=has_wire,
            )
        elif final_hue and not use_bar_labels:
            handles, lbls = ax.get_legend_handles_labels()
            seen = set()
            uniq = []
            for h, l in zip(handles, lbls):
                if l not in seen:
                    seen.add(l)
                    uniq.append((h, l))
            handles, lbls = zip(*uniq) if uniq else ([], [])
            ncol = min(len(lbls), 8) if lbls else 1
            ax.legend(
                handles, lbls,
                loc='lower center',
                bbox_to_anchor=(0.5, 1.04 if not subtitle else 1.07),
                ncol=ncol, frameon=False, fontsize=7.2,
                handlelength=1.4, columnspacing=1.3,
            )

        cls._add_footer(
            fig,
            (
                f"mean ± SE across n={n_seeds} units"
                if "_consistency_mode" in df_proc.columns
                else f"mean ± std across n={n_seeds} seeds"
            ),
        )

        if use_bar_labels:
            # Bottom room for rotated series labels + category names
            fig.tight_layout(rect=[0, 0.09, 1, 1])
        elif final_hue:
            fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        else:
            fig.tight_layout(rect=[0, 0.02, 1, 1])
        return fig, ax, summary

    @classmethod
    def _bar_tick_label(cls, name: str) -> str:
        """Ultra-short label for under-bar / y-axis identity text.

        Examples
        --------
        burrito@llamacpp (fc_model=0, wire_api=chat)  →  b-llama·fc0·ch
        vllm (fc_model=1, wire_api=responses)         →  vllm·fc1·re
        """
        s = shorten_label(name)
        # Ultra-short stems (longer first so bp-* isn't partially rewritten)
        s = s.replace('burrito-pt-llama', 'bp-llama')
        s = s.replace('burrito-pt-vllm', 'bp-vllm')
        s = s.replace('burrito-llama', 'b-llama')
        s = s.replace('burrito-vllm', 'b-vllm')
        s = s.replace('llama-default', 'l-def')
        s = s.replace('llama-fixed', 'l-fix')
        # Collapse parenthetical factors into ·tokens
        s = re.sub(r'wire_api\s*=\s*responses', 're', s, flags=re.I)
        s = re.sub(r'wire_api\s*=\s*chat', 'ch', s, flags=re.I)
        s = re.sub(r'fc_model\s*=\s*([01])', r'fc\1', s, flags=re.I)
        s = re.sub(r'fc\s*=\s*([01])', r'fc\1', s, flags=re.I)
        s = re.sub(
            r'reasoning[_\s-]?(?:effort|level)?\s*=\s*none', 'n', s, flags=re.I
        )
        s = re.sub(
            r'reasoning[_\s-]?(?:effort|level)?\s*=\s*low', 'lo', s, flags=re.I
        )
        s = re.sub(
            r'reasoning[_\s-]?(?:effort|level)?\s*=\s*medium', 'md', s, flags=re.I
        )
        s = re.sub(
            r'reasoning[_\s-]?(?:effort|level)?\s*=\s*high', 'hi', s, flags=re.I
        )
        # "(0, ch)" or "(0, re)" or "(0)" → ·0·ch
        def _paren_to_dots(m):
            inner = m.group(1)
            parts = re.split(r'[,\s]+', inner.strip())
            parts = [p for p in parts if p]
            return '·' + '·'.join(parts) if parts else ''
        s = re.sub(r'\(([^)]*)\)', _paren_to_dots, s)
        s = re.sub(r'\s+', '', s)
        return s.strip('·').strip()

    @classmethod
    def _sort_hue_pairs(cls, hue_levels: list) -> list:
        """Order combined labels by backend, fc, reasoning_effort, wire_api.

        Backend families follow BACKEND_DISPLAY_ORDER. Reasoning effort follows
        none < low < medium < high. Wire API: chat before responses.
        """
        effort_rank = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}

        def family_rank(family: str) -> tuple:
            f = family.lower().strip()
            hits = [
                (i, pat) for i, pat in enumerate(BACKEND_DISPLAY_ORDER)
                if pat.lower() in f
            ]
            if hits:
                i, _ = max(hits, key=lambda t: len(t[1]))
                return (0, i)
            return (1, f)

        def sort_key(name):
            s = str(name).lower()
            family = re.sub(r'\s*\(.*\)\s*$', '', s).strip()
            is_fc1 = bool(
                re.search(r'fc[_=]?model\s*=\s*1|(?<![\d.])fc\s*=\s*1', s)
            ) or s in ('1', '1.0')
            is_resp = bool(re.search(r'wire[_\s-]?api\s*=\s*responses', s))
            m_eff = re.search(
                r'reasoning[_\s-]?(?:effort|level)?\s*=\s*(\w+)', s
            )
            if not m_eff:
                m_eff = re.search(r'\b(none|low|medium|high)\b', s)
            eff = effort_rank.get(m_eff.group(1), 50) if m_eff else 25
            return (
                family_rank(family),
                0 if not is_fc1 else 1,
                eff,
                0 if not is_resp else 1,
                s,
            )

        return sorted(hue_levels, key=sort_key)

    @classmethod
    def plot_pareto_quadrant(
        cls,
        data: Union[pd.DataFrame, EvalPivotResult],
        x_col: str,
        y_col: str,
        label_col: Union[str, List[str]],
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Accuracy vs Latency Trade-off (Pareto Frontier)",
        subtitle: Optional[str] = None,
        xlabel: str = "Latency / Turn (s)  ·  lower is better",
        ylabel: str = "Accuracy (%)  ·  higher is better",
        figsize: Tuple[float, float] = (8.6, 5.2),
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None):
        """Pareto frontier with shaded ideal corner and collision-aware labels."""
        df_proc, seed_col, final_label = cls._prepare_data(
            data, filter_query, label_col, seed_col, value_col=y_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold,
            min_turns=min_turns,
            pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _is_consistency = '_consistency_mode' in df_proc.columns
        _footer_n_label = 'units' if _is_consistency else 'seeds'
        fig, ax = cls._new_figure(figsize, dpi=dpi)

        seed_agg = (
            df_proc.groupby([final_label, seed_col], observed=True)[[x_col, y_col]]
            .mean()
            .reset_index()
        )
        summary = (
            seed_agg.groupby(final_label, observed=True)[[x_col, y_col]]
            .mean()
            .reset_index()
        )

        med_x = float(summary[x_col].median()) if len(summary) else 0.0
        med_y = float(summary[y_col].median()) if len(summary) else 0.0
        max_y = float(summary[y_col].max()) if len(summary) else 0.0

        cls._apply_paper_style(ax, grid_axis='y')
        ax.grid(True, axis='x', linestyle='-', linewidth=0.65, alpha=0.7, color=GRID)

        if is_percent_metric(y_col, ylabel):
            y_top = percent_axis_limit(max_y, headroom=8)
        else:
            y_top = max(max_y * 1.12, 1.0)
        ax.set_ylim(0, y_top)
        x_max = float(summary[x_col].max()) if len(summary) else 1.0
        x_right = x_max * 1.14 if x_max > 0 else 1.0
        ax.set_xlim(0, x_right)

        # Ideal quadrant (low latency, high accuracy)
        ax.add_patch(Rectangle(
            (0, med_y), max(med_x, 1e-9), max(y_top - med_y, 1e-9),
            facecolor='#10b981', alpha=0.06, zorder=0, linewidth=0,
        ))
        ax.text(
            x_right * 0.015, y_top * 0.98, "faster & more accurate",
            fontsize=7, color='#059669', fontweight='bold',
            ha='left', va='top', alpha=0.95,
        )

        ax.axvline(med_x, color=SPINE, linestyle='--', linewidth=0.75, zorder=1)
        ax.axhline(med_y, color=SPINE, linestyle='--', linewidth=0.75, zorder=1)

        # Pareto frontier (min x, max y)
        sorted_pts = summary.sort_values(
            by=[x_col, y_col], ascending=[True, False]
        ).copy()
        pareto_pts, best_y = [], -np.inf
        for _, row in sorted_pts.iterrows():
            if row[y_col] > best_y:
                pareto_pts.append(row)
                best_y = row[y_col]
        pareto_df = pd.DataFrame(pareto_pts)

        if not pareto_df.empty:
            ax.step(
                pareto_df[x_col], pareto_df[y_col], where='post',
                color='#f59e0b', linestyle='-', linewidth=1.7,
                alpha=0.92, zorder=2,
            )

        texts = []
        for _, row in summary.iterrows():
            color, _, marker = get_semantic_style(row[final_label])
            ax.scatter(
                row[x_col], row[y_col], color=color, marker=marker,
                s=72, zorder=5, edgecolor='white', linewidth=1.1,
            )
            lbl = shorten_label(row[final_label])
            val_txt = fmt_metric(float(row[y_col]))
            txt = ax.text(
                row[x_col], row[y_col],
                f"  {lbl}  {val_txt}",
                fontsize=7.0, fontweight='bold', color=INK, zorder=6,
            )
            texts.append(txt)

        if HAS_ADJUST_TEXT and texts:
            adjust_text(
                texts, ax=ax,
                expand_text=(1.15, 1.3), expand_points=(1.15, 1.3),
                arrowprops=dict(arrowstyle="-", color=SPINE, lw=0.55),
            )

        cls._style_title_labels(
            ax, title, xlabel, ylabel, subtitle,
            title_pad=(12 if subtitle else 8), subtitle_y=1.004,
        )
        cls._add_footer(
            fig,
            f"mean across n={n_seeds} seeds  ·  amber step = Pareto frontier",
        )
        fig.tight_layout(rect=[0, 0.02, 1, 1])
        return fig, ax, summary

    # ---------------------------------------------------------------
    # horizontal capability bars
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # horizontal capability bars
    # ---------------------------------------------------------------

    @classmethod
    def plot_capability_bars(
        cls,
        data: Union[pd.DataFrame, EvalPivotResult],
        category_col: Optional[str],
        value_col: str,
        group_col: Union[str, List[str]],
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Capability Profile by Benchmark",
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        show_values: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        label_bars: Optional[bool] = None,
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None):
        """
        Horizontal capability bars (model-card style).

        Dense series (backend × fc_model, etc.):
          • One row per series — no legend, name on the y-axis
          • fc=0 hatched / fc=1 solid (same language as plot_bar_comparison)
          • Categories stacked as labeled bands with soft separators
          • Height scales with the number of rows

        Sparse series keep a compact clustered dodge + top legend.
        """
        df_proc, seed_col, final_group = cls._prepare_data(
            data, filter_query, group_col, seed_col, value_col=value_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold,
            min_turns=min_turns,
            pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _is_consistency = '_consistency_mode' in df_proc.columns
        _footer_n_label = 'units' if _is_consistency else 'seeds'

        if category_col is None:
            category_col = '__overall__'
            df_proc = df_proc.copy()
            df_proc[category_col] = 'overall'

        seed_agg = (
            df_proc.groupby(
                [category_col, final_group, seed_col], observed=True
            )[value_col]
            .mean()
            .reset_index()
        )
        summary = (
            seed_agg.groupby([category_col, final_group], observed=True)[value_col]
            .agg(['mean', 'std'])
            .reset_index()
        )
        summary = cls._fix_consistency_std(
            summary, df_proc, value_col, [category_col, final_group],
        )

        categories = cls._ordered_levels(
            df_proc, category_col, summary[category_col]
        )  # top-to-bottom after reverse later
        hue_levels = cls._ordered_levels(
            df_proc, final_group, summary[final_group]
        )
        hue_levels = cls._sort_hue_pairs(hue_levels)
        n_hue = len(hue_levels)
        n_cats = len(categories)

        # Expanded one-row-per-bar once a legend would hurt
        use_rows = (
            label_bars if label_bars is not None
            else bool(n_hue >= 4)
        )

        if use_rows:
            return cls._capability_bars_expanded(
                summary, categories, hue_levels, final_group,
                category_col, value_col, n_seeds,
                title, subtitle, xlabel, show_values, figsize, dpi,
            )

        # ----- sparse: classic clustered horizontal dodge -------------------
        categories = categories[::-1]
        if figsize is None:
            figsize = (9.0, max(3.8, 0.55 * n_cats + 1.5))
        fig, ax = cls._new_figure(figsize, dpi=dpi)

        max_cluster = 0.82
        g_ratio = 0.18
        denom = n_hue + (n_hue - 1) * g_ratio
        bw = max_cluster / max(denom, 1)
        bw = max(min(bw, 0.28), 0.04)
        gap = g_ratio * bw
        offsets = cls._dodge_offsets(hue_levels, bw, gap)
        cls._apply_paper_style(ax, grid_axis='x')

        labeled = set()
        max_right = 0.0
        for c_idx, cat in enumerate(categories):
            cat_sub = summary[summary[category_col] == cat]
            for h in hue_levels:
                row = cat_sub[cat_sub[final_group] == h]
                if row.empty or pd.isna(row['mean'].iloc[0]):
                    continue
                val = float(row['mean'].iloc[0])
                std = row['std'].iloc[0]
                std = 0.0 if pd.isna(std) else float(std)

                y_pos = c_idx + offsets[h]
                color, hatch, edge, face_alpha, lw = get_bar_style(h)

                lbl = shorten_label(h) if h not in labeled else None
                labeled.add(h)

                ax.barh(
                    y_pos, val, height=bw * 0.96, color=color,
                    alpha=face_alpha, edgecolor=edge, linewidth=lw,
                    hatch=hatch, zorder=3, label=lbl,
                )
                if std > 0:
                    ax.errorbar(
                        val, y_pos, xerr=std, fmt='none', ecolor='#475569',
                        elinewidth=0.75, capsize=1.6, capthick=0.75, zorder=4,
                    )
                right = val + std
                max_right = max(max_right, right)
                if show_values:
                    mean_str = fmt_metric(val)
                    ax.annotate(
                        mean_str, (right + 1.0, y_pos),
                        va='center', ha='left',
                        fontsize=6.5, fontweight='bold', color=INK, zorder=5,
                    )
                    ax.annotate(
                        f" (±{fmt_metric(std)})", (right + 1.0, y_pos),
                        xytext=(len(mean_str) * 4.2 + 2, 0),
                        textcoords='offset points',
                        va='center', ha='left',
                        fontsize=5.2, color=FAINT, zorder=5,
                    )

        ax.set_ylim(-0.55, n_cats - 0.45)
        if is_percent_metric(value_col, xlabel):
            x_right = percent_axis_limit(max_right, headroom=10)
        else:
            x_right = max(max_right + 14, 55)
        ax.set_xlim(0, x_right)
        if is_percent_metric(value_col, xlabel):
            ax.axvline(100, color=SPINE, linewidth=0.55, linestyle=':', zorder=1)
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(
            [shorten_label(c) for c in categories], fontsize=8.5
        )

        default_xlab = {
            'correct': 'Accuracy (%)',
            'mt_num_turns_success_pct': 'Multi-Turn Success Rate (%)',
            'mt_latency_success': 'Latency / Turn (s)',
        }.get(value_col, value_col.replace('_', ' ').title() + " (%)")
        cls._style_title_labels(
            ax, title, xlabel if xlabel else default_xlab, None, subtitle,
            title_pad=(12 if subtitle else 8), subtitle_y=1.004,
        )
        has_fc, has_wire = detect_encoding_factors(hue_levels)
        if has_fc or has_wire:
            cls._add_encoding_legend(
                ax, y=0.01, fig=fig, has_fc=has_fc, has_wire=has_wire,
            )
            fig.tight_layout(rect=[0, 0.10, 1, 1])
        else:
            cls._legend_top(ax, y=1.04 if subtitle else 1.02)
            fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        cls._add_footer(
            fig,
            (
                f"mean ± SE across n={n_seeds} units"
                if "_consistency_mode" in df_proc.columns
                else f"mean ± std across n={n_seeds} seeds"
            ),
        )
        return fig, ax, summary

    @classmethod
    def _capability_bars_expanded(
        cls,
        summary, categories, hue_levels, final_group,
        category_col, value_col, n_seeds,
        title, subtitle, xlabel, show_values, figsize, dpi=None,
    ):
        """One horizontal bar per (category, series) with y-axis identity labels."""
        # Build row list: category bands, first category at top
        rows = []  # (y_label, category, hue, val, std)
        for cat in categories:
            cat_sub = summary[summary[category_col] == cat]
            for h in hue_levels:
                row = cat_sub[cat_sub[final_group] == h]
                if row.empty or pd.isna(row['mean'].iloc[0]):
                    continue
                val = float(row['mean'].iloc[0])
                std = row['std'].iloc[0]
                std = 0.0 if pd.isna(std) else float(std)
                # When multiple categories, prefix is redundant on every row —
                # category bands handle that. Label = series only.
                rows.append((cls._bar_tick_label(h), cat, h, val, std))

        if not rows:
            raise ValueError("No data to plot after filtering.")

        n_rows = len(rows)
        # Reverse so first category ends up on top
        rows = rows[::-1]

        if figsize is None:
            fig_h = max(4.0, min(0.38 * n_rows + 1.6, 16.0))
            figsize = (9.2, fig_h)

        fig, ax = cls._new_figure(figsize, dpi=dpi)
        cls._apply_paper_style(ax, grid_axis='x')

        max_right = 0.0
        y_positions = list(range(n_rows))
        y_labels = []
        cat_boundaries = []  # y midpoints between categories for separators

        prev_cat = None
        for i, (ylab, cat, h, val, std) in enumerate(rows):
            if prev_cat is not None and cat != prev_cat:
                cat_boundaries.append(i - 0.5)
            prev_cat = cat

            color, hatch, edge, face_alpha, lw = get_bar_style(h)

            ax.barh(
                i, val, height=0.72, color=color, alpha=face_alpha,
                edgecolor=edge, linewidth=lw, hatch=hatch, zorder=3,
            )
            if std > 0:
                ax.errorbar(
                    val, i, xerr=std, fmt='none', ecolor='#475569',
                    elinewidth=0.75, capsize=1.6, capthick=0.75, zorder=4,
                )
            right = val + std
            max_right = max(max_right, right)
            if show_values:
                mean_str = fmt_metric(val)
                ax.annotate(
                    mean_str, (right + 1.0, i),
                    va='center', ha='left',
                    fontsize=6.5, fontweight='bold', color=INK, zorder=5,
                )
                ax.annotate(
                    f" (±{fmt_metric(std)})", (right + 1.0, i),
                    xytext=(len(mean_str) * 4.2 + 2, 0),
                    textcoords='offset points',
                    va='center', ha='left',
                    fontsize=5.2, color=FAINT, zorder=5,
                )
            y_labels.append(ylab)

        # Category band labels on the right edge when multiple categories
        if len(categories) > 1:
            # Find span of each category in the reversed row list
            from collections import defaultdict
            spans = defaultdict(list)
            for i, (_, cat, _, _, _) in enumerate(rows):
                spans[cat].append(i)
            for cat, idxs in spans.items():
                mid = (min(idxs) + max(idxs)) / 2.0
                ax.text(
                    1.01, mid, shorten_label(cat),
                    transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=7.5,
                    fontweight=700, color=MUTED, clip_on=False,
                )

        for b in cat_boundaries:
            ax.axhline(b, color=GRID, linewidth=0.8, zorder=1)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=7.5)
        # Color y-tick labels to match their bar
        for tick, (_, _, h, _, _) in zip(ax.get_yticklabels(), rows):
            color, _, _ = get_semantic_style(h)
            tick.set_color(color)
            tick.set_fontweight(400)

        ax.set_ylim(-0.6, n_rows - 0.4)
        if is_percent_metric(value_col, xlabel):
            x_right = percent_axis_limit(max_right, headroom=10)
        else:
            x_right = max(max_right + 16, 55)
        ax.set_xlim(0, x_right)
        if is_percent_metric(value_col, xlabel):
            ax.axvline(100, color=SPINE, linewidth=0.55, linestyle=':', zorder=1)

        default_xlab = {
            'correct': 'Accuracy (%)',
            'mt_num_turns_success_pct': 'Multi-Turn Success Rate (%)',
            'mt_latency_success': 'Latency / Turn (s)',
        }.get(value_col, value_col.replace('_', ' ').title() + " (%)")

        cls._style_title_labels(
            ax, title, xlabel if xlabel else default_xlab, None, subtitle,
            title_pad=(14 if subtitle else 10),
            subtitle_y=1.008,
        )
        has_fc, has_wire = detect_encoding_factors(hue_levels)
        cls._add_encoding_legend(
            ax, y=0.01, fig=fig,
            has_fc=has_fc, has_wire=has_wire,
        )
        cls._add_footer(
            fig,
            (
                f"mean ± SE across n={n_seeds} units"
                if "_consistency_mode" in df_proc.columns
                else f"mean ± std across n={n_seeds} seeds"
            ),
        )
        fig.tight_layout(rect=[0, 0.12, 0.98 if len(categories) > 1 else 1, 1])
        return fig, ax, summary

    @classmethod
    def plot_radar(
        cls,
        data: Union[pd.DataFrame, EvalPivotResult],
        category_col: str,
        value_col: str,
        group_col: Union[str, List[str]],
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Capability Profile across Benchmarks",
        subtitle: Optional[str] = None,
        figsize: Tuple[float, float] = (6.6, 6.6),
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None):
        """Wireframe radar. Spokes follow category insertion / categorical order."""
        df_proc, seed_col, final_group = cls._prepare_data(
            data, filter_query, group_col, seed_col, value_col=value_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold,
            min_turns=min_turns,
            pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _is_consistency = '_consistency_mode' in df_proc.columns
        _footer_n_label = 'units' if _is_consistency else 'seeds'

        seed_agg = (
            df_proc.groupby(
                [final_group, category_col, seed_col], observed=True
            )[value_col]
            .mean()
            .reset_index()
        )
        pivot = (
            seed_agg.groupby([final_group, category_col], observed=True)[value_col]
            .mean()
            .unstack(level=category_col)
        )

        cat_order = cls._ordered_levels(
            df_proc, category_col, seed_agg[category_col]
        )
        cat_order = [c for c in cat_order if c in pivot.columns]
        pivot = pivot[cat_order]

        # Stable group order matching bar charts
        grp_order = cls._sort_hue_pairs(list(pivot.index))
        pivot = pivot.reindex(grp_order)

        categories = list(pivot.columns)
        N = len(categories)
        if N < 3:
            raise ValueError(
                f"Radar needs ≥3 categories; got {N}: {categories}"
            )

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = cls._new_figure(figsize, dpi=dpi, subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [shorten_label(c) for c in categories],
            fontsize=8, fontweight=700, color=SLATE,
        )
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], fontsize=7, color=FAINT)
        ax.set_rlabel_position(22.5)
        ax.grid(color=GRID, linestyle='-', linewidth=0.75)
        ax.spines['polar'].set_color(SPINE)
        ax.spines['polar'].set_linewidth(0.8)

        for group_name, row in pivot.iterrows():
            color, ls, marker = get_semantic_style(group_name)
            values = [0.0 if pd.isna(v) else float(v) for v in row.values]
            # Scale fractions if needed
            if values and max(values) <= 1.0 and max(values) > 0:
                values = [v * 100.0 for v in values]
            values += values[:1]
            ax.plot(
                angles, values, linewidth=1.7, linestyle=ls,
                marker=marker, markersize=5.0,
                markeredgecolor='white', markeredgewidth=0.8,
                label=shorten_label(group_name), color=color, zorder=3,
            )
            ax.fill(angles, values, alpha=0.07, color=color, zorder=2)

        ax.set_title(
            title, size=12, color=INK, y=1.14, fontweight='bold', loc='center',
        )
        if subtitle:
            ax.text(
                0.5, 1.08, subtitle, transform=ax.transAxes,
                fontsize=7.8, color=FAINT, ha='center', va='bottom',
            )

        n_grp = len(pivot)
        ax.legend(
            bbox_to_anchor=(0.5, -0.08), loc='upper center',
            ncol=min(n_grp, 8), frameon=False, fontsize=7.0,
            handlelength=1.6, columnspacing=1.2,
        )
        fig.text(
            0.995, 0.005,
            f"mean across n={n_seeds} seeds",
            ha='right', va='bottom', fontsize=6.5, color=FAINT, style='italic',
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        return fig, ax, pivot.reset_index()

    @classmethod
    def plot_turn_survival(
        cls,
        data: Union[pd.DataFrame, EvalPivotResult],
        group_col: Optional[Union[str, List[str]]] = None,
        filter_query: Optional[str] = None,
        max_turn: Optional[int] = None,
        min_reached: int = 50,
        title: str = "Multi-Turn Step Survival",
        subtitle: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        show_cum: bool = False,
        facet_col: Optional[str] = None,
    ):
        """Tian-style multi-turn reliability curves.

        Parameters
        ----------
        min_reached : int
            Turns with fewer trajectories than this get NaN rates (default 50).
            Raise it to hide noisy tails; lower it to keep long-turn estimates.
        show_cum : bool
            Exclusive mode switch:
              False → P(pass turn t | reached t)   [default]
              True  → cumulative product ∏ rates (e2e estimate)
            Only one is drawn so the chart stays readable.
        facet_col : str, optional
            Small multiples (e.g. ``'reasoning_effort'``).
        """
        if isinstance(data, EvalPivotResult):
            df = data.filtered_df.copy()
        else:
            df = data.copy()
        df = apply_sql_filter(df, filter_query)
        df = apply_categorical_orders(df)

        def _prepare_groups(frame, gcol):
            if gcol is None:
                return frame, None, None
            if isinstance(gcol, str):
                return frame, [gcol], gcol
            gcols = list(gcol)
            if len(gcols) == 1:
                return frame, gcols, gcols[0]
            frame = frame.copy()
            # Prefer "backend (fc_model=0)" so get_semantic_style / strip
            # helpers parse factors reliably (not bare "backend · 0").
            def _combine(row):
                base = str(row[gcols[0]])
                extras = [f"{c}={row[c]}" for c in gcols[1:]]
                return f"{base} ({', '.join(extras)})"
            frame['__group__'] = frame.apply(_combine, axis=1)
            return frame, ['__group__'], '__group__'

        # Faceted layout
        if facet_col is not None:
            if facet_col not in df.columns:
                raise KeyError(f"facet_col {facet_col!r} not in data")
            # order facets
            if facet_col in ('reasoning_effort', 'reasoning_level', 'reasoning'):
                facet_levels = [
                    v for v in ['none', 'low', 'medium', 'high']
                    if v in set(df[facet_col].astype(str))
                ]
                if not facet_levels:
                    facet_levels = list(pd.unique(df[facet_col].astype(str)))
            else:
                facet_levels = list(pd.unique(df[facet_col].astype(str)))

            n_f = len(facet_levels)
            if figsize is None:
                figsize = (min(4.5 * n_f, 16), 4.6)
            fig, axes = plt.subplots(
                1, n_f, figsize=figsize, dpi=dpi or DEFAULT_DPI, sharey=True,
            )
            if n_f == 1:
                axes = [axes]

            surv_parts = []
            for ax, fval in zip(axes, facet_levels):
                sub_df = df[df[facet_col].astype(str) == str(fval)]
                sub_df, gcols, plot_group = _prepare_groups(sub_df, group_col)
                surv = compute_turn_survival(
                    sub_df, group_cols=gcols, max_turn=max_turn,
                    min_reached=min_reached,
                )
                if not surv.empty:
                    part = surv.copy()
                    part[facet_col] = fval
                    surv_parts.append(part)
                cls._draw_survival_ax(
                    ax, surv, plot_group, show_cum=show_cum, title=str(fval),
                )
            plot_data = (
                pd.concat(surv_parts, ignore_index=True) if surv_parts
                else pd.DataFrame()
            )

            fig.suptitle(title, fontsize=12, fontweight='bold', color=INK, y=1.04)
            if subtitle:
                fig.text(
                    0.5, 0.975, subtitle, ha='center', fontsize=7.5,
                    color=FAINT, style='italic',
                )
            # Merge legend entries across all facets (union of series).
            # fc0/fc1 last so encoding keys aren't mixed into backend series.
            seen, h2, l2 = set(), [], []
            for ax in axes:
                handles, labels = getattr(ax, '_survival_legend', (None, None))
                if handles is None:
                    handles, labels = ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l not in seen:
                        seen.add(l)
                        h2.append(h)
                        l2.append(l)
            if h2:
                body = [(h, l) for h, l in zip(h2, l2) if l not in ('fc0', 'fc1')]
                tail = [(h, l) for h, l in zip(h2, l2) if l in ('fc0', 'fc1')]
                tail = sorted(tail, key=lambda x: 0 if x[1] == 'fc0' else 1)
                h2, l2 = (
                    map(list, zip(*(body + tail))) if (body or tail) else ([], [])
                )
                ncol = min(len(h2), 16)
                fig.legend(
                    h2, l2, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                    ncol=ncol, frameon=False, fontsize=6.5,
                    handlelength=2.2,
                )
                fig.tight_layout(rect=[0, 0.12, 1, 0.93])
            else:
                fig.tight_layout(rect=[0, 0.02, 1, 0.93])
            mode = "cum. survival ∏ rates" if show_cum else "P(pass t | reached t)"
            notes = [mode, f"turns with n<{min_reached} omitted"]
            if any(getattr(ax, '_survival_has_fc', False) for ax in axes):
                notes.append("dotted/hollow = fc0 · solid/filled = fc1")
            cls._add_footer(fig, "  ·  ".join(notes))
            return fig, axes, plot_data

        # Single-axis layout
        df, gcols, plot_group = _prepare_groups(df, group_col)
        surv = compute_turn_survival(
            df, group_cols=gcols, max_turn=max_turn, min_reached=min_reached,
        )
        n_series = (
            1 if plot_group is None
            else surv[plot_group].nunique() if plot_group in surv.columns else 1
        )
        if figsize is None:
            figsize = (8.4, 5.0)

        fig, ax = cls._new_figure(figsize, dpi=dpi)
        cls._draw_survival_ax(ax, surv, plot_group, show_cum=show_cum, title=None)
        cls._style_title_labels(
            ax, title, None, None, subtitle,
            title_pad=(12 if subtitle else 8), subtitle_y=1.004,
        )
        handles, labels = getattr(ax, '_survival_legend', (None, None))
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        if h2:
            body = [(h, l) for h, l in zip(h2, l2) if l not in ('fc0', 'fc1')]
            tail = [(h, l) for h, l in zip(h2, l2) if l in ('fc0', 'fc1')]
            tail = sorted(tail, key=lambda x: 0 if x[1] == 'fc0' else 1)
            h2, l2 = (
                map(list, zip(*(body + tail))) if (body or tail) else ([], [])
            )
        if h2:
            if n_series >= 8:
                ax.legend(
                    h2, l2, loc='upper center', bbox_to_anchor=(0.5, -0.14),
                    ncol=min(n_series, 16), frameon=False, fontsize=6.5,
                    handlelength=2.2,
                )
                fig.tight_layout(rect=[0, 0.14, 1, 1])
            else:
                ax.legend(
                    h2, l2, ncol=16, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                    frameon=False, fontsize=7.0, handlelength=2.2,
                )
                fig.tight_layout(rect=[0, 0.02, 0.78, 1])
        else:
            fig.tight_layout(rect=[0, 0.02, 1, 1])
        mode = "cum. survival ∏ rates" if show_cum else "P(pass t | reached t)"
        notes = [mode, f"turns with n<{min_reached} omitted"]
        if getattr(ax, '_survival_has_fc', False):
            notes.append("dotted/hollow = fc0 · solid/filled = fc1")
        cls._add_footer(fig, "  ·  ".join(notes))
        return fig, ax, surv

    @classmethod
    def _draw_survival_ax(
        cls, ax, surv, plot_group, show_cum=False, title=None,
    ):
        """Draw either step pass-rate OR cumulative survival (exclusive).

        Encoding (aligned with plot_effort_story / plot_compute_scaling):
          • color     = backend family
          • linestyle = fc (dotted=fc0, solid=fc1); falls back to wire_api
            (chat→dotted, responses→solid) when no fc factor is present
          • marker    = filled for fc1 / hollow for fc0 when fc is present
          • legend    = stripped backend names + separate fc0/fc1 entries
        """
        from matplotlib.lines import Line2D

        cls._apply_paper_style(ax, grid_axis='y')
        if title:
            ax.set_title(title, fontsize=10, fontweight=700, color=INK, pad=6)

        y_col = 'cum_survival' if show_cum else 'pass_rate'
        y_label = (
            'Cumulative survival (%)' if show_cum
            else 'P(pass turn t | reached t) (%)'
        )

        legend_handles = []
        legend_labels = []
        seen_labels = set()
        has_fc = False

        if plot_group is None:
            sub = surv.dropna(subset=[y_col]).sort_values('turn')
            if not sub.empty:
                line, = ax.plot(
                    sub['turn'], sub[y_col] * 100,
                    color='#4f46e5', marker='o', linewidth=2.0, markersize=6,
                    markeredgecolor='white', markeredgewidth=1.0,
                    label=('cum. survival' if show_cum else 'step pass rate'),
                    zorder=3,
                )
                legend_handles.append(line)
                legend_labels.append(line.get_label())
                if not show_cum and sub['se'].notna().any():
                    ax.fill_between(
                        sub['turn'],
                        (sub['pass_rate'] - sub['se']).clip(lower=0) * 100,
                        (sub['pass_rate'] + sub['se']).clip(upper=1) * 100,
                        color='#4f46e5', alpha=0.12, zorder=2,
                    )
        else:
            raw = [str(g) for g in surv[plot_group].dropna().unique()]
            groups = cls._sort_hue_pairs(raw)
            has_fc, _ = detect_encoding_factors(groups)
            # Also detect bare 0/1 combined labels
            if not has_fc:
                has_fc = any(
                    bool(re.search(r'[·,\s][01](?:\s*$|\s*[·,])', str(g)))
                    or str(g).strip() in ('0', '1', '0.0', '1.0')
                    for g in groups
                )
            n = len(groups)
            lw = 1.3 if n >= 10 else 1.8
            ms = 3.2 if n >= 10 else 5.0
            for grp in groups:
                sub = (
                    surv[surv[plot_group].astype(str) == str(grp)]
                    .dropna(subset=[y_col])
                    .sort_values('turn')
                )
                if sub.empty:
                    continue
                color, ls, _ = get_semantic_style(grp)
                # Fall back to wire_api linestyle only when no fc factor
                if not has_fc:
                    s = str(grp).lower()
                    tokens = [t for t in re.split(r'[·\s,=]+', s) if t]
                    if 'chat' in tokens or 'ch' in tokens:
                        ls = ':'
                    elif 'responses' in tokens or 're' in tokens:
                        ls = '-'
                short = (
                    strip_fc_effort_label(grp) if has_fc
                    else cls._bar_tick_label(grp)
                )
                is_fc0 = ls in (':', 'dotted')
                # Line
                ax.plot(
                    sub['turn'], sub[y_col] * 100,
                    color=color, linestyle=ls, marker=None,
                    linewidth=lw, alpha=0.92, zorder=3,
                )
                # Markers: hollow for fc0, filled for fc1
                if is_fc0 and has_fc:
                    ax.scatter(
                        sub['turn'], sub[y_col] * 100,
                        s=ms ** 1.6, facecolors='none', edgecolors=color,
                        marker='o', linewidths=1.2, zorder=4, alpha=0.95,
                    )
                else:
                    ax.scatter(
                        sub['turn'], sub[y_col] * 100,
                        s=ms ** 1.5, color=color, marker='o',
                        edgecolor='white', linewidth=0.7, zorder=4, alpha=0.95,
                    )
                if short not in seen_labels:
                    seen_labels.add(short)
                    proxy = Line2D(
                        [0], [0], color=color, linestyle=ls, marker='o',
                        markersize=6, markeredgecolor='white', label=short,
                    )
                    legend_handles.append(proxy)
                    legend_labels.append(short)

            if has_fc:
                for lab, face in (('fc0', 'none'), ('fc1', SLATE)):
                    if lab not in seen_labels:
                        seen_labels.add(lab)
                        legend_handles.append(Line2D(
                            [0], [0], color=SLATE, marker='o', linestyle='None',
                            markersize=6.5,
                            markerfacecolor=face,
                            markeredgecolor=SLATE,
                            markeredgewidth=1.3 if face == 'none' else 0.8,
                            label=lab,
                        ))
                        legend_labels.append(lab)

        ax.set_ylim(0, 105)
        ax.axhline(100, color=SPINE, linewidth=0.55, linestyle=':', zorder=1)
        if len(surv) and surv[y_col].notna().any():
            valid_turns = sorted(
                surv.loc[surv[y_col].notna(), 'turn'].unique()
            )
            if valid_turns:
                ax.set_xticks(valid_turns)
        ax.set_xlabel('Turn index (0-based)', fontsize=8.5, color=SLATE)
        ax.set_ylabel(y_label, fontsize=8.5, color=SLATE)
        ax._survival_legend = (legend_handles, legend_labels)
        ax._survival_has_fc = has_fc

    @classmethod
    def plot_pass_curves(
        cls,
        data: Union[pd.DataFrame, EvalPivotResult],
        group_col: Optional[Union[str, List[str]]] = None,
        value_col: str = 'correct',
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        k_max: Optional[int] = None,
        min_turns: Optional[int] = None,
        facet_col: Optional[str] = None,
        title: str = "pass@k vs pass^k",
        subtitle: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        show_at: bool = True,
        show_hat: bool = True,
    ):
        """Lab-style reliability curves: pass@k (dotted) and pass^k (solid).

        Parameters
        ----------
        facet_col : str, optional
            Small multiples (e.g. ``'reasoning_effort'``) so series stay readable.
        show_at / show_hat : bool
            Toggle dotted pass@k and solid pass^k independently.
        """
        if isinstance(data, EvalPivotResult):
            df = data.filtered_df.copy()
        else:
            df = data.copy()

        if group_col is None:
            gcols = None
            plot_group = None
        elif isinstance(group_col, str):
            gcols = [group_col]
            plot_group = group_col
        else:
            gcols = list(group_col)
            if len(gcols) == 1:
                plot_group = gcols[0]
            else:
                df = df.copy()
                def _combine(row):
                    base = str(row[gcols[0]])
                    extras = [f"{c}={row[c]}" for c in gcols[1:]]
                    return f"{base} ({', '.join(extras)})"
                df['__group__'] = df.apply(_combine, axis=1)
                gcols = ['__group__']
                plot_group = '__group__'

        # Facet must not also be in group_cols for the curve table
        curve_groups = list(gcols) if gcols else []
        if facet_col and facet_col not in (curve_groups or []):
            # compute per facet×group by temporarily including facet in groups
            curve_groups_full = ([facet_col] + curve_groups) if curve_groups else [facet_col]
        else:
            curve_groups_full = curve_groups or None

        curves = compute_pass_curves(
            df,
            value_col=value_col,
            group_cols=curve_groups_full,
            filter_query=filter_query,
            seed_col=seed_col,
            unit_col=unit_col,
            k_max=k_max,
            min_turns=min_turns,
        )
        if curves.empty:
            raise ValueError("No pass curves to plot (empty after filter).")

        def _draw(ax, sub, legend=False):
            from matplotlib.lines import Line2D
            cls._apply_paper_style(ax, grid_axis='y')
            handles, labels = [], []
            seen = set()
            has_fc = False
            if plot_group is None or plot_group not in sub.columns:
                s = sub.sort_values('k')
                if show_at:
                    line, = ax.plot(
                        s['k'], s['pass_at'] * 100,
                        color='#4f46e5', linestyle=':', linewidth=2.0,
                        marker='o', markersize=5, markeredgecolor='white',
                        label='pass@k', zorder=3,
                    )
                    handles.append(line); labels.append('pass@k')
                if show_hat:
                    line, = ax.plot(
                        s['k'], s['pass_hat'] * 100,
                        color='#4f46e5', linestyle='-', linewidth=2.0,
                        marker='o', markersize=5, markeredgecolor='white',
                        label='pass^k', zorder=3,
                    )
                    handles.append(line); labels.append('pass^k')
            else:
                raw = [str(g) for g in sub[plot_group].unique()]
                groups = cls._sort_hue_pairs(raw)
                has_fc, _ = detect_encoding_factors(groups)
                if not has_fc:
                    has_fc = any(
                        bool(re.search(r'[·,\s][01](?:\s*$|\s*[·,])', str(g)))
                        or str(g).strip() in ('0', '1', '0.0', '1.0')
                        for g in groups
                    )
                for grp in groups:
                    s = sub[sub[plot_group].astype(str) == str(grp)].sort_values('k')
                    if s.empty:
                        continue
                    color, ls_fc, _ = get_semantic_style(grp)
                    short = (
                        strip_fc_effort_label(grp) if has_fc
                        else cls._bar_tick_label(grp)
                    )
                    is_fc0 = ls_fc in (':', 'dotted')
                    # pass@k always dotted, pass^k always solid (primary curve
                    # encoding). fc is carried by marker fill (hollow/filled).
                    for ycol, ls_curve, do_label in (
                        ('pass_at', ':', show_at),
                        ('pass_hat', '-', show_hat),
                    ):
                        if not do_label or ycol not in s.columns:
                            continue
                        ax.plot(
                            s['k'], s[ycol] * 100,
                            color=color, linestyle=ls_curve, marker=None,
                            linewidth=1.6, zorder=3, alpha=0.95,
                        )
                        if is_fc0 and has_fc:
                            ax.scatter(
                                s['k'], s[ycol] * 100,
                                s=22, facecolors='none', edgecolors=color,
                                marker='o', linewidths=1.15, zorder=4, alpha=0.95,
                            )
                        else:
                            ax.scatter(
                                s['k'], s[ycol] * 100,
                                s=20, color=color, marker='o',
                                edgecolor='white', linewidth=0.6, zorder=4,
                                alpha=0.95,
                            )
                    if short not in seen:
                        seen.add(short)
                        handles.append(Line2D(
                            [0], [0], color=color, linestyle='-', marker='o',
                            markersize=5.5, markeredgecolor='white', label=short,
                        ))
                        labels.append(short)
                if has_fc:
                    for lab, face in (('fc0', 'none'), ('fc1', SLATE)):
                        if lab not in seen:
                            seen.add(lab)
                            handles.append(Line2D(
                                [0], [0], color=SLATE, marker='o',
                                linestyle='None', markersize=6.5,
                                markerfacecolor=face, markeredgecolor=SLATE,
                                markeredgewidth=1.3 if face == 'none' else 0.8,
                                label=lab,
                            ))
                            labels.append(lab)
            ax.set_ylim(0, 105)
            ax.axhline(100, color=SPINE, linewidth=0.55, linestyle=':', zorder=1)
            ks = sorted(sub['k'].unique())
            ax.set_xticks(ks)
            ax.set_xlabel('k = #trials', fontsize=8.5, color=SLATE)
            ax.set_ylabel('Rate (%)', fontsize=8.5, color=SLATE)
            ax._pass_legend = (handles, labels)
            ax._pass_has_fc = has_fc

        # ----- faceted -----
        if facet_col and facet_col in curves.columns:
            levels = list(curves[facet_col].dropna().unique())
            # prefer order from categorical
            try:
                ordered = apply_categorical_orders(curves[[facet_col]])
                levels = list(ordered[facet_col].cat.categories) if hasattr(ordered[facet_col], 'cat') else levels
                levels = [lv for lv in levels if lv in set(curves[facet_col].unique())]
            except Exception:
                pass
            n = len(levels)
            if figsize is None:
                figsize = (min(4.2 * n, 14.0), 5.6)
            fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi or DEFAULT_DPI, sharey=True)
            if n == 1:
                axes = [axes]
            for ax, lv in zip(axes, levels):
                sub = curves[curves[facet_col] == lv]
                _draw(ax, sub)
                ax.set_title(str(lv), fontsize=10, fontweight=700, color=INK, pad=6)
                if ax is not axes[0]:
                    ax.set_ylabel('')
            fig.suptitle(title, fontsize=13, fontweight=700, color=INK, y=1.02)
            if subtitle:
                fig.text(0.5, 0.94, subtitle, ha='center', fontsize=9, color=MUTED, style='italic')

            # single compact legend: one entry per series color (style key in footer)
            seen, h2, l2 = set(), [], []
            for ax in axes:
                handles, labels = getattr(ax, '_pass_legend', ([], []))
                for h, l in zip(handles, labels):
                    if l not in seen:
                        seen.add(l); h2.append(h); l2.append(l)
            if h2:
                ncol = min(len(h2), 10)
                fig.legend(
                    h2, l2, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                    ncol=ncol, frameon=False, fontsize=6.5, handlelength=2.0,
                )
                fig.tight_layout(rect=[0, 0.10, 1, 0.94])
            else:
                fig.tight_layout(rect=[0, 0.02, 1, 0.94])
            notes = ["dotted = pass@k (≥1 of k)  ·  solid = pass^k (all k)"]
            if any(getattr(ax, '_pass_has_fc', False) for ax in axes):
                notes.append("hollow = fc0 · filled = fc1")
            cls._add_footer(fig, "  ·  ".join(notes))
            return fig, axes, curves

        # ----- single axis -----
        if figsize is None:
            figsize = (8.0, 5.0)
        fig, ax = cls._new_figure(figsize, dpi=dpi)
        _draw(ax, curves)
        cls._style_title_labels(
            ax, title, None, None, subtitle,
            title_pad=(12 if subtitle else 8), subtitle_y=1.004,
        )
        handles, labels = getattr(ax, '_pass_legend', ([], []))
        n = len(handles)
        if n >= 6:
            ax.legend(
                handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14),
                ncol=min(n, 8), frameon=False, fontsize=6.5, handlelength=2.0,
            )
            fig.tight_layout(rect=[0, 0.12, 1, 1])
        elif n:
            ax.legend(
                handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                frameon=False, fontsize=7.0, handlelength=2.0,
            )
            fig.tight_layout(rect=[0, 0.02, 0.80, 1])
        else:
            fig.tight_layout()
        notes = ["dotted = pass@k (≥1 of k)  ·  solid = pass^k (all k)"]
        if getattr(ax, '_pass_has_fc', False):
            notes.append("hollow = fc0 · filled = fc1")
        cls._add_footer(fig, "  ·  ".join(notes))
        return fig, ax, curves

    # ---------------------------------------------------------------
    # token-bin progression facets (rows × reasoning-effort cols)
    # ---------------------------------------------------------------
    @staticmethod
    def _fmt_tok(v: float) -> str:
        """Human-friendly token count in binary units (context-window style):
        256, 512, 1k (=1024), 2k, 4k … 128k, 1M."""
        v = float(v)
        if v >= 1024 * 1024:
            return f"{v / (1024 * 1024):g}M"
        if v >= 1024:
            return f"{v / 1024:g}k"
        return f"{v:g}"

    @classmethod
    def plot_metric_by_token_bins(
        cls,
        data: Union[pd.DataFrame, 'EvalPivotResult'],
        value_col: str = 'correct',
        token_col: str = 'total_token_count',
        reasoning_col: str = 'reasoning_effort',
        group_col: Optional[Union[str, List[str]]] = 'backend',
        row_col: Optional[str] = None,           # e.g. 'test_name' → grid of rows
        row_order: Optional[List[str]] = None,   # optional explicit row order
        bin_method: str = 'doubling',            # 'doubling' (×2) | 'quantile'
        bin_base: float = 256.0,
        bin_cap_quantile: float = 0.98,
        n_bins: int = 5,                         # only for bin_method='quantile'
        x_scale: str = 'log',                    # 'log' (log2 axis) | 'categorical'
        scale_markers: bool = True,              # marker area ~ #runs in bin
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Accuracy by Token Length & Reasoning Effort",
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Accuracy (%)",
        show_values: Optional[bool] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None,
    ):
        """
        Metric progression over token-length bins, faceted as a grid:
        columns = reasoning levels (low / medium / high),
        rows    = optional row_col (e.g. test_name); single row when None.
        One line per backend/factor per panel, stable identity styling.

        Binning: geometric ×2 edges from bin_base; values above the cap
        quantile fall into a final "≥X" overflow bin. Bins are computed
        globally (across all rows/facets) so every panel shares the same
        bin slots and the same X scale — directly comparable across tests.

        X axis: x_scale='log' (default, doubling bins only) → log2 axis,
        ticks at bin edges formatted like plain numbers (256, 512, 1k, 2k…),
        points at bin geometric centers. 'categorical' → one slot per bin
        with range labels.

        Honesty chrome: marker area ~ #runs in bin (a backend whose runs
        all collapse into one bin shows as one BIG dot, not a mystery);
        missing token values treated as 0 and disclosed; ±1 std bands
        clipped to [0, 100] for percent metrics.

        Returns (fig, axes): axes is a 2D list [row][col] when row_col is
        set, else a flat list of the 3 reasoning axes.
        """
        from matplotlib.lines import Line2D

        # ---- sanitize group_col against actual columns ----------------
        raw_cols = (
            data.filtered_df.columns if isinstance(data, EvalPivotResult)
            else data.columns
        )
        if isinstance(group_col, str):
            group_col = [group_col]
        if group_col:
            group_col = [c for c in group_col if c in raw_cols] or None
        multi_group = bool(group_col and len(group_col) > 1)

        df_proc, seed_col, final_group = cls._prepare_data(
            data, filter_query, group_col, seed_col, value_col=value_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold, min_turns=min_turns, pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _is_consistency = '_consistency_mode' in df_proc.columns
        _footer_n_label = 'units' if _is_consistency else 'seeds'
        _is_pct = is_percent_metric(value_col, ylabel)

        # ---- resolve / compute the token column -----------------------
        if token_col == 'total_token_count' and 'total_token_count' not in df_proc.columns:
            inp = (pd.to_numeric(df_proc['input_token_count'], errors='coerce').fillna(0)
                   if 'input_token_count' in df_proc.columns else 0)
            out = (pd.to_numeric(df_proc['output_token_count'], errors='coerce').fillna(0)
                   if 'output_token_count' in df_proc.columns else 0)
            df_proc['total_token_count'] = inp + out
        if token_col not in df_proc.columns:
            raise KeyError(f"token_col {token_col!r} not in data.")
        df_proc[token_col] = pd.to_numeric(df_proc[token_col], errors='coerce')
        n_filled = int(df_proc[token_col].isna().sum())
        df_proc[token_col] = df_proc[token_col].fillna(0)   # disclose, don't drop
        df_proc = df_proc.dropna(subset=[reasoning_col])

        # ---- build bins (global, so all panels share the same slots) --
        centers_by_label: Dict[str, float] = {}
        edge_vals: List[float] = []
        if bin_method == 'doubling':
            s = df_proc[token_col]
            cap = (float(s.quantile(bin_cap_quantile))
                   if bin_cap_quantile and bin_cap_quantile < 1 else float(s.max()))
            cap = max(cap, bin_base * 2)
            edges = [float(bin_base)]
            while edges[-1] < cap:
                edges.append(edges[-1] * 2.0)
            edge_vals = list(edges)
            cut_edges = [-np.inf] + edges + [np.inf]
            raw = pd.cut(df_proc[token_col], bins=cut_edges)
            labs = []
            for l, r in zip(cut_edges[:-1], cut_edges[1:]):
                if np.isneginf(l):
                    lab, cen = f"<={cls._fmt_tok(r)}", r / 2.0
                elif np.isposinf(r):
                    lab, cen = f">={cls._fmt_tok(l)}", l * 2.0
                else:
                    lab, cen = (f"{cls._fmt_tok(l)}-{cls._fmt_tok(r)}",
                                float(np.sqrt(l * r)))
                labs.append(lab)
                centers_by_label[lab] = cen
            df_proc['token_bin'] = raw.cat.rename_categories(
                dict(zip(raw.cat.categories, labs))
            )
        else:
            try:
                df_proc['token_bin'] = pd.qcut(
                    df_proc[token_col], q=n_bins, duplicates='drop', precision=0)
            except ValueError:
                df_proc['token_bin'] = pd.cut(df_proc[token_col], bins=n_bins, precision=0)
            labs = [
                f"{int(iv.left)}-{int(iv.right)}" if hasattr(iv, 'left') else str(iv)
                for iv in df_proc['token_bin'].cat.categories
            ]
            df_proc['token_bin'] = df_proc['token_bin'].cat.rename_categories(
                dict(zip(df_proc['token_bin'].cat.categories, labs))
            )

        use_log = (bin_method == 'doubling') and (x_scale == 'log')

        # Trim leading/trailing globally-empty bins; keep the span aligned.
        cats = list(df_proc['token_bin'].cat.categories)
        obs_idx = [i for i, c in enumerate(cats)
                   if c in set(df_proc['token_bin'].dropna().unique())]
        kept = cats[min(obs_idx):max(obs_idx) + 1]

        # ---- two-stage seed aggregation --------------------------------
        group_keys = [reasoning_col, 'token_bin', seed_col]
        if final_group:
            group_keys.append(final_group)
        if row_col:
            group_keys.append(row_col)
        seed_scores = (
            df_proc.groupby(group_keys, observed=True)[value_col]
            .mean().reset_index()
        )
        across_keys = ([reasoning_col, 'token_bin']
                       + ([final_group] if final_group else [])
                       + ([row_col] if row_col else []))
        summary = (
            seed_scores.groupby(across_keys, observed=True)[value_col]
            .agg(['mean', 'std']).reset_index()
        )
        summary = cls._fix_consistency_std(summary, df_proc, value_col, across_keys)

        # runs-per-bin counts → marker-area encoding
        counts = (
            df_proc.groupby(across_keys, observed=True).size()
            .rename('_n').reset_index()
        )
        n_max = float(counts['_n'].max()) if len(counts) else 1.0
        n_max = n_max if n_max > 0 else 1.0

        # ---- facet levels: cols (reasoning), series, rows --------------
        unique_levels = summary[reasoning_col].unique()
        target_levels = []
        for target in ('low', 'medium', 'high'):
            for lvl in unique_levels:
                if str(lvl).lower() == target or target in str(lvl).lower():
                    if lvl not in target_levels:
                        target_levels.append(lvl)
                    break
        if not target_levels:
            target_levels = sorted(unique_levels, key=str)[:3]
        if not target_levels:
            raise ValueError(f"No reasoning levels found in '{reasoning_col}'.")

        if final_group:
            grp_levels = cls._sort_hue_pairs(
                cls._ordered_levels(df_proc, final_group, summary[final_group])
            )
            has_fc_bins, _ = detect_encoding_factors(grp_levels)
            if not has_fc_bins:
                has_fc_bins = any(
                    bool(re.search(r'[·,\s][01](?:\s*$|\s*[·,])', str(g)))
                    or str(g).strip() in ('0', '1', '0.0', '1.0')
                    for g in grp_levels
                )
            label_fn = (
                strip_fc_effort_label if (multi_group or has_fc_bins)
                else shorten_label
            )
        else:
            grp_levels = [None]
            has_fc_bins = False
            label_fn = lambda g: None  # noqa: E731

        if row_col:
            if row_col not in df_proc.columns:
                raise KeyError(f"row_col {row_col!r} not in data.")
            if row_order:
                present = set(df_proc[row_col].dropna().unique())
                row_levels = [v for v in row_order if v in present]
            elif (isinstance(df_proc[row_col].dtype, CategoricalDtype)
                    and row_col in DEFAULT_CATEGORY_ORDERS):
                row_levels = cls._ordered_levels(df_proc, row_col, df_proc[row_col])
            else:
                row_levels = sorted(df_proc[row_col].dropna().unique(), key=str)
            if not row_levels:
                raise ValueError(f"No row levels found in '{row_col}'.")
        else:
            row_levels = [None]

        # ---- figure grid ------------------------------------------------
        annotate = (show_values if show_values is not None else not final_group)
        n_plots = len(target_levels)
        n_rows = len(row_levels)
        # Compact panel height: scale with row count so dense grids stay tight
        # and 1–2 row figures still have room for title/subtitle.
        if figsize is None:
            row_h = 2.6
            chrome = 1.1
            figsize = (4.35 * n_plots, row_h * max(n_rows, 1) + chrome)
        hspace = 0.28 if n_rows > 1 else 0.15
        hspace = 0.28 if n_rows > 1 else 0.15
        fig, axes2d = plt.subplots(
            n_rows, n_plots, figsize=figsize, dpi=dpi or DEFAULT_DPI,
            sharey=True, sharex=True, squeeze=False,
            gridspec_kw={'hspace': hspace, 'wspace': 0.11},
        )

        reasoning_colors = {'low': '#64748b', 'medium': '#3b82f6', 'high': '#8b5cf6'}
        cat_pos = np.arange(len(kept))
        log_pos = np.array([centers_by_label.get(l, 1.0) for l in kept])
        max_top = 0.0
        proxies = []

        if use_log:
            tick_vals = list(edge_vals)
            # Prefer every edge when few bins; thin to every other when dense.
            labeled = tick_vals[::2] if len(tick_vals) > 8 else tick_vals
            tick_labs = [cls._fmt_tok(t) for t in labeled]

        # Show x tick labels on every row of tall grids so panels are readable
        # without scrolling to the bottom (sharex alone would hide them).
        show_xticks_all_rows = n_rows > 1

        for r, rlv in enumerate(row_levels):
            for c, lvl in enumerate(target_levels):
                ax = axes2d[r][c]
                cls._apply_paper_style(ax, grid_axis='y')
                if use_log:
                    ax.set_xscale('log', base=2)
                    ax.set_xticks(labeled)
                    ax.set_xticklabels(
                        tick_labs, rotation=0, fontsize=7.0, color=SLATE,
                    )
                    ax.minorticks_off()
                    ax.grid(
                        True, axis='x', linestyle=':', linewidth=0.5,
                        alpha=0.55, color=GRID, zorder=0,
                    )
                for grp in grp_levels:
                    base_s = summary[reasoning_col] == lvl
                    base_c = counts[reasoning_col] == lvl
                    if row_col:
                        base_s &= summary[row_col] == rlv
                        base_c &= counts[row_col] == rlv
                    if final_group:
                        base_s &= summary[final_group] == grp
                        base_c &= counts[final_group] == grp
                    sub = summary[base_s]
                    csub = counts[base_c]
                    if sub.empty:
                        continue
                    s_mean = sub.set_index('token_bin').reindex(kept)['mean']
                    s_std = (
                        sub.set_index('token_bin').reindex(kept)['std']
                        .fillna(0.0)
                    )
                    s_n = csub.set_index('token_bin').reindex(kept)['_n'].fillna(0)
                    x_pos = log_pos if use_log else cat_pos

                    if final_group:
                        # Color = backend; linestyle = fc (dotted=fc0, solid=fc1).
                        # Marker fill: hollow=fc0, filled=fc1. Effort is the
                        # column facet so markers stay circular.
                        color, ls, _ = get_semantic_style(grp)
                        s_low = str(grp).lower()
                        if not has_fc_bins:
                            toks = [
                                t for t in re.split(r'[·\s,=()]+', s_low) if t
                            ]
                            if 'chat' in toks or 'ch' in toks:
                                ls = ':'
                            elif 'responses' in toks or 're' in toks:
                                ls = '-'
                        label = label_fn(grp)
                        band_alpha = 0.08
                        is_fc0 = ls in (':', 'dotted')
                        marker = 'o'
                    else:
                        color = '#4f46e5'
                        for key, cc in reasoning_colors.items():
                            if key in str(lvl).lower():
                                color = cc
                                break
                        ls, marker, label, band_alpha = '-', 'o', None, 0.12
                        is_fc0 = False

                    # Break the line across empty bins (NaN) rather than
                    # interpolating through missing data.
                    y = s_mean.to_numpy(dtype=float)
                    ax.plot(
                        x_pos, y, color=color, linestyle=ls,
                        linewidth=1.7, zorder=3, marker=None,
                    )
                    lo_band = (s_mean - s_std).clip(lower=0).to_numpy(dtype=float)
                    hi_band = (s_mean + s_std).to_numpy(dtype=float)
                    if _is_pct:
                        hi_band = np.clip(hi_band, None, 100.0)
                    valid_mask = np.isfinite(y)
                    if valid_mask.any():
                        ax.fill_between(
                            x_pos, lo_band, hi_band,
                            where=valid_mask, interpolate=False,
                            color=color, alpha=band_alpha, zorder=2,
                        )
                    if scale_markers:
                        sizes = 14.0 + 70.0 * (s_n[valid_mask].to_numpy() / n_max)
                    else:
                        sizes = np.full(int(valid_mask.sum()), 30.0)
                    if valid_mask.any():
                        if is_fc0 and has_fc_bins:
                            ax.scatter(
                                x_pos[valid_mask], y[valid_mask], s=sizes,
                                facecolors='none', edgecolors=color,
                                marker=marker, linewidths=1.2, zorder=4,
                                alpha=0.95,
                            )
                        else:
                            ax.scatter(
                                x_pos[valid_mask], y[valid_mask], s=sizes,
                                color=color, marker=marker, edgecolor='white',
                                linewidth=0.9, zorder=4, alpha=0.95,
                            )
                    if label is not None:
                        proxies.append(Line2D(
                            [0], [0], color=color, linestyle=ls, marker=marker,
                            markersize=5.5, markeredgecolor='white', label=label,
                        ))
                    if annotate:
                        for i, m in enumerate(y):
                            if np.isfinite(m):
                                ax.annotate(
                                    fmt_metric(float(m)), (x_pos[i], m),
                                    textcoords='offset points', xytext=(0, 5),
                                    ha='center', va='bottom',
                                    fontsize=6.5, fontweight='bold',
                                    color=INK, zorder=5,
                                )
                    if valid_mask.any():
                        top = float(np.nanmax(hi_band[valid_mask]))
                        max_top = max(max_top, top)

                # ---- per-panel chrome -----------------------------------
                if r == 0:
                    ax.set_title(
                        str(lvl).title(), fontsize=10.5,
                        fontweight='bold', color=INK, pad=6,
                    )
                if not use_log:
                    rot = 30 if len(kept) > 6 else 0
                    ax.set_xticks(cat_pos)
                    ax.set_xticklabels(
                        kept, rotation=rot,
                        ha=('right' if rot else 'center'),
                        fontsize=7.0, color=SLATE,
                    )

                # X tick labels: always on bottom row; also on every row of
                # multi-row grids so mid-panel scales are readable.
                if r == n_rows - 1 or show_xticks_all_rows:
                    ax.tick_params(axis='x', labelbottom=True)
                else:
                    ax.tick_params(axis='x', labelbottom=False)

                if r == n_rows - 1:
                    ax.set_xlabel(
                        xlabel if xlabel else (
                            f"{token_col.replace('_', ' ').title()} (tokens, log2)"
                            if use_log
                            else f"{token_col.replace('_', ' ').title()} bins"
                        ),
                        fontsize=8.0, color=SLATE, labelpad=4,
                    )

                # Y label only on the left column of each row (shared scale).
                if c == 0:
                    ax.set_ylabel(
                        ylabel, fontsize=8.5, fontweight=700,
                        color=SLATE, labelpad=4,
                    )

        # ---- shared axes across the whole grid --------------------------
        flat_axes = [ax for row in axes2d for ax in row]
        if use_log and len(log_pos):
            lo = float(np.nanmin(log_pos)) / 1.55
            hi = float(np.nanmax(log_pos)) * 1.55
            for ax in flat_axes:
                ax.set_xlim(lo, hi)
        if _is_pct:
            y_top = percent_axis_limit(max_top, headroom=12)
            for ax in flat_axes:
                ax.set_ylim(0, y_top)
                ax.axhline(
                    100, color=SPINE, linewidth=0.55, linestyle=':', zorder=1,
                )
        else:
            for ax in flat_axes:
                ax.set_ylim(0, max(max_top * 1.12, 1.0))

        # Title + optional subtitle in a reserved top band (no overlap with
        # column headers "Low / Medium / High"). Positions are in figure
        # fraction; keep a clear gap between title and subtitle.
        has_subtitle = bool(subtitle)
        fig.suptitle(
            title, fontsize=12.5, fontweight='bold', color=INK, y=0.995,
        )
        if has_subtitle:
            if n_rows >= 5:
                sub_y = 0.982
            elif n_rows > 1:
                sub_y = 0.955
            else:
                sub_y = 0.925
            fig.text(
                0.5, sub_y, subtitle, ha='center', fontsize=7.5,
                color=FAINT, style='italic',
            )

        # ---- merged figure legend (deduped proxies) ----------------------
        has_legend = bool(final_group)
        n_legend = 0
        if has_legend and proxies:
            seen, h2 = set(), []
            for p in proxies:
                if p.get_label() not in seen:
                    seen.add(p.get_label())
                    h2.append(p)
            if has_fc_bins:
                for lab, face in (('fc0', 'none'), ('fc1', SLATE)):
                    if lab not in seen:
                        seen.add(lab)
                        h2.append(Line2D(
                            [0], [0], color=SLATE, marker='o', linestyle='None',
                            markersize=6.5, markerfacecolor=face,
                            markeredgecolor=SLATE,
                            markeredgewidth=1.3 if face == 'none' else 0.8,
                            label=lab,
                        ))
            n_legend = len(h2)
            ncol = min(n_legend, 10)
            fig.legend(
                h2, [p.get_label() for p in h2],
                loc='upper center', bbox_to_anchor=(0.5, 0.0),
                ncol=ncol, frameon=False, fontsize=6.5,
                handlelength=2.0, columnspacing=1.1, handletextpad=0.4,
            )

        # Footer: prefer fc encoding note when present (matches line charts).
        notes = [f"shaded band = ±1 std across n={n_seeds} {_footer_n_label}"]
        if scale_markers:
            notes.append("marker area ~ #runs in bin")
        if n_filled:
            notes.append(f"{n_filled} rows w/ missing {token_col} -> 0")
        has_wire_factor = False
        if final_group and not has_fc_bins:
            _, has_wire_factor = detect_encoding_factors(grp_levels)
        if has_fc_bins:
            notes.append("dotted/hollow = fc0 · solid/filled = fc1")
        elif has_wire_factor:
            notes.append("dotted = chat / solid = responses")
        cls._add_footer(fig, "  ·  ".join(notes))

        # Top band must clear suptitle (+ subtitle) and the per-column titles.
        # Few-row figures need a larger *fraction* reserved at the top because
        # each panel is tall; many-row figures use a smaller fraction.
        grid_mode = bool(row_col) and n_rows > 1
        has_leg = bool(has_legend and n_legend)
        if n_rows == 1:
            bottom = 0.16 if has_leg else 0.10
            top = 0.86 if has_subtitle else 0.91
        elif n_rows == 2:
            bottom = 0.10 if has_leg else 0.05
            top = 0.90 if has_subtitle else 0.93
        elif n_rows <= 4:
            bottom = 0.06 if has_leg else 0.03
            top = 0.93 if has_subtitle else 0.95
        else:
            bottom = 0.04 if has_leg else 0.02
            top = 0.96 if has_subtitle else 0.97
        if has_leg and n_legend > 8:
            bottom = max(bottom, 0.07 if n_rows >= 3 else bottom)
        left = 0.055 if grid_mode else 0.06
        fig.subplots_adjust(
            left=left, right=0.99, top=top, bottom=bottom,
            hspace=hspace if n_rows > 1 else 0.15,
            wspace=0.11,
        )
        # Row headers (e.g. test names) in the reserved left margin.
        # Position after subplots_adjust so get_position() is final.
        if grid_mode:
            for r, rlv in enumerate(row_levels):
                pos = axes2d[r][0].get_position()
                fig.text(
                    0.014, (pos.y0 + pos.y1) / 2, str(rlv),
                    rotation=90, ha='center', va='center',
                    fontsize=8.5, fontweight=700, color=INK,
                )

        return fig, (axes2d if row_col else list(axes2d[0])), summary


    # ---------------------------------------------------------------
    # reasoning-effort story: effect → cost → tradeoff
    # ---------------------------------------------------------------
    @classmethod
    def _effort_summary_table(
        cls,
        df_proc: pd.DataFrame,
        value_col: str,
        token_col: str,
        effort_col: str,
        seed_col: str,
        group_col: Optional[str] = None,
        facet_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Per (facet × group × effort): mean accuracy, std, median tokens, n.

        Two-stage seed aggregation for accuracy; tokens summarized as the
        median of per-seed means so a few runaway generations don't dominate.
        """
        keys = [effort_col]
        if group_col:
            keys.append(group_col)
        if facet_col:
            keys.append(facet_col)

        seed_keys = keys + [seed_col]
        seed_acc = (
            df_proc.groupby(seed_keys, observed=True)[value_col]
            .mean()
            .rename('_acc')
            .reset_index()
        )
        seed_tok = (
            df_proc.groupby(seed_keys, observed=True)[token_col]
            .mean()
            .rename('_tok')
            .reset_index()
        )
        seed = seed_acc.merge(seed_tok, on=seed_keys)

        summary = (
            seed.groupby(keys, observed=True)
            .agg(
                accuracy=('_acc', 'mean'),
                acc_std=('_acc', 'std'),
                tokens=('_tok', 'median'),
                tokens_mean=('_tok', 'mean'),
                n_seeds=(seed_col, 'nunique'),
            )
            .reset_index()
        )
        return summary

    @classmethod
    def plot_efficiency_scatter(
        cls,
        data: Union[pd.DataFrame, 'EvalPivotResult'],
        value_col: str = 'correct',
        token_col: str = 'output_token_count',
        effort_col: str = 'reasoning_effort',
        group_col: Optional[Union[str, List[str]]] = 'backend',
        facet_col: Optional[str] = None,
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Efficiency: accuracy vs median tokens",
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Accuracy (%)",
        log_x: bool = True,
        show_labels: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None,
    ):
        """Scatter of mean accuracy vs median tokens (one point per config).

        Each point is a (group × effort [× facet]) aggregate — not a run
        conditioned on length — so the plot is a true cost/quality tradeoff
        without the selection bias of token-bin curves.

        Encoding
        --------
        • color  = backend family (stable identity palette)
        • marker = reasoning effort  (low ○ · medium ◇ · high ■)
        • linestyle N/A (markers only); optional text labels
        """
        from matplotlib.lines import Line2D

        raw_cols = (
            data.filtered_df.columns if isinstance(data, EvalPivotResult)
            else data.columns
        )
        if isinstance(group_col, str):
            group_col = [group_col]
        if group_col:
            group_col = [c for c in group_col if c in raw_cols] or None
        multi_group = bool(group_col and len(group_col) > 1)

        df_proc, seed_col, final_group = cls._prepare_data(
            data, filter_query, group_col, seed_col, value_col=value_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold, min_turns=min_turns, pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _footer_n_label = (
            'units' if '_consistency_mode' in df_proc.columns else 'seeds'
        )
        _is_pct = is_percent_metric(value_col, ylabel)

        if token_col not in df_proc.columns:
            if token_col == 'total_token_count':
                inp = (
                    pd.to_numeric(df_proc['input_token_count'], errors='coerce').fillna(0)
                    if 'input_token_count' in df_proc.columns else 0
                )
                out = (
                    pd.to_numeric(df_proc['output_token_count'], errors='coerce').fillna(0)
                    if 'output_token_count' in df_proc.columns else 0
                )
                df_proc = df_proc.copy()
                df_proc['total_token_count'] = inp + out
            else:
                raise KeyError(f"token_col {token_col!r} not in data.")
        df_proc[token_col] = pd.to_numeric(df_proc[token_col], errors='coerce').fillna(0)

        summary = cls._effort_summary_table(
            df_proc, value_col, token_col, effort_col, seed_col,
            group_col=final_group, facet_col=facet_col,
        )

        effort_order = ['none', 'low', 'medium', 'high']

        def _effort_key(v):
            s = str(v).lower()
            for i, e in enumerate(effort_order):
                if e in s:
                    return i
            return 50

        # ---- figure -----------------------------------------------------
        if facet_col and facet_col in summary.columns:
            facets = cls._ordered_levels(df_proc, facet_col, summary[facet_col])
            n = len(facets)
            if figsize is None:
                figsize = (min(4.0 * n, 14.0), 4.6)
            fig, axes = plt.subplots(
                1, n, figsize=figsize, dpi=dpi or DEFAULT_DPI,
                sharey=True, squeeze=False,
            )
            axes = list(axes[0])
        else:
            facets = [None]
            if figsize is None:
                figsize = (7.2, 5.0)
            fig, ax = cls._new_figure(figsize, dpi=dpi)
            axes = [ax]

        # Strip fc from backend legend when fc is a factor
        _groups_for_detect = (
            list(summary[final_group].unique()) if final_group else []
        )
        has_fc_eff, _ = detect_encoding_factors(_groups_for_detect)
        label_fn = (
            strip_fc_effort_label if (multi_group or has_fc_eff)
            else shorten_label
        )
        max_top = 0.0
        proxies_backend = {}
        proxies_effort = {}

        for ax, fac in zip(axes, facets):
            cls._apply_paper_style(ax, grid_axis='y')
            sub = summary if fac is None else summary[summary[facet_col] == fac]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                grp = row[final_group] if final_group else None
                effort = row[effort_col]
                color, ls, _ = (
                    get_semantic_style(grp) if grp is not None
                    else ('#4f46e5', '-', 'o')
                )
                mk = get_effort_marker(effort)
                is_fc0 = ls in (':', 'dotted')
                x = float(row['tokens'])
                y = float(row['accuracy'])
                if not np.isfinite(x) or not np.isfinite(y) or x <= 0:
                    continue
                if is_fc0 and has_fc_eff:
                    ax.scatter(
                        x, y, s=70, facecolors='none', edgecolors=color,
                        marker=mk, linewidths=1.3, zorder=4, alpha=0.95,
                    )
                else:
                    ax.scatter(
                        x, y, s=70, color=color, marker=mk,
                        edgecolor='white', linewidth=0.9, zorder=4, alpha=0.92,
                    )
                yerr = float(row['acc_std']) if pd.notna(row['acc_std']) else 0.0
                if yerr > 0:
                    ax.errorbar(
                        x, y, yerr=yerr, fmt='none', ecolor=color,
                        elinewidth=0.9, capsize=2.0, alpha=0.45, zorder=3,
                    )
                max_top = max(max_top, y + yerr)
                if show_labels and grp is not None:
                    ax.annotate(
                        f"{label_fn(grp)}·{str(effort)[:2]}",
                        (x, y), textcoords='offset points', xytext=(4, 4),
                        fontsize=6.0, color=SLATE, alpha=0.85,
                    )
                leg_key = label_fn(grp) if grp is not None else None
                if leg_key is not None and leg_key not in proxies_backend:
                    proxies_backend[leg_key] = Line2D(
                        [0], [0], color=color, marker='o', linestyle='None',
                        markersize=7, markeredgecolor='white',
                        label=leg_key,
                    )
                ek = str(effort).lower()
                if ek not in proxies_effort:
                    proxies_effort[ek] = Line2D(
                        [0], [0], color=SLATE, marker=mk, linestyle='None',
                        markersize=7, markeredgecolor='white',
                        label=str(effort),
                    )

            if log_x:
                ax.set_xscale('log', base=2)
                ax.minorticks_off()
            ax.set_xlabel(
                xlabel or f"Median {token_col.replace('_', ' ')}",
                fontsize=8.5, color=SLATE,
            )
            if fac is not None:
                ax.set_title(str(fac), fontsize=10, fontweight=700, color=INK, pad=6)

        for ax in axes:
            if _is_pct:
                ax.set_ylim(0, percent_axis_limit(max_top, headroom=10))
                ax.axhline(100, color=SPINE, linewidth=0.55, linestyle=':', zorder=1)
            else:
                ax.set_ylim(0, max(max_top * 1.12, 1.0))
        axes[0].set_ylabel(ylabel, fontsize=9, fontweight=700, color=SLATE)

        fig.suptitle(title, fontsize=13, fontweight='bold', color=INK, y=1.01)
        if subtitle:
            fig.text(
                0.5, 0.965, subtitle, ha='center', fontsize=8.0,
                color=FAINT, style='italic',
            )

        # Legend: backends (color) + effort shapes + fc hollow/filled
        handles = list(proxies_backend.values())
        effort_handles = []
        for e in effort_order:
            for k, h in proxies_effort.items():
                if e in k:
                    effort_handles.append(h)
                    break
        fc_handles = []
        if has_fc_eff:
            fc_handles = [
                Line2D(
                    [0], [0], color=SLATE, marker='o', linestyle='None',
                    markersize=7, markerfacecolor='none',
                    markeredgecolor=SLATE, markeredgewidth=1.4, label='fc0',
                ),
                Line2D(
                    [0], [0], color=SLATE, marker='o', linestyle='None',
                    markersize=7, markerfacecolor=SLATE,
                    markeredgecolor='white', label='fc1',
                ),
            ]
        all_h = handles + effort_handles + fc_handles
        if all_h:
            fig.legend(
                all_h, [h.get_label() for h in all_h],
                loc='upper center', bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(all_h), 10), frameon=False, fontsize=6.5,
                handlelength=1.6,
            )

        notes = [
            f"one point = config aggregate  ·  ±1 std across n={n_seeds} {_footer_n_label}",
            "markers = effort (○ low · ◇ med · □ high)",
        ]
        if has_fc_eff:
            notes.append("hollow = fc0 · filled = fc1")
        cls._add_footer(fig, "  ·  ".join(notes))
        fig.subplots_adjust(
            left=0.08, right=0.98,
            top=0.88 if subtitle else 0.92,
            bottom=0.16 if all_h else 0.08,
            wspace=0.12,
        )
        return fig, (axes if facet_col else axes[0]), summary

    @classmethod
    def plot_effort_story(
        cls,
        data: Union[pd.DataFrame, 'EvalPivotResult'],
        value_col: str = 'correct',
        token_col: str = 'output_token_count',
        effort_col: str = 'reasoning_effort',
        group_col: Optional[Union[str, List[str]]] = 'backend',
        facet_col: Optional[str] = None,
        facet_order: Optional[List[str]] = None,
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Reasoning effort: effect → cost → tradeoff",
        subtitle: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None,
        log_tokens: bool = True,
        human_token_ticks: bool = True,
        show_pareto: bool = True,
        show_ideal_region: bool = True,
    ):
        """Three-column figure matching o1/R1-style reporting.

        Columns
        -------
        Effect   — accuracy vs reasoning effort (lines + ±1σ band)
        Cost     — median tokens vs reasoning effort
        Tradeoff — accuracy vs median tokens (efficiency scatter)

        Optional ``facet_col`` (e.g. ``'test_name'``) stacks one row per level.

        Tradeoff annotations
        --------------------
        • **Pareto front** (``show_pareto``): configs not dominated on both
          higher accuracy and lower tokens, connected as a step curve.
        • **Ideal region** (``show_ideal_region``): top-left quadrant relative
          to the median point (high accuracy, low tokens) — the magic-quadrant
          "stars" zone.
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyBboxPatch
        import matplotlib.patheffects as pe

        raw_cols = (
            data.filtered_df.columns if isinstance(data, EvalPivotResult)
            else data.columns
        )
        if isinstance(group_col, str):
            group_col = [group_col]
        if group_col:
            group_col = [c for c in group_col if c in raw_cols] or None
        multi_group = bool(group_col and len(group_col) > 1)

        df_proc, seed_col, final_group = cls._prepare_data(
            data, filter_query, group_col, seed_col, value_col=value_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold, min_turns=min_turns, pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _footer_n_label = (
            'units' if '_consistency_mode' in df_proc.columns else 'seeds'
        )
        _is_pct = is_percent_metric(value_col, 'Accuracy')

        if token_col not in df_proc.columns:
            if token_col == 'total_token_count':
                inp = (
                    pd.to_numeric(df_proc['input_token_count'], errors='coerce').fillna(0)
                    if 'input_token_count' in df_proc.columns
                    else pd.Series(0, index=df_proc.index)
                )
                out = (
                    pd.to_numeric(df_proc['output_token_count'], errors='coerce').fillna(0)
                    if 'output_token_count' in df_proc.columns
                    else pd.Series(0, index=df_proc.index)
                )
                df_proc = df_proc.copy()
                df_proc['total_token_count'] = inp + out
            else:
                raise KeyError(f"token_col {token_col!r} not in data.")
        df_proc[token_col] = pd.to_numeric(df_proc[token_col], errors='coerce').fillna(0)
        df_proc = df_proc.dropna(subset=[effort_col])

        summary = cls._effort_summary_table(
            df_proc, value_col, token_col, effort_col, seed_col,
            group_col=final_group, facet_col=facet_col,
        )

        effort_rank = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}

        def _sort_efforts(vals):
            return sorted(vals, key=lambda v: effort_rank.get(str(v).lower(), 50))

        efforts = _sort_efforts(summary[effort_col].unique())
        effort_pos = {e: i for i, e in enumerate(efforts)}
        # Use module-level EFFORT_MARKERS via get_effort_marker

        if final_group:
            groups = cls._sort_hue_pairs(
                cls._ordered_levels(df_proc, final_group, summary[final_group])
            )
            # Always strip fc/effort from backend legend entries — they get
            # their own legend rows (linestyle / hollow-fill + shape).
            has_fc, _ = detect_encoding_factors(groups)
            label_fn = (
                strip_fc_effort_label if (multi_group or has_fc)
                else shorten_label
            )
        else:
            groups = [None]
            label_fn = lambda g: 'all'  # noqa: E731

        if facet_col:
            if facet_col not in df_proc.columns:
                raise KeyError(f"facet_col {facet_col!r} not in data.")
            if facet_order:
                present = set(summary[facet_col].dropna().unique())
                facets = [v for v in facet_order if v in present]
            else:
                facets = cls._ordered_levels(
                    df_proc, facet_col, summary[facet_col]
                )
            if not facets:
                facets = list(summary[facet_col].dropna().unique())
            else:
                facets = sorted(facets)
        else:
            facets = [None]

        n_rows = len(facets)
        if figsize is None:
            if n_rows > 1:
                row_h = 2.15 if n_rows >= 6 else 2.45
                figsize = (13.2, row_h * n_rows + 1.5)
            else:
                figsize = (13.5, 4.6)

        fig, axes2d = plt.subplots(
            n_rows, 3, figsize=figsize, dpi=dpi or DEFAULT_DPI,
            sharex=False, squeeze=False,
            gridspec_kw={
                'wspace': 0.28,
                'hspace': 0.32 if n_rows > 1 else 0.15,
            },
        )

        def _pareto_front(points: list) -> list:
            """Non-dominated set: maximize accuracy, minimize tokens.

            Returns points sorted by ascending tokens (left→right on plot).
            """
            pts = [
                (t, a) for t, a in points
                if np.isfinite(t) and np.isfinite(a) and t > 0
            ]
            if not pts:
                return []
            # sort by tokens asc, then accuracy desc
            pts = sorted(pts, key=lambda p: (p[0], -p[1]))
            front = []
            best_acc = -np.inf
            for t, a in pts:
                if a > best_acc:
                    front.append((t, a))
                    best_acc = a
            return front

        proxies = []
        effort_proxies_done = set()
        effort_proxies = []
        global_max_acc = 0.0

        for r, fac in enumerate(facets):
            ax_eff, ax_cost, ax_trade = axes2d[r]
            if fac is None:
                row_sum = summary
            else:
                row_sum = summary[summary[facet_col] == fac]
            if row_sum.empty:
                continue

            max_acc = 0.0
            max_tok = 0.0
            scatter_pts = []  # (tokens, accuracy) for Pareto

            for grp in groups:
                sub = (
                    row_sum if final_group is None
                    else row_sum[row_sum[final_group] == grp]
                )
                if sub.empty:
                    continue
                sub = sub.copy()
                sub['_x'] = sub[effort_col].map(effort_pos)
                sub = sub.dropna(subset=['_x']).sort_values('_x')

                # color = backend; linestyle = fc (dotted=fc0, solid=fc1)
                # same contract as get_semantic_style / plot_line_scaling.
                color, ls, _sem_marker = (
                    get_semantic_style(grp) if grp is not None
                    else ('#4f46e5', '-', 'o')
                )
                short = label_fn(grp) if grp is not None else 'all'
                # Effect/Cost: effort is the x-axis, so keep a single marker
                # shape and let linestyle carry fc — avoids clashing with the
                # effort-marker legend used on the scatter.
                line_marker = 'o'

                xs = sub['_x'].to_numpy()
                acc = sub['accuracy'].to_numpy(dtype=float)
                acc_std = sub['acc_std'].fillna(0).to_numpy(dtype=float)
                toks = sub['tokens'].to_numpy(dtype=float)

                ax_eff.plot(
                    xs, acc, color=color, linestyle=ls, linewidth=1.8,
                    marker=line_marker, markersize=5.5, markeredgecolor='white',
                    markeredgewidth=0.75, zorder=3, label=short,
                )
                ax_eff.fill_between(
                    xs, np.clip(acc - acc_std, 0, None), acc + acc_std,
                    color=color, alpha=0.11, zorder=2,
                )
                max_acc = max(max_acc, float(np.nanmax(acc + acc_std)))

                ax_cost.plot(
                    xs, toks, color=color, linestyle=ls, linewidth=1.8,
                    marker=line_marker, markersize=5.5, markeredgecolor='white',
                    markeredgewidth=0.75, zorder=3,
                )
                if len(toks) and np.isfinite(toks).any():
                    max_tok = max(max_tok, float(np.nanmax(toks)))

                # fc0 = hollow marker; fc1 = filled (effort still = shape).
                is_fc0 = (ls in (':', 'dotted')) or (
                    grp is not None and bool(
                        re.search(
                            r'fc[_=]?model\s*=\s*0|\bfc\s*=\s*0|\bfc0\b',
                            str(grp).lower(),
                        )
                    )
                )
                for _, row in sub.iterrows():
                    ek = str(row[effort_col]).lower()
                    mk = get_effort_marker(ek)
                    t, a = float(row['tokens']), float(row['accuracy'])
                    if not (np.isfinite(t) and np.isfinite(a) and t > 0):
                        continue
                    scatter_pts.append((t, a))
                    if is_fc0:
                        ax_trade.scatter(
                            t, a, s=38, facecolors='none', edgecolors=color,
                            marker=mk, linewidths=1.2, zorder=4, alpha=0.95,
                        )
                    else:
                        ax_trade.scatter(
                            t, a, s=42, color=color, marker=mk,
                            edgecolor='white', linewidth=0.85, zorder=4,
                            alpha=0.92,
                        )
                    yerr = (
                        float(row['acc_std']) if pd.notna(row['acc_std']) else 0.0
                    )
                    if yerr > 0:
                        ax_trade.errorbar(
                            t, a, yerr=yerr, fmt='none', ecolor=color,
                            elinewidth=0.75, capsize=1.8, alpha=0.4, zorder=3,
                        )

                if short not in {p.get_label() for p in proxies}:
                    proxies.append(Line2D(
                        [0], [0], color=color, linestyle=ls, marker=line_marker,
                        markersize=6, markeredgecolor='white', label=short,
                    ))

            for e in efforts:
                ek = str(e).lower()
                if ek in effort_proxies_done:
                    continue
                mk = get_effort_marker(ek)
                effort_proxies.append(Line2D(
                    [0], [0], color=SLATE, marker=mk, linestyle='None',
                    markersize=7, markerfacecolor=SLATE,
                    markeredgecolor='white', markeredgewidth=0.8,
                    label=f'{e}',
                ))
                effort_proxies_done.add(ek)

            # ---- tradeoff: ideal region + Pareto front ------------------
            if scatter_pts:
                toks_arr = np.array([p[0] for p in scatter_pts])
                acc_arr = np.array([p[1] for p in scatter_pts])
                med_t = float(np.median(toks_arr))
                med_a = float(np.median(acc_arr))
                t_lo, t_hi = float(toks_arr.min()), float(toks_arr.max())
                a_lo = 0.0
                a_hi = (
                    percent_axis_limit(float(acc_arr.max()), headroom=8)
                    if _is_pct else float(acc_arr.max()) * 1.12
                )
                # lock axes before drawing region so log-scale fill is correct
                ax_trade.set_ylim(0, a_hi)
                if log_tokens:
                    ax_trade.set_xlim(t_lo / 1.5, t_hi * 1.5)
                else:
                    ax_trade.set_xlim(max(0, t_lo * 0.9), t_hi * 1.08)

                if show_ideal_region:
                    # Top-left of median cross = high accuracy, low tokens
                    # (classic magic-quadrant "stars" region).
                    if log_tokens:
                        x0 = max(t_lo / 1.4, med_t / 80.0)
                    else:
                        x0 = max(0.0, t_lo * 0.85)
                    ax_trade.fill_between(
                        [x0, med_t], [med_a, med_a], [a_hi, a_hi],
                        color='#22c55e', alpha=0.08, zorder=0,
                        linewidth=0,
                    )
                    ax_trade.axhline(
                        med_a, color='#86efac', linewidth=0.7,
                        linestyle='--', alpha=0.7, zorder=1,
                    )
                    ax_trade.axvline(
                        med_t, color='#86efac', linewidth=0.7,
                        linestyle='--', alpha=0.7, zorder=1,
                    )
                    ax_trade.text(
                        0.03, 0.97, 'ideal',
                        transform=ax_trade.transAxes,
                        ha='left', va='top', fontsize=7.0, fontweight=700,
                        color='#15803d', alpha=0.85, style='italic',
                        zorder=5,
                    )

                if show_pareto:
                    front = _pareto_front(scatter_pts)
                    if len(front) >= 1:
                        fx = [p[0] for p in front]
                        fy = [p[1] for p in front]
                        if len(front) >= 2:
                            # Soft guide through the front (no step jaggies).
                            ax_trade.plot(
                                fx, fy, color='#15803d',
                                linewidth=0.75, linestyle=(0, (3, 2.5)),
                                alpha=0.55, zorder=3, solid_capstyle='round',
                            )
                        # Subtle halo on Pareto points only — keep series
                        # markers/colors intact underneath.
                        ax_trade.scatter(
                            fx, fy, s=160, facecolors='none',
                            edgecolors='#15803d', linewidths=0.75,
                            zorder=5, marker='o', alpha=0.55,
                        )

            # ---- chrome per panel ---------------------------------------
            for ax in (ax_eff, ax_cost, ax_trade):
                cls._apply_paper_style(ax, grid_axis='y')

            if r == 0:
                ax_eff.set_title('Effect', fontsize=11, fontweight=700, color=INK, pad=7)
                ax_cost.set_title('Cost', fontsize=11, fontweight=700, color=INK, pad=7)
                ax_trade.set_title(
                    'Tradeoff', fontsize=11, fontweight=700, color=INK, pad=7,
                )

            ax_eff.set_ylabel('Accuracy (%)', fontsize=8.0, fontweight=700, color=SLATE)
            ax_cost.set_ylabel(
                f"Median {token_col.replace('_', ' ')}",
                fontsize=7.5, fontweight=700, color=SLATE,
            )
            ax_trade.set_ylabel('Accuracy (%)', fontsize=8.0, fontweight=700, color=SLATE)

            for ax in (ax_eff, ax_cost):
                ax.set_xticks(range(len(efforts)))
                ax.set_xticklabels(
                    [str(e).title() for e in efforts],
                    fontsize=7.5, color=SLATE,
                )
                ax.set_xlim(-0.35, len(efforts) - 0.65)

            if r == n_rows - 1:
                ax_eff.set_xlabel('Reasoning effort', fontsize=8.0, color=SLATE)
                ax_cost.set_xlabel('Reasoning effort', fontsize=8.0, color=SLATE)
                ax_trade.set_xlabel(
                    f"Median {token_col.replace('_', ' ')}",
                    fontsize=8.0, color=SLATE,
                )

            if _is_pct:
                y_top = percent_axis_limit(max_acc, headroom=10)
                for ax in (ax_eff, ax_trade):
                    ax.set_ylim(0, y_top)
                    ax.axhline(
                        100, color=SPINE, linewidth=0.5, linestyle=':', zorder=1,
                    )
            else:
                for ax in (ax_eff, ax_trade):
                    ax.set_ylim(0, max(max_acc * 1.12, 1.0))

            if log_tokens:
                ax_cost.set_yscale('log', base=2)
                ax_cost.minorticks_off()
                ax_trade.set_xscale('log', base=2)
                ax_trade.minorticks_off()
            if max_tok > 0:
                ax_cost.set_ylim(bottom=max(1.0, max_tok / 80.0))

            if human_token_ticks:
                def _apply_human_ticks(ax, axis: str):
                    """Replace 2^n power labels with 512 / 1k / 4k / …"""
                    import matplotlib.ticker as mticker
                    if axis == 'y':
                        axis_obj = ax.yaxis
                        lim = ax.get_ylim()
                    else:
                        axis_obj = ax.xaxis
                        lim = ax.get_xlim()
                    lo, hi = lim
                    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= 0:
                        return
                    # Candidate binary ticks spanning the visible range.
                    start = max(1, int(np.floor(np.log2(max(lo, 1)))))
                    stop = int(np.ceil(np.log2(hi))) + 1
                    ticks = [2 ** e for e in range(start, stop + 1)]
                    ticks = [t for t in ticks if lo * 0.9 <= t <= hi * 1.15]
                    if len(ticks) > 7:
                        ticks = ticks[::2]
                    if not ticks:
                        return
                    axis_obj.set_major_locator(mticker.FixedLocator(ticks))
                    axis_obj.set_major_formatter(
                        mticker.FuncFormatter(lambda v, _p: cls._fmt_tok(v))
                    )
                    axis_obj.set_minor_locator(mticker.NullLocator())

                _apply_human_ticks(ax_cost, 'y')
                _apply_human_ticks(ax_trade, 'x')

            global_max_acc = max(global_max_acc, max_acc)

            # row label for faceted grids
            if facet_col and fac is not None:
                pos = ax_eff.get_position()
                # drawn after subplots_adjust below

        # ---- figure-level chrome ----------------------------------------
        has_subtitle = bool(subtitle)
        fig.suptitle(title, fontsize=13, fontweight='bold', color=INK, y=0.995)
        if has_subtitle:
            # Keep a clear gap under the title (figure fraction).
            sub_y = 0.980 if n_rows > 1 else 0.935
            fig.text(
                0.5, sub_y, subtitle, ha='center', fontsize=7.5,
                color=FAINT, style='italic',
            )

        # Explicit scatter fc encoding (hollow vs filled).
        fc_proxies = [
            Line2D(
                [0], [0], color=SLATE, marker='o', linestyle='None',
                markersize=7.5, markerfacecolor='none',
                markeredgecolor=SLATE, markeredgewidth=1.5,
                label='fc0',
            ),
            Line2D(
                [0], [0], color=SLATE, marker='o', linestyle='None',
                markersize=7.5, markerfacecolor=SLATE,
                markeredgecolor='white', markeredgewidth=0.8,
                label='fc1',
            ),
        ]
        all_h = proxies + effort_proxies + fc_proxies
        if show_pareto:
            all_h.append(Line2D(
                [0], [0], color='#15803d', linewidth=0.75,
                linestyle=(0, (3, 2.5)),
                alpha=0.5,
                marker='o', markersize=5.5, markerfacecolor='none',
                markeredgecolor='#15803d', markeredgewidth=0.75,
                label='Pareto front',
            ))
        if all_h:
            fig.legend(
                all_h, [h.get_label() for h in all_h],
                loc='upper center', bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(all_h), 16), frameon=False, fontsize=6.4,
                handlelength=1.7, columnspacing=1.0,
            )

        notes = [
            f"effect/cost: mean ±1 std across n={n_seeds} {_footer_n_label}",
            # "lines: dotted = fc0 / solid = fc1",
            # "scatter: hollow = fc0 / filled = fc1 · shape = effort",
        ]
        # if show_ideal_region:
        #     notes.append("green band = above-median accuracy & below-median tokens")
        # if show_pareto:
        #     notes.append("Pareto = not dominated on accuracy↑ tokens↓")
        cls._add_footer(fig, "  ·  ".join(notes))

        bottom = 0.10 if n_rows > 1 else 0.18
        if n_rows >= 6:
            bottom = 0.07
        # Single-row figures need a larger top fraction so title/subtitle
        # don't collide with panel titles.
        if n_rows == 1:
            top = 0.78 if has_subtitle else 0.86
        elif n_rows <= 3:
            top = 0.90 if has_subtitle else 0.93
        else:
            top = 0.95 if has_subtitle else 0.97
        left = 0.07 if facet_col else 0.06
        fig.subplots_adjust(
            left=left, right=0.99, top=top, bottom=bottom,
            wspace=0.30, hspace=0.34 if n_rows > 1 else 0.18,
        )

        if facet_col:
            for r, fac in enumerate(facets):
                pos = axes2d[r][0].get_position()
                fig.text(
                    0.012, (pos.y0 + pos.y1) / 2, str(fac),
                    rotation=90, ha='center', va='center',
                    fontsize=8.5, fontweight=700, color=INK,
                )

        return fig, axes2d, summary


    @classmethod
    def plot_compute_scaling(
        cls,
        data: Union[pd.DataFrame, 'EvalPivotResult'],
        value_col: str = 'correct',
        token_col: Optional[str] = None,
        effort_col: str = 'reasoning_effort',
        group_col: Optional[Union[str, List[str]]] = 'backend',
        facet_col: Optional[str] = None,
        facet_order: Optional[List[str]] = None,
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Test-time compute scaling",
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Accuracy (%)",
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None,
        log_x: bool = True,
        human_token_ticks: bool = True,
        show_values: bool = False,
        connect_efforts: bool = True,
        max_cols: int = 4,
        markersize: float = 20.0,
        linewidth: float = 1.8,
    ):
        """OpenAI-style test-time compute curves.

        One point per (series × reasoning effort):
          x = median CoT+answer tokens used at that effort
          y = mean accuracy
        Points for the same series are connected low → medium → high so the
        path is the intentional budget schedule (not observational length bins).

        Token column
        ------------
        Default resolves ``reasoning_token_count + response_token_count``
        (CoT + answer). Falls back to ``output_token_count`` when reasoning
        tokens are missing/zero.

        Layout
        ------
        ``facet_col`` panels wrap into a grid with at most ``max_cols`` columns
        (default 4) so many tests stay readable.
        """
        from matplotlib.lines import Line2D
        import matplotlib.ticker as mticker

        raw_cols = (
            data.filtered_df.columns if isinstance(data, EvalPivotResult)
            else data.columns
        )
        if isinstance(group_col, str):
            group_col = [group_col]
        if group_col:
            group_col = [c for c in group_col if c in raw_cols] or None
        multi_group = bool(group_col and len(group_col) > 1)

        df_proc, seed_col, final_group = cls._prepare_data(
            data, filter_query, group_col, seed_col, value_col=value_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold, min_turns=min_turns, pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _footer_n_label = (
            'units' if '_consistency_mode' in df_proc.columns else 'seeds'
        )
        _is_pct = is_percent_metric(value_col, ylabel)
        df_proc = df_proc.dropna(subset=[effort_col]).copy()

        # ---- resolve CoT + answer token column --------------------------
        tok_label = 'CoT + answer tokens'
        if token_col is not None:
            if token_col not in df_proc.columns:
                raise KeyError(f"token_col {token_col!r} not in data.")
            df_proc['_cot_answer'] = pd.to_numeric(
                df_proc[token_col], errors='coerce'
            ).fillna(0)
            tok_label = token_col.replace('_', ' ')
        else:
            has_reason = 'reasoning_token_count' in df_proc.columns
            has_resp = 'response_token_count' in df_proc.columns
            has_out = 'output_token_count' in df_proc.columns
            reason = (
                pd.to_numeric(df_proc['reasoning_token_count'], errors='coerce')
                .fillna(0)
                if has_reason else pd.Series(0.0, index=df_proc.index)
            )
            resp = (
                pd.to_numeric(df_proc['response_token_count'], errors='coerce')
                .fillna(0)
                if has_resp else pd.Series(0.0, index=df_proc.index)
            )
            cot = reason + resp
            if has_out:
                out = pd.to_numeric(
                    df_proc['output_token_count'], errors='coerce'
                ).fillna(0)
                cot = cot.where(cot > 0, out)
            df_proc['_cot_answer'] = cot

        summary = cls._effort_summary_table(
            df_proc, value_col, '_cot_answer', effort_col, seed_col,
            group_col=final_group, facet_col=facet_col,
        )

        effort_rank = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}

        def _sort_efforts(vals):
            return sorted(
                vals, key=lambda v: effort_rank.get(str(v).lower(), 50)
            )

        efforts = _sort_efforts(summary[effort_col].unique())

        if final_group:
            groups = cls._sort_hue_pairs(
                cls._ordered_levels(df_proc, final_group, summary[final_group])
            )
            has_fc_cs, _ = detect_encoding_factors(groups)
            label_fn = (
                strip_fc_effort_label if (multi_group or has_fc_cs)
                else shorten_label
            )
        else:
            groups = [None]
            has_fc_cs = False
            label_fn = lambda g: 'all'  # noqa: E731

        if facet_col:
            if facet_col not in df_proc.columns:
                raise KeyError(f"facet_col {facet_col!r} not in data.")
            if facet_order:
                present = set(summary[facet_col].dropna().unique())
                facets = [v for v in facet_order if v in present]
            else:
                facets = cls._ordered_levels(
                    df_proc, facet_col, summary[facet_col]
                )
                facets = sorted(facets)
            if not facets:
                facets = list(summary[facet_col].dropna().unique())

        else:
            facets = [None]

        n_facets = len(facets)
        max_cols = max(1, int(max_cols))
        n_cols = min(n_facets, max_cols) if n_facets > 1 else 1
        n_rows = int(np.ceil(n_facets / n_cols)) if n_facets > 1 else 1

        if figsize is None:
            if n_facets > 1:
                figsize = (3.9 * n_cols, 3.35 * n_rows + 1.1)
            else:
                figsize = (7.6, 5.0)

        fig, axes2d = plt.subplots(
            n_rows, n_cols, figsize=figsize, dpi=dpi or DEFAULT_DPI,
            sharey=True, squeeze=False,
        )
        axes_flat = [ax for row in axes2d for ax in row]
        # hide unused cells
        for k in range(n_facets, len(axes_flat)):
            axes_flat[k].set_visible(False)

        proxies = []
        global_max_acc = 0.0
        # per-panel token ranges so each facet keeps a sensible x scale
        panel_xlims = []

        def _human_xticks(ax):
            lo, hi = ax.get_xlim()
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= 0:
                return
            start = max(0, int(np.floor(np.log2(max(lo, 1)))))
            stop = int(np.ceil(np.log2(hi))) + 1
            ticks = [2 ** e for e in range(start, stop + 1)]
            ticks = [t for t in ticks if lo * 0.85 <= t <= hi * 1.2]
            if len(ticks) > 7:
                ticks = ticks[::2]
            if not ticks:
                return
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _p: cls._fmt_tok(v))
            )
            ax.xaxis.set_minor_locator(mticker.NullLocator())

        ms = float(markersize)
        lw = float(linewidth)

        for idx, fac in enumerate(facets):
            ax = axes_flat[idx]
            cls._apply_paper_style(ax, grid_axis='y')
            row = summary if fac is None else summary[summary[facet_col] == fac]
            t_lo, t_hi = np.inf, 0.0
            max_acc = 0.0

            if not row.empty:
                for grp in groups:
                    sub = (
                        row if final_group is None
                        else row[row[final_group] == grp]
                    )
                    if sub.empty:
                        continue
                    sub = sub.copy()
                    sub['_erank'] = sub[effort_col].map(
                        lambda v: effort_rank.get(str(v).lower(), 50)
                    )
                    sub = sub.sort_values('_erank')

                    color, ls, _mk = (
                        get_semantic_style(grp) if grp is not None
                        else ('#4f46e5', '-', 'o')
                    )
                    short = label_fn(grp) if grp is not None else 'all'

                    xs, ys, yerrs, mks = [], [], [], []
                    for _, r in sub.iterrows():
                        t = float(r['tokens'])
                        a = float(r['accuracy'])
                        if not (np.isfinite(t) and np.isfinite(a) and t > 0):
                            continue
                        xs.append(t)
                        ys.append(a)
                        ye = (
                            float(r['acc_std']) if pd.notna(r['acc_std']) else 0.0
                        )
                        yerrs.append(ye)
                        ek = str(r[effort_col]).lower()
                        mks.append(get_effort_marker(ek))
                        t_lo = min(t_lo, t)
                        t_hi = max(t_hi, t)
                        max_acc = max(max_acc, a + ye)

                    if not xs:
                        continue

                    if connect_efforts and len(xs) >= 2:
                        ax.plot(
                            xs, ys, color=color, linestyle=ls, linewidth=lw,
                            zorder=3, alpha=0.95, solid_capstyle='round',
                        )
                    for x, y, ye, mk in zip(xs, ys, yerrs, mks):
                        is_fc0 = ls in (':', 'dotted')
                        if is_fc0:
                            ax.scatter(
                                x, y, s=ms, facecolors='none', edgecolors=color,
                                marker=mk, linewidths=1.35, zorder=5, alpha=0.95,
                            )
                        else:
                            ax.scatter(
                                x, y, s=ms * 0.92, color=color, marker=mk,
                                edgecolor='white', linewidth=0.7, zorder=5,
                                alpha=0.95,
                            )
                        if ye > 0:
                            ax.errorbar(
                                x, y, yerr=ye, fmt='none', ecolor=color,
                                elinewidth=0.7, capsize=1.6, alpha=0.35, zorder=4,
                            )
                        if show_values:
                            ax.annotate(
                                fmt_metric(y), (x, y),
                                textcoords='offset points', xytext=(0, 6),
                                ha='center', fontsize=6.0, fontweight='bold',
                                color=INK, zorder=6,
                            )

                    if short not in {p.get_label() for p in proxies}:
                        proxies.append(Line2D(
                            [0], [0], color=color, linestyle=ls, marker='o',
                            markersize=5, markeredgecolor='white', label=short,
                        ))

            global_max_acc = max(global_max_acc, max_acc)
            panel_xlims.append((t_lo, t_hi))

            if log_x:
                ax.set_xscale('log', base=2)
                ax.minorticks_off()
            if fac is not None:
                ax.set_title(
                    str(fac), fontsize=10, fontweight=700, color=INK, pad=6,
                )
            # x labels only on bottom row of used panels
            row_i = idx // n_cols
            if row_i == n_rows - 1 or idx + n_cols >= n_facets:
                ax.set_xlabel(
                    xlabel or f"Median {tok_label}",
                    fontsize=7.5, color=SLATE, labelpad=3,
                )
            else:
                ax.set_xlabel('')

            if np.isfinite(t_lo) and t_hi > 0:
                if log_x:
                    ax.set_xlim(t_lo / 1.35, t_hi * 1.35)
                else:
                    ax.set_xlim(t_lo * 0.9, t_hi * 1.08)
            if human_token_ticks and log_x:
                _human_xticks(ax)

        # shared y limits
        for i in range(n_facets):
            ax = axes_flat[i]
            if _is_pct:
                y_top = percent_axis_limit(global_max_acc, headroom=8)
                ax.set_ylim(0, y_top)
                ax.axhline(
                    100, color=SPINE, linewidth=0.5, linestyle=':', zorder=1,
                )
            else:
                ax.set_ylim(0, max(global_max_acc * 1.12, 1.0))

        axes_flat[0].set_ylabel(
            ylabel, fontsize=9, fontweight=700, color=SLATE,
        )
        # only left column keeps ylabel text; clear others to reduce clutter
        for i in range(1, n_facets):
            if (i % n_cols) != 0:
                axes_flat[i].set_ylabel('')

        has_subtitle = bool(subtitle)
        fig.suptitle(title, fontsize=13, fontweight='bold', color=INK, y=0.995)
        if has_subtitle:
            # Sit clearly under the title, above panel headers.
            sub_y = 0.965 if n_rows > 1 else 0.935
            fig.text(
                0.5, sub_y, subtitle, ha='center', fontsize=7.5,
                color=FAINT, style='italic',
            )

        effort_h = []
        for e in efforts:
            ek = str(e).lower()
            mk = get_effort_marker(ek)
            effort_h.append(Line2D(
                [0], [0], color=SLATE, marker=mk, linestyle='None',
                markersize=6, markerfacecolor=SLATE,
                markeredgecolor='white', label=str(e),
            ))
        fc_h = [
            Line2D(
                [0], [0], color=SLATE, marker='o', linestyle='None',
                markersize=6.5, markerfacecolor='none',
                markeredgecolor=SLATE, markeredgewidth=1.4, label='fc0',
            ),
            Line2D(
                [0], [0], color=SLATE, marker='o', linestyle='None',
                markersize=6.5, markerfacecolor=SLATE,
                markeredgecolor='white', label='fc1',
            ),
        ]
        all_h = list(proxies) + effort_h
        if has_fc_cs:
            all_h = all_h + fc_h

        if all_h:
            ncol_leg = min(len(all_h), 12 if n_facets > 1 else 16)
            fig.legend(
                all_h, [h.get_label() for h in all_h],
                loc='upper center', bbox_to_anchor=(0.5, 0.0),
                ncol=ncol_leg, frameon=False, fontsize=6.2,
                handlelength=1.7, columnspacing=0.9, handletextpad=0.35,
            )

        cls._add_footer(
            fig,
            f"each point = effort level aggregate across n={n_seeds} "
            f"{_footer_n_label}  ·  x = median CoT+answer tokens  ·  "
            f"path = low → high effort  ·  "
            f"lines: dotted = fc0 / solid = fc1",
        )

        if n_facets == 1:
            top = 0.80 if has_subtitle else 0.88
            bottom = 0.16 if all_h else 0.08
            left = 0.09
        else:
            top = 0.85 if has_subtitle else 0.94
            if n_rows >= 3:
                top = 0.93 if has_subtitle else 0.96
            bottom = 0.10 if all_h else 0.05
            left = 0.07
        fig.subplots_adjust(
            left=left, right=0.99, top=top, bottom=bottom,
            wspace=0.18 if n_facets > 1 else 0.12,
            hspace=0.38 if n_rows > 1 else 0.15,
        )
        return fig, (axes2d if n_facets > 1 else axes_flat[0]), summary


    @classmethod
    def plot_effort_at_matched_tokens(
        cls,
        data: Union[pd.DataFrame, 'EvalPivotResult'],
        value_col: str = 'correct',
        token_col: str = 'reasoning_token_count',
        effort_col: str = 'reasoning_effort',
        group_col: Optional[Union[str, List[str]]] = 'backend',
        facet_col: Optional[str] = None,
        facet_order: Optional[List[str]] = None,
        filter_query: Optional[str] = None,
        seed_col: Optional[str] = None,
        title: str = "Accuracy at matched reasoning length, by effort",
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Accuracy (%)",
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        consistency: Optional[str] = None,
        unit_col: Optional[str] = None,
        pass_threshold: float = 1.0,
        min_turns: Optional[int] = None,
        pass_k: Optional[int] = None,
        binning: str = 'doubling',
        min_bin_n: int = 8,
        log_x: bool = True,
        human_token_ticks: bool = True,
        max_cols: int = 2,
        within_question: bool = True,
        question_col: str = 'test_id',
        markersize: float = 30.0,
        linewidth: float = 1.5,
        show_errorbars: bool = False,
        errorbar_cap_pct: bool = True,
    ):
        """Falsify 'effort only buys more tokens'.

        Hypothesis (H1)
        ---------------
        Conditional on roughly the same reasoning-token budget, mean accuracy
        still differs by ``reasoning_effort``.

        Encoding
        --------
        - If ``group_col`` includes backend (and optionally fc): suite colors +
          linestyle/fill for fc; marker shape = effort.
        - If ``group_col`` is None / empty: pool all runs; color by effort
          (low=slate-green, medium=amber, high=coral) with solid lines.
        - If fc is not in the grouping, lines and markers are always solid/filled.

        ``within_question=True`` averages per ``test_id`` first so each question
        contributes equally inside a token bin.
        """
        from matplotlib.lines import Line2D
        import matplotlib.ticker as mticker

        # Modern effort palette when not splitting by backend
        EFFORT_COLORS = {
            'low': '#0d9488',      # teal
            'medium': '#d97706',   # amber
            'high': '#e11d48',     # rose
            'none': '#64748b',
        }
        effort_rank = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}

        raw_cols = (
            data.filtered_df.columns if isinstance(data, EvalPivotResult)
            else data.columns
        )
        if isinstance(group_col, str):
            group_col = [group_col]
        if group_col:
            group_col = [c for c in group_col if c in raw_cols] or None
        multi_group = bool(group_col and len(group_col) > 1)
        group_has_fc = bool(
            group_col and any(
                c in ('fc_model', 'fc') or str(c).startswith('fc')
                for c in group_col
            )
        )
        group_has_backend = bool(
            group_col and any('backend' in str(c).lower() for c in group_col)
        )
        # Pool-by-effort mode when no backend split
        pool_by_effort = not group_has_backend

        df_proc, seed_col, final_group = cls._prepare_data(
            data, filter_query, group_col, seed_col, value_col=value_col,
            consistency=consistency, unit_col=unit_col,
            pass_threshold=pass_threshold, min_turns=min_turns, pass_k=pass_k,
        )
        n_seeds = df_proc[seed_col].nunique()
        _footer_n_label = (
            'units' if '_consistency_mode' in df_proc.columns else 'seeds'
        )
        _is_pct = is_percent_metric(value_col, ylabel)

        if token_col not in df_proc.columns:
            raise KeyError(f"token_col {token_col!r} not in data.")
        if effort_col not in df_proc.columns:
            raise KeyError(f"effort_col {effort_col!r} not in data.")

        use_within = bool(within_question and question_col in df_proc.columns)
        if within_question and not use_within:
            import warnings
            warnings.warn(
                f"within_question=True but {question_col!r} missing; "
                "falling back to pooled bins.",
                stacklevel=2,
            )

        df_proc = df_proc.dropna(subset=[effort_col]).copy()
        df_proc['_tok'] = pd.to_numeric(df_proc[token_col], errors='coerce')
        df_proc = df_proc[df_proc['_tok'] > 0].copy()
        if df_proc.empty:
            raise ValueError(f"No rows with {token_col} > 0 after filtering.")

        toks_all = df_proc['_tok'].to_numpy(dtype=float)
        t_min = max(1.0, float(np.nanmin(toks_all)))
        t_max = float(np.nanmax(toks_all))
        if binning == 'doubling':
            lo_exp = int(np.floor(np.log2(t_min)))
            hi_exp = int(np.ceil(np.log2(max(t_max, t_min * 2))))
            edges = sorted(set(2 ** e for e in range(lo_exp, hi_exp + 2)))
            df_proc['_bin'] = pd.cut(
                df_proc['_tok'], bins=edges, right=False, include_lowest=True,
            )
        else:
            edges = df_proc['_tok'].quantile(np.linspace(0, 1, 9)).unique()
            if len(edges) < 3:
                raise ValueError("Not enough unique token values for bins.")
            df_proc['_bin'] = pd.cut(
                df_proc['_tok'], bins=edges, include_lowest=True,
            )

        def _bin_center(cat):
            if pd.isna(cat):
                return np.nan
            try:
                return float(np.sqrt(cat.left * max(cat.right, cat.left * 1.01)))
            except Exception:
                return np.nan

        df_proc['_bin_center'] = df_proc['_bin'].map(_bin_center)

        efforts = sorted(
            df_proc[effort_col].dropna().unique(),
            key=lambda v: effort_rank.get(str(v).lower(), 50),
        )

        if final_group and not pool_by_effort:
            groups = cls._sort_hue_pairs(
                cls._ordered_levels(df_proc, final_group, df_proc[final_group])
            )
            label_fn = (
                strip_fc_effort_label if (multi_group or group_has_fc)
                else shorten_label
            )
        else:
            groups = [None]
            final_group = None
            label_fn = lambda g: 'all'  # noqa: E731

        if facet_col:
            if facet_col not in df_proc.columns:
                raise KeyError(f"facet_col {facet_col!r} not in data.")
            if facet_order:
                present = set(df_proc[facet_col].dropna().unique())
                facets = [v for v in facet_order if v in present]
            else:
                facets = cls._ordered_levels(
                    df_proc, facet_col, df_proc[facet_col]
                )
            if not facets:
                facets = list(df_proc[facet_col].dropna().unique())
        else:
            facets = [None]

        n_facets = len(facets)
        max_cols = max(1, int(max_cols))
        n_cols = min(n_facets, max_cols) if n_facets > 1 else 1
        n_rows = int(np.ceil(n_facets / n_cols)) if n_facets > 1 else 1

        if figsize is None:
            if n_facets > 1:
                figsize = (4.6 * n_cols, 3.6 * n_rows + 0.9)
            else:
                figsize = (8.0, 5.0)

        fig, axes2d = plt.subplots(
            n_rows, n_cols, figsize=figsize, dpi=dpi or DEFAULT_DPI,
            sharey=True, squeeze=False,
        )
        axes_flat = [ax for row in axes2d for ax in row]
        for k in range(n_facets, len(axes_flat)):
            axes_flat[k].set_visible(False)

        def _human_xticks(ax):
            lo, hi = ax.get_xlim()
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= 0:
                return
            start = max(0, int(np.floor(np.log2(max(lo, 1)))))
            stop = int(np.ceil(np.log2(hi))) + 1
            ticks = [2 ** e for e in range(start, stop + 1)]
            ticks = [t for t in ticks if lo * 0.85 <= t <= hi * 1.2]
            if len(ticks) > 8:
                ticks = ticks[::2]
            if not ticks:
                return
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _p: cls._fmt_tok(v))
            )
            ax.xaxis.set_minor_locator(mticker.NullLocator())

        ms = float(markersize)
        lw = float(linewidth)
        proxies = []
        effort_proxies_done = set()
        effort_proxies = []
        global_max_acc = 0.0
        overlap_notes = []
        matched_parts = []

        for idx, fac in enumerate(facets):
            ax = axes_flat[idx]
            cls._apply_paper_style(ax, grid_axis='y')
            d = df_proc if fac is None else df_proc[df_proc[facet_col] == fac]
            if d.empty:
                continue

            t_lo, t_hi = np.inf, 0.0

            for grp in groups:
                sub = (
                    d if final_group is None
                    else d[d[final_group] == grp]
                )
                if sub.empty:
                    continue

                if pool_by_effort:
                    # Series identity is effort itself — handled in loop below.
                    short = None
                    base_color, base_ls = None, '-'
                    force_solid = True
                else:
                    color, ls, _ = get_semantic_style(grp)
                    # No fc in grouping → never dotted / hollow
                    if not group_has_fc:
                        ls = '-'
                    short = label_fn(grp)
                    base_color, base_ls = color, ls
                    force_solid = not group_has_fc

                # ---- aggregate ------------------------------------------
                if use_within:
                    q_keys = [question_col, '_bin_center', effort_col, seed_col]
                    seed_lvl = (
                        sub.groupby(q_keys, observed=True)[value_col]
                        .mean().reset_index()
                    )
                    q_lvl = (
                        seed_lvl.groupby(
                            [question_col, '_bin_center', effort_col],
                            observed=True,
                        )[value_col]
                        .agg(['mean', 'count'])
                        .reset_index()
                        .rename(columns={'mean': 'q_acc', 'count': 'n_seeds'})
                    )
                    n_runs_q = (
                        sub.groupby(
                            [question_col, '_bin_center', effort_col],
                            observed=True,
                        ).size().rename('n_runs').reset_index()
                    )
                    q_lvl = q_lvl.merge(
                        n_runs_q,
                        on=[question_col, '_bin_center', effort_col],
                        how='left',
                    )
                    q_lvl = q_lvl[q_lvl['n_runs'] >= max(1, min_bin_n // 4)]
                    summary = (
                        q_lvl.groupby(['_bin_center', effort_col], observed=True)
                        .agg(
                            accuracy=('q_acc', 'mean'),
                            acc_std=('q_acc', 'std'),
                            n_questions=(question_col, 'nunique'),
                            n_runs=('n_runs', 'sum'),
                        )
                        .reset_index()
                    )
                    summary = summary[
                        (summary['n_runs'] >= min_bin_n)
                        & (summary['n_questions'] >= 1)
                    ].copy()
                else:
                    gb_keys = ['_bin_center', effort_col, seed_col]
                    seed_means = (
                        sub.groupby(gb_keys, observed=True)[value_col]
                        .mean().reset_index()
                    )
                    summary = (
                        seed_means.groupby(
                            ['_bin_center', effort_col], observed=True
                        )[value_col]
                        .agg(['mean', 'std', 'count'])
                        .reset_index()
                        .rename(columns={
                            'mean': 'accuracy', 'std': 'acc_std',
                            'count': 'n_seeds',
                        })
                    )
                    n_runs = (
                        sub.groupby(
                            ['_bin_center', effort_col], observed=True
                        ).size().rename('n_runs').reset_index()
                    )
                    summary = summary.merge(
                        n_runs, on=['_bin_center', effort_col], how='left'
                    )
                    summary = summary[summary['n_runs'] >= min_bin_n].copy()

                if summary.empty:
                    continue

                if _is_pct and summary['accuracy'].max(skipna=True) <= 1.5:
                    summary['accuracy'] = summary['accuracy'] * 100
                    summary['acc_std'] = summary['acc_std'] * 100

                part = summary.copy()
                if fac is not None:
                    part[facet_col] = fac
                if final_group is not None and grp is not None:
                    part[final_group] = grp
                matched_parts.append(part)

                for e in efforts:
                    esub = summary[summary[effort_col] == e].sort_values(
                        '_bin_center'
                    )
                    if esub.empty:
                        continue
                    ek = str(e).lower()
                    mk = get_effort_marker(ek)

                    if pool_by_effort:
                        color = EFFORT_COLORS.get(ek, '#4f46e5')
                        ls = '-'
                        is_fc0 = False
                    else:
                        color = base_color
                        ls = base_ls
                        is_fc0 = (not force_solid) and (ls in (':', 'dotted'))

                    xs = esub['_bin_center'].to_numpy(dtype=float)
                    ys = esub['accuracy'].to_numpy(dtype=float)
                    yerr = esub['acc_std'].fillna(0).to_numpy(dtype=float)

                    ax.plot(
                        xs, ys, color=color, linestyle=ls, linewidth=lw,
                        zorder=3, alpha=0.92, solid_capstyle='round',
                    )
                    for x, y in zip(xs, ys):
                        if is_fc0:
                            ax.scatter(
                                x, y, s=ms, facecolors='none', edgecolors=color,
                                marker=mk, linewidths=1.35, zorder=5, alpha=0.95,
                            )
                        else:
                            ax.scatter(
                                x, y, s=ms * 0.92, color=color, marker=mk,
                                edgecolor='white', linewidth=0.7, zorder=5,
                                alpha=0.95,
                            )

                    if show_errorbars and np.any(yerr > 0):
                        y_lo = ys - yerr
                        y_hi = ys + yerr
                        if errorbar_cap_pct and _is_pct:
                            y_lo = np.clip(y_lo, 0, 100)
                            y_hi = np.clip(y_hi, 0, 100)
                        ax.vlines(
                            xs, y_lo, y_hi, colors=color, linewidths=0.7,
                            alpha=0.35, zorder=4,
                        )
                        ax.hlines(
                            y_lo, xs - 0, xs + 0, colors=color, linewidths=0.7,
                            alpha=0.35, zorder=4,
                        )
                        # small caps
                        cap = 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if not log_x else None
                        for x, lo, hi in zip(xs, y_lo, y_hi):
                            ax.plot(
                                [x, x], [lo, hi], color=color, linewidth=0.7,
                                alpha=0.35, zorder=4,
                            )

                    t_lo = min(t_lo, float(np.nanmin(xs)))
                    t_hi = max(t_hi, float(np.nanmax(xs)))
                    global_max_acc = max(
                        global_max_acc, float(np.nanmax(ys + yerr))
                    )

                    if pool_by_effort:
                        leg = str(e)
                        if leg not in {p.get_label() for p in proxies}:
                            proxies.append(Line2D(
                                [0], [0], color=color, linestyle='-',
                                marker=mk, markersize=5.5,
                                markeredgecolor='white', label=leg,
                            ))
                    else:
                        if ek not in effort_proxies_done:
                            effort_proxies.append(Line2D(
                                [0], [0], color=SLATE, marker=mk,
                                linestyle='None', markersize=6,
                                markerfacecolor=SLATE, markeredgecolor='white',
                                label=str(e),
                            ))
                            effort_proxies_done.add(ek)

                if not pool_by_effort and short not in {
                    p.get_label() for p in proxies
                }:
                    proxies.append(Line2D(
                        [0], [0], color=base_color, linestyle=base_ls,
                        marker='o', markersize=5.5, markeredgecolor='white',
                        label=short,
                    ))

            tmp = d.copy()
            be = (
                tmp.groupby(['_bin_center', effort_col], observed=True)
                .size().reset_index(name='n')
            )
            be = be[be['n'] >= min_bin_n]
            n_ov = int(
                be.groupby('_bin_center')[effort_col].nunique().ge(2).sum()
            )
            overlap_notes.append(f"{fac or 'all'}: {n_ov} overlapping bins")

            if fac is not None:
                ax.set_title(
                    str(fac), fontsize=11, fontweight=700, color=INK, pad=6,
                )
            if log_x and np.isfinite(t_lo) and t_hi > 0:
                ax.set_xscale('log', base=2)
                ax.minorticks_off()
                ax.set_xlim(t_lo / 1.35, t_hi * 1.35)
            if human_token_ticks and log_x:
                _human_xticks(ax)

            row_i = idx // n_cols
            if row_i == n_rows - 1 or idx + n_cols >= n_facets:
                ax.set_xlabel(
                    xlabel or f"{token_col.replace('_', ' ')} (bin center)",
                    fontsize=8, color=SLATE, labelpad=3,
                )

        for i in range(n_facets):
            ax = axes_flat[i]
            if _is_pct:
                # keep 0–100 even if std would push past
                y_top = min(110.0, percent_axis_limit(
                    min(global_max_acc*1.12, 112.0), headroom=4
                ))
                y_top = max(y_top, 112.0) if global_max_acc >= 95 else y_top
                # Prefer a clean 0–100 frame for accuracy
                ax.set_ylim(0, 112)
                ax.axhline(
                    100, color=SPINE, linewidth=0.5, linestyle=':', zorder=1,
                )
            else:
                ax.set_ylim(0, max(global_max_acc * 1.12, 1.12))

        axes_flat[0].set_ylabel(ylabel, fontsize=9, fontweight=700, color=SLATE)

        # Default subtitle carries H1 so footer stays short
        if subtitle is None:
            subtitle = (
                "At matched reasoning length, higher effort still yields "
                "higher accuracy"
                + ("  ·  within-question" if use_within else "  ·  pooled runs")
            )
        has_subtitle = bool(subtitle)
        fig.suptitle(title, fontsize=13, fontweight='bold', color=INK, y=1.0)
        if has_subtitle:
            fig.text(
                0.5, 0.975 if n_rows > 1 else 0.945, subtitle,
                ha='center', fontsize=7.5, color=FAINT, style='italic',
            )

        fc_h = []
        if group_has_fc and not pool_by_effort:
            fc_h = [
                Line2D(
                    [0], [0], color=SLATE, marker='o', linestyle='None',
                    markersize=6.5, markerfacecolor='none',
                    markeredgecolor=SLATE, markeredgewidth=1.4, label='fc0',
                ),
                Line2D(
                    [0], [0], color=SLATE, marker='o', linestyle='None',
                    markersize=6.5, markerfacecolor=SLATE,
                    markeredgecolor='white', label='fc1',
                ),
            ]

        all_h = list(proxies) + (
            list(effort_proxies) if not pool_by_effort else []
        ) + fc_h
        if all_h:
            fig.legend(
                all_h, [p.get_label() for p in all_h],
                loc='upper center', bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(all_h), 12), frameon=False, fontsize=6.3,
                handlelength=1.7, columnspacing=0.9, handletextpad=0.35,
            )

        mode = (
            f"within-{question_col}" if use_within else "pooled runs"
        )
        ov = '; '.join(overlap_notes[:4]) if overlap_notes else ''
        footer = (
            f"{mode}  ·  bins n≥{min_bin_n}  ·  "
            f"mean across n={n_seeds} {_footer_n_label}"
            + ("  ·  ±1 std" if show_errorbars else "")
            + (f"  ·  {ov}" if ov else "")
        )
        cls._add_footer(fig, footer)

        if n_facets == 1:
            top = 0.78 if has_subtitle else 0.86
            bottom = 0.14 if all_h else 0.08
        else:
            top = 0.86 if has_subtitle else 0.90
            bottom = 0.11 if all_h else 0.06
        fig.subplots_adjust(
            left=0.09, right=0.98, top=top, bottom=bottom,
            wspace=0.16, hspace=0.40 if n_rows > 1 else 0.15,
        )
        plot_data = (
            pd.concat(matched_parts, ignore_index=True) if matched_parts
            else pd.DataFrame()
        )
        return fig, (axes2d if n_facets > 1 else axes_flat[0]), plot_data

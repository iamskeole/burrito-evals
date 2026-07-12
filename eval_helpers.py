from IPython.display import display  # <-- MANDATORY for helper scripts

import colorsys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
from IPython.display import display

DATA_FOLDER = "./data"

def read_dataset(eval_name):
    df = pd.read_csv(f"{DATA_FOLDER}/{eval_name.replace('.csv','')}.csv")
    df["reasoning_effort"] = pd.Categorical(
        df["reasoning_effort"], 
        categories=["low", "medium", "high"], 
        ordered=True
    )
    return df


import colorsys
import pandas as pd
from IPython.display import display, Markdown


def display_pivoted_results(
    df, index_cols, col_cols, metric_col, precision=4, 
    title=None, body=None, caption=None, return_styler=False
):
    """Pivots a flat evaluation DataFrame into a structured academic leaderboard.

    Groups configuration dimensions into nested row headers, places tasks along
    the columns, and bolds the best-performing configuration per column.
    Supports a top title, an explanatory body description, and a bottom caption.
    """
    # 1. Normalize input structures
    index_cols = [index_cols] if isinstance(index_cols, str) else index_cols
    col_cols = [col_cols] if isinstance(col_cols, str) else col_cols

    # Resolve primary metric (mean) and variability (std) columns
    mean_col = metric_col if metric_col in df.columns else f"mean_{metric_col}"
    std_col = f"std_{metric_col}"
    
    if mean_col not in df.columns or std_col not in df.columns:
        raise KeyError(
            f"Could not find both '{mean_col}' and '{std_col}' in the summary DataFrame."
        )

    # 2. Pivot the numeric data under the hood (for exact mathematical comparisons)
    numeric_pivoted_mean = df.pivot(index=index_cols, columns=col_cols, values=mean_col)
    numeric_pivoted_std = df.pivot(index=index_cols, columns=col_cols, values=std_col)

    # 3. Construct a formatted String Pivot Table using HTML-styled 'mean ± std'
    combined_pivoted = pd.DataFrame(
        "", index=numeric_pivoted_mean.index, columns=numeric_pivoted_mean.columns
    )
    
    for col in numeric_pivoted_mean.columns:
        for idx in numeric_pivoted_mean.index:
            m = numeric_pivoted_mean.loc[idx, col]
            s = numeric_pivoted_std.loc[idx, col]
            
            if pd.isna(m):
                combined_pivoted.loc[idx, col] = float("nan")
            else:
                # Rule of Thumb 1 & 2: 
                # - Keep mean at high precision (e.g. 4 decimals)
                # - Round std to exactly 2 decimals (uncertainty estimate)
                # - Mute the std color to #8c8c8d, shrink its font size to 10px, 
                #   and force its weight to normal so it stays soft when the mean is bolded.
                combined_pivoted.loc[idx, col] = (
                    f'{m:.{precision}f}<span style="font-size: 10px; color: #8c8c8d; font-weight: normal;"> ± {s:.2f}</span>'
                )

    # 4. Initialize the Styler on the formatted string DataFrame
    styler = combined_pivoted.style

    # 5. Globally format NaNs to a clean, quiet em-dash
    # Set escape=None to ensure pandas renders our inline <span> HTML tags correctly
    styler = styler.format(na_rep="—", escape=None)

    # 6. Apply Classic Academic 'booktabs' rules with strict contrast overrides
    table_styles = [
        # Main table container (Force white background, black text, monospace font)
        {'selector': '', 'props': [
            ('font-family', '"SF Mono", Consolas, Menlo, "Fira Code", "Courier New", monospace'),
            ('font-size', '12px'),
            ('border-collapse', 'collapse'),
            ('margin', '20px auto'),
            ('background-color', '#ffffff'),
            ('color', '#000000')
        ]},
        # Booktabs Top Rule (Thick top border)
        {'selector': 'thead', 'props': [
            ('border-top', '2.5px solid #000000 !important'),
            ('background-color', '#ffffff')
        ]},
        # Column headers (Align numeric task columns to the right)
        {'selector': 'th.col_heading', 'props': [
            ('border-bottom', '1px solid #000000 !important'),
            ('padding', '8px 16px'),
            ('font-weight', 'bold'),
            ('color', '#000000'),
            ('background-color', '#ffffff'),
            ('text-align', 'right')
        ]},
        # Row headers (Left-align the nested config labels)
        {'selector': 'th.row_heading', 'props': [
            ('text-align', 'left'),
            ('padding', '6px 16px'),
            ('font-weight', 'bold'),
            ('color', '#000000'),
            ('background-color', '#ffffff')
        ]},
        # Top-left corner showing the names of the index dimensions
        {'selector': 'th.index_name', 'props': [
            ('text-align', 'left'),
            ('font-size', '10px'),
            ('font-weight', 'bold'),
            ('color', '#7f8c8d'),
            ('background-color', '#ffffff'),
            ('border-bottom', '1px solid #000000 !important')
        ]},
        # Clean data cells (No gridlines, right-aligned)
        {'selector': 'td', 'props': [
            ('padding', '6px 16px'),
            ('border', 'none !important'),
            ('color', '#111111'),
            ('background-color', '#ffffff'),
            ('text-align', 'right')
        ]},
        # Booktabs Bottom Rule (Thick bottom border)
        {'selector': 'tbody', 'props': [
            ('border-bottom', '2.5px solid #000000 !important'),
            ('background-color', '#ffffff')
        ]},
        # Force table rows to stay white
        {'selector': 'tr', 'props': [
            ('background-color', '#ffffff'),
            ('color', '#000000')
        ]},
        # Bottom-aligned table footnote
        {'selector': 'caption', 'props': [
            ('caption-side', 'bottom !important'),  # Forces caption below the table
            ('font-size', '10px !important'),
            ('color', '#7f8c8d !important'),        # Muted gray color
            ('text-align', 'left !important'),      # Left-align with booktabs margins
            ('margin-top', '12px !important'),      # Space under the bottom rule
            ('font-family', '"SF Mono", Consolas, monospace !important')
        ]}
    ]

    # --- DYNAMIC CATEGORY SEGMENTATION BARS ---
    if isinstance(combined_pivoted.index, pd.MultiIndex):
        num_levels = len(combined_pivoted.index.levels)
        for idx_num in range(1, len(combined_pivoted.index)):
            prev_val = combined_pivoted.index[idx_num - 1]
            curr_val = combined_pivoted.index[idx_num]
            
            diff_level = None
            for level_idx in range(num_levels - 1):
                if prev_val[level_idx] != curr_val[level_idx]:
                    diff_level = level_idx
                    break
            
            if diff_level is not None:
                if diff_level == 0:
                    border_style = '1.5px solid #000000 !important'
                else:
                    border_style = '0.5px solid #8c8c8d !important'
                    
                table_styles.append({
                    'selector': f'.row{idx_num} td, tr.row{idx_num} th',
                    'props': [('border-top', border_style)]
                })

    styler = styler.set_table_styles(table_styles)

    # 7. Under-the-Hood Numeric Bolding (axis=None)
    def highlight_best_numeric(data):
        css_df = pd.DataFrame("", index=data.index, columns=data.columns)

        for col in data.columns:
            if any(word in metric_col for word in ["mean", "correct", "accuracy", "success"]):
                best_val = numeric_pivoted_mean[col].max()
                best_indices = numeric_pivoted_mean[
                    numeric_pivoted_mean[col] == best_val
                ].index
            elif any(word in metric_col for word in ["std", "var", "error", "latency", "cv"]):
                best_val = numeric_pivoted_mean[col].min()
                best_indices = numeric_pivoted_mean[
                    numeric_pivoted_mean[col] == best_val
                ].index

            css_df.loc[best_indices, col] = "font-weight: bold; color: #000000;"

        return css_df

    # Apply the mathematical bolding rule over our formatted string table
    styler = styler.apply(highlight_best_numeric, axis=None)

    # Apply caption if provided
    if caption is not None:
        styler = styler.set_caption(caption)

    # Render directly in the Jupyter Notebook
    if not return_styler:
        if title is not None:
            display(Markdown(f"### {title}"))
        if body is not None:
            display(Markdown(body))
        display(styler)
    else:
        return styler


def compute_group_variance(df, group_cols, variance_col, metric_col="correct", query_filter=None):
    # 1. Apply pre-filtering if a filter string was provided
    if query_filter is not None:
        df = df.query(query_filter)

    # 2. Calculate average of the metric per seed/unit within each group
    intermediate = (
        df.groupby(group_cols + [variance_col])[metric_col].mean().reset_index()
    )

    # 3. Define dynamic column names using the metric_col variable
    agg_dict = {
        f"mean_{metric_col}": (metric_col, "mean"),
        f"variance_{metric_col}": (metric_col, "var"),
        f"std_{metric_col}": (metric_col, "std"),
        f"{variance_col}_count": (metric_col, "count")
    }

    # 4. Compute variance across the variance column with dynamic names
    final_variance = (
        intermediate.groupby(group_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    # 5. Calculate Coefficient of Variation (CV) = Std Dev / Mean
    mean_col = f"mean_{metric_col}"
    std_col = f"std_{metric_col}"
    cv_col = f"cv_{metric_col}"
    
    # Calculate CV (pandas handles division by zero by returning NaN/inf)
    final_variance[cv_col] = final_variance[std_col] / final_variance[mean_col]
    
    # Optional: Reorder columns so CV sits right next to the Std Dev column
    columns_order = list(final_variance.columns)
    columns_order.remove(cv_col)
    std_index = columns_order.index(std_col)
    columns_order.insert(std_index + 1, cv_col)
    
    return final_variance[columns_order]


def plot_computed_results(
    out, group_cols, metric_col, plot_type="bar", x_order=None, figsize=(13, 6), dpi=300
):
    """Plots precomputed summary stats (mean & std) directly from `out`.

    Applies modern, premium editorial styling with custom bold monospace typography.
    """
    # 1. Shape data dynamically
    mean_col = f"mean_{metric_col}"
    std_col = f"std_{metric_col}"

    if len(group_cols) > 1:
        indexed_mean = out.set_index(group_cols)[mean_col]
        indexed_std = out.set_index(group_cols)[std_col]

        # Mode A: If x_order is provided, keep group_cols[0] on x-axis and combine rest into legend
        if x_order is not None:
            unstack_levels = list(range(1, len(group_cols)))
            pivoted_mean = indexed_mean.unstack(level=unstack_levels)
            pivoted_std = indexed_std.unstack(level=unstack_levels)
            pivoted_mean = pivoted_mean.reindex(x_order)
            pivoted_std = pivoted_std.reindex(x_order)
            
        # Mode B: Standard grouped behavior (combine all except last on x-axis)
        else:
            pivoted_mean = indexed_mean.unstack(level=-1)
            pivoted_std = indexed_std.unstack(level=-1)

        # Flatten MultiIndex index if it exists
        if isinstance(pivoted_mean.index, pd.MultiIndex):
            new_index_labels = pivoted_mean.index.map(
                lambda x: " | ".join(map(str, x))
            )
            pivoted_mean.index = new_index_labels
            pivoted_std.index = new_index_labels

        # Flatten MultiIndex columns if they exist
        if isinstance(pivoted_mean.columns, pd.MultiIndex):
            new_column_labels = pivoted_mean.columns.map(
                lambda x: " | ".join(map(str, x))
            )
            pivoted_mean.columns = new_column_labels
            pivoted_std.columns = new_column_labels
    else:
        pivoted_mean = out.set_index(group_cols[0])[mean_col]
        pivoted_std = out.set_index(group_cols[0])[std_col]

        if x_order is not None:
            pivoted_mean = pivoted_mean.reindex(x_order)
            pivoted_std = pivoted_std.reindex(x_order)

    # 2. Setup a minimalist editorial Canvas theme with monospace typography
    sns.set_theme(
        style="white",  # Solid white background
        rc={
            "font.family": "monospace",  # Force the generic monospace family
            # List preferred high-quality monospace fonts (falls back gracefully)
            "font.monospace": [
                "SF Mono",
                "Consolas",
                "Menlo",
                "Fira Code",
                "Courier New",
                "DejaVu Sans Mono"
            ],
            "text.color": "#2c3e50",
        },
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 3. Generate the Custom RPG Rarity Base Gradient
    from matplotlib.colors import LinearSegmentedColormap
    rpg_colors = [
        "#ff8000",  # Legendary Orange
        "#a335ee",  # Epic Purple
        "#0070dd",  # Rare Blue
        "#7f8c8d"   # Common Slate Gray
    ]
    cmap = LinearSegmentedColormap.from_list("rpg_rarity", rpg_colors, N=256)

    num_categories = len(pivoted_mean.columns) if isinstance(pivoted_mean, pd.DataFrame) else 1
    
    # Simple color scale fallback
    if num_categories <= 5:
        line_palette = ["#2b5c8f", "#48c9b0", "#eb984e", "#af7ac5", "#7f8c8d"]
    else:
        line_palette = [cmap(i / (num_categories - 1)) for i in range(num_categories)]

    # 4. Draw and Style the Plots
    if plot_type == "bar":
        # Draw the base bar structure
        pivoted_mean.plot(
            kind="bar",
            yerr=pivoted_std,
            width=0.85,
            ax=ax,
            edgecolor="white",  # Crisp white margins
            linewidth=0.6,
            error_kw=dict(ecolor="#555555", elinewidth=0.8, capsize=0),
        )

        num_groups = len(pivoted_mean.index)
        num_columns = len(pivoted_mean.columns)

        # Safely filter out only the BarContainer objects (excluding ErrorbarContainers)
        from matplotlib.container import BarContainer
        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]

        # Dynamically recolor every individual bar based on its X-axis group
        for col_idx, container in enumerate(bar_containers):
            # Calculate shading factors for this column (Vivid/Dark -> Soft/Lighter)
            if num_columns > 1:
                l_factor = 0.85 + (col_idx / (num_columns - 1)) * 0.45
                s_factor = 1.0 - (col_idx / (num_columns - 1)) * 0.4
            else:
                l_factor = 1.0
                s_factor = 1.0

            for group_idx, patch in enumerate(container.patches):
                # Interpolate base color from the x-axis rarity gradient
                if num_groups > 1:
                    base_color = cmap(group_idx / (num_groups - 1))
                else:
                    base_color = cmap(0.0)

                # Convert to HLS to adjust lightness and saturation
                r, g, b = base_color[:3]
                h, l, s = colorsys.rgb_to_hls(r, g, b)

                # Apply structural shading
                l_new = max(0.0, min(1.0, l * l_factor))
                s_new = max(0.0, min(1.0, s * s_factor))
                new_color = colorsys.hls_to_rgb(h, l_new, s_new)

                patch.set_facecolor(new_color)

        # Generate custom neutral legend keys representing the shading relationship
        from matplotlib.patches import Patch
        legend_handles = []
        for col_idx in range(num_columns):
            neutral_rgb = (0.27, 0.44, 0.65)  # Premium steel blue base
            h, l, s = colorsys.rgb_to_hls(*neutral_rgb)
            
            if num_columns > 1:
                l_factor = 0.85 + (col_idx / (num_columns - 1)) * 0.45
                s_factor = 1.0 - (col_idx / (num_columns - 1)) * 0.4
            else:
                l_factor = 1.0
                s_factor = 1.0
                
            l_new = max(0.0, min(1.0, l * l_factor))
            s_new = max(0.0, min(1.0, s * s_factor))
            legend_color = colorsys.hls_to_rgb(h, l_new, s_new)
            legend_handles.append(Patch(facecolor=legend_color, edgecolor="white", linewidth=0.5))
            
        legend_labels = list(pivoted_mean.columns)

    elif plot_type == "line":
        pivoted_mean.plot(
            kind="line",
            yerr=pivoted_std,
            marker="o",
            ax=ax,
            color=line_palette,
            ecolor="#555555",
            elinewidth=1.0,
            capsize=0,
        )
        from matplotlib.container import ErrorbarContainer
        handles, legend_labels = ax.get_legend_handles_labels()
        legend_handles = [h[0] if isinstance(h, ErrorbarContainer) else h for h in handles]
    else:
        raise ValueError(
            "plot_type must be 'bar' or 'line' when plotting precomputed summary statistics."
        )

    # 5. Clean up Spines & Gridlines (No vertical gridlines)
    sns.despine(left=True, bottom=False)  # Remove the left vertical spine entirely
    ax.spines["bottom"].set_color("#cccccc")  # Make the bottom axis line an ultra-soft gray
    ax.spines["bottom"].set_linewidth(0.8)

    # Add ultra-soft, dashed horizontal gridlines only
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")

    # 6. Apply Modern Typography, Alignment, and Padding
    if len(group_cols) > 1:
        if x_order is not None:
            x_axis_label = group_cols[0]
            legend_title = " | ".join(group_cols[1:])
        else:
            x_axis_label = " | ".join(group_cols[:-1])
            legend_title = group_cols[-1]
    else:
        x_axis_label = group_cols[0]
        legend_title = ""

    # Editorial Typography and Alignment (Using clean semi-bold and dark charcoal colors)
    title_text = f"{metric_col.replace('_', ' ').title()} by {x_axis_label.replace('_', ' ').title()}"
    ax.set_title(title_text, fontsize=9, pad=20, fontweight="bold", color="#1a252f", loc="left")
    ax.set_xlabel(x_axis_label.replace("_", " ").title(), fontsize=8.5, fontweight="bold", color="#7f8c8d", labelpad=15)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=8.5, fontweight="bold", color="#7f8c8d", labelpad=15)

    # --- FORCED MONOSPACE TICK LABEL STYLING ---
    # Iterating over the labels directly guarantees we override any pandas defaults.
    x_rot = 90 if (" | " in x_axis_label or len(pivoted_mean.index) > 5) else 0
    for label in ax.get_xticklabels():
        label.set_rotation(x_rot)
        label.set_fontsize(6.5)          # Compact size
        label.set_weight("semibold")     # Strong semibold weight to match mono aesthetics
        label.set_color("#4f4f4f")
        if x_rot == 90:
            label.set_ha("center")
            label.set_va("top")

    for label in ax.get_yticklabels():
        label.set_fontsize(6.5)
        label.set_weight("semibold")
        label.set_color("#4f4f4f")

    # Push labels down slightly so they don't touch the bars
    ax.tick_params(axis="x", pad=8)

    # --- DYNAMIC Y-AXIS LIMITS ---
    if metric_col in ["correct", "accuracy", "success_rate"]:
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(bottom=0)

    # 7. Clean up the legend (Floating, Borderless, Compact Bold Keys)
    if len(group_cols) > 1:
        ax.legend(
            legend_handles,
            legend_labels,
            title=legend_title.replace("_", " ").title(),
            bbox_to_anchor=(1.02, 1),  # Positioned cleanly to the right
            loc="upper left",
            frameon=False,  # REMOVE the clunky black border frame entirely
            prop={"weight": "semibold", "size": 6.5},  # Bold, compact legend labels
            title_fontproperties={"weight": "bold", "size": 8.5},  # Bold, compact legend title
        )

    plt.tight_layout()
    plt.show()


#!/usr/bin/env python3
"""
Enhanced grouped-bar chart for any metric (default: *Quality*) stored in
`results.txt` files.  Run this script from **inside** the directory that holds
folders like `aamas2021/`, `iclr2018/`, etc. Example:

```bash
cd test_results
python plot_quality.py --metric "Quality" --outfile quality.png
```

✨  Enhanced Visualization Features  ✨
---------------------------------
1. Modern styling with improved aesthetics
2. Numeric labels on each bar with customizable formatting
3. Improved color schemes with greater contrast
4. Enhanced typography and layout
5. Multiple theme options (default, dark, publication)
6. Error handling and data validation
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

title_size_def = 16

# Set up the visualization styles
THEMES = {
    "default": {
        "fig_size": (12, 7),
        "bg_color": "#f8f9fa",
        "grid_color": "#dcdee0",
        "text_color": "#333333",
        "title_size": title_size_def,
        "label_size": 12,
        "tick_size": 11,
        "x_tick_size": 13,
        "palette": "Set2",
        "bar_alpha": 0.85,
        "edge_width": 0.8,
        "value_fmt": ".2f"
    },
    "dark": {
        "fig_size": (12, 7),
        "bg_color": "#2d3035",
        "grid_color": "#3d4044",
        "text_color": "#e5e5e5",
        "title_size": title_size_def,
        "label_size": 12,
        "tick_size": 11,
        "x_tick_size": 13,
        "palette": "Set2",
        "bar_alpha": 0.85,
        "edge_width": 0.8,
        "value_fmt": ".2f"
    },
    "publication": {
        "fig_size": (10, 6),
        "bg_color": "white",
        "grid_color": "#dddddd",
        "text_color": "black",
        "title_size": title_size_def,
        "label_size": 11,
        "tick_size": 10,
        "x_tick_size": 12,
        "palette": "colorblind",
        "bar_alpha": 0.85,
        "edge_width": 0.5,
        "value_fmt": ".2f"
    }
}


## Function to extract the data
def extract_metric(results_path: Path, metric_key: str) -> Optional[float]:
    pattern = rf"^{re.escape(metric_key)}:\s*([0-9.+-Ee]+)"
    try:
        text = results_path.read_text(errors="ignore")
    except Exception as exc:
        print(f"Error reading file {results_path}: {exc}")
        return None

    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


## Collect data
def collect_records(root: Path, metric_key: str) -> pd.DataFrame:
    records: List[Dict] = []
    
    # Validate that the root directory exists
    if not root.exists() or not root.is_dir():
        print(f"Warning: Directory {root} doesn't exist or is not a directory.")
        return pd.DataFrame()
    
    dataset_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not dataset_dirs:
        print(f"Warning: No dataset directories found in {root}")
        return pd.DataFrame()

    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        for algo_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
            algorithm = algo_dir.name
            results_txt = algo_dir / "results.txt"
            
            if not results_txt.exists():
                print(f"Warning: No results.txt found for {dataset}/{algorithm}")
                continue
            
            if metric_key == 'Seniority':
                value = extract_metric(results_txt, '#papers_with_max_reviewer_seniority_0') +  extract_metric(results_txt, '#papers_with_max_reviewer_seniority_1')
            elif metric_key == 'Region':
                h1 = extract_metric(results_txt, '#papers_such_that_reviewers_come_from_1_region')
                h2 = extract_metric(results_txt, '#papers_such_that_reviewers_come_from_2_regions')
                h3 = extract_metric(results_txt, '#papers_such_that_reviewers_come_from_3_regions')
                h4 = extract_metric(results_txt, '#papers_such_that_reviewers_come_from_4_regions')
                value = (h1 * 1 + h2 * 2 + h3 * 3 + h4 * 4) / (h1 + h2 + h3 + h4)
            else:
                value = extract_metric(results_txt, metric_key)
            if value is not None:
                records.append({
                    "dataset": dataset,
                    "algorithm": algorithm,
                    metric_key: value,
                })
    
    if not records:
        print(f"Warning: No valid data found for metric '{metric_key}'")
    
    return pd.DataFrame(records)


## Add value labels to the bars
def add_value_labels(ax, rects, value_fmt: str = ".2f", font_size: int = 8, 
                     color: str = "black", offset: Tuple[float, float] = (0, 0),
                     rotation: int = 0) -> None:
    """
    Add value labels to the top of each bar.
    
    Parameters:
    -----------
    ax : matplotlib Axes
        The axes to add labels to
    rects : list
        List of matplotlib Rectangle objects (the bars)
    value_fmt : str
        Format string for the values
    font_size : int
        Font size for labels
    color : str
        Text color for labels
    offset : Tuple[float, float]
        (x, y) offset for fine-tuning label position
    rotation : int
        Rotation angle for labels
    """
    for rect in rects:
        height = rect.get_height()
        if np.isnan(height):
            continue  # Skip missing bars
        
        # Determine vertical position based on whether the value is positive or negative
        va = "bottom" if height >= 0 else "top"
        y_pos = height + (0.01 if height >= 0 else -0.01)
        
        # Format the value
        text = f"{height:{value_fmt}}"
        
        # Add the label
        ax.text(
            rect.get_x() + rect.get_width() / 2 + offset[0],
            y_pos + offset[1],
            text,
            ha="center",
            va=va,
            fontsize=font_size,
            color=color,
            rotation=rotation,
            fontweight='normal',
            zorder=5
        )


## Create custom colormap
def create_custom_cmap(name: str, n_colors: int = 8) -> Any:
    """
    Create a custom colormap based on predefined options or seaborn palettes.
    
    Parameters:
    -----------
    name : str
        Name of the colormap or seaborn palette
    n_colors : int
        Number of colors to extract
        
    Returns:
    --------
    colors : list
        List of colors
    """
    if name == "custom_blue":
        # Custom blue gradient
        return LinearSegmentedColormap.from_list(
            "custom_blue", 
            ["#d0e6f5", "#2980b9", "#1a5276"], 
            N=n_colors
        )
    elif name == "gradient":
        # Custom gradient
        return LinearSegmentedColormap.from_list(
            "gradient", 
            ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"], 
            N=n_colors
        )
    elif name in plt.colormaps():
        # Use a matplotlib colormap
        return plt.get_cmap(name)
    else:
        # Try to use a seaborn palette
        try:
            return sns.color_palette(name, n_colors=n_colors)
        except:
            print(f"Warning: Unknown palette '{name}', falling back to default.")
            return plt.get_cmap("tab10")


## Apply theme to figure
def apply_theme(fig, ax, theme: Dict) -> None:
    """
    Apply a theme to the figure and axes.
    
    Parameters:
    -----------
    fig : matplotlib Figure
        The figure to style
    ax : matplotlib Axes
        The axes to style
    theme : Dict
        Dictionary containing theme specifications
    """
    # Figure background
    fig.patch.set_facecolor(theme["bg_color"])
    ax.set_facecolor(theme["bg_color"])
    
    # Text colors
    ax.title.set_color(theme["text_color"])
    ax.xaxis.label.set_color(theme["text_color"])
    ax.yaxis.label.set_color(theme["text_color"])
    
    # Text sizes
    ax.title.set_fontsize(40)
    ax.xaxis.label.set_fontsize(theme["label_size"])
    ax.yaxis.label.set_fontsize(theme["label_size"])
    
    # Tick colors and sizes
    for tick in ax.get_xticklabels():
        tick.set_color(theme["text_color"])
        tick.set_fontsize(theme["x_tick_size"])  # Larger size for x-axis labels
    
    for tick in ax.get_yticklabels():
        tick.set_color(theme["text_color"])
        tick.set_fontsize(theme["tick_size"])
    
    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=theme["grid_color"])
    ax.set_axisbelow(True)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_color(theme["grid_color"])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


## Main plotting function
def plot_metric(
    df: pd.DataFrame, 
    metric_key: str, 
    outfile: Optional[str] = None,
    theme_name: str = "default",
    bar_width_scale: float = 0.9,
    add_labels: bool = True,
    sort_values: bool = False,
    highlight_max: bool = False,
    use_hatch: bool = False,
    custom_palette: Optional[str] = None,
    custom_title: Optional[str] = None,
    horizontal_labels: bool = True,
    normalize: bool = False
) -> None:
    """
    Plot a grouped bar chart of the metric values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the data to plot
    metric_key : str
        Name of the metric to plot
    outfile : Optional[str]
        If provided, save the plot to this file
    theme_name : str
        Name of the theme to use ('default', 'dark', 'publication')
    bar_width_scale : float
        Scale factor for bar width (0-1)
    add_labels : bool
        Whether to add value labels to the bars
    sort_values : bool
        Whether to sort datasets by their mean value
    highlight_max : bool
        Whether to highlight the maximum value in each dataset
    use_hatch : bool
        Whether to add hatching patterns to bars
    custom_palette : Optional[str]
        Name of a custom color palette to use
    custom_title : Optional[str]
        Custom title for the plot
    """
    # Apply theme
    theme = THEMES.get(theme_name, THEMES["default"])
    
    # Sort datasets if requested
    if df.empty:
        print("Error: No data to plot.")
        return

    print(df)

    if normalize:
        df = normalize_data(df, metric_key)
        y_label = f"{metric_key} (% of Default)"
        value_fmt = '.1f'  # One decimal place for percentages
        is_percentage = True
        print("NORMALIZING")
    else:
        y_label = metric_key
        # Choose format based on data range
        max_val = df[metric_key].max()
        if max_val >= 100:
            value_fmt = '.0f'  # No decimal places for large values
        elif max_val >= 10:
            value_fmt = '.1f'  # One decimal place for medium values
        else:
            value_fmt = '.2f'  # Three decimal places for small values
        is_percentage = False

    print(df)
    
    def sort_key(d):
        if d == 'Default':
            return 0
        if d == 'MILP':
            return 1
        if d == 'Randomized':
            return 2
        if d == 'Perturbed Maximization':
            return 3
        if d == 'Ours':
            return 4
        return 5

    datasets = sorted(df["dataset"].unique())
    algorithms = sorted(df["algorithm"].unique(), key=sort_key)
    
    if sort_values:
        # Sort datasets by mean value
        dataset_means = df.groupby("dataset")[metric_key].mean().sort_values(ascending=False)
        datasets = dataset_means.index.tolist()
    
    # Calculate bar positions
    x = np.arange(len(datasets), dtype=float)
    bar_group_width = 0.75
    bar_width = bar_group_width / max(len(algorithms), 1) * bar_width_scale
    x_offsets = (np.arange(len(algorithms)) - (len(algorithms) - 1) / 2) * bar_width / bar_width_scale

    # Set up the figure
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Get figure size from theme
    fig, ax = plt.subplots(figsize=theme["fig_size"])
    
    # Choose color palette
    if custom_palette:
        colors = create_custom_cmap(custom_palette, len(algorithms))
    else:
        colors = sns.color_palette(theme["palette"], len(algorithms))
    
    colors = list(colors)  # Convert to mutable list if not already
    colors[-1] = "#FFEB3B"  # Bright yellow (Gold)
    
    # Track max value bars for highlighting
    max_bars = []
    
    # Plot bars for each algorithm
    for i, algo in enumerate(algorithms):
        # Get data for this algorithm
        algo_data = df[df["algorithm"] == algo].set_index("dataset")
        vals = [
            algo_data.loc[d, metric_key] if d in algo_data.index else np.nan
            for d in datasets
        ]
        
        # Create the bars
        rects = ax.bar(
            x + x_offsets[i],
            vals,
            bar_width,
            label=algo,
            color=colors[i] if not isinstance(colors, LinearSegmentedColormap) else colors(i/len(algorithms)),
            alpha=theme["bar_alpha"],
            edgecolor="black",
            linewidth=theme["edge_width"],
            hatch=["//", "\\\\", "xx", "++", "oo", "OO", ".."][i % 7] if use_hatch else None,
        )
        
        # Add value labels if requested
        if add_labels:
            add_value_labels(
                ax, 
                rects, 
                value_fmt=theme["value_fmt"],
                font_size=theme["tick_size"] - 1,
                color=theme["text_color"],
            )
        
        # Track this for max highlighting
        if highlight_max:
            max_bars.append((rects, vals))
    
    # Highlight max values if requested
    if highlight_max and len(datasets) > 0:
        # Create a mask for max values in each dataset
        all_values = np.array([vals for _, vals in max_bars])
        max_indices = np.nanargmax(all_values, axis=0)
        
        # Highlight max bars
        for dataset_idx, algo_idx in enumerate(max_indices):
            if np.isnan(all_values[algo_idx, dataset_idx]):
                continue  # Skip if the max is NaN
            
            # Get the bar to highlight
            bar = max_bars[algo_idx][0][dataset_idx]
            
            # Highlight with a thicker edge
            bar.set_edgecolor('black')
            bar.set_linewidth(theme["edge_width"] * 2)
            
            # Add a star or other marker above the bar
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height * 1.05,
                "★",
                ha="center",
                va="bottom",
                fontsize=theme["tick_size"],
                color=theme["text_color"],
                weight="bold"
            )

    # Apply theme styling
    apply_theme(fig, ax, theme)
    
    # Set labels and title
    title = custom_title if custom_title else f"{metric_key} by Algorithm and Dataset"
    ax.set_title(title, pad=15, fontsize = theme["title_size"])
    
    # Set tick positions and labels
    ax.set_xticks(x)
    if horizontal_labels:
        ax.set_xticklabels(datasets, rotation=0, ha="center")
    else:
        ax.set_xticklabels(datasets, rotation=25, ha="right")
    
    # Place legend outside plot area with a title
    legend = ax.legend(
        title="Algorithm", 
        bbox_to_anchor=(1.02, 1), 
        loc="upper left", 
        frameon=False,
        fontsize=13,
        title_fontsize=theme["label_size"]
    )
    legend.get_title().set_color(theme["text_color"])
    
    # Adjust layout with extra bottom padding for horizontal labels
    if horizontal_labels and len(datasets) > 0 and max(len(d) for d in datasets) > 10:
        # Add extra bottom padding for long labels
        fig.tight_layout(pad=1.1, rect=[0, 0.05, 1, 1])
    else:
        fig.tight_layout()
    
    # Save or show
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved figure to {outfile}")
    else:
        plt.show()

## Normalize data relative to baseline algorithm
def normalize_data(df: pd.DataFrame, metric_key, baseline_algo = "Default") -> pd.DataFrame:
    """
    Normalize data relative to the baseline algorithm.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with columns: "dataset", "algorithm", and metric_key
    metric_key : str 
        Name of the metric column to normalize
    baseline_algo : str
        Name of the algorithm to use as baseline
        
    Returns:
    --------
    df_normalized : pd.DataFrame
        Dataframe with normalized values
    """
    # Create a copy to avoid modifying the original
    df_norm = df.copy()
    
    # Get list of datasets
    datasets = df_norm["dataset"].unique()
    
    # Check if baseline algorithm exists
    if baseline_algo not in df_norm["algorithm"].unique():
        print(f"Warning: Baseline algorithm '{baseline_algo}' not found. Using raw values.")
        return df_norm
    
    # For each dataset, compute normalized values
    for dataset in datasets:
        # Get the baseline value for this dataset
        baseline_value = df_norm.loc[
            (df_norm["dataset"] == dataset) & 
            (df_norm["algorithm"] == baseline_algo), 
            metric_key
        ].values
        
        # Check if baseline value exists and is valid
        if len(baseline_value) == 0 or np.isnan(baseline_value[0]) or baseline_value[0] == 0:
            print(f"Warning: No valid baseline for dataset '{dataset}'. Skipping normalization.")
            continue
            
        # Get the actual baseline value
        base_val = baseline_value[0]
        
        # Normalize all values for this dataset
        mask = df_norm["dataset"] == dataset
        df_norm.loc[mask, metric_key] = (df_norm.loc[mask, metric_key] / base_val) 
    
    return df_norm


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enhanced visualization of metrics across datasets and algorithms.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Path containing dataset folders (default: current directory)")
    parser.add_argument("--metric", default="Quality", help="Metric key to extract, e.g. 'Quality', 'Max Quality'")
    parser.add_argument("--outfile", help="If provided, save the plot instead of displaying it interactively")
    parser.add_argument("--theme", default="default", choices=list(THEMES.keys()), help="Visual theme to use")
    parser.add_argument("--no-labels", action="store_true", help="Don't add value labels to bars")
    parser.add_argument("--sort", action="store_true", help="Sort datasets by their mean metric value")
    parser.add_argument("--highlight-max", action="store_true", help="Highlight maximum value in each dataset")
    parser.add_argument("--use-hatch", action="store_true", help="Add hatching patterns to differentiate bars")
    parser.add_argument("--horizontal-labels", action="store_true", default=True, help="Use horizontal (straight) labels on x-axis")
    parser.add_argument("--palette", help="Color palette to use (any matplotlib or seaborn palette)")
    parser.add_argument("--title", help="Custom title for the plot")
    parser.add_argument("--format", default=".2f", help="Format for value labels (e.g., '.2f', '.1%')")
    parser.add_argument("--normalize", action="store_true", help = "should we normalize?")
    
    args = parser.parse_args()

    # Update theme format if specified
    if args.format and args.theme in THEMES:
        THEMES[args.theme]["value_fmt"] = args.format

    # Collect data
    df = collect_records(args.root, args.metric)
    if df.empty:
        raise SystemExit(f"No data found under {args.root.resolve()} for metric '{args.metric}'.")

    # Plot the data
    plot_metric(
        df, 
        args.metric, 
        args.outfile,
        theme_name=args.theme,
        add_labels=not args.no_labels,
        sort_values=args.sort,
        highlight_max=args.highlight_max,
        use_hatch=args.use_hatch,
        custom_palette=args.palette,
        custom_title=args.title,
        horizontal_labels=args.horizontal_labels,
        normalize = args.normalize
    )

if __name__ == "__main__":
    main()


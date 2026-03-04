"""Shared plotting utilities for hierarchical benchmark and ablation scripts."""

import logging

import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for plotting scripts."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def plot_task_panel(
    ax,
    df: pd.DataFrame,
    metric: str,
    groups: list,
    group_colors: dict,
    x_column: str,
    x_label: str,
    group_column: str = "algorithm",
    label_map: dict | None = None,
    show_title: bool = False,
    title: str = "",
    show_ylabel: bool = False,
    ylabel: str = "",
    use_scientific_x: bool = True,
    show_xlabel: bool = True,
):
    """Plot a single task panel with all groups overlaid as lines with bootstrap CIs.

    Args:
        ax: Matplotlib axes object
        df: DataFrame with columns: group_column, x_column, metric
        metric: Name of the metric column to plot
        groups: List of group values to plot (controls order)
        group_colors: Dict mapping group value to color
        x_column: Column name for x-axis values
        x_label: Label for x-axis
        group_column: Column name to group by (e.g. "algorithm" or "method")
        label_map: Dict mapping group values to display labels. If None, uses
            the group value uppercased.
        show_title: Whether to show subplot title
        title: Title text for subplot
        show_ylabel: Whether to show y-axis label
        ylabel: Label for y-axis
        use_scientific_x: Whether to use scientific notation for x-axis ticks
        show_xlabel: Whether to show x-axis label
    """
    if label_map is None:
        label_map = {}

    for group in groups:
        group_df = df[df[group_column] == group]

        if len(group_df) == 0:
            continue

        # Group by x_column and compute bootstrap CIs
        x_values = []
        means = []
        lower_errs = []
        upper_errs = []

        for x_val, sub in group_df.groupby(x_column):
            data = sub[metric].values
            mean = np.mean(data)
            x_values.append(x_val)
            means.append(mean)

            if len(data) > 1:
                res = bootstrap(
                    (data,),
                    np.mean,
                    confidence_level=0.95,
                    n_resamples=1000,
                    random_state=42,
                )
                lower_errs.append(mean - res.confidence_interval.low)
                upper_errs.append(res.confidence_interval.high - mean)
            else:
                lower_errs.append(np.nan)
                upper_errs.append(np.nan)

        color = group_colors[group]
        label = label_map.get(group, group.upper())

        ax.errorbar(
            x_values,
            means,
            yerr=[lower_errs, upper_errs],
            marker="o",
            color=color,
            markersize=5,
            linewidth=2,
            capsize=3,
            label=label,
        )

    # Formatting
    if show_xlabel:
        ax.set_xlabel(x_label, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks to actual values
    x_ticks = sorted(df[x_column].unique())
    ax.set_xticks(x_ticks)
    if use_scientific_x:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    else:
        ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    if show_title:
        ax.set_title(title, fontsize=10, fontweight="bold")

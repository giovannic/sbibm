#!/usr/bin/env python
"""
Grid visualization script for hierarchical benchmark results.

This script aggregates CSV results from parallel HPC jobs and creates
publication-quality grid visualizations with tasks as rows. All algorithms
are overlaid on the same axes with a shared legend.

Example usage:
    python scripts/plot_hierarchical_benchmark.py \
        --input_dir results \
        --metric reverse_kl \
        --output_path results/reverse_kl_grid.png \
        --config manuscript
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the plotting script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_all_results(input_dir: Path, n_l: int = 1) -> dict:
    """Load benchmark results for all hierarchical tasks.

    Args:
        input_dir: Directory containing benchmark CSV files
        n_l: Scaling factor for adjusting num_simulations

    Returns:
        Dict mapping task_name -> DataFrame with all results
    """
    log = logging.getLogger(__name__)

    # Find all hierarchical_*.csv files
    pattern = "hierarchical_*.csv"
    csv_files = list(input_dir.glob(pattern))

    if not csv_files:
        raise ValueError(
            f"No CSV files found matching pattern '{pattern}' " f"in {input_dir}"
        )

    log.info(f"Found {len(csv_files)} result files total")

    # Load all CSVs and group by task name from data
    results = {}
    for csv_file in csv_files:
        log.debug(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Extract task name from 'task' column (first row)
        if len(df) > 0 and "task" in df.columns:
            task_name = df["task"].iloc[0]

            if task_name not in results:
                results[task_name] = []
            results[task_name].append(df)

    # Concatenate all DataFrames per task
    for task_name in sorted(results.keys()):
        results[task_name] = pd.concat(results[task_name], ignore_index=True)
        log.info(f"Loaded {len(results[task_name])} total runs for " f"'{task_name}'")

        # Apply scaling to num_simulations if n_l > 1
        if n_l > 1:
            df = results[task_name]
            snpe_mask = df["algorithm"] == "snpe"
            deepset_mask = df["algorithm"] == "deepset"

            if snpe_mask.any():
                df.loc[snpe_mask, "num_simulations"] *= n_l
                log.info(f"  Scaled SNPE num_simulations by {n_l}")

            if deepset_mask.any():
                deepset_scale = (n_l + 1) // 2
                df.loc[deepset_mask, "num_simulations"] *= deepset_scale
                log.info(f"  Scaled DeepSet num_simulations by {deepset_scale}")

    return results


def load_n_l_scaling_results(
    input_dir: Path, n_l_values: list[int] | None = None
) -> dict:
    """Load n_l scaling benchmark results for all hierarchical tasks.

    Args:
        input_dir: Directory containing n_l scaling CSV files
                   Filename pattern: {task}_{algorithm}_{num_simulations}_{n_l}.csv
        n_l_values: List of n_l values corresponding to rows in CSV files.
                    Row index i maps to n_l_values[i]. If None, uses row index + 1.

    Returns:
        Dict mapping task_name -> DataFrame with all results
    """
    log = logging.getLogger(__name__)

    # Find all hierarchical_*.csv files
    pattern = "hierarchical_*.csv"
    csv_files = list(input_dir.glob(pattern))

    if not csv_files:
        raise ValueError(
            f"No CSV files found matching pattern '{pattern}' in {input_dir}"
        )

    log.info(f"Found {len(csv_files)} n_l scaling result files")

    # Load all CSVs and group by task name
    results = {}
    for csv_file in csv_files:
        log.debug(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Add n_l column based on row index and provided n_l_values
        if n_l_values is not None:
            # Truncate to minimum of CSV rows and n_l_values length
            n_rows = min(len(df), len(n_l_values))
            df = df.head(n_rows).copy()
            df["n_l"] = n_l_values[:n_rows]
            log.debug(f"  Mapped {n_rows} rows to n_l values: {n_l_values[:n_rows]}")
        else:
            # Fallback: use row index + 1 as n_l
            df = df.copy()
            df["n_l"] = range(1, len(df) + 1)
            log.debug(f"  Using row index + 1 as n_l for {len(df)} rows")

        # Extract task name from 'task' column
        if len(df) > 0 and "task" in df.columns:
            task_name = df["task"].iloc[0]

            if task_name not in results:
                results[task_name] = []
            results[task_name].append(df)

    # Concatenate all DataFrames per task
    for task_name in sorted(results.keys()):
        results[task_name] = pd.concat(results[task_name], ignore_index=True)
        df = results[task_name]
        log.info(f"Loaded {len(df)} n_l configurations for '{task_name}'")

        # Apply scaling based on algorithm using the n_l column
        snpe_mask = df["algorithm"] == "snpe"
        deepset_mask = df["algorithm"] == "deepset"

        if snpe_mask.any():
            # SNPE: scale by n_l
            df.loc[snpe_mask, "num_simulations"] *= df.loc[snpe_mask, "n_l"].astype(int)
            log.info("  Scaled SNPE num_simulations by n_l")

        if deepset_mask.any():
            # DeepSet: scale by (n_l + 1) // 2
            deepset_scale = (df.loc[deepset_mask, "n_l"].astype(int) + 1) // 2
            df.loc[deepset_mask, "num_simulations"] *= deepset_scale
            log.info("  Scaled DeepSet num_simulations by (n_l + 1) // 2")

    return results


def _plot_task_panel(
    ax,
    df: pd.DataFrame,
    metric: str,
    algorithms: list,
    algo_colors: dict,
    x_column: str,
    x_label: str,
    show_title: bool = False,
    title: str = "",
    show_ylabel: bool = False,
    ylabel: str = "",
    use_scientific_x: bool = True,
):
    """Plot a single task panel with all algorithms overlaid.

    Args:
        ax: Matplotlib axes object
        df: DataFrame with columns: algorithm, x_column, metric
        metric: Name of the metric column to plot
        algorithms: List of algorithm names to plot
        algo_colors: Dict mapping algorithm name to color
        x_column: Column name for x-axis values
        x_label: Label for x-axis
        show_title: Whether to show subplot title
        title: Title text for subplot
        show_ylabel: Whether to show y-axis label
        ylabel: Label for y-axis
        use_scientific_x: Whether to use scientific notation for x-axis ticks
    """
    for algorithm in algorithms:
        # Filter data for this algorithm
        algo_df = df[df["algorithm"] == algorithm]

        if len(algo_df) == 0:
            continue

        # Group by x_column and compute bootstrap CIs
        x_values = []
        means = []
        lower_errs = []
        upper_errs = []

        for x_val, group in algo_df.groupby(x_column):
            data = group[metric].values
            mean = np.mean(data)
            x_values.append(x_val)
            means.append(mean)

            if len(data) > 1:
                # Compute bootstrap 95% CI
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
                # Single data point, no CI (consistent with previous std-based behavior)
                lower_errs.append(np.nan)
                upper_errs.append(np.nan)

        # Get color for this algorithm
        color = algo_colors[algorithm]

        # Plot line with error bars
        label = "NPE" if algorithm == "snpe" else algorithm.upper()
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


def create_grid_plot(
    results: dict,
    metric: str,
    config: str = "manuscript",
    n_l_results: dict | None = None,
    row_labels: tuple | None = None,
):
    """Create a grid of line plots for all tasks with algorithms overlaid.

    Args:
        results: Dict mapping task_name -> DataFrame with
                 columns: algorithm, num_simulations, metric
        metric: Name of the metric column to plot
        config: Styling configuration ('manuscript' or 'streamlit')
        n_l_results: Optional dict for n_l scaling data (enables second row)
        row_labels: Labels for rows when using two-row layout

    Returns:
        matplotlib Figure object
    """
    # Set style based on config
    if config == "manuscript":
        plt.rcParams["font.size"] = 9
        cell_width = 3.0
        cell_height = 2.5
    else:  # streamlit
        plt.rcParams["font.size"] = 11
        cell_width = 4.0
        cell_height = 3.0

    plt.style.use("seaborn-v0_8-whitegrid")

    # Determine number of rows
    n_rows = 2 if n_l_results else 1

    # Extract unique tasks and algorithms from both result sets
    tasks = sorted(results.keys())
    all_algorithms = set()
    for df in results.values():
        all_algorithms.update(df["algorithm"].unique())
    if n_l_results:
        for df in n_l_results.values():
            all_algorithms.update(df["algorithm"].unique())
    algorithms = sorted(all_algorithms)

    # Assign colors to algorithms in alphabetical order using tab10 palette
    cmap = plt.colormaps["tab10"]
    algo_colors = {algo: cmap(i) for i, algo in enumerate(algorithms)}

    n_tasks = len(tasks)

    # Create figure with appropriate number of rows
    figsize = (cell_width * n_tasks, cell_height * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_tasks,
        figsize=figsize,
        squeeze=False,
        sharey="row",  # Share y-axis within each row independently
    )

    # Format metric label
    metric_label = metric.replace("_", " ").title()

    # Row 0: Simulation budget scaling
    for task_idx, task_name in enumerate(tasks):
        ax = axes[0, task_idx]
        df = results[task_name]

        # Format task title
        title = task_name.replace("_", " ").title()
        title = title.replace("Sir", "SIR").replace("Slcp", "SLCP")

        _plot_task_panel(
            ax=ax,
            df=df,
            metric=metric,
            algorithms=algorithms,
            algo_colors=algo_colors,
            x_column="num_simulations",
            x_label="Number of Simulations",
            show_title=True,
            title=title,
            show_ylabel=(task_idx == 0),
            ylabel=metric_label,
        )

    # Row 1: n_l scaling (if provided)
    if n_l_results:
        for task_idx, task_name in enumerate(tasks):
            ax = axes[1, task_idx]

            if task_name in n_l_results:
                df = n_l_results[task_name]
                _plot_task_panel(
                    ax=ax,
                    df=df,
                    metric=metric,
                    algorithms=algorithms,
                    algo_colors=algo_colors,
                    x_column="n_l",
                    x_label="n_l",
                    show_title=False,
                    show_ylabel=(task_idx == 0),
                    ylabel=metric_label,
                    use_scientific_x=False,
                )
            else:
                # Empty panel for missing task
                ax.set_visible(False)

        # Add row labels on the left side
        if row_labels:
            for row_idx, label in enumerate(row_labels):
                # Add text annotation to the left of the first column
                ax = axes[row_idx, 0]
                ax.annotate(
                    label,
                    xy=(-0.35, 0.5),
                    xycoords="axes fraction",
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )

    # Add shared figure legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Adjust legend position for two-row layout
    legend_y = -0.02 if n_rows == 1 else -0.01
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(algorithms),
        fontsize=9,
    )

    plt.tight_layout()

    return fig


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Aggregate and plot hierarchical benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="results",
        help="Directory containing benchmark CSV files",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "reverse_kl",
            "lc2st_p_value",
            "lc2st_test_statistic",
            "lc2st_reject",
        ],
        required=True,
        help="Metric to plot",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output figure (PNG format)",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["manuscript", "streamlit"],
        default="manuscript",
        help="Styling configuration",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=1,
        help="Scaling factor for adjusting num_simulations",
    )
    parser.add_argument(
        "--n_l_input_dir",
        type=str,
        default=None,
        help="Optional directory for n_l scaling data (enables second row)",
    )
    parser.add_argument(
        "--row_labels",
        type=str,
        nargs=2,
        default=["Simulation Budget", "n_l"],
        help="Labels for rows when using two-row layout",
    )
    parser.add_argument(
        "--n_l_values",
        type=int,
        nargs="+",
        default=None,
        help="List of n_l values corresponding to rows in n_l scaling CSVs",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("Hierarchical Benchmark Grid Plotter")
    log.info("=" * 80)
    log.info(f"Input directory: {args.input_dir}")
    log.info(f"Metric: {args.metric}")
    log.info(f"Output path: {args.output_path}")
    log.info(f"Config: {args.config}")
    log.info(f"n_l scaling factor: {args.n_l}")
    if args.n_l_input_dir:
        log.info(f"n_l scaling directory: {args.n_l_input_dir}")
        log.info(f"n_l values: {args.n_l_values}")
        log.info(f"Row labels: {args.row_labels}")
    log.info("=" * 80)

    # Validate output path
    output_path = Path(args.output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Output path must end with .png, got: {output_path.suffix}")

    # Load all results
    results = load_all_results(input_dir=Path(args.input_dir), n_l=args.n_l)

    # Print summary statistics
    log.info("\nSummary Statistics (Simulation Budget):")
    log.info(f"  Tasks: {list(results.keys())}")
    for task_name, df in results.items():
        log.info(f"  {task_name}:")
        log.info(f"    Algorithms: {df['algorithm'].unique().tolist()}")
        log.info(
            f"    Simulation budgets: "
            f"{sorted(df['num_simulations'].unique().tolist())}"
        )
        log.info(f"    Total runs: {len(df)}")

    # Load n_l scaling results if provided
    n_l_results = None
    if args.n_l_input_dir:
        n_l_results = load_n_l_scaling_results(
            input_dir=Path(args.n_l_input_dir),
            n_l_values=args.n_l_values,
        )

        log.info("\nSummary Statistics (n_l Scaling):")
        log.info(f"  Tasks: {list(n_l_results.keys())}")
        for task_name, df in n_l_results.items():
            log.info(f"  {task_name}:")
            log.info(f"    Algorithms: {df['algorithm'].unique().tolist()}")
            log.info(
                f"    n_l values: "
                f"{sorted(df['n_l'].unique().tolist())}"
            )
            log.info(f"    Total configurations: {len(df)}")

    # Create grid plot
    log.info(f"Creating grid plot for metric: {args.metric}")
    fig = create_grid_plot(
        results=results,
        metric=args.metric,
        config=args.config,
        n_l_results=n_l_results,
        row_labels=tuple(args.row_labels) if n_l_results else None,
    )

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    log.info(f"Saving plot to {output_path}")
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info("=" * 80)
    log.info("Grid plotting completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

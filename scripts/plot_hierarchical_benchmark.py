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
import pandas as pd


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


def create_grid_plot(
    results: dict,
    metric: str,
    config: str = "manuscript",
):
    """Create a grid of line plots for all tasks with algorithms overlaid.

    Args:
        results: Dict mapping task_name -> DataFrame with
                 columns: algorithm, num_simulations, metric
        metric: Name of the metric column to plot
        config: Styling configuration ('manuscript' or 'streamlit')

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

    # Extract unique tasks and algorithms
    tasks = sorted(results.keys())
    all_algorithms = set()
    for df in results.values():
        all_algorithms.update(df["algorithm"].unique())
    algorithms = sorted(all_algorithms)

    # Assign colors to algorithms in alphabetical order using tab10 palette
    cmap = plt.colormaps["tab10"]
    algo_colors = {algo: cmap(i) for i, algo in enumerate(algorithms)}

    n_tasks = len(tasks)

    # Create figure with one row of subplots (tasks as columns)
    figsize = (cell_width * n_tasks, cell_height)
    fig, axes = plt.subplots(
        1,
        n_tasks,
        figsize=figsize,
        squeeze=False,
        sharey=True,
    )

    # Plot each task with all algorithms overlaid
    for task_idx, task_name in enumerate(tasks):
        ax = axes[0, task_idx]
        df = results[task_name]

        for algorithm in algorithms:
            # Filter data for this algorithm
            algo_df = df[df["algorithm"] == algorithm]

            if len(algo_df) == 0:
                continue

            # Group by num_simulations and compute stats
            grouped = (
                algo_df.groupby("num_simulations")[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            # Compute 95% CI
            grouped["ci"] = 1.96 * grouped["std"] / (grouped["count"] ** 0.5)

            # Get color for this algorithm
            color = algo_colors[algorithm]

            # Plot line with error bars
            label = "NPE" if algorithm == "snpe" else algorithm.upper()
            ax.errorbar(
                grouped["num_simulations"],
                grouped["mean"],
                yerr=grouped["ci"],
                marker="o",
                color=color,
                markersize=5,
                linewidth=2,
                capsize=3,
                label=label,
            )

        # Formatting
        ax.set_xlabel("Number of Simulations", fontsize=9)
        if task_idx == 0:
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set x-axis ticks to actual simulation values in standard form
        x_ticks = sorted(df["num_simulations"].unique())
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        # Task name as subplot title (keep SIR and SLCP uppercase)
        title = task_name.replace("_", " ").title()
        title = title.replace("Sir", "SIR").replace("Slcp", "SLCP")
        ax.set_title(
            title,
            fontsize=10,
            fontweight="bold",
        )

    # Add shared figure legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
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
    log.info("=" * 80)

    # Validate output path
    output_path = Path(args.output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Output path must end with .png, got: {output_path.suffix}")

    # Load all results
    results = load_all_results(input_dir=Path(args.input_dir), n_l=args.n_l)

    # Print summary statistics
    log.info("\nSummary Statistics:")
    log.info(f"  Tasks: {list(results.keys())}")
    for task_name, df in results.items():
        log.info(f"  {task_name}:")
        log.info(f"    Algorithms: {df['algorithm'].unique().tolist()}")
        log.info(
            f"    Simulation budgets: "
            f"{sorted(df['num_simulations'].unique().tolist())}"
        )
        log.info(f"    Total runs: {len(df)}")

    # Create grid plot
    log.info(f"Creating grid plot for metric: {args.metric}")
    fig = create_grid_plot(
        results=results,
        metric=args.metric,
        config=args.config,
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

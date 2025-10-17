#!/usr/bin/env python
"""
Aggregation and plotting script for hierarchical benchmark results.

This script aggregates CSV results from parallel HPC jobs and creates
publication-quality visualizations of benchmark metrics.

Example usage:
    python scripts/plot_hierarchical_benchmark.py \
        --input_dir results \
        --task hierarchical_two_moons \
        --metric reverse_kl \
        --output_path results/reverse_kl.png \
        --config manuscript
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the plotting script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_results(input_dir: Path, task_name: str) -> pd.DataFrame:
    """Load and concatenate all benchmark results for a task.

    Args:
        input_dir: Directory containing benchmark CSV files
        task_name: Task name to filter results

    Returns:
        Concatenated DataFrame with all results
    """
    log = logging.getLogger(__name__)

    # Find all CSV files matching the task pattern
    pattern = f"{task_name}_*.csv"
    csv_files = list(input_dir.glob(pattern))

    if not csv_files:
        raise ValueError(
            f"No CSV files found matching pattern '{pattern}' "
            f"in {input_dir}"
        )

    log.info(f"Found {len(csv_files)} result files for task '{task_name}'")

    # Load and concatenate all CSVs
    dfs = []
    for csv_file in csv_files:
        log.debug(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    log.info(f"Loaded {len(combined_df)} total benchmark runs")

    return combined_df


def create_metric_plot(
    df: pd.DataFrame,
    metric: str,
    title: str = None,
    config: str = "manuscript",
) -> plt.Figure:
    """Create a line plot for the specified metric.

    Args:
        df: DataFrame with columns: algorithm, num_simulations, metric
        metric: Name of the metric column to plot
        title: Optional title for the plot
        config: Styling configuration ('manuscript' or 'streamlit')

    Returns:
        matplotlib Figure object
    """
    # Set style
    if config == "manuscript":
        plt.rcParams["font.size"] = 10
        figsize = (8, 4)
    else:  # streamlit
        plt.rcParams["font.size"] = 14
        figsize = (10, 6)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Get unique algorithms
    algorithms = sorted(df["algorithm"].unique())
    n_algorithms = len(algorithms)

    # Create subplots (one per algorithm)
    fig, axes = plt.subplots(
        1, n_algorithms, figsize=(figsize[0] * n_algorithms / 2, figsize[1]),
        squeeze=False
    )
    axes = axes.flatten()

    # Plot each algorithm
    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx]
        algo_df = df[df["algorithm"] == algorithm]

        # Group by num_simulations and compute mean/CI
        grouped = (
            algo_df.groupby("num_simulations")[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Compute 95% CI
        grouped["ci"] = 1.96 * grouped["std"] / (grouped["count"] ** 0.5)

        # Plot line with error bars
        ax.errorbar(
            grouped["num_simulations"],
            grouped["mean"],
            yerr=grouped["ci"],
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            label=algorithm,
        )

        # Formatting
        ax.set_xlabel("Number of Simulations")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(algorithm.upper())
        ax.grid(True, alpha=0.3)

        # Set x-axis to use actual values (not continuous)
        ax.set_xticks(sorted(df["num_simulations"].unique()))
        ax.tick_params(axis="x", rotation=45)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

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
        "--task",
        type=str,
        default="hierarchical_two_moons",
        help="Task name to filter results",
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
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("Hierarchical Benchmark Plotter")
    log.info("=" * 80)
    log.info(f"Input directory: {args.input_dir}")
    log.info(f"Task: {args.task}")
    log.info(f"Metric: {args.metric}")
    log.info(f"Output path: {args.output_path}")
    log.info(f"Config: {args.config}")
    log.info("=" * 80)

    # Validate output path
    output_path = Path(args.output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError(
            f"Output path must end with .png, got: {output_path.suffix}"
        )

    # Load results
    df = load_results(
        input_dir=Path(args.input_dir), task_name=args.task
    )

    # Print summary statistics
    log.info("\nSummary Statistics:")
    log.info(f"  Algorithms: {df['algorithm'].unique().tolist()}")
    log.info(
        f"  Simulation budgets: "
        f"{sorted(df['num_simulations'].unique().tolist())}"
    )
    log.info(
        f"  Observations: {sorted(df['num_observation'].unique().tolist())}"
    )
    log.info(f"  Total runs: {len(df)}")

    # Create plot
    log.info(f"Creating plot for metric: {args.metric}")
    fig = create_metric_plot(
        df=df,
        metric=args.metric,
        title=args.title,
        config=args.config,
    )

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    log.info(f"Saving plot to {output_path}")
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info("=" * 80)
    log.info("Plotting completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Ablation study visualization script.

This script creates line plots with bootstrap CIs comparing ablation methods
(TFMPE, MLP, Direct, Joint) across hierarchical tasks, with increasing
simulation budgets on the x-axis.

Results are loaded from a flat directory where the ablation variant is encoded
in the CSV filename.

Filename pattern:
    hierarchical_<task>_bottom_up_[<ablation>_]<num_sims>_<seed>.csv
    - no ablation suffix → TFMPE
    - direct → Direct
    - joint → Joint
    - mlp → MLP

Example usage:
    python scripts/plot_ablation.py \
        --input_dir /mnt/gc1610/home/ablation_results \
        --metric lc2st_test_statistic \
        --output_path ablation_test.png \
        --config manuscript
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import plot_task_panel, setup_logging

# Ablation tokens in filenames and their display labels
ABLATION_TOKENS = {
    "_direct_": "Direct",
    "_joint_": "Joint",
    "_mlp_": "MLP",
}

# Fixed method order and label map
METHODS = ["TFMPE", "Direct", "Joint", "MLP"]
METHOD_LABEL_MAP = {m: m for m in METHODS}


def load_ablation_results(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load ablation results from a flat directory of CSV files.

    Parses the ablation method from the filename and groups results by task.

    Args:
        input_dir: Directory containing hierarchical_*.csv files

    Returns:
        Dict mapping task_name -> DataFrame with a 'method' column added
    """
    log = logging.getLogger(__name__)

    csv_files = list(input_dir.glob("hierarchical_*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    log.info(f"Found {len(csv_files)} CSV files in {input_dir}")

    results: dict[str, list[pd.DataFrame]] = {}
    for csv_file in csv_files:
        log.debug(f"Loading {csv_file.name}")

        # Determine ablation method from filename
        fname = csv_file.name
        method = "TFMPE"  # default: no ablation token
        for token, label in ABLATION_TOKENS.items():
            if token in fname:
                method = label
                break

        df = pd.read_csv(csv_file)
        df["method"] = method

        # Group by task name from CSV
        if len(df) > 0 and "task" in df.columns:
            task_name = df["task"].iloc[0]
            if task_name not in results:
                results[task_name] = []
            results[task_name].append(df)

    # Concatenate per task
    for task_name in sorted(results.keys()):
        results[task_name] = pd.concat(results[task_name], ignore_index=True)
        log.info(
            f"Loaded {len(results[task_name])} total runs for '{task_name}'"
        )

    return results


def create_ablation_plot(
    results: dict[str, pd.DataFrame],
    metric: str = "lc2st_test_statistic",
    config: str = "manuscript",
) -> plt.Figure:
    """Create line plots with bootstrap CIs for ablation study.

    Args:
        results: Dict mapping task_name -> DataFrame
        metric: Name of the metric column to plot
        config: Styling configuration ('manuscript' or 'streamlit')

    Returns:
        matplotlib Figure object
    """
    if config == "manuscript":
        plt.rcParams["font.size"] = 9
        cell_width = 3.0
        cell_height = 2.5
    else:  # streamlit
        plt.rcParams["font.size"] = 11
        cell_width = 4.0
        cell_height = 3.0

    plt.style.use("seaborn-v0_8-whitegrid")

    tasks = sorted(results.keys())
    n_tasks = len(tasks)

    # Assign colors using tab10 palette in fixed method order
    cmap = plt.colormaps["tab10"]
    method_colors = {method: cmap(i) for i, method in enumerate(METHODS)}

    figsize = (cell_width * n_tasks, cell_height)
    fig, axes = plt.subplots(
        1,
        n_tasks,
        figsize=figsize,
        squeeze=False,
        sharey=True,
    )

    metric_label = metric.replace("_", " ").title()

    for task_idx, task_name in enumerate(tasks):
        ax = axes[0, task_idx]
        df = results[task_name]

        # Format task title
        title = task_name.replace("hierarchical_", "").replace("_", " ").title()
        title = title.replace("Sir", "SIR").replace("Slcp", "SLCP")

        plot_task_panel(
            ax=ax,
            df=df,
            metric=metric,
            groups=METHODS,
            group_colors=method_colors,
            x_column="num_simulations",
            x_label="Number of Simulations",
            group_column="method",
            label_map=METHOD_LABEL_MAP,
            show_title=True,
            title=title,
            show_ylabel=(task_idx == 0),
            ylabel=metric_label,
        )

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(METHODS),
        fontsize=9,
    )

    plt.tight_layout()

    return fig


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Plot ablation study results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing ablation CSV files (flat layout)",
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
        default="lc2st_test_statistic",
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

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("Ablation Study Plotter")
    log.info("=" * 80)
    log.info(f"Input directory: {args.input_dir}")
    log.info(f"Metric: {args.metric}")
    log.info(f"Output path: {args.output_path}")
    log.info(f"Config: {args.config}")
    log.info("=" * 80)

    # Validate output path
    output_path = Path(args.output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Output path must end with .png, got: {output_path.suffix}")

    # Load all results
    results = load_ablation_results(input_dir=Path(args.input_dir))

    # Print summary statistics
    log.info("\nSummary Statistics:")
    log.info(f"  Tasks: {list(results.keys())}")
    for task_name, df in sorted(results.items()):
        log.info(f"  {task_name}:")
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            budgets = sorted(method_df["num_simulations"].unique().tolist())
            log.info(f"    {method}: {len(method_df)} runs, budgets={budgets}")

    # Create plot
    log.info(f"Creating ablation plot for metric: {args.metric}")
    fig = create_ablation_plot(
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
    log.info("Ablation plotting completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

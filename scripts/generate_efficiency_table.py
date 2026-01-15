#!/usr/bin/env python
"""
LaTeX table generation script for hierarchical benchmark results.

This script aggregates CSV results from parallel HPC jobs and creates
publication-quality LaTeX tables with proper statistical formatting.

Example usage:
    python scripts/generate_efficiency_table.py \
        --input_dir /mnt/gc1610/home/sbibm_results_8 \
        --output_path results/efficiency_table.tex \
        --metric lc2st_test_statistic
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the table generation script."""
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

            if snpe_mask.any():
                df.loc[snpe_mask, "num_simulations"] *= n_l
                log.info(f"  Scaled SNPE num_simulations by {n_l}")

            deepset_mask = df["algorithm"] == "deepset"
            if deepset_mask.any():
                scale = n_l // 2
                df.loc[deepset_mask, "num_simulations"] *= scale
                log.info(f"  Scaled Deepset num_simulations by {scale}")

    return results


def compute_statistics(values: pd.Series) -> tuple[float, float, float]:
    """Compute mean and 95% confidence interval.

    Args:
        values: Series of metric values from multiple runs

    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    import numpy as np

    mean = values.mean()
    n = len(values)

    # For single runs, return mean with no margin
    if n == 1:
        return mean, mean, mean

    std = values.std()

    # Compute 95% CI: mean +/- 1.96 * std / sqrt(n)
    margin = 1.96 * std / np.sqrt(n)
    lower_ci = mean - margin
    upper_ci = mean + margin

    return mean, lower_ci, upper_ci


def determine_bold_direction(metric: str) -> str:
    """Determine if lower or higher values are better for a metric.

    Args:
        metric: Name of the metric

    Returns:
        'min' if lower is better, 'max' if higher is better
    """
    metric_lower = metric.lower()

    # Metrics where lower is better
    if any(
        keyword in metric_lower
        for keyword in ["kl", "error", "loss", "mse", "mae", "test_statistic"]
    ):
        return "min"

    # Metrics where higher is better
    if any(keyword in metric_lower for keyword in ["accuracy", "score", "auc", "r2"]):
        return "max"

    # Default: lower is better
    return "min"


def format_cell(
    mean: float, lower_ci: float, upper_ci: float, is_best: bool = False
) -> str:
    """Format a table cell with mean and confidence interval.

    Args:
        mean: Mean value
        lower_ci: Lower bound of 95% CI
        upper_ci: Upper bound of 95% CI
        is_best: Whether this is the best value (should be bolded)

    Returns:
        Formatted LaTeX string
    """
    # For single runs (where CI equals mean), just show the mean
    if lower_ci == mean and upper_ci == mean:
        cell_text = f"{mean:.3f}"
    else:
        # Format with 3 decimal places
        cell_text = f"{mean:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]"

    # Apply bold if this is the best result
    if is_best:
        cell_text = f"\\textbf{{{cell_text}}}"

    return cell_text


def format_number(num: int) -> str:
    """Format large numbers with comma separators.

    Args:
        num: Number to format

    Returns:
        Formatted string (e.g., 1000 -> "1,000")
    """
    return f"{num:,}"


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text.

    Args:
        text: Text to escape

    Returns:
        Escaped text
    """
    # Replace underscores with escaped underscores
    text = text.replace("_", "\\_")
    return text


def generate_latex_table(
    results: dict,
    metric: str,
    tasks: Optional[list[str]] = None,
    algorithms: Optional[list[str]] = None,
    budgets: Optional[list[int]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """Generate LaTeX table from benchmark results.

    Args:
        results: Dict mapping task_name -> DataFrame with results
        metric: Name of metric column to display
        tasks: List of tasks to include (None = all)
        algorithms: List of algorithms to include (None = all)
        budgets: List of simulation budgets to include (None = all)
        caption: Custom caption (None = auto-generate)
        label: Custom LaTeX label (None = auto-generate)

    Returns:
        Complete LaTeX table as string
    """
    log = logging.getLogger(__name__)

    # Determine if lower or higher is better for this metric
    bold_direction = determine_bold_direction(metric)
    log.info(f"Metric '{metric}': boldface direction = {bold_direction}")

    # Extract unique tasks, algorithms, budgets
    all_tasks = sorted(results.keys())
    if tasks:
        all_tasks = [t for t in all_tasks if t in tasks]

    all_algorithms = set()
    all_budgets = set()
    for task_name in all_tasks:
        df = results[task_name]
        all_algorithms.update(df["algorithm"].unique())
        all_budgets.update(df["num_simulations"].unique())

    # Filter algorithms and budgets
    if algorithms:
        all_algorithms = [a for a in sorted(all_algorithms) if a in algorithms]
    else:
        all_algorithms = sorted(all_algorithms)

    if budgets:
        all_budgets = [b for b in sorted(all_budgets) if b in budgets]
    else:
        all_budgets = sorted(all_budgets)

    log.info(f"Tasks: {all_tasks}")
    log.info(f"Algorithms: {all_algorithms}")
    log.info(f"Budgets: {all_budgets}")

    # Build LaTeX table
    lines = []

    # Header comment
    lines.append("% LaTeX table generated by generate_efficiency_table.py")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Table environment
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")

    # Caption
    if caption is None:
        caption = f"Results for metric: {escape_latex(metric)}"
    lines.append(f"\\caption{{{caption}}}")

    # Label
    if label is None:
        label = f"table:{metric.replace('_', '_')}"
    lines.append(f"\\label{{{label}}}")

    # Column specification: first column + one per algorithm
    n_cols = len(all_algorithms)
    col_spec = "l|" + "c|" * (n_cols - 1) + "c"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header row
    lines.append("\\toprule")
    header_parts = ["n\\_simulations"] + [
        escape_latex(algo.upper()) for algo in all_algorithms
    ]
    lines.append(" & ".join(header_parts) + " \\\\")

    # Process each task
    for _, task_name in enumerate(all_tasks):
        df = results[task_name]

        # Task section header
        lines.append("\\midrule")
        task_display = escape_latex(task_name.replace("_", " ").title())
        lines.append(f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textbf{{{task_display}}}}} \\\\")
        lines.append("\\midrule")

        # Process each budget
        for budget in all_budgets:
            row_parts = [format_number(budget)]

            # Compute statistics for each algorithm
            algo_stats = {}
            for algo in all_algorithms:
                # Filter data for this task, algorithm, and budget
                mask = (
                    (df["algorithm"] == algo)
                    & (df["num_simulations"] == budget)
                    & (df[metric].notna())
                )
                values = df[mask][metric]

                if len(values) > 0:
                    mean, lower_ci, upper_ci = compute_statistics(values)
                    algo_stats[algo] = (mean, lower_ci, upper_ci)
                else:
                    algo_stats[algo] = None

            # Determine best algorithm for this budget
            valid_algos = {k: v for k, v in algo_stats.items() if v is not None}
            if valid_algos:
                if bold_direction == "min":
                    best_algo = min(valid_algos.keys(), key=lambda k: valid_algos[k][0])
                else:
                    best_algo = max(valid_algos.keys(), key=lambda k: valid_algos[k][0])
            else:
                best_algo = None

            # Format cells for each algorithm
            for algo in all_algorithms:
                if algo_stats[algo] is not None:
                    mean, lower_ci, upper_ci = algo_stats[algo]
                    is_best = algo == best_algo
                    cell = format_cell(mean, lower_ci, upper_ci, is_best)
                    row_parts.append(cell)
                else:
                    row_parts.append("--")

            lines.append(" & ".join(row_parts) + " \\\\")

    # Footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from hierarchical benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="results",
        help="Directory containing benchmark CSV files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output LaTeX file (.tex format)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="lc2st_test_statistic",
        help="Metric to display in table",
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=1,
        help="Scaling factor for adjusting num_simulations",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="List of tasks to include (default: all)",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=None,
        help="List of algorithms to include (default: all)",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="List of simulation budgets to include (default: all)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Custom caption for the table",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Custom LaTeX label for the table",
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
    log.info("LaTeX Table Generator for Hierarchical Benchmarks")
    log.info("=" * 80)
    log.info(f"Input directory: {args.input_dir}")
    log.info(f"Metric: {args.metric}")
    log.info(f"Output path: {args.output_path}")
    log.info(f"n_l scaling factor: {args.n_l}")
    log.info("=" * 80)

    # Validate output path
    output_path = Path(args.output_path)
    if output_path.suffix.lower() != ".tex":
        raise ValueError(f"Output path must end with .tex, got: {output_path.suffix}")

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

    # Generate LaTeX table
    log.info("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(
        results=results,
        metric=args.metric,
        tasks=args.tasks,
        algorithms=args.algorithms,
        budgets=args.budgets,
        caption=args.caption,
        label=args.label,
    )

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    log.info(f"Writing table to {output_path}")
    with open(output_path, "w") as f:
        f.write(latex_table)

    log.info("=" * 80)
    log.info("Table generation completed successfully!")
    log.info(f"Output written to: {output_path}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

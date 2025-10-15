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

NPE_SCALED_ALGORITHMS = {"snpe", "snpe_5r", "fmpe", "fmpe_transformer", "simformer"}

# Label map for algorithm display names (mirrors plot_hierarchical_benchmark.py)
ALGO_LABEL_MAP = {
    "snpe": "NPE",
    "snpe_5r": "SNPE (5 rounds)",
    "fmpe": "FMPE",
    "fmpe_transformer": "FMPE Transformer",
    "simformer": "Simformer",
    "bottom_up": "LF",
    "deepset": "PF",
}


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
            npe_mask = df["algorithm"].isin(NPE_SCALED_ALGORITHMS)
            deepset_mask = df["algorithm"] == "deepset"

            if npe_mask.any():
                df.loc[npe_mask, "num_simulations"] *= n_l
                scaled_algorithms = sorted(df.loc[npe_mask, "algorithm"].unique())
                log.info(
                    f"  Scaled {', '.join(scaled_algorithms)} num_simulations by {n_l}"
                )

            if deepset_mask.any():
                deepset_scale = (n_l + 1) // 2
                df.loc[deepset_mask, "num_simulations"] *= deepset_scale
                log.info(f"  Scaled DeepSet num_simulations by {deepset_scale}")

    return results


def load_n_l_scaling_results(input_dir: Path, n_l_values: Optional[list[int]] = None) -> dict:
    """Load n_l scaling benchmark results for all hierarchical tasks.

    Args:
        input_dir: Directory containing n_l scaling CSV files
        n_l_values: Optional list of n_l values to include (None = all)

    Returns:
        Dict mapping task_name -> DataFrame with all results
    """
    log = logging.getLogger(__name__)

    pattern = "hierarchical_*.csv"
    csv_files = list(input_dir.glob(pattern))

    if not csv_files:
        raise ValueError(
            f"No CSV files found matching pattern '{pattern}' in {input_dir}"
        )

    log.info(f"Found {len(csv_files)} n_l scaling result files")

    results = {}
    for csv_file in csv_files:
        log.debug(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Verify n_l column exists
        if "n_l" not in df.columns:
            log.warning(f"Skipping {csv_file.name}: missing 'n_l' column")
            continue

        # Filter by n_l values if specified
        if n_l_values is not None:
            df = df[df["n_l"].isin(n_l_values)]
            if len(df) == 0:
                continue

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
        npe_mask = df["algorithm"].isin(NPE_SCALED_ALGORITHMS)
        deepset_mask = df["algorithm"] == "deepset"

        if npe_mask.any():
            # NPE-style methods: scale by n_l
            df.loc[npe_mask, "num_simulations"] *= df.loc[npe_mask, "n_l"].astype(int)
            scaled_algorithms = sorted(df.loc[npe_mask, "algorithm"].unique())
            log.info(
                "  Scaled "
                f"{', '.join(scaled_algorithms)} num_simulations by n_l"
            )

        if deepset_mask.any():
            # DeepSet: scale by (n_l + 1) // 2
            deepset_scale = (df.loc[deepset_mask, "n_l"].astype(int) + 1) // 2
            df.loc[deepset_mask, "num_simulations"] *= deepset_scale
            log.info("  Scaled DeepSet num_simulations by (n_l + 1) // 2")

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
    """Format a table cell with mean and confidence interval in floating point notation.

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
        cell_text = f"{mean:.2e}"
    else:
        margin = upper_ci - mean
        cell_text = f"{mean:.2e} $\\pm$ {margin:.2e}"

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
    x_column: str = "num_simulations",
    x_label: Optional[str] = None,
) -> str:
    """Generate a LaTeX tabular block from benchmark results.

    Emits only the ``tabular`` environment (plus a header comment). The
    surrounding ``table`` environment, caption, and label are owned by the
    consuming LaTeX document.
    """
    log = logging.getLogger(__name__)

    # Determine if lower or higher is better for this metric
    bold_direction = determine_bold_direction(metric)
    log.info(f"Metric '{metric}': boldface direction = {bold_direction}")

    # Extract unique tasks, algorithms, x_values
    all_tasks = sorted(results.keys())
    if tasks:
        all_tasks = [t for t in all_tasks if t in tasks]

    all_algorithms = set()
    all_x_values = set()
    for task_name in all_tasks:
        df = results[task_name]
        all_algorithms.update(df["algorithm"].unique())
        all_x_values.update(df[x_column].unique())

    # Filter algorithms
    if algorithms:
        all_algorithms = [a for a in sorted(all_algorithms) if a in algorithms]
    else:
        all_algorithms = sorted(all_algorithms)

    # Filter x_values (budgets parameter for backwards compatibility)
    if budgets:
        all_x_values = [x for x in sorted(all_x_values) if x in budgets]
    else:
        all_x_values = sorted(all_x_values)

    log.info(f"Tasks: {all_tasks}")
    log.info(f"Algorithms: {all_algorithms}")
    log.info(f"{x_column} values: {all_x_values}")

    # Build LaTeX table
    lines = []

    # Header comment
    lines.append("% LaTeX table generated by generate_efficiency_table.py")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Format x_value for headers
    def format_x_value(x_value):
        if x_column == "num_simulations":
            return format_number(int(x_value))
        return str(int(x_value))

    # Column specification: algorithm label column + one per x_value
    n_cols = len(all_x_values)
    col_spec = "l|" + "c|" * (n_cols - 1) + "c"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header row: algorithm col + "<x_label> = <value>" per numeric column
    lines.append("\\toprule")
    if x_label is None:
        x_label = escape_latex(x_column.replace("_", "\\_"))
    header_parts = ["Method"] + [
        f"{x_label} = {format_x_value(x)}" for x in all_x_values
    ]
    lines.append(" & ".join(header_parts) + " \\\\")

    # Process each task
    for _, task_name in enumerate(all_tasks):
        df = results[task_name]

        # Task section header (spans algorithm col + x_value cols)
        lines.append("\\midrule")
        task_display = task_name.replace("_", " ").title()
        task_display = task_display.replace("Sir", "SIR").replace("Slcp", "SLCP")
        task_display = escape_latex(task_display)
        lines.append(
            f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textbf{{{task_display}}}}} \\\\"
        )
        lines.append("\\midrule")

        # Precompute stats for every (algorithm, x_value) cell so we can pick
        # the best algorithm per column for bolding.
        stats = {algo: {} for algo in all_algorithms}
        for algo in all_algorithms:
            for x_value in all_x_values:
                mask = (
                    (df["algorithm"] == algo)
                    & (df[x_column] == x_value)
                    & (df[metric].notna())
                )
                values = df[mask][metric]
                if len(values) > 0:
                    stats[algo][x_value] = compute_statistics(values)
                else:
                    stats[algo][x_value] = None

        # Best algorithm per x_value (column-wise bolding)
        best_per_x = {}
        for x_value in all_x_values:
            valid = {a: stats[a][x_value] for a in all_algorithms if stats[a][x_value] is not None}
            if not valid:
                best_per_x[x_value] = None
                continue
            if bold_direction == "min":
                best_per_x[x_value] = min(valid, key=lambda k: valid[k][0])
            else:
                best_per_x[x_value] = max(valid, key=lambda k: valid[k][0])

        # One row per algorithm
        for algo in all_algorithms:
            algo_label = escape_latex(ALGO_LABEL_MAP.get(algo, algo.upper()))
            row_parts = [algo_label]
            for x_value in all_x_values:
                cell_stats = stats[algo][x_value]
                if cell_stats is None:
                    row_parts.append("--")
                    continue
                mean, lower_ci, upper_ci = cell_stats
                is_best = algo == best_per_x[x_value]
                row_parts.append(format_cell(mean, lower_ci, upper_ci, is_best))
            lines.append(" & ".join(row_parts) + " \\\\")

    # Footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
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
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--n_l_mode",
        action="store_true",
        help="Enable n_l scaling mode (uses n_l as row index)",
    )
    parser.add_argument(
        "--n_l_values",
        type=int,
        nargs="+",
        default=None,
        help="List of n_l values to include (default: all)",
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
    if args.n_l_mode:
        log.info("Mode: n_l scaling")
        if args.n_l_values:
            log.info(f"n_l values filter: {args.n_l_values}")
    else:
        log.info(f"n_l scaling factor: {args.n_l}")
    log.info("=" * 80)

    # Validate output path
    output_path = Path(args.output_path)
    if output_path.suffix.lower() != ".tex":
        raise ValueError(f"Output path must end with .tex, got: {output_path.suffix}")

    # Load results based on mode
    if args.n_l_mode:
        results = load_n_l_scaling_results(
            input_dir=Path(args.input_dir),
            n_l_values=args.n_l_values,
        )
        x_column = "n_l"
        x_label = "$n_s$"
    else:
        results = load_all_results(input_dir=Path(args.input_dir), n_l=args.n_l)
        x_column = "num_simulations"
        x_label = "N"

    # Print summary statistics
    log.info("\nSummary Statistics:")
    log.info(f"  Tasks: {list(results.keys())}")
    for task_name, df in results.items():
        log.info(f"  {task_name}:")
        log.info(f"    Algorithms: {df['algorithm'].unique().tolist()}")
        if args.n_l_mode:
            log.info(f"    n_l values: {sorted(df['n_l'].unique().tolist())}")
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
        x_column=x_column,
        x_label=x_label,
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

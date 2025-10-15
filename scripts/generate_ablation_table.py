#!/usr/bin/env python
"""
LaTeX table generation script for ablation study results.

This script aggregates CSV results from ablation experiments and creates
publication-quality LaTeX tables with proper statistical formatting.

Ablation variants are parsed from filenames using the token after
``_bottom_up`` in the CSV filename:

- no token → TFMPE (baseline)
- ``_direct_`` → Direct
- ``_joint_`` → Joint
- ``_mlp_`` → MLP
- ``_linear_`` → Linear
- ``_no_grouping_`` → No Grouping

Example usage:
    python scripts/generate_ablation_table.py \
        --input_dir /mnt/gc1610/home/ablation_results_4 \
        --output_path ablation_table.tex \
        --metric lc2st_test_statistic
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ablation tokens in filenames and their display labels
ABLATION_TOKENS = {
    "_direct_": "Direct",
    "_joint_": "Joint",
    "_mlp_": "MLP",
    "_linear_": "Linear",
    "_no_grouping_": "No Grouping",
}

ALL_METHODS = ["TFMPE", "Direct", "Joint", "MLP", "Linear", "No Grouping"]


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the table generation script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
        # Only match tokens after "_bottom_up" to avoid false matches
        # in task names (e.g. "gaussian_linear" contains "_linear_")
        fname = csv_file.name
        method = "TFMPE"  # default: no ablation token
        bu_idx = fname.find("_bottom_up")
        if bu_idx != -1:
            suffix = fname[bu_idx + len("_bottom_up") :]
            for token, label in ABLATION_TOKENS.items():
                if token in suffix:
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
        log.info(f"Loaded {len(results[task_name])} total runs for '{task_name}'")

    return results


def compute_statistics(values: pd.Series) -> tuple[float, float, float]:
    """Compute mean and 95% confidence interval.

    Args:
        values: Series of metric values from multiple runs

    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    mean = values.mean()
    n = len(values)

    if n == 1:
        return mean, mean, mean

    std = values.std()
    margin = 1.96 * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


def format_cell(mean: float, lower_ci: float, upper_ci: float) -> str:
    """Format a table cell with mean and confidence interval.

    Args:
        mean: Mean value
        lower_ci: Lower bound of 95% CI
        upper_ci: Upper bound of 95% CI

    Returns:
        Formatted LaTeX string
    """
    if lower_ci == mean and upper_ci == mean:
        return f"{mean:.2e}"
    return f"{mean:.2e} [{lower_ci:.2e}, {upper_ci:.2e}]"


def format_number(num: int) -> str:
    """Format large numbers with comma separators."""
    return f"{num:,}"


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    return text.replace("_", "\\_")


def generate_ablation_table(
    results: dict[str, pd.DataFrame],
    metric: str,
    tasks: Optional[list[str]] = None,
    methods: Optional[list[str]] = None,
    budgets: Optional[list[int]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """Generate LaTeX table from ablation results.

    Args:
        results: Dict mapping task_name -> DataFrame with results
        metric: Name of metric column to display
        tasks: List of tasks to include (None = all)
        methods: List of methods to include (None = all from ALL_METHODS)
        budgets: List of simulation budgets to include (None = all)
        caption: Custom caption (None = auto-generate)
        label: Custom LaTeX label (None = auto-generate)

    Returns:
        Complete LaTeX table as string
    """
    log = logging.getLogger(__name__)

    # Determine tasks
    all_tasks = sorted(results.keys())
    if tasks:
        all_tasks = [t for t in all_tasks if t in tasks]

    # Determine methods present in data
    all_methods_in_data = set()
    all_budgets = set()
    for task_name in all_tasks:
        df = results[task_name]
        all_methods_in_data.update(df["method"].unique())
        all_budgets.update(df["num_simulations"].unique())

    # Use requested methods (preserving order), falling back to ALL_METHODS
    if methods:
        selected_methods = [m for m in methods if m in all_methods_in_data]
    else:
        selected_methods = [m for m in ALL_METHODS if m in all_methods_in_data]

    # Filter budgets
    if budgets:
        selected_budgets = sorted(b for b in all_budgets if b in budgets)
    else:
        selected_budgets = sorted(all_budgets)

    log.info(f"Tasks: {all_tasks}")
    log.info(f"Methods: {selected_methods}")
    log.info(f"Budgets: {selected_budgets}")

    # Build LaTeX table
    lines = []

    lines.append("% LaTeX table generated by generate_ablation_table.py")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")

    if caption is None:
        caption = f"Ablation study results for metric: {escape_latex(metric)}"
    lines.append(f"\\caption{{{caption}}}")

    if label is None:
        label = f"table:ablation_{metric}"
    lines.append(f"\\label{{{label}}}")

    n_cols = len(selected_methods)
    col_spec = "l|" + "c|" * (n_cols - 1) + "c"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    lines.append(r"\toprule")
    header = [r"n\_simulations"] + [m.upper() for m in selected_methods]
    lines.append(" & ".join(header) + r" \\")

    # Process each task
    for task_name in all_tasks:
        df = results[task_name]

        lines.append(r"\midrule")
        task_display = task_name.replace("_", " ").title()
        lines.append(
            f"\\multicolumn{{{n_cols + 1}}}{{l}}"
            f"{{\\textbf{{{task_display}}}}} \\\\"
        )
        lines.append(r"\midrule")

        for budget in selected_budgets:
            row_parts = [format_number(int(budget))]

            for method in selected_methods:
                mask = (
                    (df["method"] == method)
                    & (df["num_simulations"] == budget)
                    & (df[metric].notna())
                )
                values = df[mask][metric]

                if len(values) > 0:
                    mean, lower_ci, upper_ci = compute_statistics(values)
                    row_parts.append(format_cell(mean, lower_ci, upper_ci))
                else:
                    row_parts.append("--")

            lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from ablation study results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing ablation CSV files (flat layout)",
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
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="List of tasks to include (default: all)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="List of ablation methods to include (default: all)",
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

    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("LaTeX Table Generator for Ablation Study")
    log.info("=" * 80)
    log.info(f"Input directory: {args.input_dir}")
    log.info(f"Metric: {args.metric}")
    log.info(f"Output path: {args.output_path}")
    log.info("=" * 80)

    # Validate output path
    output_path = Path(args.output_path)
    if output_path.suffix.lower() != ".tex":
        raise ValueError(f"Output path must end with .tex, got: {output_path.suffix}")

    # Load results
    results = load_ablation_results(input_dir=Path(args.input_dir))

    # Print summary
    log.info("\nSummary Statistics:")
    log.info(f"  Tasks: {list(results.keys())}")
    for task_name, df in sorted(results.items()):
        log.info(f"  {task_name}:")
        for method in sorted(df["method"].unique()):
            method_df = df[df["method"] == method]
            task_budgets = sorted(method_df["num_simulations"].unique().tolist())
            log.info(f"    {method}: {len(method_df)} runs, budgets={task_budgets}")

    # Generate table
    log.info("\nGenerating LaTeX table...")
    latex_table = generate_ablation_table(
        results=results,
        metric=args.metric,
        tasks=args.tasks,
        methods=args.methods,
        budgets=args.budgets,
        caption=args.caption,
        label=args.label,
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Writing table to {output_path}")
    with open(output_path, "w") as f:
        f.write(latex_table)

    log.info("=" * 80)
    log.info("Table generation completed successfully!")
    log.info(f"Output written to: {output_path}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

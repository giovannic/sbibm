#!/usr/bin/env python
"""
CLI script for visualizing hierarchical task posteriors.

This script trains an algorithm on a hierarchical task and visualizes
the resulting posterior distribution using custom parameter labels that
indicate global vs local parameter structure.

Currently only supports hierarchical_two_moons.

Example usage:
    python scripts/visualize_hierarchical_posterior.py \
        --task hierarchical_two_moons \
        --algorithm snpe \
        --num_simulations 5000 \
        --num_observation 1 \
        --n_l 5 \
        --output_path results/hierarchical_posterior.png\
        --max_local_contexts 2
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import sbibm


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the visualization script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_hierarchical_two_moons_labels(
    n_l: int,
    max_local_contexts: int,
) -> List[str]:
    """Generate custom parameter labels for hierarchical_two_moons.

    Args:
        n_l: Number of local contexts
        max_local_contexts: Max local contexts to show in visualization

    Returns:
        List of parameter labels (e.g., ["global_loc₀", "local₁_θ₀", ...])
    """
    numbers_unicode = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    labels = []

    # Global: 4 pooling params (2 locs, 2 scales)
    labels.extend(
        [
            f"global_loc{numbers_unicode[0]}",
            f"global_loc{numbers_unicode[1]}",
            f"global_scale{numbers_unicode[0]}",
            f"global_scale{numbers_unicode[1]}",
        ]
    )

    # Local: 2 params per context
    num_local_to_show = min(max_local_contexts, n_l)
    for ctx in range(num_local_to_show):
        labels.extend(
            [
                f"local{numbers_unicode[ctx]}_θ{numbers_unicode[0]}",
                f"local{numbers_unicode[ctx]}_θ{numbers_unicode[1]}",
            ]
        )

    return labels


def visualize_posterior(
    task_name: str,
    algorithm: str,
    num_simulations: int,
    num_observation: int,
    n_l: int,
    output_path: Path,
    num_samples: int = 1000,
    max_local_contexts: int = 2,
    config: str = "manuscript",
    seed: Optional[int] = None,
    **algorithm_kwargs,
) -> None:
    """Train algorithm and visualize hierarchical posterior.

    Args:
        task_name: Name of the hierarchical task (must be hierarchical_two_moons)
        algorithm: Algorithm name (one of: "snpe", "snle", "snre")
        num_simulations: Simulation budget
        num_observation: Observation number
        n_l: Number of local contexts for hierarchical task
        output_path: Path to save figure
        num_samples: Number of samples to plot
        max_local_contexts: Max local contexts to show (to avoid clutter)
        config: Config for styling (default: "manuscript")
        seed: Random seed for reproducibility
        **algorithm_kwargs: Additional kwargs for the algorithm
    """
    log = logging.getLogger(__name__)

    # Validate task name
    if task_name != "hierarchical_two_moons":
        raise ValueError(
            f"Task '{task_name}' not supported. "
            f"Only 'hierarchical_two_moons' is currently implemented."
        )

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        log.info(f"Set random seed to {seed}")

    # Load task
    log.info(f"Loading task: {task_name} with n_l={n_l}")
    task = sbibm.get_task(task_name, n_l=n_l)

    # Import algorithm
    log.info(f"Importing algorithm: {algorithm}")
    if algorithm == "snpe":
        from sbibm.algorithms.sbi.snpe import run as run_algorithm
    elif algorithm == "snle":
        from sbibm.algorithms.sbi.snle import run as run_algorithm
    elif algorithm == "snre":
        from sbibm.algorithms.sbi.snre import run as run_algorithm
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. " f"Choose from: snpe, snle, snre"
        )

    # Run algorithm
    log.info(
        f"Running {algorithm} with {num_simulations} simulations "
        f"on observation {num_observation}"
    )
    samples, actual_num_sims, log_prob_true, posterior = run_algorithm(
        task=task,
        num_samples=num_samples,
        num_simulations=num_simulations,
        num_observation=num_observation,
        automatic_transforms_enabled=True,
        **algorithm_kwargs,
    )

    log.info(f"Algorithm completed ({actual_num_sims} simulations)")

    # Generate custom labels for hierarchical_two_moons
    log.info("Generating custom parameter labels")
    labels = generate_hierarchical_two_moons_labels(
        n_l=n_l,
        max_local_contexts=max_local_contexts,
    )

    # For hierarchical_two_moons:
    # - 4 global params (2 locs, 2 scales)
    # - 2 params per local context
    num_global = 4
    num_local_per_ctx = 2
    num_local_to_show = min(max_local_contexts, n_l)
    num_dims_to_plot = num_global + num_local_per_ctx * num_local_to_show

    log.info(
        f"Plotting {num_dims_to_plot} dimensions: "
        f"{num_global} global + "
        f"{num_local_per_ctx * num_local_to_show} local "
        f"({num_local_to_show} contexts)"
    )

    # Slice samples to dimensions we want to plot
    samples_sliced = samples[:, :num_dims_to_plot].numpy()

    # Get true parameters if available
    try:
        true_params = (
            task.get_true_parameters(num_observation=num_observation)
            .numpy()
            .flatten()[:num_dims_to_plot]
        )
        has_true_params = True
    except Exception:
        has_true_params = False
        log.warning("True parameters not available")

    # Create visualization
    log.info("Creating posterior visualization with matplotlib")
    fig, axes = plt.subplots(
        num_dims_to_plot,
        num_dims_to_plot,
        figsize=(2 * num_dims_to_plot, 2 * num_dims_to_plot),
    )

    # Plot pairwise comparisons
    for i in range(num_dims_to_plot):
        for j in range(num_dims_to_plot):
            ax = axes[i, j] if num_dims_to_plot > 1 else axes

            if i == j:
                # Diagonal: histogram
                ax.hist(
                    samples_sliced[:, i],
                    bins=30,
                    color="#0035FD",
                    alpha=0.6,
                    density=True,
                )
                if has_true_params:
                    ax.axvline(
                        true_params[i],
                        color="#f92700",
                        linestyle="--",
                        linewidth=2,
                        label="True",
                    )
                ax.set_yticks([])
            elif i > j:
                # Lower triangle: scatter plot
                ax.scatter(
                    samples_sliced[:, j],
                    samples_sliced[:, i],
                    s=1,
                    alpha=0.3,
                    color="#0035FD",
                )
                if has_true_params:
                    ax.scatter(
                        true_params[j],
                        true_params[i],
                        s=50,
                        color="#f92700",
                        marker="x",
                        linewidths=2,
                    )
            else:
                # Upper triangle: hide
                ax.axis("off")

            # Add labels on edges
            if i == num_dims_to_plot - 1:
                ax.set_xlabel(labels[j], fontsize=10)
            else:
                ax.set_xticks([])

            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=10)
            else:
                ax.set_yticks([])

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving figure to {output_path}")
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Visualization complete!")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize hierarchical_two_moons posterior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="hierarchical_two_moons",
        help="Task name (currently only hierarchical_two_moons supported)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="snpe",
        choices=["snpe", "snle", "snre"],
        help="Algorithm to use",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=5000,
        help="Simulation budget",
    )
    parser.add_argument(
        "--num_observation",
        type=int,
        default=1,
        help="Observation number",
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=5,
        help="Number of local contexts",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/hierarchical_posterior.png",
        help="Path to save figure",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to plot",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="manuscript",
        help="Config for styling (manuscript or streamlit)",
    )
    parser.add_argument(
        "--max_local_contexts",
        type=int,
        default=2,
        help="Max local contexts to show (to avoid clutter)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of rounds for sequential algorithms (SNPE/SNLE/SNRE)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
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
    log.info("Hierarchical Posterior Visualization")
    log.info("=" * 80)
    log.info(f"Task: {args.task}")
    log.info(f"Algorithm: {args.algorithm}")
    log.info(f"Simulations: {args.num_simulations}")
    log.info(f"Observation: {args.num_observation}")
    log.info(f"Local contexts (n_l): {args.n_l}")
    log.info(f"Max contexts to show: {args.max_local_contexts}")
    log.info(f"Output path: {args.output_path}")
    log.info(f"Seed: {args.seed}")
    log.info("=" * 80)

    # Visualize posterior
    visualize_posterior(
        task_name=args.task,
        algorithm=args.algorithm,
        num_simulations=args.num_simulations,
        num_observation=args.num_observation,
        n_l=args.n_l,
        output_path=Path(args.output_path),
        num_samples=args.num_samples,
        max_local_contexts=args.max_local_contexts,
        config=args.config,
        seed=args.seed,
        num_rounds=args.num_rounds,
    )

    log.info("=" * 80)
    log.info("Visualization completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

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
from typing import Optional

import torch

import sbibm
from sbibm.visualisation import (
    generate_hierarchical_labels,
    plot_hierarchical_posterior,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the visualization script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        log.info(f"Set random seed to {seed}")

    # Load task
    log.info(f"Loading task: {task_name} with n_l={n_l}")
    task = sbibm.get_task(
        task_name,
        n_l=n_l,
        device=algorithm_kwargs['device']
    )

    # Import algorithm
    log.info(f"Importing algorithm: {algorithm}")
    if algorithm == "snpe":
        from sbibm.algorithms.sbi.snpe import run as run_algorithm
    elif algorithm == "snle":
        from sbibm.algorithms.sbi.snle import run as run_algorithm
    elif algorithm == "snre":
        from sbibm.algorithms.sbi.snre import run as run_algorithm
    elif algorithm == "deepset":
        from sbibm.algorithms.deepset import run as run_algorithm
    elif algorithm == "bottom_up":
        from sbibm.algorithms.tfmpe.bottom_up import run as run_algorithm
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Choose from: snpe, snle, snre, deepset, bottom_up"
        )

    # Run algorithm
    log.info(
        f"Running {algorithm} with {num_simulations} simulations "
        f"on observation {num_observation}"
    )

    # Only pass automatic_transforms_enabled to algorithms that support it
    if algorithm in ["snpe", "snle", "snre", "bottom_up"]:
        algorithm_kwargs["automatic_transforms_enabled"] = True

    samples, actual_num_sims, log_prob_true, posterior = run_algorithm(
        task=task,
        num_samples=num_samples,
        num_simulations=num_simulations,
        num_observation=num_observation,
        **algorithm_kwargs,
    )

    log.info(f"Algorithm completed ({actual_num_sims} simulations)")

    # Generate custom labels for hierarchical task
    log.info("Generating custom parameter labels")
    labels = generate_hierarchical_labels(
        task=task,
        max_local_contexts=max_local_contexts,
    )

    # Get dimensions from task.prior_dist
    num_global = task.prior_dist.dim_global
    num_local_per_ctx = task.prior_dist.dim_local
    num_local_to_show = min(max_local_contexts, n_l)
    num_dims_to_plot = num_global + num_local_per_ctx * num_local_to_show

    log.info(
        f"Plotting {num_dims_to_plot} dimensions: "
        f"{num_global} global + "
        f"{num_local_per_ctx * num_local_to_show} local "
        f"({num_local_to_show} contexts)"
    )

    # Slice samples to dimensions we want to plot
    samples_sliced = samples[:, :num_dims_to_plot]

    # Get true parameters if available
    try:
        true_params = (
            task.get_true_parameters(num_observation=num_observation)
            .numpy()
            .flatten()[:num_dims_to_plot]
        )
    except Exception:
        true_params = None
        log.warning("True parameters not available")

    # Create visualization
    log.info("Creating posterior visualization")
    log.info(f"Saving figure to {output_path}")
    plot_hierarchical_posterior(
        samples=samples_sliced,
        labels=labels,
        output_path=Path(output_path),
        true_params=true_params,
    )
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
        choices=["snpe", "snle", "snre", "deepset", "bottom_up"],
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
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device",
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
        device=args.device
    )

    log.info("=" * 80)
    log.info("Visualization completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

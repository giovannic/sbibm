#!/usr/bin/env python
"""
CLI script for prior predictive checks on hierarchical tasks.

This script samples from the prior of a hierarchical task and visualizes
the prior distribution using custom parameter labels that indicate
global vs local parameter structure.

Currently only supports hierarchical_two_moons.

Example usage:
    python scripts/visualize_hierarchical_prior.py \
        --task hierarchical_two_moons \
        --num_samples 1000 \
        --num_observation 1 \
        --n_l 5 \
        --output_path results/hierarchical_prior.png \
        --max_local_contexts 2
"""
import argparse
import logging
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


def visualize_prior(
    task_name: str,
    n_l: int,
    output_path: Path,
    num_samples: int = 1000,
    num_observation: int = 1,
    max_local_contexts: int = 2,
    seed: Optional[int] = None,
) -> None:
    """Sample from prior and visualize the distribution.

    Args:
        task_name: Name of the hierarchical task (must be hierarchical_two_moons)
        n_l: Number of local contexts for hierarchical task
        output_path: Path to save figure
        num_samples: Number of samples to draw from prior
        num_observation: Observation number (for true parameter overlay)
        max_local_contexts: Max local contexts to show (to avoid clutter)
        seed: Random seed for reproducibility
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
    )

    # Get prior and sample
    log.info(f"Sampling {num_samples} parameters from prior")
    prior = task.get_prior()
    samples = prior(num_samples=num_samples)

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

    # Get true parameters
    true_params = (
        task.get_true_parameters(num_observation=num_observation)
        .numpy()
        .flatten()[:num_dims_to_plot]
    )

    # Create visualization
    log.info("Creating prior visualization")
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
        description="Prior predictive check for hierarchical tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="hierarchical_two_moons",
        help="Task name (currently only hierarchical_two_moons supported)",
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=5,
        help="Number of local contexts",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to draw from prior",
    )
    parser.add_argument(
        "--num_observation",
        type=int,
        default=1,
        help="Observation number (for true parameter overlay)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/hierarchical_prior.png",
        help="Path to save figure",
    )
    parser.add_argument(
        "--max_local_contexts",
        type=int,
        default=2,
        help="Max local contexts to show (to avoid clutter)",
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
    log.info("Hierarchical Prior Predictive Check")
    log.info("=" * 80)
    log.info(f"Task: {args.task}")
    log.info(f"Local contexts (n_l): {args.n_l}")
    log.info(f"Num samples: {args.num_samples}")
    log.info(f"Observation (for true params): {args.num_observation}")
    log.info(f"Max contexts to show: {args.max_local_contexts}")
    log.info(f"Output path: {args.output_path}")
    log.info(f"Seed: {args.seed}")
    log.info("=" * 80)

    # Visualize prior
    visualize_prior(
        task_name=args.task,
        n_l=args.n_l,
        output_path=Path(args.output_path),
        num_samples=args.num_samples,
        num_observation=args.num_observation,
        max_local_contexts=args.max_local_contexts,
        seed=args.seed,
    )

    log.info("=" * 80)
    log.info("Prior predictive check completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

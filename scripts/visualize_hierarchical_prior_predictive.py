#!/usr/bin/env python
"""
CLI script for prior predictive checks on hierarchical task observations.

This script samples parameters from the prior, runs the simulator to generate
observations, and visualizes the prior predictive distribution of observations.

Example usage:
    python scripts/visualize_hierarchical_prior_predictive.py \
        --task hierarchical_two_moons \
        --num_samples 1000 \
        --num_observation 1 \
        --n_l 5 \
        --output_path results/hierarchical_prior_predictive.png \
        --max_local_contexts 5
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

import sbibm
from sbibm.visualisation import (
    generate_observation_labels,
    plot_hierarchical_posterior,
    plot_observation_grid,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the visualization script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def visualize_prior_predictive(
    task_name: str,
    n_l: int,
    output_path: Path,
    num_samples: int = 1000,
    num_observation: int = 1,
    max_local_contexts: int = 5,
    seed: Optional[int] = None,
) -> None:
    """Sample from prior, run simulator, and visualize observations.

    Args:
        task_name: Name of the hierarchical task
        n_l: Number of local contexts for hierarchical task
        output_path: Path to save figure
        num_samples: Number of samples to draw from prior
        num_observation: Observation number (for true observation overlay)
        max_local_contexts: Max local contexts to show
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

    # Sample parameters from prior
    log.info(f"Sampling {num_samples} parameters from prior")
    prior = task.get_prior()
    parameters = prior(num_samples=num_samples)

    # Run simulator to generate observations
    log.info("Running simulator to generate observations")
    simulator = task.get_simulator()
    observations = simulator(parameters)

    # Auto-detect dim_per_context
    dim_per_context = task.dim_data // n_l
    log.info(f"Detected dim_per_context: {dim_per_context}")

    # Get true observation for overlay
    true_observation = (
        task.get_observation(num_observation=num_observation)
        .numpy()
        .flatten()
    )

    # Determine number of contexts to show
    num_contexts_to_show = min(max_local_contexts, n_l)
    num_dims_to_plot = num_contexts_to_show * dim_per_context

    log.info(
        f"Plotting {num_contexts_to_show} contexts "
        f"({num_dims_to_plot} observation dimensions)"
    )

    # Slice observations to dimensions we want to plot
    observations_sliced = observations[:, :num_dims_to_plot]
    true_obs_sliced = true_observation[:num_dims_to_plot]

    # Choose visualization based on dim_per_context
    if dim_per_context <= 2:
        # Use context grid for 1D or 2D observations per context
        log.info(
            f"Using {'2D scatter' if dim_per_context == 2 else '1D histogram'} "
            f"grid visualization"
        )
        plot_observation_grid(
            samples=observations_sliced,
            n_l=num_contexts_to_show,
            dim_per_context=dim_per_context,
            output_path=output_path,
            true_observation=true_obs_sliced,
            max_local_contexts=max_local_contexts,
        )
    else:
        # Fall back to pairplot for higher-dimensional observations
        log.info("Using pairplot visualization (dim_per_context > 2)")
        labels = generate_observation_labels(
            task=task,
            max_local_contexts=max_local_contexts,
        )
        plot_hierarchical_posterior(
            samples=observations_sliced,
            labels=labels,
            output_path=output_path,
            true_params=true_obs_sliced,
        )

    log.info(f"Saved figure to {output_path}")
    log.info("Prior predictive visualization complete!")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Prior predictive check for hierarchical task observations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="hierarchical_two_moons",
        help="Task name",
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
        help="Observation number (for true observation overlay)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/hierarchical_prior_predictive.png",
        help="Path to save figure",
    )
    parser.add_argument(
        "--max_local_contexts",
        type=int,
        default=5,
        help="Max local contexts to show",
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
    log.info("Hierarchical Prior Predictive Check (Observations)")
    log.info("=" * 80)
    log.info(f"Task: {args.task}")
    log.info(f"Local contexts (n_l): {args.n_l}")
    log.info(f"Num samples: {args.num_samples}")
    log.info(f"Observation (for true obs): {args.num_observation}")
    log.info(f"Max contexts to show: {args.max_local_contexts}")
    log.info(f"Output path: {args.output_path}")
    log.info(f"Seed: {args.seed}")
    log.info("=" * 80)

    # Visualize prior predictive
    visualize_prior_predictive(
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

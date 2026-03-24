#!/usr/bin/env python
"""
CLI script for running single hierarchical benchmark experiment.

This script is designed to be HPC-friendly, allowing parallel execution
of individual (algorithm, budget, observation) combinations. Results are
saved to CSV files that can be aggregated later.

Example usage:
    python scripts/run_hierarchical_benchmark.py \
        --task hierarchical_two_moons \
        --algorithm snpe \
        --num_simulations 5000 \
        --num_observation 1 \
        --output_dir results \
        --seed 42
"""
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

import sbibm
from sbibm.metrics.lc2st import lc2st
from sbibm.metrics.reverse_kl import reverse_kl
from sbibm.visualisation import (
    generate_hierarchical_labels,
    plot_hierarchical_posterior,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the benchmark script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_benchmark(
    task_name: str,
    algorithm: str,
    num_simulations: int,
    num_observation: int,
    output_dir: Path,
    seed: Optional[int] = None,
    num_samples: int = 1000,
    n_l: int = 5,
    plot_posterior: bool = False,
    max_local_contexts: int = 2,
    plot_output_dir: Optional[Path] = None,
    **algorithm_kwargs,
) -> Dict:
    """Run a single benchmark experiment.

    Args:
        task_name: Name of the task (e.g., "hierarchical_two_moons")
        algorithm: Algorithm name (one of: "snpe", "snle", "snre")
        num_simulations: Simulation budget
        num_observation: Observation number
        output_dir: Directory to save results
        seed: Random seed for reproducibility
        num_samples: Number of posterior samples for metrics
        n_l: Number of local contexts (for hierarchical tasks)
        plot_posterior: Whether to generate posterior visualization
        max_local_contexts: Max local contexts to show in visualization
        plot_output_dir: Directory for plots (uses output_dir if None)
        **algorithm_kwargs: Additional kwargs for the algorithm

    Returns:
        Dict with benchmark results including metrics
    """
    log = logging.getLogger(__name__)

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        log.info(f"Set random seed to {seed}")

    # Load task
    log.info(f"Loading task: {task_name} with n_l={n_l}")
    task = sbibm.get_task(task_name, n_l=n_l, device=algorithm_kwargs.get('device', 'cpu'))

    # Extract n_l for hierarchical tasks (reduce simulations for NPE-style methods)
    n_l = getattr(task, 'n_l', 1)
    adjusted_num_simulations = num_simulations
    npe_scaled_algorithms = {"snpe", "fmpe", "fmpe_transformer", "simformer"}
    if algorithm in npe_scaled_algorithms and n_l > 1:
        n_sim_per_sample = n_l
        adjusted_num_simulations = num_simulations // n_sim_per_sample
        log.info(
            f"Reducing {algorithm} simulations by factor of n_l={n_l}: "
            f"{num_simulations} -> {adjusted_num_simulations}"
        )
    if algorithm == "deepset" and n_l > 1:
        n_sim_per_sample = ((n_l + 1) // 2)
        adjusted_num_simulations = num_simulations // n_sim_per_sample
        log.info(
            f"Reducing DeepSet simulations to account for set sampling: "
            f"{num_simulations} -> {adjusted_num_simulations}"
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
    elif algorithm == "fmpe":
        from sbibm.algorithms.sbi.fmpe import run as run_algorithm
    elif algorithm == "fmpe_transformer":
        from sbibm.algorithms.sbi.fmpe import run as run_algorithm
        algorithm_kwargs["vf_estimator"] = "transformer"
    elif algorithm == "simformer":
        from sbibm.algorithms.simformer import run as run_algorithm
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Choose from: snpe, snle, snre, deepset, bottom_up, fmpe, fmpe_transformer, simformer"
        )

    # Run algorithm
    log.info(
        f"Running {algorithm} with {adjusted_num_simulations} simulations "
        f"on observation {num_observation}"
    )
    start_time = time.time()

    if algorithm == 'bottom_up':
        if task_name in {'hierarchical_sir', 'hierarchical_slcp', 'hierarchical_two_moons'}:
            algorithm_kwargs['sample_batch_size'] = 100
        else:
            algorithm_kwargs['sample_batch_size'] = 1_000

    samples, actual_num_sims, log_prob_true, posterior = run_algorithm(
        task=task,
        num_samples=num_samples,
        num_simulations=adjusted_num_simulations,
        num_observation=num_observation,
        **algorithm_kwargs,
    )

    elapsed_time = time.time() - start_time
    log.info(
        f"Algorithm completed in {elapsed_time:.2f}s "
        f"({actual_num_sims} simulations)"
    )

    # Compute metrics
    results = {
        "task": task_name,
        "algorithm": algorithm,
        "num_simulations": actual_num_sims,
        "num_observation": num_observation,
        "n_l": n_l,
        "elapsed_time": elapsed_time,
        "seed": seed,
    }

    # Reverse KL divergence
    log.info("Computing reverse KL divergence...")
    # Some algorithms (like bottom_up) may not provide log_prob
    if log_prob_true is not None:
        rkl = reverse_kl(
            posterior=posterior,
            task=task,
            num_observation=num_observation,
            num_samples=num_samples,
            device=algorithm_kwargs.get('device', 'cpu'),
        )
        results["reverse_kl"] = rkl.item()
        log.info(f"Reverse KL: {rkl.item():.6f}")
    else:
        results["reverse_kl"] = float("nan")
        log.warning("Algorithm does not provide log_prob, skipping reverse KL")

    # LC2ST
    log.info("Computing LC2ST...")
    lc2st_result = lc2st(
        posterior=posterior,
        task=task,
        num_observation=num_observation,
        num_calibration_samples=num_samples,
        posterior_samples=samples
    )
    # Extract key metrics from lc2st result dict
    results["lc2st_p_value"] = lc2st_result["p_value"].item()
    results["lc2st_test_statistic"] = lc2st_result["test_statistic"].item()
    results["lc2st_critical_value"] = lc2st_result["critical_value"].item()
    results["lc2st_reject"] = lc2st_result["reject"].item()
    log.info(
        f"LC2ST p-value: {results['lc2st_p_value']:.4f}, "
        f"critical_value: {results['lc2st_critical_value']:.4f}, "
        f"test_statistic: {results['lc2st_test_statistic']:.4f}, "
        f"critical_value: {results['lc2st_critical_value']:.4f}, "
        f"reject: {results['lc2st_reject']}"
    )

    # Generate posterior visualization if requested
    if plot_posterior:
        log.info("Generating posterior visualization...")

        # Generate hierarchical labels
        labels = generate_hierarchical_labels(
            task=task,
            max_local_contexts=max_local_contexts,
        )

        # Determine dimensions to plot
        num_global = task.prior_dist.dim_global
        num_local_per_ctx = task.prior_dist.dim_local
        num_local_to_show = min(max_local_contexts, n_l)
        num_dims_to_plot = (
            num_global + num_local_per_ctx * num_local_to_show
        )

        # Slice samples and true params
        samples_sliced = samples[:, :num_dims_to_plot]

        true_params = (
            task.get_true_parameters(num_observation=num_observation)
            .numpy()
            .flatten()[:num_dims_to_plot]
        )

        # Create plot output path
        plot_dir = plot_output_dir if plot_output_dir else output_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = (
            plot_dir
            / f"{task_name}_{algorithm}_"
            f"{actual_num_sims}_obs{num_observation}_posterior.png"
        )

        # Generate plot
        plot_hierarchical_posterior(
            samples=samples_sliced,
            labels=labels,
            output_path=plot_path,
            true_params=true_params,
        )

        log.info(f"Posterior plot saved to {plot_path}")

    return results


def save_results(
    results: Dict, output_dir: Path, task_name: str, algorithm: str
) -> None:
    """Save results to CSV file.

    Args:
        results: Dictionary of results
        output_dir: Directory to save results
        task_name: Task name for filename
        algorithm: Algorithm name for filename
    """
    log = logging.getLogger(__name__)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    output_file = (
        output_dir / f"{task_name}_{algorithm}_"
        f"{results['num_simulations']}_"
        f"{results['num_observation']}.csv"
    )

    # Convert to DataFrame
    df = pd.DataFrame([results])

    # Append if file exists, otherwise create new
    if output_file.exists():
        log.info(f"Appending to existing file: {output_file}")
        existing_df = pd.read_csv(output_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    else:
        log.info(f"Creating new file: {output_file}")

    df.to_csv(output_file, index=False)
    log.info(f"Results saved to {output_file}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run single hierarchical benchmark experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="hierarchical_two_moons",
        help="Task name",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["snpe", "snle", "snre", "deepset", "bottom_up", "fmpe", "fmpe_transformer", "simformer"],
        required=True,
        help="Algorithm to use",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        required=True,
        help="Simulation budget",
    )
    parser.add_argument(
        "--num_observation",
        type=int,
        default=1,
        help="Observation number",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of posterior samples for metrics",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of rounds for sequential algorithms (SNPE/SNLE/SNRE)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu, cuda, cuda:0, cuda:1, etc.)",
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=5,
        help="Number of local contexts (for hierarchical tasks)",
    )
    parser.add_argument(
        "--plot_posterior",
        action="store_true",
        help="Generate posterior visualization",
    )
    parser.add_argument(
        "--max_local_contexts",
        type=int,
        default=2,
        help="Max local contexts to show in visualization",
    )
    parser.add_argument(
        "--plot_output_dir",
        type=str,
        default=None,
        help="Directory for plots (uses output_dir if not specified)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--ablation",
        choices=['none', 'mlp', 'joint', 'linear', 'sequential', 'direct', 'pf', 'no_grouping'],
        default='none',
        help="Ablation experiment",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("Hierarchical Benchmark Runner")
    log.info("=" * 80)
    log.info(f"Task: {args.task}")
    log.info(f"Algorithm: {args.algorithm}")
    log.info(f"Simulations: {args.num_simulations}")
    log.info(f"Observation: {args.num_observation}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Seed: {args.seed}")
    log.info(f"Device: {args.device}")
    log.info(f"n_l (local contexts): {args.n_l}")
    log.info("=" * 80)

    # Run benchmark
    extra_kwargs = {}
    if args.algorithm == 'bottom_up':
        extra_kwargs['ablation'] = args.ablation
    results = run_benchmark(
        task_name=args.task,
        algorithm=args.algorithm,
        num_simulations=args.num_simulations,
        num_observation=args.num_observation,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        num_samples=args.num_samples,
        n_l=args.n_l,
        plot_posterior=args.plot_posterior,
        max_local_contexts=args.max_local_contexts,
        plot_output_dir=(
            Path(args.plot_output_dir) if args.plot_output_dir else None
        ),
        num_rounds=args.num_rounds,
        device=args.device,
        automatic_transforms_enabled=True,
        **extra_kwargs
    )

    if args.ablation != 'none':
        algo_name = f'{args.algorithm}_{args.ablation}'
    else:
        algo_name = args.algorithm

    # Save results
    save_results(
        results=results,
        output_dir=Path(args.output_dir),
        task_name=args.task,
        algorithm=algo_name,
    )

    log.info("=" * 80)
    log.info("Benchmark completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

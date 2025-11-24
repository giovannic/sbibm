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
    log.info(f"Loading task: {task_name}")
    task = sbibm.get_task(task_name)

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
            f"Choose from: snpe, snle, snre, deepset"
        )

    # Run algorithm
    log.info(
        f"Running {algorithm} with {num_simulations} simulations "
        f"on observation {num_observation}"
    )
    start_time = time.time()

    samples, actual_num_sims, log_prob_true, posterior = run_algorithm(
        task=task,
        num_samples=num_samples,
        num_simulations=num_simulations,
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
        num_posterior_samples=num_samples,
    )
    # Extract key metrics from lc2st result dict
    results["lc2st_p_value"] = lc2st_result["p_value"].item()
    results["lc2st_test_statistic"] = lc2st_result["test_statistic"].item()
    results["lc2st_reject"] = lc2st_result["reject"].item()
    log.info(
        f"LC2ST p-value: {results['lc2st_p_value']:.4f}, "
        f"test_statistic: {results['lc2st_test_statistic']:.4f}, "
        f"reject: {results['lc2st_reject']}"
    )

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
        choices=["snpe", "snle", "snre", "deepset", "bottom_up"],
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
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
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
    log.info("=" * 80)

    # Run benchmark
    results = run_benchmark(
        task_name=args.task,
        algorithm=args.algorithm,
        num_simulations=args.num_simulations,
        num_observation=args.num_observation,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        num_samples=args.num_samples,
        num_rounds=args.num_rounds,
        device=args.device,
        automatic_transforms_enabled=True,
    )

    # Save results
    save_results(
        results=results,
        output_dir=Path(args.output_dir),
        task_name=args.task,
        algorithm=args.algorithm,
    )

    log.info("=" * 80)
    log.info("Benchmark completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

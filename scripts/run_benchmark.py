#!/usr/bin/env python
"""
Hydra-based benchmark runner for hierarchical inference experiments.

Replaces the argparse-based run_hierarchical_benchmark.py with structured
Hydra configuration. Supports single runs and multirun sweeps.

Examples:
    # Single run
    python scripts/run_benchmark.py algorithm=snpe task=hierarchical_two_moons \
        num_simulations=1000 num_observation=1

    # Full benchmark sweep
    python scripts/run_benchmark.py +experiment=benchmark -m

    # Ablation sweep
    python scripts/run_benchmark.py +experiment=ablation -m

    # n_l scaling sweep
    python scripts/run_benchmark.py +experiment=n_l_scaling -m

    # HPC with SLURM
    python scripts/run_benchmark.py +experiment=benchmark hydra/launcher=submitit_slurm -m
"""
import importlib
import logging
import shutil
import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Keys in algorithm config that are not passed as kwargs to run()
ALGORITHM_META_KEYS = {
    "name", "_target_", "npe_scaled", "deepset_scaled", "ablation",
}


def resolve_algorithm_kwargs(cfg: DictConfig) -> dict:
    """Extract algorithm-specific kwargs from the algorithm config.

    Filters out meta-keys and returns only kwargs that the algorithm's
    run() function accepts.
    """
    kwargs = {}
    for k, v in cfg.algorithm.items():
        if k not in ALGORITHM_META_KEYS:
            kwargs[k] = v
    return kwargs


def compute_adjusted_simulations(cfg: DictConfig, n_l: int) -> int:
    """Adjust simulation budget based on algorithm type and n_l."""
    num_sims = cfg.num_simulations
    if cfg.algorithm.get("npe_scaled", False) and n_l > 1:
        return num_sims // n_l
    if cfg.algorithm.get("deepset_scaled", False) and n_l > 1:
        return num_sims // ((n_l + 1) // 2)
    return num_sims


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    import torch
    import sbibm
    from sbibm.metrics.lc2st import lc2st
    from sbibm.metrics.reverse_kl import reverse_kl

    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        log.info(f"Set random seed to {cfg.seed}")

    # Load task
    log.info(f"Loading task: {cfg.task.name} with n_l={cfg.n_l}")
    task = sbibm.get_task(
        cfg.task.name, n_l=cfg.n_l, device=cfg.device
    )
    n_l = getattr(task, "n_l", 1)

    # Adjust simulation budget
    adjusted_sims = compute_adjusted_simulations(cfg, n_l)
    if adjusted_sims != cfg.num_simulations:
        log.info(
            f"Adjusted simulations: {cfg.num_simulations} -> {adjusted_sims} "
            f"(n_l={n_l})"
        )

    # Import algorithm via _target_
    module_path, func_name = cfg.algorithm._target_.rsplit(".", 1)
    module = importlib.import_module(module_path)
    run_algorithm = getattr(module, func_name)

    # Build algorithm kwargs
    algo_kwargs = resolve_algorithm_kwargs(cfg)
    algo_kwargs["device"] = cfg.device
    algo_kwargs["automatic_transforms_enabled"] = cfg.automatic_transforms_enabled

    # Handle bottom_up specifics
    if cfg.algorithm.name == "bottom_up":
        algo_kwargs["sample_batch_size"] = cfg.task.bottom_up_sample_batch_size
        algo_kwargs["ablation"] = cfg.algorithm.ablation.variant

    # Run algorithm
    log.info(
        f"Running {cfg.algorithm.name} with {adjusted_sims} simulations "
        f"on observation {cfg.num_observation}"
    )
    start_time = time.time()

    samples, actual_num_sims, log_prob_true, posterior = run_algorithm(
        task=task,
        num_samples=cfg.num_samples,
        num_simulations=adjusted_sims,
        num_observation=cfg.num_observation,
        **algo_kwargs,
    )

    elapsed_time = time.time() - start_time
    log.info(
        f"Algorithm completed in {elapsed_time:.2f}s "
        f"({actual_num_sims} simulations)"
    )

    # Compute metrics
    results = {
        "task": cfg.task.name,
        "algorithm": cfg.algorithm.name,
        "num_simulations": actual_num_sims,
        "num_observation": cfg.num_observation,
        "n_l": n_l,
        "elapsed_time": elapsed_time,
        "seed": cfg.seed,
    }

    # Reverse KL divergence
    if log_prob_true is not None:
        log.info("Computing reverse KL divergence...")
        rkl = reverse_kl(
            posterior=posterior,
            task=task,
            num_observation=cfg.num_observation,
            num_samples=cfg.num_samples,
            device=cfg.device,
        )
        results["reverse_kl"] = rkl.item()
        log.info(f"Reverse KL: {rkl.item():.6f}")
    else:
        results["reverse_kl"] = float("nan")
        log.warning(
            "Algorithm does not provide log_prob, skipping reverse KL"
        )

    # LC2ST
    log.info("Computing LC2ST...")
    lc2st_result = lc2st(
        posterior=posterior,
        task=task,
        num_observation=cfg.num_observation,
        num_calibration_samples=cfg.num_samples,
        posterior_samples=samples,
    )
    results["lc2st_p_value"] = lc2st_result["p_value"].item()
    results["lc2st_test_statistic"] = lc2st_result["test_statistic"].item()
    results["lc2st_critical_value"] = lc2st_result["critical_value"].item()
    results["lc2st_reject"] = lc2st_result["reject"].item()
    log.info(
        f"LC2ST p-value: {results['lc2st_p_value']:.4f}, "
        f"test_statistic: {results['lc2st_test_statistic']:.4f}, "
        f"reject: {results['lc2st_reject']}"
    )

    # Posterior visualization
    if cfg.plot_posterior:
        _plot_posterior(cfg, task, samples, actual_num_sims, n_l)

    # Determine algorithm display name for CSV filename
    algo_name = cfg.algorithm.name
    if (
        cfg.algorithm.name == "bottom_up"
        and "ablation" in cfg.algorithm
        and cfg.algorithm.ablation.variant != "none"
    ):
        algo_name = f"bottom_up_{cfg.algorithm.ablation.variant}"

    # Save to Hydra output directory
    output_dir = Path.cwd()  # Hydra changes cwd to output dir
    csv_name = (
        f"{cfg.task.name}_{algo_name}_"
        f"{actual_num_sims}_{cfg.num_observation}.csv"
    )

    df = pd.DataFrame([results])
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)
    log.info(f"Results saved to {csv_path}")

    # Copy to aggregate directory for analysis scripts
    agg_dir = Path(cfg.aggregate_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_path = agg_dir / csv_name

    if agg_path.exists():
        log.info(f"Appending to existing aggregate file: {agg_path}")
        existing_df = pd.read_csv(agg_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(agg_path, index=False)
    log.info(f"Aggregated results: {agg_path}")


def _plot_posterior(cfg, task, samples, actual_num_sims, n_l):
    """Generate posterior visualization."""
    from sbibm.visualisation import (
        generate_hierarchical_labels,
        plot_hierarchical_posterior,
    )

    labels = generate_hierarchical_labels(
        task=task,
        max_local_contexts=cfg.max_local_contexts,
    )

    num_global = task.prior_dist.dim_global
    num_local_per_ctx = task.prior_dist.dim_local
    num_local_to_show = min(cfg.max_local_contexts, n_l)
    num_dims_to_plot = num_global + num_local_per_ctx * num_local_to_show

    samples_sliced = samples[:, :num_dims_to_plot]
    true_params = (
        task.get_true_parameters(num_observation=cfg.num_observation)
        .numpy()
        .flatten()[:num_dims_to_plot]
    )

    plot_dir = Path.cwd()
    plot_path = (
        plot_dir
        / f"{cfg.task.name}_{cfg.algorithm.name}_"
        f"{actual_num_sims}_obs{cfg.num_observation}_posterior.png"
    )

    plot_hierarchical_posterior(
        samples=samples_sliced,
        labels=labels,
        output_path=plot_path,
        true_params=true_params,
    )
    log.info(f"Posterior plot saved to {plot_path}")


if __name__ == "__main__":
    main()

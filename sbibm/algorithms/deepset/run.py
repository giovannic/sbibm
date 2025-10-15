"""
Hierarchical DeepSet algorithm wrapper for sbibm.

This module provides the sbibm interface for the hierarchical DeepSet
algorithm, enabling inference on tasks with global and local parameters.

References:
    - Paper: arXiv:2306.12584
    - Implementation: https://github.com/smsharma/hierarchical-inference
"""

from typing import Tuple

import torch

from sbibm.tasks.task import Task

from .hierarchical_deepset import HierarchicalDeepSetInference
from .posterior import HierarchicalPosterior
from .training import train_hierarchical_deepset
from .utils import reshape_observations


def run(
    task: Task,
    num_samples: int = 1000,
    num_simulations: int = 10000,
    num_observation: int = 1,
    dim_hidden: int = 128,
    num_transforms: int = 6,
    condition_local_on_global: bool = True,
    training_batch_size: int = 128,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, int, torch.Tensor, HierarchicalPosterior]:
    """
    Run hierarchical DeepSet inference on hierarchical task.

    Hierarchical DeepSet performs density estimation on tasks with both
    global (shared) and local (context-specific) parameters using a
    permutation-invariant set encoder with separate normalizing flows.

    Args:
        task: HierarchicalTask instance with global/local structure
        num_samples: Number of posterior samples to return
        num_simulations: Simulation budget for training
        num_observation: Which observation to condition on (1-indexed)
        dim_hidden: Hidden dimension for encoder (default: 128)
        num_transforms: Number of MAF transforms (default: 6)
        condition_local_on_global: Condition local flow on global params
        training_batch_size: Batch size during training (default: 128)
        max_epochs: Maximum training epochs (default: 50)
        learning_rate: Adam learning rate (default: 1e-3)
        device: "cpu" or "cuda" (default: "cpu")
        verbose: Print training progress (default: False)
        **kwargs: Additional arguments (ignored)

    Returns:
        samples: Posterior samples, shape (num_samples, dim_global +
                 dim_local*num_events)
        actual_num_sims: Actual simulations used
        log_prob_true: Log prob at true parameters (torch.tensor(0.0))
        posterior: HierarchicalPosterior object for querying

    Raises:
        ValueError: If task doesn't have hierarchical structure
        RuntimeError: If training fails
    """
    prior = task.get_prior()
    simulator = task.get_simulator()

    theta_all = prior(num_samples=num_simulations)
    x_all = simulator(theta_all)

    theta_global, theta_local = _split_parameters(task, theta_all)
    x_set = reshape_observations(task, x_all)

    n_in = x_set.shape[-1]
    dim_global = theta_global.shape[-1]
    dim_local = theta_local.shape[-1]
    n_set_max = x_set.shape[1]

    model = HierarchicalDeepSetInference(
        n_in=n_in,
        dim_global=dim_global,
        dim_local=dim_local,
        n_set_max=n_set_max,
        dim_hidden=dim_hidden,
        num_transforms=num_transforms,
        max_epochs=max_epochs,
        lr=learning_rate,
        local_loss=True,
        global_loss=True,
    )

    trained_model = train_hierarchical_deepset(
        model,
        x_set,
        theta_local,
        theta_global,
        num_epochs=max_epochs,
        batch_size=training_batch_size,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose,
    )

    observation_flat = task.get_observation(num_observation)
    observation = reshape_observations(task, observation_flat.unsqueeze(0))[0]

    posterior = HierarchicalPosterior(trained_model, observation, task, device=device)

    samples = posterior.sample((num_samples,))

    return samples, num_simulations, torch.tensor(0.0), posterior


def _split_parameters(task: Task, theta: torch.Tensor):
    """
    Split parameters into global and local components.

    Args:
        task: Task instance
        theta: Parameters, shape (batch, dim_parameters)

    Returns:
        theta_global: Global parameters, shape (batch, dim_global)
        theta_local: Local parameters, shape (batch, num_events,
                     dim_local)
    """
    prior_dist = task.get_prior_dist()
    dim_global = prior_dist.dim_global
    n_l = task.n_l

    theta_global = theta[:, :dim_global]
    theta_local_flat = theta[:, dim_global:]

    dim_local = theta_local_flat.shape[-1] // n_l
    theta_local = theta_local_flat.reshape(theta.shape[0], n_l, dim_local)

    return theta_global, theta_local

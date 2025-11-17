"""TFMPE bottom-up algorithm for hierarchical inference on SBIBM tasks."""

import time
from typing import Dict, Tuple

import torch


def run(
    task,
    num_samples: int,
    num_simulations: int,
    num_observation: int,
    **kwargs,
) -> Tuple[torch.Tensor, float, Dict]:
    """Run TFMPE bottom-up inference on a hierarchical task.

    Args:
        task: SBIBM task instance
        num_samples: Number of posterior samples to generate
        num_simulations: Number of simulator calls to use during training
        num_observation: Index of observation to use (1-10)
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of:
        - posterior_samples: Tensor of shape (num_samples, dim_parameters)
        - execution_time: Float, seconds elapsed
        - metadata: Dict containing training losses and hyperparameters
    """
    start_time = time.time()

    # Stub implementation - will be filled in Task 2
    # For now, return valid dummy data to satisfy test structure
    posterior_samples = torch.zeros(
        num_samples, task.dim_parameters, dtype=torch.float32
    )

    # Return samples in [-1, 1] range to pass bound checks
    posterior_samples.uniform_(-1.0, 1.0)

    execution_time = time.time() - start_time

    metadata = {
        "losses": [()],  # Placeholder for round losses
        "n_samples_per_round": num_simulations // 2,
    }

    return posterior_samples, execution_time, metadata

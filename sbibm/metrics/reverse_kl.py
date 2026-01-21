import logging
from typing import Any, Optional

import torch

from sbibm.tasks.task import Task

log = logging.getLogger(__name__)


def reverse_kl(
    posterior: Any,
    task: Task,
    num_observation: int,
    num_samples: int = 10000,
    device: str = 'cpu'
) -> torch.Tensor:
    """Estimate reverse KL divergence without reference posterior samples

    Estimates E_q[log q(θ|y) - log p(θ|y)] using samples from the
    estimated posterior q. This metric is reference-free and only
    requires the posterior estimator to have a log_prob method.

    Args:
        posterior: Posterior estimator object with log_prob method
        task: Task instance
        num_observation: Observation number
        num_samples: Number of samples for Monte Carlo estimation

    Returns:
        Estimated reverse KL divergence (lower is better). Returns NaN
        if posterior does not have log_prob method.
    """
    # Check if posterior has log_prob method
    if not hasattr(posterior, "log_prob"):
        log.warning("Posterior lacks log_prob method, returning NaN")
        return torch.tensor(float("nan"))

    # Sample from estimated posterior
    samples = posterior.sample((num_samples,))
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    # Compute log q(theta|y) - log probability under estimated posterior
    log_q = posterior.log_prob(samples)
    if log_q.ndim == 0:
        log_q = log_q.unsqueeze(0)

    # Compute log p(theta|y) = log p(theta) + log p(y|theta)
    prior_dist = task.get_prior_dist()
    log_p_prior = prior_dist.log_prob(samples)

    # Get observation and compute likelihood
    observation = task.get_observation(num_observation).to(device=device)
    # Observations already have batch dimension [1, dim_data]
    # Expand to match sample batch size
    observation_expanded = observation.expand(samples.shape[0], -1)
    log_p_likelihood = task._likelihood(samples, observation_expanded)
    log_p = log_p_prior + log_p_likelihood

    # Compute per-sample KL contributions
    kl_per_sample = log_q - log_p

    # Estimate reverse KL: E_q[log q - log p]
    reverse_kl_estimate = kl_per_sample.mean()

    return reverse_kl_estimate

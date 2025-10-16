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
    invalid_penalty: Optional[float] = 1000.0,
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
        invalid_penalty: Penalty value for samples with invalid log
            probability (NaN or -inf). Default: 1000.0. Set to None
            to include invalid samples as-is (will result in NaN).

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

    # Compute log p(theta|y) - true unnormalized log posterior
    log_p_fn = task._get_log_prob_fn(
        num_observation=num_observation,
        implementation="pyro",
        posterior=True,
        jit_compile=False,
        automatic_transform_enabled=False,
    )
    log_p = log_p_fn(samples)

    # Compute per-sample KL contributions
    kl_per_sample = log_q - log_p

    # Handle invalid samples (NaN or -inf log_p)
    invalid_mask = ~torch.isfinite(kl_per_sample)
    if invalid_mask.any():
        num_invalid = invalid_mask.sum().item()
        log.warning(
            f"Found {num_invalid}/{num_samples} samples with invalid "
            f"log probability"
        )

        if invalid_penalty is not None:
            # Replace invalid values with penalty
            kl_per_sample = torch.where(
                invalid_mask,
                torch.tensor(invalid_penalty, dtype=kl_per_sample.dtype),
                kl_per_sample,
            )

    # Estimate reverse KL: E_q[log q - log p]
    reverse_kl_estimate = kl_per_sample.mean()

    return reverse_kl_estimate

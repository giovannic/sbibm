import logging
from typing import Any, Dict

from sbibm.tasks.task import Task

import torch

from tfmpe.metrics.lc2st import run_lc2st

from jax import random as jr
from jax import numpy as jnp

log = logging.getLogger(__name__)


def lc2st(
    posterior: Any,
    task: Task,
    num_observation: int,
    posterior_samples: torch.Tensor,
    num_calibration_samples: int = 10000,
    num_trials: int = 100,
    n_epochs: int = 1000,
    alpha: float = 0.05,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Local Classifier Two-Sample Test (LC2ST) for posterior validation

    Tests if a classifier can distinguish between samples from
    (prior, simulations) vs (posterior|observation, observation).
    A well-calibrated posterior should make these indistinguishable.

    This is a reference-free metric that uses the LC2ST diagnostic
    from the sbi library to test posterior calibration for a specific
    observation.

    Args:
        posterior: Posterior estimator object with sample method
        task: Task instance
        num_observation: Observation number
        num_calibration_samples: Number of (theta, x) pairs to
            generate from prior for calibration
        num_posterior_samples: Number of samples to draw from
            posterior for testing
        num_trials: Number of permutation trials for null hypothesis
        alpha: Significance level for hypothesis test
        **kwargs: Additional arguments passed to LC2ST

    Returns:
        Dictionary containing:
            - p_value: P-value for the hypothesis test
            - test_statistic: Test statistic on observed data
            - critical_value: Critical value from null distribution
            - reject: Boolean rejection decision at specified alpha
    """
    # Get observation
    observation = task.get_observation(num_observation)

    # Generate calibration data: sample from prior and run simulator
    prior = task.get_prior()
    simulator = task.get_simulator()

    thetas = prior(num_samples=num_calibration_samples)
    xs = simulator(thetas)

    # Sample from posterior conditioned on observation
    theta_q = posterior.sample((num_calibration_samples,), x=xs)

    key = jr.PRNGKey(0)

    d_cal = (
        jnp.array(xs),
        jnp.array(thetas),
        jnp.array(theta_q)
    )

    response = run_lc2st(
        key,
        d_cal,
        jnp.array(observation)[0],
        jnp.array(posterior_samples),
        latent_dim = 32,
        n_layers = 2,
        num_folds = 10,
        num_ensemble = 10,
        num_null = num_trials,
        n_epochs = n_epochs,
    )
    critical_value = response.critical_value(alpha)
    return {
        "p_value": response.pvalue,
        "test_statistic": response.main_stat,
        "critical_value": critical_value,
        "reject": response.main_stat > critical_value,
    }

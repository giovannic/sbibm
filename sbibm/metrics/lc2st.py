import logging
from typing import Any, Dict

import torch
from sbi.diagnostics.lc2st import LC2ST

from sbibm.tasks.task import Task

log = logging.getLogger(__name__)


def lc2st(
    posterior: Any,
    task: Task,
    num_observation: int,
    num_calibration_samples: int = 1000,
    num_posterior_samples: int = 1000,
    classifier: str = "mlp",
    num_ensemble: int = 1,
    num_trials: int = 100,
    alpha: float = 0.05,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
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
        classifier: Classifier type ("mlp" or "rf")
        num_ensemble: Number of classifiers in ensemble
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
    posterior_samples = posterior.sample((num_posterior_samples,), x=xs)

    # Initialize LC2ST
    lc2st_test = LC2ST(
        thetas=thetas,
        xs=xs,
        posterior_samples=posterior_samples,
        classifier=classifier,
        num_ensemble=num_ensemble,
        num_trials_null=num_trials,
    )

    # Train under null hypothesis for permutation test
    lc2st_test.train_under_null_hypothesis()

    # Train on observed data
    lc2st_test.train_on_observed_data()

    # Get statistics (theta_o and x_o are passed to evaluation methods)
    theta_o = task.get_true_parameters(num_observation)
    x_o = observation

    p_val = lc2st_test.p_value(theta_o=theta_o, x_o=x_o)
    test_stat = lc2st_test.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o)

    # Get null distribution statistics for critical value
    null_stats = lc2st_test.get_statistics_under_null_hypothesis(
        theta_o=theta_o, x_o=x_o
    )
    critical_value = torch.quantile(torch.tensor(null_stats), 1.0 - alpha)

    reject = lc2st_test.reject_test(theta_o=theta_o, x_o=x_o, alpha=alpha)

    return {
        "p_value": torch.tensor(p_val),
        "test_statistic": torch.tensor(test_stat),
        "critical_value": critical_value,
        "reject": torch.tensor(reject),
    }

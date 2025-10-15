import pytest
import torch
from jax import numpy as jnp

from sbibm.metrics import lc2st
from sbibm.tasks import get_task

class MockPosterior:
    """Mock posterior for testing LC2ST"""

    def __init__(self, mean, std):
        self.dist = torch.distributions.Normal(mean, std)

    def sample(self, shape, x=None):
        return self.dist.sample(shape)

@pytest.mark.slow
def test_lc2st_with_hierarchical_two_moons():
    """Test lc2st with hierarchical_two_moons task"""
    task = get_task("hierarchical_two_moons", n_l=5)

    # Create a mock posterior with appropriate dimensionality
    true_params = task.get_true_parameters(num_observation=1)
    mock_posterior = MockPosterior(
        mean=true_params.squeeze(),
        std=torch.ones(task.dim_parameters) * 0.1,
    )

    # Compute LC2ST with small sample sizes for speed
    posterior_samples = mock_posterior.sample((200,))
    result = lc2st(
        posterior=mock_posterior,
        task=task,
        num_observation=1,
        posterior_samples=posterior_samples,
        num_calibration_samples=200,
        num_trials=10,
    )

    # Check that all expected keys are present
    assert "p_value" in result
    assert "test_statistic" in result
    assert "critical_value" in result
    assert "reject" in result

    # Check that values are finite
    assert jnp.isfinite(result["p_value"])
    assert jnp.isfinite(result["test_statistic"])
    assert jnp.isfinite(result["critical_value"])

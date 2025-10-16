import torch

from sbibm.metrics import lc2st
from sbibm.tasks import get_task


class MockPosterior:
    """Mock posterior for testing LC2ST"""

    def __init__(self, mean, std):
        self.dist = torch.distributions.Normal(mean, std)

    def sample(self, shape):
        return self.dist.sample(shape)


def test_lc2st_with_two_moons():
    """Test lc2st with two_moons task and mock posterior"""
    task = get_task("two_moons")

    # Create a mock posterior centered at true parameters
    true_params = task.get_true_parameters(num_observation=1)
    mock_posterior = MockPosterior(mean=true_params.squeeze(), std=torch.ones(2) * 0.1)

    # Compute LC2ST with small sample sizes for speed
    result = lc2st(
        posterior=mock_posterior,
        task=task,
        num_observation=1,
        num_calibration_samples=100,
        num_posterior_samples=100,
        num_trials=10,
    )

    # Check that all expected keys are present
    assert "p_value" in result
    assert "test_statistic" in result
    assert "critical_value" in result
    assert "reject" in result

    # Check that values are finite and have correct types
    assert torch.isfinite(result["p_value"])
    assert torch.isfinite(result["test_statistic"])
    assert torch.isfinite(result["critical_value"])

    # p_value should be in [0, 1]
    assert 0.0 <= result["p_value"].item() <= 1.0

    # reject should be boolean
    assert result["reject"].dtype == torch.bool


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
    result = lc2st(
        posterior=mock_posterior,
        task=task,
        num_observation=1,
        num_calibration_samples=100,
        num_posterior_samples=100,
        num_trials=10,
    )

    # Check that all expected keys are present
    assert "p_value" in result
    assert "test_statistic" in result
    assert "critical_value" in result
    assert "reject" in result

    # Check that values are finite
    assert torch.isfinite(result["p_value"])
    assert torch.isfinite(result["test_statistic"])
    assert torch.isfinite(result["critical_value"])

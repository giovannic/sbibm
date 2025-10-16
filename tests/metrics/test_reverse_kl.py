import torch

from sbibm.metrics import reverse_kl
from sbibm.tasks import get_task


class MockPosterior:
    """Mock posterior with log_prob method for testing"""

    def __init__(self, mean, std):
        # Use Uniform distribution for bounded support to avoid -inf log_prob
        half_width = std * torch.sqrt(torch.tensor(3.0))
        self.dist = torch.distributions.Uniform(mean - half_width, mean + half_width)

    def sample(self, shape):
        return self.dist.sample(shape)

    def log_prob(self, samples):
        return self.dist.log_prob(samples).sum(dim=-1)


class MockPosteriorWithoutLogProb:
    """Mock posterior without log_prob method for testing"""

    def __init__(self, mean, std):
        self.dist = torch.distributions.Normal(mean, std)

    def sample(self, shape):
        return self.dist.sample(shape)


def test_reverse_kl_with_two_moons():
    """Test reverse_kl with two_moons task and mock posterior"""
    task = get_task("two_moons")

    # Create a mock posterior centered at true parameters
    # Use very small std to keep log_prob finite
    true_params = task.get_true_parameters(num_observation=1)
    mock_posterior = MockPosterior(mean=true_params.squeeze(), std=torch.ones(2) * 0.01)

    # Compute reverse KL
    rkl = reverse_kl(
        posterior=mock_posterior,
        task=task,
        num_observation=1,
        num_samples=1000,
    )

    # Should return a finite value
    assert torch.isfinite(rkl)
    # Should be a scalar
    assert rkl.ndim == 0


def test_reverse_kl_returns_nan_without_log_prob():
    """Test that reverse_kl returns NaN when posterior lacks log_prob"""
    task = get_task("two_moons")

    # Create a mock posterior without log_prob method
    true_params = task.get_true_parameters(num_observation=1)
    mock_posterior = MockPosteriorWithoutLogProb(
        mean=true_params.squeeze(), std=torch.ones(2) * 0.1
    )

    # Compute reverse KL
    rkl = reverse_kl(
        posterior=mock_posterior,
        task=task,
        num_observation=1,
        num_samples=1000,
    )

    # Should return NaN
    assert torch.isnan(rkl)


def test_reverse_kl_with_hierarchical_two_moons():
    """Test reverse_kl with hierarchical_two_moons task"""
    task = get_task("hierarchical_two_moons", n_l=3)

    # Create a mock posterior with appropriate dimensionality
    # hierarchical_two_moons has dim_parameters = 4 + 2*n_l = 10
    true_params = task.get_true_parameters(num_observation=1)
    mock_posterior = MockPosterior(
        mean=true_params.squeeze(),
        std=torch.ones(task.dim_parameters) * 0.1,
    )

    # Compute reverse KL
    rkl = reverse_kl(
        posterior=mock_posterior,
        task=task,
        num_observation=1,
        num_samples=500,
    )

    # Should return a finite value
    assert torch.isfinite(rkl)
    # Should be a scalar
    assert rkl.ndim == 0

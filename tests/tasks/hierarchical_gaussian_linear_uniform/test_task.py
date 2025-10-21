import pyro
import pytest
import torch

from sbibm.tasks.hierarchical_gaussian_linear_uniform.task import (
    HierarchicalGaussianLinearUniform,
)

pyro.util.set_rng_seed(42)


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_prior_shape(n_l):
    """Test prior returns correct shape."""
    task = HierarchicalGaussianLinearUniform(n_l=n_l)
    prior = task.get_prior()

    num_samples = 100
    samples = prior(num_samples=num_samples)

    expected_dim = 10 + n_l
    assert samples.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_no_nan(n_l):
    """Test prior samples contain no NaN or Inf values."""
    task = HierarchicalGaussianLinearUniform(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_simulator_shape(n_l):
    """Test simulator returns correct shape."""
    task = HierarchicalGaussianLinearUniform(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_samples = 20
    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    expected_dim = 10 * n_l
    assert observations.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_simulator_no_nan(n_l):
    """Test simulator output contains no NaN values."""
    task = HierarchicalGaussianLinearUniform(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    parameters = prior(num_samples=20)
    observations = simulator(parameters)

    assert not torch.isnan(observations).any()


def test_prior_structure():
    """Test prior structure: global mean (uniform) and local scales."""
    n_l = 5
    dim = 10
    prior_bound = 10.0
    task = HierarchicalGaussianLinearUniform(
        n_l=n_l, dim=dim, prior_bound=prior_bound
    )
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First dim params are global mean: should be Uniform(-10, 10)
    global_mean = samples[:, :dim]
    assert global_mean.shape[1] == dim
    # Check within bounds
    assert torch.all(global_mean >= -prior_bound)
    assert torch.all(global_mean <= prior_bound)
    # Check roughly uniform distribution (mean should be near 0)
    assert abs(global_mean.mean()) < 1.0

    # Next n_l params are local scales: should be positive (HalfNormal)
    local_scales = samples[:, dim:]
    assert local_scales.shape[1] == n_l
    assert (local_scales >= 0).all()


def test_prior_dist_log_prob():
    """Test prior distribution log_prob."""
    n_l = 3
    task = HierarchicalGaussianLinearUniform(n_l=n_l)
    prior = task.get_prior()

    # Sample from prior
    samples = prior(num_samples=10)

    # Compute log_prob via prior_dist
    log_probs = task.prior_dist.log_prob(samples)
    assert log_probs.shape == torch.Size([10])
    assert not torch.isnan(log_probs).any()
    assert torch.all(torch.isfinite(log_probs))


def test_reference_posterior_not_implemented():
    """Test that reference posterior raises NotImplementedError."""
    task = HierarchicalGaussianLinearUniform(n_l=3)

    with pytest.raises(NotImplementedError):
        task._sample_reference_posterior(num_samples=100, num_observation=1)


def test_get_prior_dist():
    """Test get_prior_dist returns the prior distribution."""
    task = HierarchicalGaussianLinearUniform(n_l=3)
    prior_dist = task.get_prior_dist()

    assert prior_dist is not None
    assert prior_dist is task.prior_dist

    # Test sampling from returned distribution
    samples = prior_dist.sample((10,))
    assert samples.shape == (10, 13)  # 10 global + 3 local


def test_likelihood():
    """Test likelihood computation."""
    n_l = 3
    task = HierarchicalGaussianLinearUniform(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Generate some parameters and data
    parameters = prior(num_samples=5)
    data = simulator(parameters)

    # Compute log-likelihood
    log_lik = task._likelihood(parameters, data, log=True)
    assert log_lik.shape == torch.Size([5])
    assert not torch.isnan(log_lik).any()
    assert torch.all(torch.isfinite(log_lik))

    # Non-log likelihood
    lik = task._likelihood(parameters, data, log=False)
    assert lik.shape == torch.Size([5])
    assert not torch.isnan(lik).any()
    assert (lik >= 0).all()

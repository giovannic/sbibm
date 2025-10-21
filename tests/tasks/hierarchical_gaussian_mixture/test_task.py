import pyro
import pytest
import torch

from sbibm.tasks.hierarchical_gaussian_mixture.task import (
    HierarchicalGaussianMixture,
)

pyro.util.set_rng_seed(47)


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_prior_shape(n_l):
    """Test prior returns correct shape."""
    dim = 2
    task = HierarchicalGaussianMixture(n_l=n_l, dim=dim)
    prior = task.get_prior()

    num_samples = 100
    samples = prior(num_samples=num_samples)

    expected_dim = 2 * dim + dim * n_l
    assert samples.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_no_nan(n_l):
    """Test prior samples contain no NaN or Inf values."""
    task = HierarchicalGaussianMixture(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_simulator_shape(n_l):
    """Test simulator returns correct shape."""
    dim = 2
    task = HierarchicalGaussianMixture(n_l=n_l, dim=dim)
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_samples = 20
    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    expected_dim = dim * n_l
    assert observations.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_simulator_no_nan(n_l):
    """Test simulator output contains no NaN values."""
    task = HierarchicalGaussianMixture(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    parameters = prior(num_samples=20)
    observations = simulator(parameters)

    assert not torch.isnan(observations).any()


def test_prior_structure():
    """Test prior structure: global locs, scales, and local params."""
    n_l = 5
    dim = 2
    prior_bound = 10.0
    task = HierarchicalGaussianMixture(n_l=n_l, dim=dim, prior_bound=prior_bound)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First dim params are global locs: should be Uniform(-10, 10)
    global_locs = samples[:, :dim]
    assert global_locs.min() >= -prior_bound
    assert global_locs.max() <= prior_bound

    # Next dim params are global scales: should be positive (HalfNormal)
    global_scales = samples[:, dim : 2 * dim]
    assert (global_scales >= 0).all()

    # Remaining dim*n_l params are local: should be Normal(global_loc,
    # global_scale)
    local_params = samples[:, 2 * dim :]
    assert local_params.shape[1] == dim * n_l


def test_prior_dist_log_prob():
    """Test prior distribution log_prob."""
    n_l = 3
    task = HierarchicalGaussianMixture(n_l=n_l)
    prior = task.get_prior()

    # Sample from prior
    samples = prior(num_samples=1000)

    # Compute log_prob via prior_dist
    log_probs = task.prior_dist.log_prob(samples)
    assert log_probs.shape == torch.Size([1000])
    assert not torch.isnan(log_probs).any()
    assert torch.all(torch.isfinite(log_probs))


def test_reference_posterior_raises():
    """Test that _sample_reference_posterior raises NotImplementedError."""
    task = HierarchicalGaussianMixture(n_l=3)

    with pytest.raises(NotImplementedError):
        task._sample_reference_posterior(num_samples=100, num_observation=1)


def test_likelihood():
    """Test likelihood computation."""
    n_l = 3
    task = HierarchicalGaussianMixture(n_l=n_l)
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

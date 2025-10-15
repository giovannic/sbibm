import pyro
import pytest
import torch

from sbibm.tasks.hierarchical_gaussian_linear.task import (
    HierarchicalGaussianLinear,
)

pyro.util.set_rng_seed(42)


@pytest.mark.parametrize(
    "n_l,dim",
    [(3, 16), (5, 26), (10, 51)],
)
def test_prior_shape(n_l, dim):
    """Test prior returns correct shape."""
    task = HierarchicalGaussianLinear(n_l=n_l, dim_local_per_context=(dim - 1) // n_l)
    prior = task.get_prior()

    num_samples = 100
    samples = prior(num_samples=num_samples)

    assert samples.shape == (num_samples, dim)


@pytest.mark.parametrize("n_l,dim", [(3, 16), (5, 26)])
def test_prior_no_nan(n_l, dim):
    """Test prior samples contain no NaN or Inf values."""
    task = HierarchicalGaussianLinear(n_l=n_l, dim_local_per_context=(dim - 1) // n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


@pytest.mark.parametrize(
    "n_l,dim",
    [(3, 16), (5, 26), (10, 51)],
)
def test_simulator_shape(n_l, dim):
    """Test simulator returns correct shape."""
    task = HierarchicalGaussianLinear(n_l=n_l, dim_local_per_context=(dim - 1) // n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_samples = 20
    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    assert observations.shape == (num_samples, task.dim_data)


@pytest.mark.parametrize("n_l,dim", [(3, 16), (5, 26)])
def test_simulator_no_nan(n_l, dim):
    """Test simulator output contains no NaN values."""
    task = HierarchicalGaussianLinear(n_l=n_l, dim_local_per_context=(dim - 1) // n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    parameters = prior(num_samples=20)
    observations = simulator(parameters)

    assert not torch.isnan(observations).any()


def test_prior_structure():
    """Test prior structure: global scales and local means."""
    n_l = 5
    dim = 26
    task = HierarchicalGaussianLinear(
        n_l=n_l, dim_local_per_context=(dim - 1) // n_l, prior_scale=0.1
    )
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First dim_global params are global scale: should be positive (HalfNormal)
    global_scale = samples[:, : task.dim_global]
    assert global_scale.shape[1] == task.dim_global
    assert (global_scale > 0).all()

    # Remaining params are local means: should be Normal(0, prior_scale)
    local_means = samples[:, task.dim_global :]  # noqa: E203
    assert local_means.shape[1] == task.dim_local_total
    # Check roughly centered at 0 with expected variance
    # prior_scale=0.1 is std, so variance = 0.01
    assert abs(local_means.mean()) < 0.05  # Close to 0
    assert 0.08 < local_means.std() < 0.15  # Close to prior_scale=0.1


def test_prior_dist_log_prob():
    """Test prior distribution log_prob."""
    n_l = 3
    dim = 16  # (16-1)=15, 15%3=0 ✓
    task = HierarchicalGaussianLinear(n_l=n_l, dim_local_per_context=(dim - 1) // n_l)
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
    n_l = 3
    dim = 16  # (16-1)=15, 15%3=0 ✓
    task = HierarchicalGaussianLinear(n_l=n_l, dim_local_per_context=(dim - 1) // n_l)

    with pytest.raises(NotImplementedError):
        task._sample_reference_posterior(num_samples=100, num_observation=1)


def test_get_prior_dist():
    """Test get_prior_dist returns the prior distribution."""
    n_l = 3
    dim = 16  # 1 + 3*5 = 16
    task = HierarchicalGaussianLinear(n_l=n_l, dim_local_per_context=(dim - 1) // n_l)
    prior_dist = task.get_prior_dist()

    assert prior_dist is not None
    assert prior_dist is task.prior_dist

    # Test sampling from returned distribution
    samples = prior_dist.sample((10,))
    assert samples.shape == (10, dim)


def test_likelihood():
    """Test likelihood computation."""
    n_l = 3
    dim = 16
    task = HierarchicalGaussianLinear(n_l=n_l, dim_local_per_context=(dim - 1) // n_l)
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

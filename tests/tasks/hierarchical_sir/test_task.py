import pyro
import pytest
import torch

from sbibm.tasks.hierarchical_sir.task import HierarchicalSIR

pyro.util.set_rng_seed(42)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_shape(n_l):
    """Test prior returns correct shape."""
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    num_samples = 100
    samples = prior(num_samples=num_samples)

    expected_dim = 1 + n_l  # 1 global (beta) + n_l local (gamma)
    assert samples.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_no_nan(n_l):
    """Test prior samples contain no NaN or Inf values."""
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


@pytest.mark.parametrize("n_l", [3, 5])
def test_simulator_shape(n_l):
    """Test simulator returns correct shape."""
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_samples = 5
    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    expected_dim = 10 * n_l  # 10 subsampled points per region
    assert observations.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [2, 3])
def test_simulator_no_nan(n_l):
    """Test simulator output contains no NaN values (most of the time).

    Note: SIR ODE can occasionally fail for extreme parameter values,
    but should succeed for most prior samples.
    """
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    parameters = prior(num_samples=20)
    observations = simulator(parameters)

    # Allow some NaN values due to ODE failures, but most should succeed
    nan_rate = torch.isnan(observations).any(dim=1).float().mean()
    assert nan_rate < 0.5  # At least 50% should succeed


def test_prior_structure():
    """Test prior structure: global beta and local gammas."""
    n_l = 5
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First param is global beta: LogNormal(log(0.4), 0.5)
    beta = samples[:, 0]
    assert beta.shape[0] == 1000
    # Check positive (LogNormal support)
    assert (beta > 0).all()
    # Check roughly centered around 0.4 in log-space
    log_beta = torch.log(beta)
    assert -1.5 < log_beta.mean() < -0.5  # log(0.4) ≈ -0.916

    # Next n_l params are local gammas: LogNormal(log(0.125), 0.2)
    gamma = samples[:, 1:]
    assert gamma.shape == (1000, n_l)
    # Check positive
    assert (gamma > 0).all()
    # Check roughly centered around 0.125 in log-space
    log_gamma = torch.log(gamma)
    assert -2.5 < log_gamma.mean() < -1.5  # log(0.125) ≈ -2.079


def test_prior_dist_log_prob():
    """Test prior distribution log_prob."""
    n_l = 3
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    # Sample from prior
    samples = prior(num_samples=10)

    # Compute log_prob via prior_dist
    log_probs = task.prior_dist.log_prob(samples)
    assert log_probs.shape == torch.Size([10])
    assert not torch.isnan(log_probs).any()
    assert torch.all(torch.isfinite(log_probs))


def test_likelihood_function():
    """Test _likelihood function returns finite values.

    This test ensures the prior is tuned such that ODE integration
    succeeds reliably. If this test fails, the prior may need adjustment.
    """
    n_l = 2
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Generate parameters and observations
    parameters = prior(num_samples=5)
    observations = simulator(parameters)

    # Compute likelihood
    log_lik = task._likelihood(parameters, observations, log=True)

    assert log_lik.shape == torch.Size([5])
    # Likelihood should be finite for all valid observations
    valid_obs = ~torch.isnan(observations).any(dim=1)
    assert torch.all(torch.isfinite(log_lik[valid_obs]))


def test_reference_posterior_not_implemented():
    """Test that reference posterior raises NotImplementedError."""
    task = HierarchicalSIR(n_l=3)

    with pytest.raises(NotImplementedError):
        task._sample_reference_posterior(num_samples=100, num_observation=1)


def test_get_prior_dist():
    """Test get_prior_dist returns the prior distribution."""
    task = HierarchicalSIR(n_l=3)
    prior_dist = task.get_prior_dist()

    assert prior_dist is not None
    assert prior_dist is task.prior_dist

    # Test sampling from returned distribution
    samples = prior_dist.sample((10,))
    assert samples.shape == (10, 4)  # 1 global + 3 local

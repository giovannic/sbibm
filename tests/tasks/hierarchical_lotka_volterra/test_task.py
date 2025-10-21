import pyro
import pytest
import torch

from sbibm.tasks.hierarchical_lotka_volterra.task import (
    HierarchicalLotkaVolterra,
)

pyro.util.set_rng_seed(47)


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_prior_shape(n_l):
    """Test prior returns correct shape."""
    task = HierarchicalLotkaVolterra(n_l=n_l)
    prior = task.get_prior()

    num_samples = 100
    samples = prior(num_samples=num_samples)

    expected_dim = 2 + 2 * n_l  # 2 global (beta, gamma) + 2*n_l local
    assert samples.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_no_nan(n_l):
    """Test prior samples contain no NaN or Inf values."""
    task = HierarchicalLotkaVolterra(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


@pytest.mark.parametrize("n_l", [3, 5])
def test_simulator_shape(n_l):
    """Test simulator returns correct shape."""
    task = HierarchicalLotkaVolterra(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_samples = 10
    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    expected_dim = 10 * n_l  # 10 observations per context (2 species * 5 times)
    assert observations.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_simulator_no_nan(n_l):
    """Test simulator output contains no NaN values (statistically)."""
    task = HierarchicalLotkaVolterra(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()
    num_samples = 10

    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    # Allow small fraction of NaN due to ODE failures
    nan_fraction = torch.isnan(observations).any(dim=1).float().mean()
    assert nan_fraction < 0.1, f"Too many NaN values: {nan_fraction:.2%}"


def test_prior_structure():
    """Test prior structure: global predation params and local birth params."""
    n_l = 5
    task = HierarchicalLotkaVolterra(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First 2 params are global: beta (predation), gamma (predator death)
    # beta ~ LogNormal(log(0.028), 0.5)
    # gamma ~ LogNormal(log(0.5), 0.5)
    global_params = samples[:, :2]
    assert global_params.shape == (1000, 2)

    # Beta should be centered around 0.028 (in log space)
    assert torch.abs(torch.log(global_params[:, 0]).mean() - torch.log(
        torch.tensor(0.028)
    )) < 0.2

    # Gamma should be centered around 0.5 (in log space)
    assert torch.abs(torch.log(global_params[:, 1]).mean() - torch.log(
        torch.tensor(0.5)
    )) < 0.2

    # All global params should be positive (LogNormal)
    assert (global_params > 0).all()

    # Local params: alpha_i (prey birth) and delta_i (predator birth) per site
    # alpha_i ~ LogNormal(log(1.0), 0.2)
    # delta_i ~ LogNormal(log(0.01), 0.2)
    local_params = samples[:, 2:]
    assert local_params.shape == (1000, 2 * n_l)

    # All local params should be positive (LogNormal)
    assert (local_params > 0).all()


def test_prior_dist_log_prob():
    """Test prior distribution log_prob."""
    n_l = 3
    task = HierarchicalLotkaVolterra(n_l=n_l)
    prior = task.get_prior()
    n_samples = 10

    # Sample from prior
    samples = prior(num_samples=n_samples)

    # Compute log_prob via prior_dist
    log_probs = task.prior_dist.log_prob(samples)
    assert log_probs.shape == torch.Size([n_samples])
    assert not torch.isnan(log_probs).any()
    assert torch.all(torch.isfinite(log_probs))


def test_likelihood_computation():
    """Test likelihood computation with Poisson observations."""
    n_l = 3
    task = HierarchicalLotkaVolterra(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Generate some test data
    parameters = prior(num_samples=5)
    observations = simulator(parameters)

    # Compute log-likelihood
    log_lik = task._likelihood(parameters, observations, log=True)

    assert log_lik.shape == (5,)
    # Likelihood should be finite (Poisson is naturally bounded)
    # or -inf for failed ODE solves
    assert not torch.isnan(log_lik).any()


def test_reference_posterior_not_implemented():
    """Test that reference posterior raises NotImplementedError."""
    task = HierarchicalLotkaVolterra(n_l=3)

    with pytest.raises(NotImplementedError):
        task._sample_reference_posterior(num_samples=100, num_observation=1)

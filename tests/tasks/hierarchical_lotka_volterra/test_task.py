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

    expected_dim = 8 + 4 * n_l  # 8 global hyperpriors + 4*n_l local
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
    num_samples = 100

    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    # Allow small fraction of NaN due to ODE failures
    nan_fraction = torch.isnan(observations).any(dim=1).float().mean()
    assert nan_fraction < 0.1, f"Too many NaN values: {nan_fraction:.2%}"


def test_prior_structure():
    """Test prior structure: hyperpriors for means/scales and local
    site-specific parameters."""
    n_l = 5
    task = HierarchicalLotkaVolterra(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First 8 params are global hyperpriors:
    # [mu_alpha, mu_beta, mu_gamma, mu_delta, sigma_alpha,
    # sigma_beta, sigma_gamma, sigma_delta]
    global_params = samples[:, :8]
    assert global_params.shape == (1000, 8)

    # First 4 are means: mu ~ Normal(0, 1) (unbounded)
    means = global_params[:, :4]
    # Means should have reasonable range for Normal(0,1)
    assert means.mean() < 0.5  # Should be near 0
    assert means.std() > 0.5  # Should have non-zero spread

    # Last 4 are scales: sigma ~ HalfNormal(1) (positive)
    scales = global_params[:, 4:]
    assert (scales > 0).all()  # All positive
    assert scales.mean() < 1.0  # Should be positive but not huge

    # Local params: [alpha_1, beta_1, gamma_1, delta_1, ...,
    # alpha_n_l, beta_n_l, gamma_n_l, delta_n_l]
    # Each parameter ~ LogNormal(mu_global, sigma_global) so all positive
    local_params = samples[:, 8:]
    assert local_params.shape == (1000, 4 * n_l)

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
    """Test likelihood computation with LogNormal observations."""
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
    # Likelihood should be finite (LogNormal is naturally bounded)
    # or -inf for failed ODE solves
    assert not torch.isnan(log_lik).any()


def test_reference_posterior_not_implemented():
    """Test that reference posterior raises NotImplementedError."""
    task = HierarchicalLotkaVolterra(n_l=3)

    with pytest.raises(NotImplementedError):
        task._sample_reference_posterior(num_samples=100, num_observation=1)

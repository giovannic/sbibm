import pyro
import pytest
import torch

from sbibm.tasks.hierarchical_two_moons.task import HierarchicalTwoMoons

pyro.util.set_rng_seed(47)


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_task_constructs(n_l):
    """Test hierarchical_two_moons can be instantiated with various n_l."""
    task = HierarchicalTwoMoons(n_l=n_l)
    assert task is not None
    assert task.n_l == n_l
    assert task.dim_parameters == 4 + 2 * n_l
    assert task.dim_data == 2 * n_l


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_prior_shape(n_l):
    """Test prior returns correct shape."""
    task = HierarchicalTwoMoons(n_l=n_l)
    prior = task.get_prior()

    num_samples = 100
    samples = prior(num_samples=num_samples)

    expected_dim = 4 + 2 * n_l
    assert samples.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_no_nan(n_l):
    """Test prior samples contain no NaN or Inf values."""
    task = HierarchicalTwoMoons(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=50)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_simulator_shape(n_l):
    """Test simulator returns correct shape."""
    task = HierarchicalTwoMoons(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_samples = 20
    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    expected_dim = 2 * n_l
    assert observations.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_simulator_no_nan(n_l):
    """Test simulator output contains no NaN values."""
    task = HierarchicalTwoMoons(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    parameters = prior(num_samples=20)
    observations = simulator(parameters)

    assert not torch.isnan(observations).any()


def test_prior_structure():
    """Test prior structure: global locs, scales, and local params."""
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First 2 params are global locs: should be Uniform(-1, 1)
    global_locs = samples[:, :2]
    assert global_locs.min() >= -1.0
    assert global_locs.max() <= 1.0

    # Next 2 params are global scales: should be positive (HalfNormal)
    global_scales = samples[:, 2:4]
    assert (global_scales >= 0).all()

    # Remaining 2*n_l params are local: should be TruncatedNormal[-1, 1]
    local_params = samples[:, 4:]
    assert local_params.shape[1] == 2 * n_l
    assert local_params.min() >= -1.0
    assert local_params.max() <= 1.0


def test_likelihood():
    """Test likelihood computation."""
    n_l = 3
    task = HierarchicalTwoMoons(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Generate some parameters and data
    parameters = prior(num_samples=5)
    data = simulator(parameters)

    # Compute likelihood
    log_lik = task._likelihood(parameters, data, log=True)
    assert log_lik.shape == torch.Size([5])
    assert not torch.isnan(log_lik).any()
    assert torch.all(torch.isfinite(log_lik) | torch.isneginf(log_lik))

    # Non-log likelihood
    lik = task._likelihood(parameters, data, log=False)
    assert lik.shape == torch.Size([5])
    assert not torch.isnan(lik).any()
    assert (lik >= 0).all()


def test_prior_dist_log_prob():
    """Test prior distribution log_prob."""
    n_l = 3
    task = HierarchicalTwoMoons(n_l=n_l)
    prior = task.get_prior()

    # Sample from prior
    samples = prior(num_samples=10)

    # Compute log_prob via prior_dist
    log_probs = task.prior_dist.log_prob(samples)
    assert log_probs.shape == torch.Size([10])
    assert not torch.isnan(log_probs).any()
    assert torch.all(torch.isfinite(log_probs))

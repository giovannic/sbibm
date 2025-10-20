import pyro
import pytest
import torch

from sbibm.tasks.hierarchical_bernoulli_glm.task import (
    HierarchicalBernoulliGLM,
)

pyro.util.set_rng_seed(47)


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_prior_shape(n_l):
    """Test prior returns correct shape."""
    task = HierarchicalBernoulliGLM(n_l=n_l)
    prior = task.get_prior()

    num_samples = 100
    samples = prior(num_samples=num_samples)

    expected_dim = 9 + n_l  # 9 global (filter) + n_l local (offsets)
    assert samples.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_no_nan(n_l):
    """Test prior samples contain no NaN or Inf values."""
    task = HierarchicalBernoulliGLM(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=100)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


@pytest.mark.parametrize("n_l", [3, 5, 10])
def test_simulator_shape(n_l):
    """Test simulator returns correct shape."""
    task = HierarchicalBernoulliGLM(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_samples = 20
    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    expected_dim = 10 * n_l  # 10 summary stats per context
    assert observations.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_simulator_no_nan(n_l):
    """Test simulator output contains no NaN values."""
    task = HierarchicalBernoulliGLM(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    parameters = prior(num_samples=100)
    observations = simulator(parameters)

    assert not torch.isnan(observations).any()


def test_prior_structure():
    """Test prior structure: global filter and local offsets."""
    n_l = 5
    task = HierarchicalBernoulliGLM(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First 9 params are global filter coefficients (MVN with precision)
    filter_params = samples[:, :9]
    assert filter_params.shape == (1000, 9)
    # Filter params should be centered around 0 with reasonable variance
    assert torch.abs(filter_params.mean(0)).max() < 0.5

    # Remaining n_l params are local offsets: Normal(0, 0.5)
    offset_params = samples[:, 9:]
    assert offset_params.shape == (1000, n_l)
    # Offsets should be centered around 0
    assert torch.abs(offset_params.mean(0)).max() < 0.2
    # Most offsets should be within 3 std devs: [-1.5, 1.5]
    assert (torch.abs(offset_params) < 2.0).float().mean() > 0.95


def test_prior_dist_log_prob():
    """Test prior distribution log_prob."""
    n_l = 3
    task = HierarchicalBernoulliGLM(n_l=n_l)
    prior = task.get_prior()

    # Sample from prior
    samples = prior(num_samples=1000)

    # Compute log_prob via prior_dist
    log_probs = task.prior_dist.log_prob(samples)
    assert log_probs.shape == torch.Size([1000])
    assert not torch.isnan(log_probs).any()
    assert torch.all(torch.isfinite(log_probs))


def test_raw_summary():
    """Test raw summary mode."""
    n_l = 3
    task_raw = HierarchicalBernoulliGLM(n_l=n_l, summary="raw")
    prior = task_raw.get_prior()
    simulator = task_raw.get_simulator()

    parameters = prior(num_samples=5)
    observations = simulator(parameters)

    # Raw observations should be 100 points per context
    expected_dim = 100 * n_l
    assert observations.shape == (5, expected_dim)
    assert not torch.isnan(observations).any()


def test_reference_posterior_not_implemented():
    """Test that reference posterior raises NotImplementedError."""
    task = HierarchicalBernoulliGLM(n_l=3)

    with pytest.raises(NotImplementedError):
        task._sample_reference_posterior(num_samples=100, num_observation=1)

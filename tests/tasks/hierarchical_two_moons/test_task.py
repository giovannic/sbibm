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

    # Remaining 2*n_l params are local: should be roughly Normal-distributed
    local_params = samples[:, 4:]
    assert local_params.shape[1] == 2 * n_l

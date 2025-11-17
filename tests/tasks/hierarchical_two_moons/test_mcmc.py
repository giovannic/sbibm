"""Test MCMC sampling for hierarchical_two_moons."""

import pytest
import torch

from sbibm.tasks.hierarchical_two_moons.task import HierarchicalTwoMoons


@pytest.mark.parametrize("n_l", [3, 5])
def test_mcmc_sampling_basic(n_l):
    """Test that MCMC sampling runs and returns valid samples."""
    task = HierarchicalTwoMoons(n_l=n_l)

    # Generate a test observation
    prior = task.get_prior()
    true_params = prior(num_samples=1)
    simulator = task.get_simulator()
    observation = simulator(true_params)

    # Run very short MCMC (just to test it works)
    samples = task._sample_reference_posterior(
        num_samples=10,
        observation=observation,
        num_chains=1,
        num_warmup=50,
    )

    # Check shape
    expected_shape = (10, 4 + 2 * n_l)
    assert (
        samples.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {samples.shape}"

    # Check no NaN or Inf
    assert not torch.isnan(samples).any(), "Samples contain NaN"
    assert not torch.isinf(samples).any(), "Samples contain Inf"

    # Check scale parameters are positive (dims 2-3)
    assert (samples[:, 2:4] > 0).all(), "Scale parameters should be positive"


def test_mcmc_sampling_multiple_chains():
    """Test that MCMC sampling runs and returns valid samples."""
    n_l = 3
    task = HierarchicalTwoMoons(n_l=n_l)

    # Generate a test observation
    prior = task.get_prior()
    true_params = prior(num_samples=1)
    simulator = task.get_simulator()
    observation = simulator(true_params)

    # Run very short MCMC (just to test it works)
    samples = task._sample_reference_posterior(
        num_samples=10,
        observation=observation,
        num_chains=2,
        num_warmup=50,
    )

    # Check shape
    expected_shape = (10, 4 + 2 * n_l)
    assert (
        samples.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {samples.shape}"

    # Check no NaN or Inf
    assert not torch.isnan(samples).any(), "Samples contain NaN"
    assert not torch.isinf(samples).any(), "Samples contain Inf"

    # Check scale parameters are positive (dims 2-3)
    assert (samples[:, 2:4] > 0).all(), "Scale parameters should be positive"


def test_potential_function():
    """Test that potential function evaluates correctly at true parameters."""
    task = HierarchicalTwoMoons(n_l=3)

    # Generate a test observation from true params
    prior = task.get_prior()
    true_params = prior(num_samples=1)
    simulator = task.get_simulator()
    observation = simulator(true_params)

    # Get potential function
    potential_fn = task._get_potential_fn(observation)

    # Test potential at true params (should be finite since generated from it)
    true_params_unconstrained = task.composite_transform.inv(true_params.squeeze())
    z_dict = {"parameters": true_params_unconstrained}
    potential = potential_fn(z_dict)

    # Check it's a scalar
    assert potential.ndim == 0, "Potential should be scalar"

    # At true params, should be finite (since observation was generated from them)
    assert torch.isfinite(
        potential
    ), f"Potential at true params should be finite: {potential}"


def test_transform_round_trip():
    """Test that composite transform round-trips correctly."""
    task = HierarchicalTwoMoons(n_l=3)

    # Sample from prior (constrained space)
    prior = task.get_prior()
    params_constrained = prior(num_samples=10)

    # Transform to unconstrained
    params_unconstrained = task.composite_transform.inv(params_constrained)

    # Transform back to constrained
    params_reconstructed = task.composite_transform(params_unconstrained)

    # Check they match
    assert torch.allclose(
        params_constrained, params_reconstructed, atol=1e-5
    ), "Transform round-trip failed"

    # Check scale parameters stay positive
    assert (
        params_reconstructed[:, 2:4] > 0
    ).all(), "Scale parameters should remain positive after round-trip"

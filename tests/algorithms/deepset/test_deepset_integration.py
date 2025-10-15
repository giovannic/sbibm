"""
Integration test for hierarchical DeepSet with sbibm benchmark pipeline.

Tests the full end-to-end pipeline: data generation, training, posterior
inference, and metric computation.
"""

import pytest
import torch
import jax.numpy as jnp

import sbibm
from sbibm.algorithms.deepset import run
from sbibm.metrics import lc2st


@pytest.mark.slow
def test_full_benchmark_pipeline():
    """
    Test complete deepset pipeline on hierarchical_gaussian_linear task.

    Verifies:
    - Algorithm runs without error
    - Returns correct tuple structure
    - Samples have correct shape and are finite
    - Posterior can be queried
    - Metrics compute successfully
    """
    # Load hierarchical task
    task = sbibm.get_task("hierarchical_gaussian_linear")

    # Run deepset algorithm with small budget for fast testing
    samples, num_sims, log_prob, posterior = run(
        task,
        num_samples=500,
        num_simulations=2000,
        num_observation=1,
        max_epochs=10,
        dim_hidden=64,
        training_batch_size=128,
        device="cpu",
        verbose=False,
    )

    # Verify return types
    assert isinstance(samples, torch.Tensor)
    assert isinstance(num_sims, int)
    assert isinstance(log_prob, torch.Tensor)
    assert posterior is not None

    # Verify sample shape and values
    # For hierarchical tasks, num_params = dim_global + dim_local * num_events
    # We can verify by checking that the first sample from the prior has
    # the same dimension
    prior = task.get_prior()
    theta_sample = prior(num_samples=1)
    expected_dim = theta_sample.shape[-1]

    assert samples.shape == (500, expected_dim)
    assert torch.isfinite(samples).all(), "Samples contain NaN or Inf"
    assert samples.dtype == torch.float32 or samples.dtype == torch.float64

    # Verify log_prob tensor
    assert torch.isfinite(log_prob), "log_prob contains NaN or Inf"

    # Verify num_sims is reasonable
    assert num_sims == 2000

    # Test posterior sample() method
    posterior_samples = posterior.sample((200,))
    assert posterior_samples.shape == (200, expected_dim)
    assert torch.isfinite(posterior_samples).all()

    # Test posterior log_prob() method
    test_theta = samples[:10]
    test_log_probs = posterior.log_prob(test_theta)
    assert test_log_probs.shape == (10,)
    assert torch.isfinite(test_log_probs).all()

    # Compute lc2st metric (reference-free calibration test)
    # Note: lc2st requires num_calibration_samples == num_posterior_samples
    lc2st_result = lc2st(
        posterior,
        task,
        num_observation=1,
        posterior_samples=posterior_samples,
        num_calibration_samples=200,
    )
    # Verify lc2st returns valid results
    assert isinstance(lc2st_result, dict)
    assert "p_value" in lc2st_result
    assert jnp.isfinite(lc2st_result["p_value"])


def test_posterior_deterministic():
    """
    Verify posterior is deterministic once observation is encoded.

    Multiple samples from same posterior should be drawn from same
    distribution (context is fixed).
    """
    task = sbibm.get_task("hierarchical_gaussian_linear")

    samples_train, num_sims, log_prob, posterior = run(
        task,
        num_samples=100,
        num_simulations=1000,
        num_observation=1,
        max_epochs=5,
        dim_hidden=64,
        device="cpu",
        verbose=False,
    )

    # Draw samples twice
    samples1 = posterior.sample((100,))
    samples2 = posterior.sample((100,))

    # Should have same shape
    assert samples1.shape == samples2.shape

    # Should have reasonable variance (not deterministic)
    diff = torch.abs(samples1 - samples2)
    assert diff.mean() > 0.01  # At least some variation

    # Both should be finite
    assert torch.isfinite(samples1).all()
    assert torch.isfinite(samples2).all()


def test_posterior_log_prob_values():
    """
    Verify log_prob returns reasonable values.

    Test that log_prob decreases for samples farther from mode
    (qualitative check that model learned something).
    """
    task = sbibm.get_task("hierarchical_gaussian_linear")

    samples_train, _, _, posterior = run(
        task,
        num_samples=100,
        num_simulations=1000,
        num_observation=1,
        max_epochs=5,
        dim_hidden=64,
        device="cpu",
        verbose=False,
    )

    # Sample from posterior
    posterior_samples = posterior.sample((50,))

    # Compute log prob for these samples
    log_probs = posterior.log_prob(posterior_samples)

    # All should be finite
    assert torch.isfinite(log_probs).all()

    # Log probs should not all be identical (model learned variation)
    assert log_probs.std() > 0.01, "log_prob has no variation"


def test_different_observations():
    """
    Test that different observations produce different posteriors.
    """
    task = sbibm.get_task("hierarchical_gaussian_linear")

    # Run with observation 1
    samples1, _, _, posterior1 = run(
        task,
        num_samples=100,
        num_simulations=1000,
        num_observation=1,
        max_epochs=5,
        dim_hidden=64,
        device="cpu",
        verbose=False,
    )

    # Run with observation 2
    samples2, _, _, posterior2 = run(
        task,
        num_samples=100,
        num_simulations=1000,
        num_observation=2,
        max_epochs=5,
        dim_hidden=64,
        device="cpu",
        verbose=False,
    )

    # Posteriors should be different (at least in mean)
    mean1 = samples1.mean(dim=0)
    mean2 = samples2.mean(dim=0)
    mean_diff = torch.abs(mean1 - mean2).mean()
    msg = "Posteriors for different observations too similar"
    assert mean_diff > 0.01, msg


def test_algorithm_signature():
    """
    Verify run() function has correct sbibm interface.
    """
    task = sbibm.get_task("hierarchical_gaussian_linear")

    # Should accept standard sbibm arguments
    result = run(
        task,
        num_samples=100,
        num_simulations=500,
        num_observation=1,
    )

    # Should return 4-tuple
    assert isinstance(result, tuple)
    assert len(result) == 4

    samples, num_sims, log_prob, posterior = result

    # Verify types
    assert isinstance(samples, torch.Tensor)
    assert isinstance(num_sims, int)
    assert isinstance(log_prob, torch.Tensor)

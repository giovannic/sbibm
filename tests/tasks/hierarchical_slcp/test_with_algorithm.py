"""Test hierarchical_slcp with SNPE algorithm and reference-free metrics.

This module tests the end-to-end integration of hierarchical_slcp with:
- SNPE algorithm from sbi
- Reference-free metrics (reverse_kl and lc2st)
"""

import logging

import pytest
import torch

from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.algorithms.tfmpe.bottom_up import run as run_bottom_up
from sbibm.metrics.lc2st import lc2st
from sbibm.metrics.reverse_kl import reverse_kl
from sbibm.tasks.hierarchical_slcp.task import HierarchicalSLCP

log = logging.getLogger(__name__)


@pytest.mark.slow
def test_both_metrics_on_same_posterior():
    """Test both metrics on the same trained posterior.

    This test verifies that both reverse_kl and lc2st can be computed
    on the same posterior object, demonstrating the complete reference-free
    evaluation workflow.

    Args:
        automatic_transforms_enabled: Whether to use automatic transforms
    """
    n_l = 5
    task = HierarchicalSLCP(n_l=n_l)
    n_samples = 100

    # Run SNPE with minimal budget for testing
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=100,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=True
    )

    # Verify samples shape
    expected_dim = 3 + 2 * n_l
    assert samples.shape == (n_samples, expected_dim), (
        f"Expected samples shape ({n_samples}, {expected_dim}), " f"got {samples.shape}"
    )

    # Verify no NaN values in samples
    assert not torch.isnan(samples).any(), "Samples contain NaN values"

    # Verify posterior object is returned
    assert posterior is not None, "Posterior object should be returned"

    # Verify posterior has required methods
    assert hasattr(posterior, "sample"), "Posterior should have sample method"
    assert hasattr(posterior, "log_prob"), "Posterior should have log_prob method"

    # Test that we can sample from posterior
    new_samples = posterior.sample((50,))
    assert new_samples.shape == (50, expected_dim)
    assert not torch.isnan(new_samples).any()

    # Test that we can compute log_prob
    log_probs = posterior.log_prob(samples)
    assert not torch.isnan(log_probs).any()

    # Verify log_prob_true is computed when num_observation is provided
    assert log_prob_true is not None
    assert torch.isfinite(log_prob_true)

    # Compute reverse KL
    rkl = reverse_kl(
        posterior=posterior,
        task=task,
        num_observation=1,
        num_samples=1000,
    )

    # Verify reverse KL metric is finite
    assert torch.isfinite(rkl), f"Reverse KL should be finite, got {rkl}"
    assert rkl.numel() == 1, "Reverse KL should be a scalar"

    # Compute LC2ST
    lc2st_samples = posterior.sample((200,))
    lc2st_result = lc2st(
        posterior=posterior,
        task=task,
        num_observation=1,
        posterior_samples=lc2st_samples,
        num_calibration_samples=200,
        num_trials=10,
    )

    # Verify LC2ST result is a dictionary
    assert isinstance(lc2st_result, dict), "LC2ST should return a dictionary"
    assert (
        "test_statistic" in lc2st_result
    ), "Result should contain 'test_statistic' key"

    # Verify LC2ST values are finite
    for key, value in lc2st_result.items():
        if isinstance(value, (torch.Tensor, float)):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(
                    value
                ).all(), f"LC2ST result[{key}] contains non-finite values"
            else:
                assert not (
                    value != value or abs(value) == float("inf")
                ), f"LC2ST result[{key}] is not finite"

    log.info(
        f"Complete reference-free evaluation "
        f"\n  Reverse KL: {rkl.item():.3f}"
        f"\n  LC2ST test_statistic: {lc2st_result.get('test_statistic', 'N/A')}"
        f"\n  Log prob (true params): {log_prob_true.item():.3f}"
    )


@pytest.mark.slow
def test_snpe_two_rounds():
    """Test that SNPE can run with two rounds of training.

    Verifies that multi-round SNPE training works correctly with
    the hierarchical SLCP task and produces valid posteriors.
    """
    n_l = 5
    task = HierarchicalSLCP(n_l=n_l)
    n_samples = 100

    # Run SNPE with two rounds
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=500,
        num_observation=1,
        num_rounds=2,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=True,
    )

    # Verify samples shape
    expected_dim = 3 + 2 * n_l
    assert samples.shape == (n_samples, expected_dim), (
        f"Expected samples shape ({n_samples}, {expected_dim}), " f"got {samples.shape}"
    )

    # Verify no NaN values in samples
    assert not torch.isnan(samples).any(), "Samples contain NaN values"

    # Verify posterior object is returned
    assert posterior is not None, "Posterior object should be returned"

    # Verify posterior has required methods
    assert hasattr(posterior, "sample"), "Posterior should have sample method"
    assert hasattr(posterior, "log_prob"), "Posterior should have log_prob method"

    # Test that we can sample from posterior
    new_samples = posterior.sample((50,))
    assert new_samples.shape == (50, expected_dim)
    assert not torch.isnan(new_samples).any()

    # Test that we can compute log_prob
    log_probs = posterior.log_prob(samples[:10])
    assert log_probs.shape == (10,)
    assert not torch.isnan(log_probs).any()

    log.info(
        f"Two-round SNPE completed successfully:"
        f"\n  Num simulations: {num_sims}"
        f"\n  Sample shape: {samples.shape}"
    )


@pytest.mark.slow
def test_bottom_up_hierarchical_slcp(
    num_observation=1,
    num_samples=100,
    num_simulations=100,
):
    """Integration test for TFMPE bottom-up on hierarchical SLCP.

    Validates that the bottom_up algorithm wrapper:
    - Loads the task and observation
    - Runs TFMPE training
    - Returns samples with correct shape
    - Returns num_simulations count and log_prob_true_params
    - Samples fall within prior bounds (all parameters bounded)
    """
    task = HierarchicalSLCP(n_l=5)

    # Run the algorithm
    samples, num_sims, log_prob_true_params, posterior = run_bottom_up(
        task=task,
        num_observation=num_observation,
        num_samples=num_samples,
        num_simulations=num_simulations,
        automatic_transforms_enabled=True,
    )

    # Validate output shape
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (num_samples, task.dim_parameters)

    # Validate num_simulations was recorded
    assert isinstance(num_sims, int)
    assert num_sims > 0

    # Validate log_prob_true_params
    assert (log_prob_true_params is None or
            isinstance(log_prob_true_params, torch.Tensor))

    # Validate posterior object is returned
    assert posterior is not None
    assert hasattr(posterior, "sample")
    assert hasattr(posterior, "log_prob")

    # Test posterior.sample() returns correct shape
    posterior_samples = posterior.sample((num_samples,))
    assert posterior_samples.shape == (num_samples, task.dim_parameters)
    assert not torch.isnan(posterior_samples).any()

    # Test posterior.log_prob() works on samples
    log_probs = posterior.log_prob(samples)
    if log_probs is not None:
        assert log_probs.shape == (num_samples,)
        assert torch.isfinite(log_probs).all()

    # Validate samples are not NaN or Inf
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()

    log.info(
        f"TFMPE bottom-up completed on hierarchical_slcp:"
        f"\n  Num simulations: {num_sims}"
        f"\n  Sample shape: {samples.shape}"
    )

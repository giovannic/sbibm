"""Test hierarchical_bernoulli_glm with SNPE algorithm and reference-free
metrics.

This module tests the end-to-end integration of hierarchical_bernoulli_glm
with:
- SNPE algorithm from sbi
- Reference-free metrics (reverse_kl and lc2st)
"""

import logging

import pytest
import torch

from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.metrics.lc2st import lc2st
from sbibm.metrics.reverse_kl import reverse_kl
from sbibm.tasks.hierarchical_bernoulli_glm.task import (
    HierarchicalBernoulliGLM,
)

log = logging.getLogger(__name__)


def test_both_metrics_on_same_posterior():
    """Test both metrics on the same trained posterior.

    This test verifies that both reverse_kl and lc2st can be computed
    on the same posterior object, demonstrating the complete reference-free
    evaluation workflow.

    Note: All parameters are unconstrained (MVN for filter, Normal for
    offsets), so automatic_transforms_enabled=False is used.
    """
    n_l = 5
    task = HierarchicalBernoulliGLM(n_l=n_l)
    n_samples = 100

    # Run SNPE with minimal budget for testing
    # All params unconstrained, no need for transforms
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=100,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=False,
    )

    # Verify samples shape
    expected_dim = 9 + n_l  # 9 global (filter) + n_l local (offsets)
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
    lc2st_result = lc2st(
        posterior=posterior,
        task=task,
        num_observation=1,
        num_calibration_samples=10,
        num_posterior_samples=10,
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
        f"Complete reference-free evaluation:"
        f"\n  Reverse KL: {rkl.item():.3f}"
        f"\n  LC2ST test_statistic: "
        f"{lc2st_result.get('test_statistic', 'N/A')}"
        f"\n  Log prob (true params): {log_prob_true.item():.3f}"
    )


def test_snpe_samples_reasonable():
    """Test that SNPE samples are reasonable.

    Verifies that samples from the trained posterior are in reasonable
    ranges for the Bernoulli GLM parameters.
    """
    n_l = 5
    task = HierarchicalBernoulliGLM(n_l=n_l)

    # Run SNPE with minimal budget for testing
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=100,
        num_simulations=1000,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=False,
    )

    # Check filter parameters (dims 0-8)
    filter_params = samples[:, :9]
    # Filter params should be reasonably bounded (MVN prior)
    assert (
        torch.abs(filter_params) < 10.0
    ).all(), f"Filter params too large: max={torch.abs(filter_params).max().item()}"

    # Check offset parameters (dims 9+, should be Normal(0, 0.5))
    offset_params = samples[:, 9:]
    # Most offsets should be within a few std devs
    assert (
        torch.abs(offset_params) < 5.0
    ).all(), f"Offset params too large: max={torch.abs(offset_params).max().item()}"

    log.info(
        f"Sample ranges:"
        f"\n  Filter params: [{filter_params.min().item():.3f}, "
        f"{filter_params.max().item():.3f}]"
        f"\n  Offset params: [{offset_params.min().item():.3f}, "
        f"{offset_params.max().item():.3f}]"
    )


def test_snpe_two_rounds():
    """Test that SNPE can run with two rounds of training.

    Verifies that multi-round SNPE training works correctly with
    the hierarchical Bernoulli GLM task and produces valid posteriors.
    """
    n_l = 5
    task = HierarchicalBernoulliGLM(n_l=n_l)
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
        automatic_transforms_enabled=False,
    )

    # Verify samples shape
    expected_dim = 9 + n_l
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

    # Verify samples are reasonable
    filter_params = samples[:, :9]
    offset_params = samples[:, 9:]

    assert (torch.abs(filter_params) < 10.0).all()
    assert (torch.abs(offset_params) < 5.0).all()

    log.info(
        f"Two-round SNPE completed successfully:"
        f"\n  Num simulations: {num_sims}"
        f"\n  Sample shape: {samples.shape}"
        f"\n  Filter params range: [{filter_params.min().item():.3f}, "
        f"{filter_params.max().item():.3f}]"
        f"\n  Offset params range: [{offset_params.min().item():.3f}, "
        f"{offset_params.max().item():.3f}]"
    )

"""Test hierarchical_gaussian_mixture with SNPE algorithm and
reference-free metrics.

This module tests the end-to-end integration of hierarchical_gaussian_mixture
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
from sbibm.tasks.hierarchical_gaussian_mixture.task import (
    HierarchicalGaussianMixture,
)

log = logging.getLogger(__name__)


def test_snpe_with_reference_free_metrics():
    """Test SNPE with both reference-free metrics.

    This test verifies that both reverse_kl and lc2st can be computed
    on a trained posterior, demonstrating the complete reference-free
    evaluation workflow.
    """
    n_l = 5
    dim = 2
    task = HierarchicalGaussianMixture(n_l=n_l, dim=dim)
    n_samples = 100

    # Run SNPE with minimal budget for testing
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=1000,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=True,
    )

    # Verify samples shape
    expected_dim = 2 * dim + dim * n_l
    assert samples.shape == (n_samples, expected_dim)

    # Verify no NaN values in samples
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()

    # Verify posterior object is returned
    assert posterior is not None

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

    # Verify global scale parameters are positive
    global_scale = samples[:, dim : 2 * dim]
    assert (global_scale >= 0.0).all()

    # Compute reverse KL
    rkl = reverse_kl(
        posterior=posterior,
        task=task,
        num_observation=1,
        num_samples=1000,
    )

    # Verify reverse KL metric is finite
    assert torch.isfinite(rkl)
    assert rkl.numel() == 1

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
    assert isinstance(lc2st_result, dict)
    assert "test_statistic" in lc2st_result

    # Verify LC2ST values are finite
    for key, value in lc2st_result.items():
        if isinstance(value, (torch.Tensor, float)):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
            else:
                assert not (value != value or abs(value) == float("inf"))

    log.info(
        f"Complete reference-free evaluation:"
        f"\n  Reverse KL: {rkl.item():.3f}"
        f"\n  LC2ST test_statistic: "
        f"{lc2st_result.get('test_statistic', 'N/A')}"
        f"\n  Log prob (true params): {log_prob_true.item():.3f}"
    )

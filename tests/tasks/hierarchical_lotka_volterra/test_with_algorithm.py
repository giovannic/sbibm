"""Test hierarchical_lotka_volterra with SNPE algorithm and reference-free
metrics.

This module tests the end-to-end integration of hierarchical_lotka_volterra
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
from sbibm.tasks.hierarchical_lotka_volterra.task import (
    HierarchicalLotkaVolterra,
)

log = logging.getLogger(__name__)


def test_both_metrics_on_same_posterior():
    """Test both metrics on the same trained posterior.

    This test verifies that both reverse_kl and lc2st can be computed
    on the same posterior object, demonstrating the complete reference-free
    evaluation workflow.
    """
    n_l = 5
    task = HierarchicalLotkaVolterra(n_l=n_l)
    n_samples = 100

    # Run SNPE with minimal budget for testing
    # All params are LogNormal (positive), use transforms
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=500,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=True,
    )

    # Verify samples shape
    expected_dim = 2 + 2 * n_l  # 2 global + 2*n_l local
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
    ranges for the Lotka-Volterra parameters (all positive due to LogNormal).
    """
    n_l = 5
    task = HierarchicalLotkaVolterra(n_l=n_l)

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
        automatic_transforms_enabled=True,
    )

    # All params should be positive (LogNormal)
    assert (samples > 0).all(), "All Lotka-Volterra params should be positive"

    # Check global parameters (dims 0-1: beta, gamma)
    global_params = samples[:, :2]
    # Beta (predation) should be around 0.028 (can vary widely)
    assert (
        global_params[:, 0] < 1.0
    ).all(), "Beta (predation) seems too large"

    # Gamma (predator death) should be around 0.5 (can vary)
    assert (
        global_params[:, 1] < 5.0
    ).all(), "Gamma (predator death) seems too large"

    # Check local parameters (dims 2+: alpha_i, delta_i per site)
    local_params = samples[:, 2:]
    # All should be positive and bounded
    assert (
        local_params < 10.0
    ).all(), f"Local params too large: max={local_params.max().item()}"

    log.info(
        f"Sample ranges:"
        f"\n  Beta (predation): [{global_params[:, 0].min().item():.4f}, "
        f"{global_params[:, 0].max().item():.4f}]"
        f"\n  Gamma (predator death): [{global_params[:, 1].min().item():.4f}, "
        f"{global_params[:, 1].max().item():.4f}]"
        f"\n  Local params: [{local_params.min().item():.4f}, "
        f"{local_params.max().item():.4f}]"
    )

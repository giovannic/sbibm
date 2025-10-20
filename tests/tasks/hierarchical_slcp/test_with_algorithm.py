"""Test hierarchical_slcp with SNPE algorithm and reference-free metrics.

This module tests the end-to-end integration of hierarchical_slcp with:
- SNPE algorithm from sbi
- Reference-free metrics (reverse_kl and lc2st)
"""

import logging

import pytest
import torch

from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.metrics.lc2st import lc2st
from sbibm.metrics.reverse_kl import reverse_kl
from sbibm.tasks.hierarchical_slcp.task import HierarchicalSLCP

log = logging.getLogger(__name__)


@pytest.mark.parametrize("automatic_transforms_enabled", [True, False])
def test_both_metrics_on_same_posterior(automatic_transforms_enabled):
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
        automatic_transforms_enabled=automatic_transforms_enabled,
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
        f"Complete reference-free evaluation "
        f"(transforms={'enabled' if automatic_transforms_enabled else 'disabled'}):"
        f"\n  Reverse KL: {rkl.item():.3f}"
        f"\n  LC2ST test_statistic: {lc2st_result.get('test_statistic', 'N/A')}"
        f"\n  Log prob (true params): {log_prob_true.item():.3f}"
    )


def test_snpe_samples_in_valid_range():
    """Test that SNPE samples are within the valid parameter ranges.

    Verifies that samples from the trained posterior respect the parameter
    constraints:
    - s1, s2: [0.5, 3.0]
    - rho: [-3, 3]
    - Local means: [-3, 3]
    """
    automatic_transforms_enabled = True
    n_l = 5
    task = HierarchicalSLCP(n_l=n_l)

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
        automatic_transforms_enabled=automatic_transforms_enabled,
    )

    # Check s1, s2 parameters (dims 0-1, should be in [0.5, 3.0])
    s_params = samples[:, :2]
    assert (s_params >= 0.5).all(), f"s params below 0.5: min={s_params.min().item()}"
    assert (s_params <= 3.0).all(), f"s params above 3.0: max={s_params.max().item()}"

    # Check rho parameter (dim 2, should be in [-3, 3])
    rho_param = samples[:, 2]
    assert (rho_param >= -3.0).all(), f"rho below -3: min={rho_param.min().item()}"
    assert (rho_param <= 3.0).all(), f"rho above 3: max={rho_param.max().item()}"

    # Check local means (dims 3+, should be in [-3, 3])
    local_means = samples[:, 3:]
    assert (
        local_means >= -3.0
    ).all(), f"Local means below -3: min={local_means.min().item()}"
    assert (
        local_means <= 3.0
    ).all(), f"Local means above 3: max={local_means.max().item()}"

    log.info(
        f"Sample ranges (transforms={'enabled' if automatic_transforms_enabled else 'disabled'}):"
        f"\n  s params: [{s_params.min().item():.3f}, {s_params.max().item():.3f}]"
        f"\n  rho: [{rho_param.min().item():.3f}, {rho_param.max().item():.3f}]"
        f"\n  Local means: [{local_means.min().item():.3f}, {local_means.max().item():.3f}]"
    )


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

    # Verify samples respect parameter constraints
    # s1, s2 (dims 0-1): [0.5, 3.0]
    s_params = samples[:, :2]
    assert (s_params >= 0.5).all(), f"s params below 0.5: min={s_params.min().item()}"
    assert (s_params <= 3.0).all(), f"s params above 3.0: max={s_params.max().item()}"

    # rho (dim 2): [-3, 3]
    rho_param = samples[:, 2]
    assert (rho_param >= -3.0).all(), f"rho below -3: min={rho_param.min().item()}"
    assert (rho_param <= 3.0).all(), f"rho above 3: max={rho_param.max().item()}"

    # Local means (dims 3+): [-3, 3]
    local_means = samples[:, 3:]
    assert (
        local_means >= -3.0
    ).all(), f"Local means below -3: min={local_means.min().item()}"
    assert (
        local_means <= 3.0
    ).all(), f"Local means above 3: max={local_means.max().item()}"

    log.info(
        f"Two-round SNPE completed successfully:"
        f"\n  Num simulations: {num_sims}"
        f"\n  Sample shape: {samples.shape}"
        f"\n  s params range: [{s_params.min().item():.3f}, "
        f"{s_params.max().item():.3f}]"
        f"\n  rho range: [{rho_param.min().item():.3f}, "
        f"{rho_param.max().item():.3f}]"
        f"\n  Local means range: [{local_means.min().item():.3f}, "
        f"{local_means.max().item():.3f}]"
    )

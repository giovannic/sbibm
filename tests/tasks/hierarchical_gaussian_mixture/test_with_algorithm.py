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
from sbibm.algorithms.tfmpe.bottom_up import run as run_bottom_up
from sbibm.metrics.lc2st import lc2st
from sbibm.metrics.reverse_kl import reverse_kl
from sbibm.tasks.hierarchical_gaussian_mixture.task import (
    HierarchicalGaussianMixture,
)

log = logging.getLogger(__name__)


@pytest.mark.slow
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


@pytest.mark.slow
def test_bottom_up_hierarchical_gaussian_mixture(
    num_observation=1,
    num_samples=100,
    num_simulations=100,
):
    """Integration test for TFMPE bottom-up on hierarchical Gaussian
    mixture.

    Validates that the bottom_up algorithm wrapper:
    - Loads the task and observation
    - Runs TFMPE training
    - Returns samples with correct shape
    - Returns num_simulations count and log_prob_true_params
    - Samples fall within prior bounds (global locs/local bounded,
      global scales positive)
    """
    dim = 2
    task = HierarchicalGaussianMixture(n_l=5, dim=dim)

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

    # Validate global location parameters are bounded [-10, 10]
    global_loc = samples[:, :dim]
    assert (
        global_loc.min() >= -10.0
    ), f"global loc must be >= -10 but found {global_loc.min()}"
    assert (
        global_loc.max() <= 10.0
    ), f"global loc must be <= 10 but found {global_loc.max()}"

    # Validate global scale parameters are positive
    global_scale = samples[:, dim : 2 * dim]
    assert (
        global_scale.min() >= 0.0
    ), f"global scale must be positive but found {global_scale.min()}"

    # Validate local parameters are bounded [-10, 10]
    local_params = samples[:, 2 * dim :]
    assert (
        local_params.min() >= -10.0
    ), f"local params must be >= -10 but found {local_params.min()}"
    assert (
        local_params.max() <= 10.0
    ), f"local params must be <= 10 but found {local_params.max()}"

    log.info(
        f"TFMPE bottom-up completed on hierarchical_gaussian_mixture:"
        f"\n  Num simulations: {num_sims}"
        f"\n  Sample shape: {samples.shape}"
        f"\n  Global loc range: [{global_loc.min().item():.3f}, "
        f"{global_loc.max().item():.3f}]"
        f"\n  Global scale range: [{global_scale.min().item():.3f}, "
        f"{global_scale.max().item():.3f}]"
        f"\n  Local params range: [{local_params.min().item():.3f}, "
        f"{local_params.max().item():.3f}]"
    )

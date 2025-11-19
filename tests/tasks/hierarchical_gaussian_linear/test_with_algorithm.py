"""Test hierarchical_gaussian_linear with SNPE algorithm.

This module tests the end-to-end integration of hierarchical_gaussian_linear
with SNPE algorithm from sbi.
"""

import logging

import pytest
import torch

from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.algorithms.tfmpe.bottom_up import run as run_bottom_up
from sbibm.tasks.hierarchical_gaussian_linear.task import (
    HierarchicalGaussianLinear,
)

log = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.parametrize("automatic_transforms_enabled", [True, False])
def test_snpe_integration(automatic_transforms_enabled):
    """Test hierarchical_gaussian_linear with SNPE algorithm.

    This test verifies that SNPE can be trained on the hierarchical task
    and produces valid posterior samples, with and without automatic
    transforms.

    Args:
        automatic_transforms_enabled: Whether to use automatic transforms
    """
    # Use n_l=5 to match pre-generated observations
    n_l = 5
    task = HierarchicalGaussianLinear(n_l=n_l)
    n_samples = 100

    # Run SNPE with minimal budget for testing
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=500,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=5,
        automatic_transforms_enabled=automatic_transforms_enabled,
    )

    # Verify samples shape
    expected_dim = 10 + n_l
    assert samples.shape == (n_samples, expected_dim), (
        f"Expected samples shape ({n_samples}, {expected_dim}), " f"got {samples.shape}"
    )

    # Verify no NaN values in samples
    assert not torch.isnan(samples).any(), "Samples contain NaN values"

    # Verify posterior object is returned
    assert posterior is not None, "Posterior object should be returned"

    # Verify global parameters (mean structure) are reasonable
    global_params = samples[:, :10]
    assert torch.all(
        torch.abs(global_params) < 10
    ), "Global parameters should be reasonable"

    # Verify local parameters (noise scales) are positive
    local_scales = samples[:, 10:]
    assert torch.all(local_scales >= 0), "Local noise scales should be positive"

    log.info(f"SNPE integration test passed with {num_sims} simulations")


def test_bottom_up_hierarchical_gaussian_linear(
    num_observation=1,
    num_samples=100,
    num_simulations=100,
):
    """Integration test for TFMPE bottom-up on hierarchical Gaussian
    linear.

    Validates that the bottom_up algorithm wrapper:
    - Loads the task and observation
    - Runs TFMPE training
    - Returns samples with correct shape
    - Returns execution time and metadata
    - Samples fall within prior bounds (global scale > 0)
    """
    task = HierarchicalGaussianLinear(n_l=5)

    # Run the algorithm
    samples, execution_time, metadata = run_bottom_up(
        task=task,
        num_observation=num_observation,
        num_samples=num_samples,
        num_simulations=num_simulations,
        automatic_transforms_enabled=True,
    )

    # Validate output shape
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (num_samples, task.dim_parameters)

    # Validate execution time was recorded
    assert isinstance(execution_time, float)
    assert execution_time > 0.0

    # Validate metadata
    assert isinstance(metadata, dict)
    assert "losses" in metadata
    assert "n_samples_per_round" in metadata

    # Validate samples are not NaN or Inf
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()

    # Validate global scale (first parameter) is positive
    global_scale = samples[:, 0]
    assert (
        global_scale.min() >= 0
    ), f"global scale must be positive but found {global_scale.min()}"

    # Validate local means are within reasonable bounds
    # (Normal distribution, so shouldn't be extreme)
    local_means = samples[:, 1:]
    assert not torch.isnan(local_means).any()
    assert not torch.isinf(local_means).any()

    log.info(
        f"TFMPE bottom-up completed on hierarchical_gaussian_linear:"
        f"\n  Num simulations: {num_simulations}"
        f"\n  Sample shape: {samples.shape}"
        f"\n  Global scale range: [{global_scale.min().item():.3f}, "
        f"{global_scale.max().item():.3f}]"
        f"\n  Local means range: [{local_means.min().item():.3f}, "
        f"{local_means.max().item():.3f}]"
    )

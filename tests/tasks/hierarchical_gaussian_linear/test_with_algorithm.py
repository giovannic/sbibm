"""Test hierarchical_gaussian_linear with SNPE algorithm.

This module tests the end-to-end integration of hierarchical_gaussian_linear
with SNPE algorithm from sbi.
"""

import logging

import pytest
import torch

from sbibm.algorithms.sbi.snpe import run as run_snpe
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
        neural_net="maf",
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
    assert torch.all(torch.abs(global_params) < 10), "Global parameters should be reasonable"

    # Verify local parameters (noise scales) are positive
    local_scales = samples[:, 10:]
    assert torch.all(local_scales >= 0), "Local noise scales should be positive"

    log.info(f"SNPE integration test passed with {num_sims} simulations")

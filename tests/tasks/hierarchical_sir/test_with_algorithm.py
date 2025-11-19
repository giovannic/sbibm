"""Test hierarchical_sir with SNPE and TFMPE algorithms.

This module tests the end-to-end integration of hierarchical_sir
with SNPE algorithm from sbi and TFMPE bottom-up algorithm.
"""

import logging

import pytest
import torch

from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.algorithms.tfmpe.bottom_up import run as run_bottom_up
from sbibm.tasks.hierarchical_sir.task import HierarchicalSIR

log = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.parametrize("automatic_transforms_enabled", [True, False])
def test_snpe_integration(automatic_transforms_enabled):
    """Test hierarchical_sir with SNPE algorithm.

    This test verifies that SNPE can be trained on the hierarchical task
    and produces valid posterior samples, with and without automatic
    transforms.

    Args:
        automatic_transforms_enabled: Whether to use automatic transforms
    """
    # Use n_l=5 to match pre-generated observations
    n_l = 5
    task = HierarchicalSIR(n_l=n_l)
    n_samples = 100

    # Run SNPE with minimal budget for testing
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=10,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=5,
        automatic_transforms_enabled=automatic_transforms_enabled,
    )

    # Verify samples shape
    expected_dim = 1 + n_l  # 1 global (beta) + n_l local (gamma)
    assert samples.shape == (n_samples, expected_dim), (
        f"Expected samples shape ({n_samples}, {expected_dim}), "
        f"got {samples.shape}"
    )

    # Verify no NaN values in samples
    assert not torch.isnan(samples).any(), "Samples contain NaN values"

    # Verify posterior object is returned
    assert posterior is not None, "Posterior object should be returned"

    # Verify global parameter (beta) is positive
    beta = samples[:, 0]
    assert torch.all(beta > 0), "Beta (transmission rate) should be positive"

    # Verify local parameters (gammas) are positive
    gammas = samples[:, 1:]
    assert torch.all(gammas > 0), "Gammas (recovery rates) should be positive"

    log.info(f"SNPE integration test passed with {num_sims} simulations")


def test_bottom_up_hierarchical_sir(
    task_name="hierarchical_sir",
    num_observation=1,
    num_samples=100,
    num_simulations=100,
):
    """Integration test for TFMPE bottom-up on hierarchical SIR.

    Validates that the bottom_up algorithm wrapper:
    - Loads the task and observation
    - Runs TFMPE training
    - Returns samples with correct shape
    - Returns execution time and metadata
    - Samples fall within prior bounds (all params positive)
    """
    task = HierarchicalSIR(n_l=5)

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
    expected_dim = 1 + task.n_l  # 1 global (beta) + n_l local (gammas)
    assert samples.shape == (num_samples, expected_dim), (
        f"Expected samples shape ({num_samples}, {expected_dim}), "
        f"got {samples.shape}"
    )

    # Validate execution time was recorded
    assert isinstance(execution_time, float)
    assert execution_time > 0.0

    # Validate metadata
    assert isinstance(metadata, dict)
    assert "losses" in metadata
    assert "n_samples_per_round" in metadata

    # Validate samples are not NaN or Inf
    assert not torch.isnan(samples).any(), "Samples contain NaN values"
    assert not torch.isinf(samples).any(), "Samples contain Inf values"

    # Verify global parameter (beta) is positive
    beta = samples[:, 0]
    assert torch.all(beta > 0), (
        f"Beta (transmission rate) should be positive, "
        f"but found min={beta.min().item()}"
    )

    # Verify local parameters (gammas) are positive
    gammas = samples[:, 1:]
    assert torch.all(gammas > 0), (
        f"Gammas (recovery rates) should be positive, "
        f"but found min={gammas.min().item()}"
    )

    log.info(
        f"TFMPE bottom-up completed on hierarchical_sir:"
        f"\n  Num simulations: {num_simulations}"
        f"\n  Sample shape: {samples.shape}"
        f"\n  Beta range: [{beta.min().item():.3f}, "
        f"{beta.max().item():.3f}]"
        f"\n  Gammas range: [{gammas.min().item():.3f}, "
        f"{gammas.max().item():.3f}]"
    )

"""Test hierarchical_gaussian_linear with SNPE algorithm.

This module tests the end-to-end integration of hierarchical_gaussian_linear
with SNPE algorithm from sbi.
"""

import logging

import pytest
import torch

from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.algorithms.simformer.run import run as run_simformer
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
    expected_dim = task.dim_global + task.dim_local_total
    assert samples.shape == (n_samples, expected_dim), (
        f"Expected samples shape ({n_samples}, {expected_dim}), " f"got {samples.shape}"
    )

    # Verify no NaN values in samples
    assert not torch.isnan(samples).any(), "Samples contain NaN values"

    # Verify posterior object is returned
    assert posterior is not None, "Posterior object should be returned"

    # Verify global parameters (noise scale) is reasonable
    global_params = samples[:, :task.dim_global]
    assert torch.all(
        torch.abs(global_params) > 0
    ), "Global parameters should be reasonable"

    # Verify local parameters (mus) are positive
    local_scales = samples[:, task.dim_global:]
    assert torch.isfinite(local_scales).all(), "Local mu should be finite"

    log.info(f"SNPE integration test passed with {num_sims} simulations")


@pytest.mark.slow
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
    - Returns num_simulations count and log_prob_true_params
    - Samples fall within prior bounds (global scale > 0)
    """
    task = HierarchicalGaussianLinear(n_l=5)

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
        f"\n  Num simulations: {num_sims}"
        f"\n  Sample shape: {samples.shape}"
        f"\n  Global scale range: [{global_scale.min().item():.3f}, "
        f"{global_scale.max().item():.3f}]"
        f"\n  Local means range: [{local_means.min().item():.3f}, "
        f"{local_means.max().item():.3f}]"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_simulations,training_batch_size,sampling_steps,"
    "train_step_floor,train_step_cap",
    [(50, 25, 10, 10, 20)],
)
def test_simformer(
    num_simulations,
    training_batch_size,
    sampling_steps,
    train_step_floor,
    train_step_cap,
):
    """Smoke test that simformer runs on hierarchical_gaussian_linear.

    Parameterised with tiny budgets to keep test time short.
    """
    num_samples = 20
    task = HierarchicalGaussianLinear(n_l=5)

    samples, num_sims, log_prob_true_params, posterior = run_simformer(
        task=task,
        num_samples=num_samples,
        num_simulations=num_simulations,
        num_observation=1,
        training_batch_size=training_batch_size,
        sampling_steps=sampling_steps,
        train_step_floor=train_step_floor,
        train_step_cap=train_step_cap,
        automatic_transforms_enabled=True,
    )

    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (num_samples, task.dim_parameters)
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()
    assert num_sims == num_simulations
    assert log_prob_true_params is None
    assert posterior is not None
    assert hasattr(posterior, "sample")

    resampled = posterior.sample((num_samples,))
    assert resampled.shape == (num_samples, task.dim_parameters)
    assert not torch.isnan(resampled).any()

    log.info(
        f"Simformer ran on hierarchical_gaussian_linear with "
        f"num_simulations={num_sims}, sample shape={samples.shape}"
    )

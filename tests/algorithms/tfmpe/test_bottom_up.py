import torch

import sbibm
from sbibm.algorithms.tfmpe.bottom_up import run


def test_bottom_up_hierarchical_two_moons(
    task_name="hierarchical_two_moons",
    num_observation=1,
    num_samples=100,
    num_simulations=100,
):
    """Integration test for TFMPE bottom-up on hierarchical two moons.

    Validates that the bottom_up algorithm wrapper:
    - Loads the task and observation
    - Runs TFMPE training
    - Returns samples with correct shape
    - Returns execution time and metadata
    - Samples fall within prior bounds
    """
    task = sbibm.get_task(task_name)

    # Run the algorithm
    samples, execution_time, metadata = run(
        task=task,
        num_observation=num_observation,
        num_samples=num_samples,
        num_simulations=num_simulations,
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

    # Validate samples fall strictly within prior bounds [-1, 1]
    assert samples.min() >= -1.0, f"Sample min {samples.min()} < -1.0"
    assert samples.max() <= 1.0, f"Sample max {samples.max()} > 1.0"

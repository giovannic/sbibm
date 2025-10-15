"""Tests for in-memory observation generation.

Verifies that observations generated in-memory from seeds are deterministic
and that RNG state is properly isolated.
"""
import numpy as np
import pytest
import torch

import sbibm


HIERARCHICAL_TASKS = [
    "hierarchical_two_moons",
    "hierarchical_gaussian_linear",
    "hierarchical_gaussian_linear_uniform",
    "hierarchical_gaussian_mixture",
    "hierarchical_slcp",
]


@pytest.mark.parametrize("task_name", HIERARCHICAL_TASKS)
def test_in_memory_is_deterministic(task_name):
    """Two independent task instances produce identical observations."""
    task_a = sbibm.get_task(task_name)
    task_b = sbibm.get_task(task_name)

    for num_obs in [1, 2]:
        obs_a = task_a.get_observation(num_obs)
        obs_b = task_b.get_observation(num_obs)
        assert torch.allclose(obs_a, obs_b, atol=1e-6), (
            f"{task_name} obs {num_obs}: not deterministic across instances"
        )

        params_a = task_a.get_true_parameters(num_obs)
        params_b = task_b.get_true_parameters(num_obs)
        assert torch.allclose(params_a, params_b, atol=1e-6), (
            f"{task_name} params {num_obs}: not deterministic across instances"
        )


@pytest.mark.parametrize("task_name", HIERARCHICAL_TASKS)
def test_in_memory_correct_shapes(task_name):
    """In-memory observations have correct shapes matching task dimensions."""
    task = sbibm.get_task(task_name)
    obs = task.get_observation(1)
    params = task.get_true_parameters(1)

    assert obs.shape == (1, task.dim_data), (
        f"{task_name}: obs shape {obs.shape} != (1, {task.dim_data})"
    )
    assert params.shape == (1, task.dim_parameters), (
        f"{task_name}: params shape {params.shape} != (1, {task.dim_parameters})"
    )


@pytest.mark.parametrize("task_name", HIERARCHICAL_TASKS)
def test_in_memory_is_cached(task_name):
    """Repeated calls return the same tensor object (cached)."""
    task = sbibm.get_task(task_name)
    obs1 = task.get_observation(1)
    obs2 = task.get_observation(1)
    assert obs1 is obs2


def test_rng_isolation():
    """Calling get_observation does not perturb the ambient RNG state."""
    task = sbibm.get_task("hierarchical_two_moons")

    # Draw a reference random tensor
    torch.manual_seed(12345)
    np.random.seed(12345)
    ref = torch.randn(10)
    ref_np = np.random.randn(5)

    # Now interleave get_observation and check we get the same stream
    torch.manual_seed(12345)
    np.random.seed(12345)
    task._observation_cache.clear()  # force regeneration
    task.get_observation(1)
    actual = torch.randn(10)
    actual_np = np.random.randn(5)

    assert torch.allclose(ref, actual), "torch RNG state was corrupted"
    assert np.allclose(ref_np, actual_np), "numpy RNG state was corrupted"


def test_different_n_l_different_shapes():
    """Different n_l values produce different observation shapes."""
    task_5 = sbibm.get_task("hierarchical_two_moons", n_l=5)
    task_10 = sbibm.get_task("hierarchical_two_moons", n_l=10)

    obs_5 = task_5.get_observation(1)
    obs_10 = task_10.get_observation(1)

    assert obs_5.shape[-1] == 10  # 2 * 5
    assert obs_10.shape[-1] == 20  # 2 * 10

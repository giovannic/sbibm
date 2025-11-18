import jax
import jax.numpy as jnp
import pytest
import torch

import sbibm
from sbibm.algorithms.tfmpe.bottom_up import (
    run,
    make_prior_fn,
    make_simulator_fn,
    make_local_fn,
)


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


@pytest.mark.parametrize(
    "n_local,n_samples,transforms",
    [
        (1, 10, False),
        (5, 5, False),
        (10, 3, False),
        (1, 10, True),
        (5, 5, True),
        (10, 3, True),
    ],
)
def test_prior_fn_pytree_shapes(n_local, n_samples, transforms):
    """Test PyTree structure and shapes from prior_fn.

    Validates that prior_fn returns correctly structured JAX
    PyTrees:
    - Returns dict with keys: ['p_g_0', 'p_g_1', 'p_l_0']
    - 'p_g_0' and 'p_g_1' have shape (n_samples, 1, 1)
    - 'p_l_0' has shape (n_samples, n_local, 2, 1)
    """
    task = sbibm.get_task("hierarchical_two_moons")
    prior_fn = make_prior_fn(task, transforms)
    rng = jax.random.PRNGKey(0)

    # Call prior_fn
    param_dict = prior_fn(rng, n=n_local, n_samples=n_samples)

    # Validate it's a dict
    assert isinstance(param_dict, dict)

    # Validate exact keys
    expected_keys = {"p_g_0", "p_g_1", "p_l_0"}
    actual_keys = set(param_dict.keys())
    assert expected_keys == actual_keys, (
        f"Expected keys {expected_keys}, got {actual_keys}"
    )

    # Validate global parameter 0 (locs)
    assert isinstance(param_dict["p_g_0"], jnp.ndarray)
    assert param_dict["p_g_0"].shape == (n_samples, 2, 1), (
        f"p_g_0 has wrong shape: {param_dict['p_g_0'].shape} "
        f"vs ({n_samples}, 2, 1)"
    )

    # Validate global parameter 1 (scales)
    assert isinstance(param_dict["p_g_1"], jnp.ndarray)
    assert param_dict["p_g_1"].shape == (n_samples, 2, 1), (
        f"p_g_1 has wrong shape: {param_dict['p_g_1'].shape} "
        f"vs ({n_samples}, 2, 1)"
    )

    # Validate local parameters
    assert isinstance(param_dict["p_l_0"], jnp.ndarray)
    assert param_dict["p_l_0"].shape == (
        n_samples,
        n_local,
        2,
        1,
    ), (
        f"p_l_0 has wrong shape: {param_dict['p_l_0'].shape} "
        f"vs ({n_samples}, {n_local}, 2, 1)"
    )


@pytest.mark.parametrize(
    "n_local,n_samples,transforms",
    [
        (1, 10, False),
        (5, 5, False),
        (10, 3, False),
        (1, 10, True),
        (5, 5, True),
        (10, 3, True),
    ],
)
def test_simulator_fn_pytree_shapes(n_local, n_samples, transforms):
    """Test PyTree structure and shapes from simulator_fn.

    Validates that simulator_fn returns correctly structured JAX
    PyTrees:
    - Returns dict with key 'y'
    - Observations have shape (n_samples, n_local, 2, 1) in
      TFMPE format
    """
    task = sbibm.get_task("hierarchical_two_moons")
    prior_fn = make_prior_fn(task, automatic_transforms_enabled=transforms)
    simulator_fn = make_simulator_fn(task, automatic_transforms_enabled=transforms)

    rng = jax.random.PRNGKey(0)

    # Generate parameters from prior
    params_dict = prior_fn(rng, n=n_local, n_samples=n_samples)

    # Call simulator_fn
    obs_dict = simulator_fn(rng, params_dict, n=n_local)

    # Validate it's a dict with 'y' key
    assert isinstance(obs_dict, dict)
    assert set(obs_dict.keys()) == {"y"}, (
        f"Expected keys {{'y'}}, got {set(obs_dict.keys())}"
    )

    # Validate observation shape for TFMPE format
    obs = obs_dict["y"]
    assert obs.shape == (n_samples, n_local, 2, 1), (
        f"Expected obs shape ({n_samples}, {n_local}, 2, 1), "
        f"got {obs.shape}"
    )

    # Validate no NaN or Inf
    assert not jnp.any(jnp.isnan(obs))
    assert not jnp.any(jnp.isinf(obs))


@pytest.mark.parametrize(
    "n_local,n_samples,transforms",
    [
        (1, 10, False),
        (5, 5, False),
        (10, 3, False),
        (1, 10, True),
        (5, 5, True),
        (10, 3, True),
    ],
)
def test_local_fn_pytree_shapes(n_local, n_samples, transforms):
    """Test PyTree structure and shapes from local_fn.

    Validates that local_fn returns correctly structured JAX
    PyTrees:
    - Returns dict with key 'p_l_0'
    - Local params have shape (n_samples, n_local, 2, 1)
    """
    task = sbibm.get_task("hierarchical_two_moons")
    prior_fn = make_prior_fn(task, transforms)
    local_fn = make_local_fn(task, transforms)

    rng = jax.random.PRNGKey(0)

    # Generate global parameters from prior
    global_params_dict = prior_fn(rng, n=n_local, n_samples=n_samples)
    # Extract only global parameters
    global_only = {
        name: values
        for name, values in global_params_dict.items()
        if name.startswith("p_g_")
    }

    # Call local_fn
    local_params_dict = local_fn(rng, global_only, n=n_local)

    # Validate exact keys
    assert set(local_params_dict.keys()) == {"p_l_0"}, (
        f"Expected keys {{'p_l_0'}}, got {set(local_params_dict.keys())}"
    )

    # Validate shape
    assert isinstance(local_params_dict['p_l_0'], jnp.ndarray)
    assert local_params_dict["p_l_0"].shape == (
        n_samples,
        n_local,
        2,
        1,
    ), (
        f"p_l_0 has wrong shape: {local_params_dict['p_l_0'].shape} "
        f"vs ({n_samples}, {n_local}, 2, 1)"
    )

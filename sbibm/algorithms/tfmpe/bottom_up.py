"""TFMPE bottom-up algorithm for hierarchical inference on SBIBM tasks."""

import time
from math import prod
from typing import Dict, List, Tuple

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import nnx
from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from tfmpe.estimators.training import fit_bottom_up as tfmpe_fit_bottom_up
from tfmpe.nn.transformer import Transformer, TransformerConfig
from tfmpe.preprocessing.tokens import Tokens
from tfmpe.preprocessing.utils import Independence, Labeller

from sbibm.algorithms.sbi.utils import wrap_prior_dist
from sbibm.tasks import Task
from sbibm.tasks.distributions import BlockwiseDistribution


def _get_blockwise_components(dist) -> List[Tuple[int, int]]:
    """Get component slices from a distribution.

    If dist is BlockwiseDistribution, returns list of (offset,
    end) slices for each component. Otherwise returns single
    (0, event_dim) slice.

    Returns:
        List of (start_idx, end_idx) tuples
    """
    if isinstance(dist, BlockwiseDistribution):
        components = []
        offset = 0
        for comp in dist.distributions:
            event_dim = prod(comp.event_shape)
            components.append((offset, offset + event_dim))
            offset += event_dim
        return components
    else:
        # Non-blockwise: single component with full event size
        event_dim = prod(dist.event_shape)
        return [(0, event_dim)]


def _get_slices(task: Task, n_local: int) -> List[Tuple[str, Tuple[int, int]]]:
    """Get the parameter names and indices for a hierarchical distribution"""
    global_dim = task.prior_dist.dim_global
    global_dist = task.prior_dist.global_dist
    global_components = _get_blockwise_components(global_dist)
    global_slices = [(f"p_g_{i}", s) for i, s in enumerate(global_components)]

    # Generate sample local distribution to get component names
    sample_global = global_dist.sample(torch.Size([1]))
    local_dist = task.prior_dist.local_dist_fn(sample_global, n_local)
    local_components = _get_blockwise_components(local_dist)
    local_slices = [
        (f"p_l_{i}", (start + global_dim, end + global_dim))
        for i, (start, end) in enumerate(local_components)
    ]

    return global_slices + local_slices


def make_prior_fn(task, automatic_transforms_enabled: bool = False):
    """Create prior sampling function for TFMPE.

    Args:
        task: SBIBM task instance with hierarchical prior

    Returns:
        prior_fn(rng, n, n_samples) -> dict of JAX arrays
    """

    def prior_fn(rng, n, n_samples):
        """Sample from prior for n local groups.

        Args:
            rng: JAX random key
            n: Number of local groups to sample
            n_samples: Number of samples to generate

        Returns:
            Dictionary where:
            - 'p_g_{i}': i-th global component with shape
              (n_samples, batch_shape, event_shape)
            - 'p_l_{j}': j-th local component with shape
              (n_samples, n_local, batch_shape, event_shape)
        """
        slices = _get_slices(task, n)
        prior_dist = task.prior_dist.for_n_local(n)
        if automatic_transforms_enabled:
            transforms = task._get_transforms(n_l=n)["parameters"]
            prior_dist = wrap_prior_dist(prior_dist, transforms)
        samples = prior_dist.sample(sample_shape=torch.Size([n_samples]))

        # Convert to JAX arrays
        samples_jax = jnp.asarray(samples)

        # Create structured dict for TFMPE
        param_dict = {}

        # Add global parameters, grouped by component
        for name, (start, end) in slices:
            # Extract this component's parameters
            component_params = samples_jax[:, start:end]
            # Add batch dimension for TFMPE format
            if str.startswith(name, "p_l_"):
                component_params = component_params.reshape(n_samples, n, -1)
            param_dict[name] = component_params[..., None]

        return param_dict

    return prior_fn


def make_simulator_fn(task, automatic_transforms_enabled: bool = False):
    """Create simulator function for TFMPE.

    Args:
        task: SBIBM task instance

    Returns:
        simulator_fn(rng, params_dict, n) -> dict with 'y' key
    """

    def simulator_fn(rng, params_dict, n):
        """Simulate observations for n local groups.

        Args:
            rng: JAX random key
            params_dict: Dictionary of JAX arrays with global
                and local parameters
            n: Number of local groups

        Returns:
            Dictionary with 'y' key containing observations
            shaped (n_samples, n, 2, 1)
        """
        slices = _get_slices(task, n)
        params_list = [
            params_dict[name].reshape(params_dict[name].shape[0], -1)
            for name, _ in slices
        ]
        params_flat = jnp.concatenate(params_list, axis=1)

        # Convert to torch and call task simulator
        params_torch = torch.from_numpy(np.array(params_flat)).float()

        if automatic_transforms_enabled:
            transforms = task._get_transforms(n_l=n)["parameters"]
            params_torch = transforms.inv(params_torch)

        obs_torch = task.get_simulator()(params_torch)

        # Convert back to JAX and reshape to n groups
        obs_jax = jnp.asarray(obs_torch.numpy()).reshape(
            obs_torch.shape[0], n, -1, 1
        )

        return {"y": obs_jax}

    return simulator_fn


def make_local_fn(task, automatic_transforms_enabled: bool = False):
    """Create local parameter sampling function for TFMPE.

    Args:
        task: SBIBM task instance with hierarchical prior

    Returns:
        local_fn(rng, global_samples, n) -> dict of JAX arrays
    """
    prior_dist = task.prior_dist

    def local_fn(rng, global_samples, n):
        """Sample local parameters conditioned on global.

        Args:
            rng: JAX random key
            global_samples: Dictionary of JAX arrays with
                global parameters
            n: Number of local groups to sample

        Returns:
            Dictionary with local parameter names as keys and
            JAX arrays as values
        """
        slices = _get_slices(task, n)

        # Reconstruct global params tensor from JAX arrays
        global_list = [
            global_samples[name].reshape(global_samples[name].shape[0], -1)
            for name, _ in slices
            if str.startswith(name, "p_g_")
        ]
        global_params = jnp.concatenate(global_list, axis=1)

        # Convert to torch and use prior_dist's local_dist_fn
        global_torch = torch.from_numpy(np.array(global_params)).float()
        local_dist = prior_dist.local_dist_fn(global_torch, n)
        local_torch = local_dist.sample().reshape(global_torch.shape[0], -1)

        samples = torch.cat([global_torch, local_torch], 1)

        if automatic_transforms_enabled:
            transforms = task._get_transforms(n_l=n)
            samples = transforms["parameters"](samples)

        local_params_dict = {}
        for name, (start, end) in slices:
            if str.startswith(name, "p_l_"):
                component_params = samples[:, start:end]
                reshaped = component_params.reshape(samples.shape[0], n, -1)
                reshaped_jax = jnp.asarray(reshaped[..., None])
                local_params_dict[name] = reshaped_jax

        return local_params_dict

    return local_fn


def run(
    task,
    num_samples: int,
    num_simulations: int,
    num_observation: int,
    automatic_transforms_enabled: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, float, Dict]:
    """Run TFMPE bottom-up inference on a hierarchical task.

    Args:
        task: SBIBM task instance (must be hierarchical)
        num_samples: Number of posterior samples to generate
        num_simulations: Number of simulator calls to use during
            training
        num_observation: Index of observation to use (1-10)
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of:
        - posterior_samples: Tensor of shape
            (num_samples, dim_parameters)
        - execution_time: Float, seconds elapsed
        - metadata: Dict containing training losses and
            hyperparameters
    """
    start_time = time.time()

    # Load observation
    y_obs_torch = task.get_observation(num_observation=num_observation)

    # Get number of groups
    n_local = task.n_l

    # Reshape observation to structured format
    # (n_local, dims_per_group, 1)
    y_obs_dict = {
        "y": y_obs_torch.reshape(1, n_local, -1, 1).numpy(),
    }

    slices = _get_slices(task, n_local)

    all_param_names = [name for name, _ in slices]
    all_param_names.append("y")

    # Create callback functions for TFMPE using helpers
    prior_fn = make_prior_fn(task, automatic_transforms_enabled)
    simulator_fn = make_simulator_fn(task, automatic_transforms_enabled)
    local_fn = make_local_fn(task, automatic_transforms_enabled)

    # Define which parameters are global
    global_names = [
        name for name in all_param_names
        if str.startswith(name, "p_g_")
    ]
    local_names = [
        name for name in all_param_names
        if str.startswith(name, "p_l_")
    ]

    # Generate sample data for token creation
    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)
    sample_params = prior_fn(
        key, n=n_local, n_samples=10
    )

    # Create labeller and independence structure
    labeller = Labeller.for_keys(all_param_names)

    # Define independence: each local param[i] attends
    # only to y[i]
    independence = Independence(
        cross_local=[(name, "y", (0, 0)) for name in local_names]
    )

    # Create tokens from sample data
    params_tokens = Tokens.from_pytree(
        sample_params,
        sample_ndims=1,
        labeller=labeller,
        independence=independence,
    )

    # Initialize TFMPE model
    config = TransformerConfig(
        latent_dim=64,
        n_encoder=1,
        n_decoder=1,
        n_heads=2,
        n_ff=2,
    )

    rngs = nnx.Rngs(
        params=jax.random.PRNGKey(0),
        dropout=jax.random.PRNGKey(1),
    )
    transformer = Transformer(
        config=config,
        tokens=params_tokens,
        rngs=rngs,
    )

    base_dist = NormalDistribution(rngs=rngs)

    tfmpe = TFMPE(
        vf_network=transformer,
        base_dist=base_dist,
        solver=diffrax.Dopri5(),
    )

    # Setup optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt = nnx.Optimizer(tfmpe, optimizer, wrt=nnx.Param)

    # Training parameters
    n_samples_per_round = num_simulations // 2
    n_val_samples = min(1000, num_simulations // 10)
    n_iter_per_round = 50
    batch_size = 100

    # Train TFMPE
    rng = jax.random.PRNGKey(42)
    trained_tfmpe, all_losses = tfmpe_fit_bottom_up(
        tfmpe=tfmpe,
        y_obs=y_obs_dict,
        simulator_fn=simulator_fn,
        prior_fn=prior_fn,
        local_fn=local_fn,
        global_names=global_names,
        n_groups=n_local,
        n_rounds=1,
        n_samples_per_round=n_samples_per_round,
        n_val_samples=n_val_samples,
        opt=opt,
        n_iter_per_round=n_iter_per_round,
        batch_size=batch_size,
        rng=rng,
        independence=independence,
        labeller=labeller,
    )

    # Generate posterior samples using trained TFMPE
    # Create context tokens from observation
    context_tokens = Tokens.from_pytree(
        y_obs_dict,
        sample_ndims=1,
        labeller=labeller,
    )

    # Create parameter tokens template for sampling
    param_dict_template = prior_fn(rng, n=n_local, n_samples=1)
    param_dict_samples = {
        key: jnp.tile(value, (num_samples,) + (1,) * (value.ndim - 1))
        for key, value in param_dict_template.items()
    }

    param_tokens = Tokens.from_pytree(
        param_dict_samples,
        sample_ndims=1,
        labeller=labeller,
        independence=independence,
    )

    # Sample from posterior
    rng_key = jax.random.PRNGKey(42)
    nnx.reseed(trained_tfmpe, params=rng_key)
    posterior_tokens = trained_tfmpe.sample_posterior(
        context=context_tokens,
        params=param_tokens,
    )

    # Convert tokens back to flat tensor format
    posterior_dict = posterior_tokens.decode()
    params_list = []
    for name in global_names + local_names:
        params_list.append(posterior_dict[name].reshape(num_samples, -1))

    posterior_flat = jnp.concatenate(params_list, axis=1)
    posterior_samples = torch.from_numpy(np.array(posterior_flat)).float()

    if automatic_transforms_enabled:
        transform = task._get_transforms(n_l=n_local)
        posterior_samples = transform["parameters"].inv(posterior_samples)

    execution_time = time.time() - start_time

    metadata = {
        "losses": all_losses,
        "n_samples_per_round": n_samples_per_round,
        "n_iter_per_round": n_iter_per_round,
        "batch_size": batch_size,
        "n_local": n_local,
    }

    return posterior_samples, execution_time, metadata

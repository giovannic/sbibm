"""TFMPE bottom-up algorithm for hierarchical inference on SBIBM tasks."""

import time
from typing import Dict, Tuple

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import nnx

from sbibm.tasks.distributions import BlockwiseDistribution
from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from tfmpe.estimators.training import fit_bottom_up as tfmpe_fit_bottom_up
from tfmpe.nn.transformer import Transformer, TransformerConfig
from tfmpe.preprocessing.tokens import Tokens
from tfmpe.preprocessing.utils import Independence, Labeller


def _wrap_prior_dist(prior_dist, transforms):
    """Wrap prior distribution with transforms."""
    return TransformedDistribution(prior_dist, transforms)


def _wrap_simulator_fn(simulator_fn, transforms):
    """Wrap simulator to apply transforms to parameters."""

    class SimulatorWrapper:
        def __init__(self, simulator_fn, transforms):
            self.simulator_fn = simulator_fn
            self.transforms = transforms

        def __call__(self, parameters):
            # Apply inverse transform to move from
            # unconstrained to constrained space
            constrained_params = self.transforms.inv(parameters)
            return self.simulator_fn(constrained_params)

    return SimulatorWrapper(simulator_fn, transforms)


def _get_blockwise_components(dist):
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
            event_dim = comp.event_shape[0]
            components.append((offset, offset + event_dim))
            offset += event_dim
        return components
    else:
        # Non-blockwise: single component with full event size
        event_dim = dist.event_shape[0]
        return [(0, event_dim)]


def make_prior_fn(task):
    """Create prior sampling function for TFMPE.

    Args:
        task: SBIBM task instance with hierarchical prior

    Returns:
        prior_fn(rng, n, n_samples) -> dict of JAX arrays
    """
    prior_dist = task.prior_dist
    dim_global = prior_dist.dim_global
    global_dist = prior_dist.global_dist

    # Get component structure for global distribution
    global_components = _get_blockwise_components(global_dist)

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
        # Sample global parameters
        global_params_torch = global_dist.sample(
            torch.Size([n_samples])
        )

        # Sample local parameters for n groups
        local_dist = prior_dist.local_dist_fn(
            global_params_torch, n
        )
        local_params_torch = local_dist.sample()

        # Get component structure for local distribution
        local_components = _get_blockwise_components(
            local_dist
        )

        # Convert to JAX arrays
        global_params_jax = jnp.asarray(
            global_params_torch.numpy()
        )
        local_params_jax = jnp.asarray(
            local_params_torch.numpy()
        )

        # Create structured dict for TFMPE
        param_dict = {}

        # Add global parameters, grouped by component
        for i, (start, end) in enumerate(global_components):
            # Extract this component's parameters
            component_params = global_params_jax[:, start:end]
            # Add batch dimension for TFMPE format
            param_dict[f"p_g_{i}"] = component_params[
                :, None, :
            ]

        # Add local parameters, grouped by component
        for i, (start, end) in enumerate(local_components):
            # Extract this component's parameters
            component_params = local_params_jax[:, start:end]
            # Reshape to (n_samples, n_local, -1) and add
            # batch dimension
            reshaped = component_params.reshape(
                n_samples, n, -1
            )
            param_dict[f"p_l_{i}"] = reshaped[:, :, None, :]

        return param_dict

    return prior_fn


def make_simulator_fn(task):
    """Create simulator function for TFMPE.

    Args:
        task: SBIBM task instance

    Returns:
        simulator_fn(rng, params_dict, n) -> dict with 'y' key
    """
    prior_dist = task.prior_dist
    dim_global = prior_dist.dim_global
    dim_local = prior_dist.dim_local

    # Extract parameter names
    global_dist = prior_dist.global_dist
    param_names_global = []
    for i, comp in enumerate(global_dist.distributions):
        if hasattr(comp, "event_shape") and len(
            comp.event_shape
        ) > 0:
            n_params = comp.event_shape[0]
            for j in range(n_params):
                param_names_global.append(f"p_g_{i}_{j}")

    param_names_local = [f"p_l_{i}" for i in range(dim_local)]

    def simulator_fn(rng, params_dict, n):
        """Simulate observations for n local groups.

        Args:
            rng: JAX random key
            params_dict: Dictionary of JAX arrays with global
                and n*2 local parameters
            n: Number of local groups

        Returns:
            Dictionary with 'y' key containing observations
            shaped (n_samples, n, 2, 1)
        """
        n_samples = params_dict[param_names_global[0]].shape[0]

        # Reconstruct flat parameter tensor from JAX arrays
        params_list = []
        for name in param_names_global:
            params_list.append(
                params_dict[name].reshape(n_samples, 1)
            )
        for i in range(n * 2):
            name = param_names_local[i]
            params_list.append(
                params_dict[name].reshape(n_samples, 1)
            )

        params_flat = jnp.concatenate(params_list, axis=1)

        # Convert to torch and call task simulator
        params_torch = torch.from_numpy(
            np.array(params_flat)
        ).float()
        obs_torch = task.get_simulator()(params_torch)

        # Convert back to JAX and reshape to n groups
        obs_jax = jnp.asarray(obs_torch.numpy()).reshape(
            n_samples, n, 2, 1
        )

        return {"y": obs_jax}

    return simulator_fn


def make_local_fn(task):
    """Create local parameter sampling function for TFMPE.

    Args:
        task: SBIBM task instance with hierarchical prior

    Returns:
        local_fn(rng, global_samples, n) -> dict of JAX arrays
    """
    prior_dist = task.prior_dist
    dim_global = prior_dist.dim_global
    dim_local = prior_dist.dim_local

    # Extract parameter names
    global_dist = prior_dist.global_dist
    param_names_global = []
    for i, comp in enumerate(global_dist.distributions):
        if hasattr(comp, "event_shape") and len(
            comp.event_shape
        ) > 0:
            n_params = comp.event_shape[0]
            for j in range(n_params):
                param_names_global.append(f"p_g_{i}_{j}")

    param_names_local = [f"p_l_{i}" for i in range(dim_local)]

    def local_fn(rng, global_samples, n):
        """Sample local parameters conditioned on global.

        Args:
            rng: JAX random key
            global_samples: Dictionary of JAX arrays with
                global parameters
            n: Number of local groups to sample

        Returns:
            Dictionary with local parameter names as keys and
            JAX arrays of shape (n_samples, 1, 1, 1) as values
        """
        n_samples = global_samples[param_names_global[0]].shape[
            0
        ]

        # Reconstruct global params tensor from JAX arrays
        global_list = []
        for name in param_names_global:
            global_list.append(
                global_samples[name].reshape(n_samples, 1)
            )
        global_params = jnp.concatenate(global_list, axis=1)

        # Convert to torch and use prior_dist's local_dist_fn
        global_torch = torch.from_numpy(
            np.array(global_params)
        ).float()
        local_dist = prior_dist.local_dist_fn(
            global_torch, n
        )
        local_samples_torch = local_dist.sample(
            (n_samples,)
        )

        # Convert to JAX and extract first n*2 local params
        local_samples = jnp.asarray(
            local_samples_torch.numpy()
        )
        local_params_dict = {}
        for i in range(min(n * 2, dim_local)):
            name = param_names_local[i]
            local_params_dict[name] = local_samples[
                :, i : i + 1, None, None
            ]

        return local_params_dict

    return local_fn


def run(
    task,
    num_samples: int,
    num_simulations: int,
    num_observation: int,
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
    y_obs_torch = task.get_observation(
        num_observation=num_observation
    )

    # Get prior distribution (already in constrained space)
    prior_dist = task.prior_dist

    # Get hierarchical structure from prior
    dim_global = prior_dist.dim_global
    dim_local = prior_dist.dim_local
    n_groups = dim_local // 2  # For two_moons: 2 per group

    # Reshape observation to structured format
    # (n_groups, dims_per_group, 1)
    y_obs_dict = {
        "y": y_obs_torch.reshape(1, n_groups, -1, 1).numpy(),
    }

    # Extract parameter names from prior_dist
    # For HierarchicalDistribution, we need to get names from
    # the underlying global_dist (BlockwiseDistribution)
    global_dist = prior_dist.global_dist
    param_names_global = []
    for i, comp in enumerate(global_dist.distributions):
        if hasattr(comp, "event_shape") and len(
            comp.event_shape
        ) > 0:
            n_params = comp.event_shape[0]
            for j in range(n_params):
                param_names_global.append(f"p_g_{i}_{j}")

    param_names_local = [f"p_l_{i}" for i in range(dim_local)]
    all_param_names = param_names_global + param_names_local
    all_param_names.append("y")

    # Create callback functions for TFMPE using helpers
    prior_fn = make_prior_fn(task)
    simulator_fn = make_simulator_fn(task)
    local_fn = make_local_fn(task)

    # Define which parameters are global
    global_names = param_names_global

    # Generate sample data for token creation
    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)
    sample_params = prior_fn(key, n=n_groups, n_samples=10)

    # Create labeller and independence structure
    labeller = Labeller.for_keys(all_param_names)

    # Define independence: each local param[i] attends
    # only to y[i]
    independence = Independence(
        cross_local=[
            (name, "y", (0, 0))
            for name in param_names_local
        ]
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
        n_groups=n_groups,
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

    # Generate posterior samples
    # For now, use prior samples as placeholder
    rng, key = jax.random.split(rng)
    posterior_params_dict = prior_fn(key, n=n_groups, n_samples=num_samples)

    # Flatten parameters to SBIBM format
    params_list = []
    for name in param_names_global:
        params_list.append(posterior_params_dict[name].reshape(num_samples, 1))
    for i in range(n_groups * 2):
        name = param_names_local[i]
        params_list.append(posterior_params_dict[name].reshape(num_samples, 1))

    posterior_flat = jnp.concatenate(params_list, axis=1)
    posterior_samples = torch.from_numpy(jnp.asarray(posterior_flat)).float()

    execution_time = time.time() - start_time

    metadata = {
        "losses": all_losses,
        "n_samples_per_round": n_samples_per_round,
        "n_iter_per_round": n_iter_per_round,
        "batch_size": batch_size,
        "n_groups": n_groups,
    }

    return posterior_samples, execution_time, metadata

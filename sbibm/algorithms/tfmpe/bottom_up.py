"""TFMPE bottom-up algorithm for hierarchical inference on SBIBM tasks."""

from math import prod
from typing import List, Tuple

import diffrax
import jax
import jax.numpy as jnp
from jax import tree
import numpy as np
import optax
import torch
from flax import nnx
from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from tfmpe.estimators.training import fit_bottom_up as tfmpe_fit_bottom_up, fit_directly as tfmpe_fit_directly, fit_pf as tfmpe_fit_pf
from tfmpe.nn.transformer import Transformer, TransformerConfig
from tfmpe.nn.mlp import MLP
from tfmpe.preprocessing.tokens import Tokens
from tfmpe.preprocessing.utils import Labeller

from sbibm.algorithms.sbi.utils import wrap_prior_dist
from sbibm.tasks import Task
from sbibm.tasks.distributions import BlockwiseDistribution, SummedStackTransform


class TFMPEPosterior:
    """Wrapper for TFMPE model to provide posterior interface.

    Handles conversion between flat tensors and token format required
    by TFMPE, and computes log probabilities with proper transform
    handling.
    """

    def __init__(
        self,
        tfmpe_model,
        labeller,
        slices,
        global_names,
        local_names,
        n_local,
        params_f_in,
        context_f_in,
        transforms=None,
        context=None,
        sample_batch_size=100
    ):
        """Initialize TFMPE posterior wrapper.

        Args:
            tfmpe_model: Trained TFMPE model
            labeller: Labeller for token creation
            slices: List of (name, (start, end)) tuples for parameter
                slicing
            global_names: Names of global parameters
            local_names: Names of local parameters
            n_local: Number of local groups
            transforms: Transform object for handling constrained
                parameters
            context: Dict dictionary of (observations) for
                computing log prob
        """
        self.tfmpe_model = tfmpe_model
        self.labeller = labeller
        self.slices = slices
        self.global_names = global_names
        self.local_names = local_names
        self.n_local = n_local
        self.transforms = transforms
        self.context = context
        self.params_f_in = params_f_in
        self.context_f_in = context_f_in
        self.sample_batch_size = sample_batch_size

    def sample(self, shape, x=None):
        """Sample from posterior.

        Args:
            shape: Tuple of sample shape
            x: (optional) torch matrix of contexts to sample from

        Returns:
            Flat tensor of samples with shape (num_samples, n_params)
        """
        num_samples = shape[0]

        # Create parameter tokens template with correct shapes
        # For each parameter, calculate event_dim from the slice size
        param_dict_template = {}
        for name, (start, end) in self.slices:
            event_dim = end - start
            if name.startswith("p_l_"):
                # Local params: (1, n_local, event_dim // n_local, 1)
                param_dict_template[name] = jnp.ones(
                    (1, self.n_local, event_dim // self.n_local, 1)
                )
            else:
                # Global params: (1, event_dim, 1)
                param_dict_template[name] = jnp.ones((1, event_dim, 1))

        param_dict_samples = {
            key: jnp.tile(
                value, (num_samples,) + (1,) * (value.ndim - 1)
            )
            for key, value in param_dict_template.items()
        }

        # Sample from posterior
        # TODO: Handle RNG seeding properly
        if x is not None:
            torch_context = x.reshape(x.shape[0], self.n_local, -1, 1)
            context = { "y":  jnp.asarray(torch_context) }
        else:
            context = self.context
            context = tree.map(
                lambda leaf: jnp.broadcast_to(
                    leaf,
                    (num_samples,) + leaf.shape[1:]
                ),
                context
            )
            
        if self.context_f_in is not None:
            f_in = {**self.params_f_in, **self.context_f_in}
        else:
            f_in = None

        tokens, decoder = Tokens.from_pytree_with_decoder(
            {**param_dict_samples, **context},
            condition=list(context.keys()),
            sample_ndims=1,
            labeller=self.labeller,
            functional_inputs=f_in,
        )
            
        posterior_tokens = self.tfmpe_model.sample_posterior_batched(
            tokens=tokens,
            batch_size=self.sample_batch_size
        )

        # Convert tokens back to flat tensor format
        posterior_dict = decoder(posterior_tokens)
        posterior_dict = {
            k: v for k, v in posterior_dict.items()
            if k in param_dict_samples.keys()
        }
        params_list = []
        for name, (start, end) in self.slices:
            # Extract component from dict and reshape correctly
            # The slice (start, end) tells us how many dimensions this
            # component should have in the flat representation
            component = posterior_dict[name]
            # Reshape to (num_samples, -1) to flatten all dimensions
            # except the first (sample) dimension
            component_flat = component.reshape(num_samples, -1)
            params_list.append(component_flat)

        posterior_flat = jnp.concatenate(params_list, axis=1)
        posterior_samples = torch.from_numpy(
            np.array(posterior_flat)
        ).float()

        # Apply inverse transform to get constrained space samples
        if self.transforms is not None:
            posterior_samples = self.transforms.inv(posterior_samples)

        return posterior_samples

    def log_prob(self, parameters):
        """Compute log probability at given parameters.

        Note: TFMPE log probability computation through the ODE solver
        encounters JAX tracing issues with stateful neural network
        modules (Flax NNX). This limitation is inherent to the current
        TFMPE implementation. Returns None to indicate unavailability.

        Args:
            parameters: Flat tensor of shape (n_samples, n_params)
                in constrained space

        Returns:
            None (log probability computation not supported due to JAX
            tracing constraints in stateful modules)
        """
        return None


class TFMPEPosteriorPF:
    """Wrapper for factored TFMPE posterior (global + local models).

    Samples from the posterior by first sampling global parameters from
    the global model, then sampling local parameters from the local model
    conditioned on the sampled globals and per-group observations.
    """

    def __init__(
        self,
        tfmpe_global,
        tfmpe_local,
        labeller,
        slices,
        global_names,
        local_names,
        n_local,
        transforms=None,
        context=None,
        sample_batch_size=100
    ):
        self.tfmpe_global = tfmpe_global
        self.tfmpe_local = tfmpe_local
        self.labeller = labeller
        self.slices = slices
        self.global_names = global_names
        self.local_names = local_names
        self.n_local = n_local
        self.transforms = transforms
        self.context = context
        self.sample_batch_size = sample_batch_size

    def sample(self, shape, x=None):
        num_samples = shape[0]

        # Build global param templates
        global_param_template = {}
        for name, (start, end) in self.slices:
            if name in self.global_names:
                event_dim = end - start
                global_param_template[name] = jnp.ones((1, event_dim, 1))
        global_param_samples = {
            key: jnp.tile(
                value, (num_samples,) + (1,) * (value.ndim - 1)
            )
            for key, value in global_param_template.items()
        }

        # Get observation context
        if x is not None:
            torch_context = x.reshape(x.shape[0], self.n_local, -1, 1)
            y_obs = {"y": jnp.asarray(torch_context)}
        else:
            y_obs = self.context
            y_obs = tree.map(
                lambda leaf: jnp.broadcast_to(
                    leaf, (num_samples,) + leaf.shape[1:]
                ),
                y_obs,
            )

        # 1. Sample global params
        global_tokens, global_decoder = Tokens.from_pytree_with_decoder(
            {**y_obs, **global_param_samples},
            condition=list(y_obs.keys()),
            sample_ndims=1,
            labeller=self.labeller,
        )
        global_posterior_tokens = self.tfmpe_global.sample_posterior_batched(
            tokens=global_tokens, batch_size=self.sample_batch_size
        )
        global_posterior_dict = global_decoder(global_posterior_tokens)
        theta_g_star = {
            k: v for k, v in global_posterior_dict.items()
            if k in self.global_names
        }

        # 2. Reshape y_obs to single-group
        y_single = tree.map(
            lambda v: v.reshape(
                num_samples * self.n_local, 1, *v.shape[2:]
            ),
            y_obs,
        )

        # 3. Repeat global params for each local group
        theta_g_expanded = tree.map(
            lambda v: jnp.repeat(v, self.n_local, axis=0),
            theta_g_star,
        )

        # 4. Build local param templates (single-group shape)
        local_param_template = {}
        for name, (start, end) in self.slices:
            if name in self.local_names:
                event_dim = (end - start) // self.n_local
                local_param_template[name] = jnp.ones((1, 1, event_dim, 1))
        local_param_samples = {
            key: jnp.tile(
                value,
                (num_samples * self.n_local,) + (1,) * (value.ndim - 1),
            )
            for key, value in local_param_template.items()
        }

        # 5. Sample local params
        local_tokens, local_decoder = Tokens.from_pytree_with_decoder(
            {**y_single, **theta_g_expanded, **local_param_samples},
            condition=(
                list(y_single.keys()) + list(theta_g_expanded.keys())
            ),
            sample_ndims=1,
            labeller=self.labeller,
        )
        local_posterior_tokens = self.tfmpe_local.sample_posterior_batched(
            tokens=local_tokens, batch_size=self.sample_batch_size
        )
        local_posterior_dict = local_decoder(local_posterior_tokens)
        theta_l_star = {
            k: v for k, v in local_posterior_dict.items()
            if k in self.local_names
        }

        # 6. Reshape local params back to (num_samples, n_local, ...)
        theta_l_reshaped = tree.map(
            lambda v: v.reshape(num_samples, self.n_local, *v.shape[2:]),
            theta_l_star,
        )

        # 7. Flatten and concatenate
        params_list = []
        for name in self.global_names:
            params_list.append(
                theta_g_star[name].reshape(num_samples, -1)
            )
        for name in self.local_names:
            params_list.append(
                theta_l_reshaped[name].reshape(num_samples, -1)
            )

        posterior_flat = jnp.concatenate(params_list, axis=1)
        posterior_samples = torch.from_numpy(
            np.array(posterior_flat)
        ).float()

        if self.transforms is not None:
            posterior_samples = self.transforms.inv(posterior_samples)

        return posterior_samples

    def log_prob(self, parameters):
        return None


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

    def prior_fn(rng, n, n_samples, f_in):
        """Sample from prior for n local groups.

        Args:
            rng: JAX random key
            n: Number of local groups to sample
            n_samples: Number of samples to generate

        Returns:
            Dictionary where:
            - 'p_g_{i}': i-th global component with shape
              (n_samples, event_shape, 1)
            - 'p_l_{j}': j-th local component with shape
              (n_samples, n_local, event_shape, 1)
        """
        slices = _get_slices(task, n)
        prior_dist = task.prior_dist.for_n_local(n)
        if automatic_transforms_enabled:
            transforms = task._get_transforms(n_l=n)["parameters"]
            prior_dist = wrap_prior_dist(prior_dist, transforms)
        samples = prior_dist.sample(sample_shape=torch.Size([n_samples]))

        # Convert to JAX arrays
        samples_jax = jnp.asarray(samples.cpu())

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


def make_simulator_fn(
    task,
    automatic_transforms_enabled: bool = False,
    device: str = 'cpu'
    ):
    """Create simulator function for TFMPE.

    Args:
        task: SBIBM task instance

    Returns:
        simulator_fn(rng, params_dict, n) -> dict with 'y' key
    """

    def simulator_fn(rng, params_dict, n, f_in):
        """Simulate observations for n local groups.

        Args:
            rng: JAX random key
            params_dict: Dictionary of JAX arrays with global
                and local parameters
            n: Number of local groups

        Returns:
            Dictionary with 'y' key containing observations
            shaped (n_samples, n, n_events, 1)
        """
        slices = _get_slices(task, n)
        params_list = [
            params_dict[name].reshape(params_dict[name].shape[0], -1)
            for name, _ in slices
        ]
        params_flat = jnp.concatenate(params_list, axis=1)

        # Convert to torch and call task simulator
        params_torch = torch.from_numpy(np.array(params_flat)).float().to(device=device)

        if automatic_transforms_enabled:
            transforms = task._get_transforms(n_l=n)["parameters"]
            params_torch = transforms.inv(params_torch)

        obs_torch = task.get_simulator()(params_torch)

        # Convert back to JAX and reshape to n groups
        obs_jax = jnp.asarray(obs_torch.cpu().numpy()).reshape(
            obs_torch.shape[0], n, -1, 1
        )
        obs_dict = {"y": obs_jax}

        return obs_dict

    return simulator_fn


def make_local_fn(task, automatic_transforms_enabled: bool = False, device='cpu'):
    """Create local parameter sampling function for TFMPE.

    Args:
        task: SBIBM task instance with hierarchical prior

    Returns:
        local_fn(rng, global_samples, n) -> dict of JAX arrays
    """
    prior_dist = task.prior_dist

    def local_fn(rng, global_samples, n, f_in):
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
        global_params_unconstrained = jnp.concatenate(global_list, axis=1)

        # Convert to torch
        global_torch_unconstrained = (
            torch.from_numpy(np.array(global_params_unconstrained)).float()
        )

        # Apply global transform if enabled (convert unconstrained to
        # constrained)
        if automatic_transforms_enabled:
            transforms_full = task._get_transforms(n_l=n)
            transform_full = transforms_full["parameters"]
            # Get base transform from Inverse wrapper
            base_transform = transform_full._inv
            # Extract just the global part of the transform
            global_dim = global_torch_unconstrained.shape[1]
            global_transforms_list = base_transform.transforms[:global_dim]
            # Create new SummedStackTransform with just global
            global_transform = SummedStackTransform(
                global_transforms_list, dim=-1
            )
            # Apply forward to convert unconstrained global to constrained
            global_torch_constrained = global_transform(
                global_torch_unconstrained
            )
        else:
            global_torch_constrained = global_torch_unconstrained
        global_torch_constrained = global_torch_constrained.to(device=device)

        # Sample local parameters conditioned on constrained global
        local_dist = prior_dist.local_dist_fn(
            global_torch_constrained, n
        )
        local_torch = local_dist.sample().reshape(
            global_torch_constrained.shape[0], -1
        )

        # Combine global (constrained) and local (constrained)
        samples = torch.cat([global_torch_constrained, local_torch], 1)

        # Apply full transform if NOT already applied to global
        if automatic_transforms_enabled:
            transforms = task._get_transforms(n_l=n)
            samples = transforms["parameters"](samples)

        local_params_dict = {}
        for name, (start, end) in slices:
            if str.startswith(name, "p_l_"):
                component_params = samples[:, start:end]
                reshaped = component_params.reshape(samples.shape[0], n, -1)
                reshaped_jax = jnp.asarray(reshaped[..., None].cpu())
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
) -> Tuple[torch.Tensor, int, torch.Tensor, TFMPEPosterior]:
    """Run TFMPE bottom-up inference on a hierarchical task.

    Args:
        task: SBIBM task instance (must be hierarchical)
        num_samples: Number of posterior samples to generate
        num_simulations: Number of simulator calls to use during
            training
        num_observation: Index of observation to use (1-10)
        automatic_transforms_enabled: Whether to enable automatic
            transforms
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of:
        - posterior_samples: Tensor of shape
            (num_samples, dim_parameters)
        - num_simulations: Number of simulator calls used
        - log_prob_true_params: Log probability at true parameters
            if num_observation provided, else None
        - posterior: Posterior object with sample() and log_prob()
            methods
    """
    device = kwargs.get('device', 'cpu')
    ablation = kwargs.get('ablation', 'none')
    sample_batch_size = kwargs.get('sample_batch_size', 100) 

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
    simulator_fn = make_simulator_fn(task, automatic_transforms_enabled, device=device)
    local_fn = make_local_fn(task, automatic_transforms_enabled, device=device)

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
        key, n_local, 10, None
    )
    sample_obs = simulator_fn(
        key, sample_params, n_local, None
    )

    # Create labeller
    labeller = Labeller.for_keys(all_param_names)

    # Create tokens from sample data
    f_in = None
    tokens = Tokens.from_pytree(
        {**sample_params, **sample_obs},
        condition=list(sample_obs.keys()),
        sample_ndims=1,
        labeller=labeller,
        functional_inputs=f_in
    )

    # Initialize TFMPE model
    if ablation == 'linear':
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_heads=16,
            n_ff=2,
            attention='linear',
        )
    else:
        if ablation == 'no_grouping':
            group_dim = 0
        else:
            group_dim = 8
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_heads=16,
            n_ff=2,
            group_dim=group_dim
        )

    rngs = nnx.Rngs(
        params=jax.random.PRNGKey(0),
        dropout=jax.random.PRNGKey(1),
    )
    base_dist = NormalDistribution(rngs=rngs)

    if ablation == 'joint' or ablation == 'direct':
        estimator = Transformer(
            config=config,
            tokens=tokens,
            rngs=rngs,
        )
        tfmpe = TFMPE(
            vf_network=estimator,
            base_dist=base_dist,
            solver=diffrax.Dopri5(),
        )
        optimizer = optax.adam(learning_rate=1e-4)
        opt = nnx.Optimizer(tfmpe, optimizer, wrt=nnx.Param)
    else:
        if ablation != 'mlp':
            local_estimator = Transformer(
                config=config,
                tokens=tokens,
                rngs=rngs,
            )
            global_estimator = Transformer(
                config=config,
                tokens=tokens,
                rngs=rngs,
            )
            estimator = [local_estimator, global_estimator]
        else:
            local_sample_params = prior_fn(
                key, 1, 10, None
            )
            local_sample_obs = simulator_fn(
                key, local_sample_params, 1, None
            )
            local_tokens = Tokens.from_pytree(
                {**local_sample_params, **local_sample_obs},
                condition=list(local_sample_params.keys()),
                sample_ndims=1,
                labeller=labeller,
                functional_inputs=f_in
            )
            local_estimator = MLP(
                n_ff=config.n_ff,
                latent_dim=config.latent_dim,
                tokens=local_tokens,
                rngs=rngs,
            )
            global_estimator = MLP(
                n_ff=config.n_ff,
                latent_dim=config.latent_dim,
                tokens=tokens,
                rngs=rngs,
            )

        tfmpe_local = TFMPE(
            vf_network=local_estimator,
            base_dist=base_dist,
            solver=diffrax.Dopri5(),
        )
        tfmpe_global = TFMPE(
            vf_network=global_estimator,
            base_dist=base_dist,
            solver=diffrax.Dopri5(),
        )
        tfmpe = [tfmpe_local, tfmpe_global]
        local_optimizer = optax.adam(learning_rate=1e-4)
        local_opt = nnx.Optimizer(tfmpe_local, local_optimizer, wrt=nnx.Param)
        global_optimizer = optax.adam(learning_rate=1e-4)
        global_opt = nnx.Optimizer(tfmpe_global, global_optimizer, wrt=nnx.Param)
        opt = [local_opt, global_opt]

    # Training parameters
    if ablation == 'sequential':
        n_rounds = 5
    else:
        n_rounds = 1
    n_val_samples = num_simulations // 10
    n_samples_per_round = num_simulations // n_rounds - n_val_samples
    n_iter_per_round = 1000
    batch_size = 100

    # Get transforms
    transforms = task._get_transforms(n_l=n_local)["parameters"]

    # probability transformation for truncated proposals
    if automatic_transforms_enabled:
        def prob_transform(params_dict: dict, log_prob: float) -> jnp.ndarray:
            params_list = []
            for name, (start, end) in slices:
                # Extract component from dict and reshape correctly
                # The slice (start, end) tells us how many dimensions this
                # component should have in the flat representation
                component = params_dict[name]
                # Reshape to (num_samples, -1) to flatten all dimensions
                # except the first (sample) dimension
                component_flat = component.reshape(component.shape[0], -1)
                params_list.append(component_flat)

            flattened = jnp.concatenate(params_list, axis=1)
            unconstrained = torch.from_numpy(np.array(flattened)).float()
            constrained = transforms(unconstrained)
            delta = transforms.log_abs_det_jacobian(
                constrained,
                unconstrained
            )
            return log_prob + jnp.array(delta)
    else:
        prob_transform = None

    def prior_log_prob(params_dict: dict) -> jnp.ndarray:
        params_list = []
        for name, (start, end) in slices:
            # Extract component from dict and reshape correctly
            # The slice (start, end) tells us how many dimensions this
            # component should have in the flat representation
            component = params_dict[name]
            # Reshape to (num_samples, -1) to flatten all dimensions
            # except the first (sample) dimension
            component_flat = component.reshape(component.shape[0], -1)
            params_list.append(component_flat)

        flattened = jnp.concatenate(params_list, axis=1)
        unconstrained = torch.from_numpy(np.array(flattened)).float().to(device=device)

        prior_dist = task.prior_dist.for_n_local(n_local)
        if automatic_transforms_enabled:
            transforms = task._get_transforms(n_l=n_local)["parameters"]
            prior_dist = wrap_prior_dist(prior_dist, transforms)

        log_prob = prior_dist.log_prob(unconstrained)

        return jnp.asarray(log_prob)

    # Train TFMPE
    rng = jax.random.PRNGKey(42)
    if ablation == 'direct':
        trained_tfmpe, all_losses = tfmpe_fit_directly(
            tfmpe=tfmpe,
            simulator_fn=simulator_fn,
            prior_fn=prior_fn,
            n_groups=n_local,
            n_samples_per_round=n_samples_per_round,
            n_val_samples=n_val_samples,
            opt=opt,
            n_iter_per_round=n_iter_per_round,
            batch_size=batch_size,
            rng=rng,
            labeller=labeller,
            delta=1e-3,
            patience=100
        )

    elif ablation == 'pf':
        trained_global, trained_local, all_losses = tfmpe_fit_pf(
            tfmpe_global=tfmpe_global,
            tfmpe_local=tfmpe_local,
            simulator_fn=simulator_fn,
            prior_fn=prior_fn,
            local_fn=local_fn,
            global_names=global_names,
            n_groups=n_local,
            n_samples=n_samples_per_round,
            n_val_samples=n_val_samples,
            opt_global=global_opt,
            opt_local=local_opt,
            n_iter=n_iter_per_round,
            batch_size=batch_size,
            rng=rng,
            labeller=labeller,
            delta=1e-3,
            patience=100,
            sample_batch_size=sample_batch_size
        )
    else:
        trained_tfmpe, all_losses = tfmpe_fit_bottom_up(
            tfmpe=tfmpe,
            y_obs=y_obs_dict,
            simulator_fn=simulator_fn,
            prior_fn=prior_fn,
            local_fn=local_fn,
            global_names=global_names,
            n_groups=n_local,
            n_rounds=n_rounds,
            n_samples_per_round=n_samples_per_round,
            n_val_samples=n_val_samples,
            opt=opt,
            n_iter_per_round=n_iter_per_round,
            batch_size=batch_size,
            rng=rng,
            labeller=labeller,
            prob_transform=prob_transform,
            prior_log_prob=prior_log_prob,
            sample_batch_size=sample_batch_size
        )
        

    # Create posterior wrapper and sample
    if ablation == 'pf':
        posterior_wrapped = TFMPEPosteriorPF(
            tfmpe_global=trained_global,
            tfmpe_local=trained_local,
            labeller=labeller,
            slices=slices,
            global_names=global_names,
            local_names=local_names,
            n_local=n_local,
            transforms=transforms if automatic_transforms_enabled else None,
            context=y_obs_dict,
            sample_batch_size=sample_batch_size
        )
    else:
        posterior_wrapped = TFMPEPosterior(
            tfmpe_model=trained_tfmpe,
            labeller=labeller,
            slices=slices,
            global_names=global_names,
            local_names=local_names,
            n_local=n_local,
            transforms=transforms if automatic_transforms_enabled else None,
            context=y_obs_dict,
            params_f_in=None,
            context_f_in=None,
            sample_batch_size=sample_batch_size
        )

    posterior_samples = posterior_wrapped.sample((num_samples,))

    # Compute log probability at true parameters
    true_parameters = task.get_true_parameters(
        num_observation=num_observation
    )
    log_prob_true_params = posterior_wrapped.log_prob(true_parameters)

    return (
        posterior_samples,
        num_simulations,
        log_prob_true_params,
        posterior_wrapped
    )

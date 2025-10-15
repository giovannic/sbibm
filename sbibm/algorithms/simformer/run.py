from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from omegaconf import OmegaConf
from scoresbibm.methods.score_transformer import train_transformer_model
from scoresbibm.tasks.base_task import base_batch_sampler

from sbibm.tasks.task import Task


_MODEL_CFG = {
    "token_dim": 40,
    "condition_token_dim": 10,
    "condition_token_init_scale": 0.1,
    "condition_token_init_mean": 0.0,
    "condition_mode": "concat",
    "time_embedding_dim": 128,
    "num_heads": 4,
    "num_layers": 6,
    "attn_size": 10,
    "widening_factor": 3,
    "num_hidden_layers": 1,
    "skip_connection_attn": True,
    "skip_connection_mlp": True,
    "layer_norm": True,
}

_TRAIN_CFG = {
    "max_number_steps": 100000,
    "min_number_steps": 5000,
    "total_number_steps_scaling": 3,
    "training_batch_size": 1000,
    "learning_rate": 1e-3,
    "min_learning_rate": 1e-6,
    "clip_max_norm": 10.0,
    "validation_fraction": 0.05,
    "val_repeat": 5,
    "val_every": 50,
    "stop_early_count": 5,
    "rebalance_loss": False,
    "z_score_data": False,
    "edge_mask_fn": {"name": "none"},
}

_POSTERIOR_CFG = {
    "sampling_method": "sde",
    "num_steps": 500,
    "method": "euler_maruyama",
}

_SDE_CFG = {
    "name": "vesde",
    "sigma_max": 15.0,
    "sigma_min": 0.0001,
    "T_max": 1.0,
    "T_min": 1e-5,
    "scale_min": 1e-3,
}


@dataclass
class _SimformerTaskShim:
    """Minimal task interface required by simformer training."""

    task: Task

    def __post_init__(self) -> None:
        self.name = self.task.name
        self.theta_dim = self.task.dim_parameters
        self.x_dim = self.task.dim_data
        self.n_l = getattr(self.task, "n_l", 1)

    def get_theta_dim(self):
        return self.theta_dim

    def get_x_dim(self):
        return self.x_dim

    def get_batch_sampler(self):
        return base_batch_sampler

    def get_node_id(self):
        return jnp.arange(self.theta_dim + self.x_dim, dtype=jnp.int32)


class SimformerPosterior:
    """Benchmark-compatible posterior wrapper around simformer."""

    def __init__(
        self,
        model,
        observation: torch.Tensor,
        parameter_transform=None,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.parameter_transform = parameter_transform
        self.seed = 0 if seed is None else int(seed)
        self.observation = observation.detach().cpu().float()
        self.model.set_default_x_o(self._to_jax(self.observation))

    @staticmethod
    def _to_jax(tensor: torch.Tensor):
        return jnp.asarray(tensor.detach().cpu().numpy())

    @staticmethod
    def _to_torch(array) -> torch.Tensor:
        return torch.from_numpy(np.array(array, copy=True)).float().cpu()

    def _to_constrained(self, samples: torch.Tensor) -> torch.Tensor:
        if self.parameter_transform is None:
            return samples
        return self.parameter_transform.inv(samples)

    def sample(self, shape, x=None):
        if isinstance(shape, int):
            shape = (shape,)

        num_samples = int(np.prod(shape)) if shape else 1
        key = jax.random.PRNGKey(self.seed)

        if x is None:
            samples = self.model.sample(num_samples, rng=key)
            samples = self._to_torch(samples)
            samples = self._to_constrained(samples)
            return samples.reshape(*shape, -1)

        x = x.detach().cpu().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_jax = self._to_jax(x)
        batched = self.model.sample_batched(1, x_o=x_jax, rng=key)
        batched = self._to_torch(batched).squeeze(1)
        batched = self._to_constrained(batched)
        return batched


def _build_method_cfg(
    device: str,
    num_simulations: int,
    training_batch_size: int,
    sampling_steps: int,
    train_step_floor: int,
    train_step_cap: int,
    condition_mask_name: str,
):
    train_cfg = dict(_TRAIN_CFG)
    train_cfg["training_batch_size"] = training_batch_size
    train_cfg["min_number_steps"] = max(train_step_floor, 10)
    train_cfg["max_number_steps"] = max(train_step_cap, 10)
    if num_simulations < 100:
        train_cfg["validation_fraction"] = 0.0
    train_cfg["condition_mask_fn"] = {"name": condition_mask_name}

    posterior_cfg = dict(_POSTERIOR_CFG)
    posterior_cfg["num_steps"] = sampling_steps

    return OmegaConf.create({
        "name": "score_transformer_posterior",
        "backend": "jax",
        "device": "gpu" if device.startswith("cuda") else "cpu",
        "model": dict(_MODEL_CFG),
        "train": train_cfg,
        "sde": dict(_SDE_CFG),
        "posterior": posterior_cfg,
    })


def _ensure_min_rows(theta: torch.Tensor, x: torch.Tensor, min_rows: int = 10):
    if theta.shape[0] >= min_rows:
        return theta, x

    repeats = (min_rows + theta.shape[0] - 1) // theta.shape[0]
    theta = theta.repeat((repeats, 1))[:min_rows]
    x = x.repeat((repeats, 1))[:min_rows]
    return theta, x


def _get_parameter_transform(task: Task, automatic_transforms_enabled: bool):
    if not automatic_transforms_enabled or not hasattr(task, "_get_transforms"):
        return None
    transforms = task._get_transforms(automatic_transforms_enabled=True)
    return transforms.get("parameters")


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    training_batch_size: int = 1000,
    sampling_steps: int = 500,
    train_step_floor: int = 5000,
    train_step_cap: int = 100000,
    automatic_transforms_enabled: bool = True,
    condition_mask_name: str = "posterior",
    **kwargs,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], SimformerPosterior]:
    """Run simformer on an existing hierarchical sbibm task."""
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    prior = task.get_prior()
    simulator = task.get_simulator()
    parameter_transform = _get_parameter_transform(
        task, automatic_transforms_enabled=automatic_transforms_enabled
    )
    theta = prior(num_samples=num_simulations).detach().cpu().float()
    x = simulator(theta.to(device=device)).detach().cpu().float()
    if parameter_transform is not None:
        theta = parameter_transform(theta)
    theta_train, x_train = _ensure_min_rows(theta, x)

    if observation is None:
        observation = task.get_observation(num_observation).detach().cpu().float()
    else:
        observation = observation.detach().cpu().float()

    data = {
        "theta": jnp.asarray(theta_train.numpy()),
        "x": jnp.asarray(x_train.numpy()),
    }
    shim = _SimformerTaskShim(task=task)
    method_cfg = _build_method_cfg(
        device=device,
        num_simulations=num_simulations,
        training_batch_size=min(training_batch_size, num_simulations),
        sampling_steps=sampling_steps,
        train_step_floor=train_step_floor,
        train_step_cap=train_step_cap,
        condition_mask_name=condition_mask_name,
    )
    rng = jr.PRNGKey(0 if seed is None else int(seed))

    model = train_transformer_model(shim, data, method_cfg, rng)
    posterior = SimformerPosterior(
        model,
        observation.squeeze(0),
        parameter_transform=parameter_transform,
        seed=seed,
    )
    samples = posterior.sample((num_samples,))

    return samples, num_simulations, None, posterior

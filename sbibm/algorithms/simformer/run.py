import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from sbibm.tasks.task import Task


def _configure_runtime(device: str) -> None:
    """Set cache and JAX runtime defaults before importing simformer."""
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    if "JAX_PLATFORMS" not in os.environ:
        if device.startswith("cpu"):
            os.environ["JAX_PLATFORMS"] = "cpu"
        elif device.startswith("cuda"):
            os.environ["JAX_PLATFORMS"] = "cuda,cpu"


def _simformer_config_path() -> Path:
    return Path.home() / "projects" / "simformer" / "src" / "scoresbibm" / "config"


def _load_yaml_config(relative_path: str) -> dict:
    config_path = _simformer_config_path() / relative_path
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


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
        from scoresbibm.tasks.base_task import base_batch_sampler

        return base_batch_sampler

    def get_node_id(self):
        import jax.numpy as jnp

        return jnp.arange(self.theta_dim + self.x_dim, dtype=jnp.int32)


class SimformerPosterior:
    """Benchmark-compatible posterior wrapper around simformer."""

    def __init__(self, model, observation: torch.Tensor, seed: Optional[int] = None):
        self.model = model
        self.seed = 0 if seed is None else int(seed)
        self.observation = observation.detach().cpu().float()
        self.model.set_default_x_o(self._to_jax(self.observation))

    @staticmethod
    def _to_jax(tensor: torch.Tensor):
        import jax.numpy as jnp

        return jnp.asarray(tensor.detach().cpu().numpy())

    @staticmethod
    def _to_torch(array) -> torch.Tensor:
        return torch.from_numpy(np.array(array, copy=True)).float().cpu()

    def sample(self, shape, x=None):
        import jax

        if isinstance(shape, int):
            shape = (shape,)

        num_samples = int(np.prod(shape)) if shape else 1
        key = jax.random.PRNGKey(self.seed)

        if x is None:
            samples = self.model.sample(num_samples, rng=key)
            samples = self._to_torch(samples)
            return samples.reshape(*shape, -1)

        x = x.detach().cpu().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_jax = self._to_jax(x)
        batched = self.model.sample_batched(1, x_o=x_jax, rng=key)
        batched = self._to_torch(batched).squeeze(1)
        return batched


def _build_method_cfg(
    device: str,
    num_simulations: int,
    training_batch_size: int,
    sampling_steps: int,
    train_step_floor: int,
    train_step_cap: int,
):
    model_cfg = _load_yaml_config("method/model/score_transformer_small.yaml")
    train_cfg = _load_yaml_config("method/train/train_score_transformer.yaml")
    posterior_cfg = _load_yaml_config("method/posterior/backward_sde.yaml")
    sde_cfg = _load_yaml_config("method/sde/vesde.yaml")

    train_cfg["training_batch_size"] = training_batch_size
    train_cfg["min_number_steps"] = max(train_step_floor, 10)
    train_cfg["max_number_steps"] = max(train_step_cap, 10)
    train_cfg["total_number_steps_scaling"] = min(
        train_cfg.get("total_number_steps_scaling", 3), 2
    )
    if num_simulations < 100:
        train_cfg["validation_fraction"] = 0.0
    train_cfg.setdefault("condition_mask_fn", {"name": "structured_random"})
    train_cfg.setdefault("edge_mask_fn", {"name": "none"})

    posterior_cfg["num_steps"] = sampling_steps

    method_cfg = {
        "name": "score_transformer",
        "backend": "jax",
        "device": "gpu" if device.startswith("cuda") else "cpu",
        "model": model_cfg,
        "train": train_cfg,
        "sde": sde_cfg,
        "posterior": posterior_cfg,
    }
    return OmegaConf.create(method_cfg)


def _ensure_min_rows(theta: torch.Tensor, x: torch.Tensor, min_rows: int = 10):
    if theta.shape[0] >= min_rows:
        return theta, x

    repeats = (min_rows + theta.shape[0] - 1) // theta.shape[0]
    theta = theta.repeat((repeats, 1))[:min_rows]
    x = x.repeat((repeats, 1))[:min_rows]
    return theta, x


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    training_batch_size: int = 128,
    sampling_steps: int = 100,
    train_step_floor: int = 200,
    train_step_cap: int = 1000,
    **kwargs,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], SimformerPosterior]:
    """Run simformer on an existing hierarchical sbibm task."""
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    _configure_runtime(device)

    import jax.numpy as jnp
    import jax.random as jr
    from scoresbibm.methods.score_transformer import train_transformer_model

    prior = task.get_prior()
    simulator = task.get_simulator()
    theta = prior(num_samples=num_simulations).detach().cpu().float()
    x = simulator(theta.to(device=device)).detach().cpu().float()
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
    )
    rng = jr.PRNGKey(0 if seed is None else int(seed))

    model = train_transformer_model(shim, data, method_cfg, rng)
    posterior = SimformerPosterior(model, observation.squeeze(0), seed=seed)
    samples = posterior.sample((num_samples,))

    return samples, num_simulations, None, posterior

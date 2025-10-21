from pathlib import Path
from typing import Any, Optional

import diffrax
import jax.numpy as jnp
import numpy
import pyro
import torch
from pyro import distributions as pdist
from pyro.distributions import constraints
from pyro.distributions.transforms import biject_to

from sbibm.tasks.distributions import (
    HierarchicalDistribution,
    SummedStackTransform,
)
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class HierarchicalLotkaVolterra(Task):
    def __init__(
        self,
        n_l: int = 5,
        days: float = 20.0,
        saveat: float = 0.2,
        total_count: int = 100,
    ):
        """Hierarchical Lotka-Volterra model

        Hierarchical extension of the Lotka-Volterra task where each
        observation consists of n_l local contexts (sites). Uses Strategy 1
        (natural global/local split): predation rate (beta) and predator
        death rate (gamma) are global (shared across sites), while prey
        birth rate (alpha) and predator birth rate (delta) are local
        (site-specific).

        Global parameters (2 total):
            - beta: Predation rate ~ LogNormal(log(0.028), 0.5)
            - gamma: Predator death rate ~ LogNormal(log(0.5), 0.5)

        Local parameters (2*n_l total):
            - alpha_i: Prey birth rate per site ~ LogNormal(log(1.0), 0.2)
            - delta_i: Predator birth rate per site ~ LogNormal(log(0.01), 0.2)

        This represents multi-site ecological studies where predation dynamics
        are consistent but birth rates vary by local environmental conditions.

        Observations are LogNormal-distributed with the trajectory as the
        mean in log-space, ensuring bounded likelihood for numerical stability.

        Args:
            n_l: Number of local contexts/sites (default: 5)
            days: Number of days to simulate (default: 20.0)
            saveat: Time step for saving trajectory (default: 0.2)
            total_count: Scaling factor for Poisson observations (default: 100)
        """
        self.n_l = n_l
        self.days = days
        self.saveat = saveat
        self.total_count = total_count

        # Calculate raw data dimensions
        # For subsampling: every 21st time point from trajectory
        # Original: int(days/saveat + 1) = 101 points, subsample -> 5 points
        # Per site: 2 species * 5 time points = 10 observations
        dim_data = 10 * n_l

        # Observation seeds
        observation_seeds = [
            1000020,
            1000030,
            1000034,
            1000013,
            1000004,
            1000011,
            1000012,
            1000039,
            1000041,
            1000009,
        ]

        super().__init__(
            dim_parameters=2 + 2 * n_l,  # 2 global + 2*n_l local
            dim_data=dim_data,
            name="hierarchical_lotka_volterra",
            name_display="Hierarchical Lotka-Volterra",
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            num_posterior_samples=10000,
            num_observations=10,
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        # Global distribution: predation rate (beta) and predator death (gamma)
        # beta ~ LogNormal(log(0.028), 0.5)
        # gamma ~ LogNormal(log(0.5), 0.5)
        mu_beta = torch.log(torch.tensor(0.028))
        mu_gamma = torch.log(torch.tensor(0.5))
        sigma_global = 0.5

        global_dist = pdist.Independent(
            pdist.LogNormal(
                loc=torch.tensor([mu_beta, mu_gamma]),
                scale=torch.tensor([sigma_global, sigma_global]),
            ),
            1,
        )
        global_dist.set_default_validate_args(False)

        # Local distribution: prey birth (alpha_i) and predator birth (delta_i)
        # per site
        # alpha_i ~ LogNormal(log(1.0), 0.2)
        # delta_i ~ LogNormal(log(0.01), 0.2)
        def local_dist_fn(global_params):
            # global_params shape: [..., 2]
            batch_shape = global_params.shape[:-1]
            mu_alpha = torch.log(torch.tensor(1.0))
            mu_delta = torch.log(torch.tensor(0.01))
            sigma_local = 0.2

            # Each site has 2 parameters: alpha_i, delta_i
            loc = torch.stack(
                [
                    torch.full(list(batch_shape) + [n_l], mu_alpha.item()),
                    torch.full(list(batch_shape) + [n_l], mu_delta.item()),
                ],
                dim=-1,
            ).reshape(
                list(batch_shape) + [2 * n_l]
            )  # [..., 2*n_l]

            scale = torch.full_like(loc, sigma_local)

            return pdist.Independent(pdist.LogNormal(loc, scale), 1)

        self.prior_dist = HierarchicalDistribution(
            global_dist, local_dist_fn, dim_global=2, dim_local=2 * n_l
        )
        self.prior_dist.set_default_validate_args(False)

        # Build composite transform (constrained <-> unconstrained)
        # All parameters are log-scale (positive real) -> R
        transforms_list = []

        # Global parameters: 2 (beta, gamma) - all LogNormal
        for _ in range(2):
            transforms_list.append(biject_to(constraints.positive))

        # Local parameters: 2*n_l - all LogNormal (alpha_i, delta_i per site)
        for _ in range(2 * n_l):
            transforms_list.append(biject_to(constraints.positive))

        # Use custom wrapper to ensure Jacobian is properly summed
        self.composite_transform = SummedStackTransform(transforms_list, dim=-1)

        # Initial conditions for ODE
        self.u0 = torch.tensor([30.0, 1.0])

    def _lotka_volterra_ode_func(
        self, t: jnp.ndarray, u: jnp.ndarray, args
    ) -> jnp.ndarray:
        """Vectorized Lotka-Volterra ODE function for batch solving.

        For hierarchical LV, we batch across all (sample, site) pairs.
        Parameters are [alpha, beta, gamma, delta] per batch element.

        Args:
            t: Time (scalar)
            u: State vector shape (batch, 2) with [prey, predator]
            args: Parameters shape (batch, 4) with
                [alpha, beta, gamma, delta]

        Returns:
            du/dt: State derivatives shape (batch, 2)
        """
        x = u[:, 0]
        y = u[:, 1]
        alpha = args[:, 0]
        beta = args[:, 1]
        gamma = args[:, 2]
        delta = args[:, 3]

        dx = alpha * x - beta * x * y
        dy = -gamma * y + delta * x * y

        return jnp.stack([dx, dy], axis=1)

    def solve_ode_trajectories(self, parameters: torch.Tensor) -> torch.Tensor:
        """Solve hierarchical LV ODE for batched parameters.

        Parameters are structured as:
        - [:, 0:2]: global params [beta, gamma]
        - [:, 2:]: local params [alpha_1, delta_1, ..., alpha_n_l,
          delta_n_l]

        We expand this to (num_samples * n_l, 4) with
        [alpha_i, beta, gamma, delta_i] for each batch element.

        Args:
            parameters: Shape (num_samples, 2 + 2*n_l) with
                [beta, gamma, alpha_1, delta_1, ..., alpha_n_l,
                delta_n_l]

        Returns:
            Trajectories shape (num_samples, n_l, 2, num_timepoints)
        """
        num_samples = parameters.shape[0]
        t_save = torch.linspace(0, self.days, int(self.days / self.saveat) + 1)

        # Extract global and local parameters
        global_params = parameters[:, :2]  # (num_samples, 2)
        local_params = parameters[:, 2:]  # (num_samples, 2*n_l)

        # Reshape local params to (num_samples, n_l, 2)
        # where each site has [alpha_i, delta_i]
        local_reshaped = local_params.reshape(num_samples, self.n_l, 2)

        # Create flattened batch: (num_samples * n_l, 4)
        # For each (sample, site) pair, we need [alpha, beta,
        # gamma, delta]
        alpha_flat = local_reshaped[:, :, 0].reshape(-1)
        delta_flat = local_reshaped[:, :, 1].reshape(-1)
        beta_expanded = global_params[:, 0].repeat_interleave(self.n_l)
        gamma_expanded = global_params[:, 1].repeat_interleave(self.n_l)

        # Stack into args format: (num_samples * n_l, 4)
        params_jax = jnp.stack(
            [
                alpha_flat.numpy(),
                beta_expanded.numpy(),
                gamma_expanded.numpy(),
                delta_flat.numpy(),
            ],
            axis=1,
        )

        # Initial conditions for all batch elements
        u0_batch = jnp.tile(
            jnp.array(self.u0.numpy()),
            (num_samples * self.n_l, 1),
        )
        t_save_jax = jnp.array(t_save.numpy())

        # Define ODE term
        vector_field = diffrax.ODETerm(self._lotka_volterra_ode_func)

        # Solve ODE for all (sample, site) pairs in batch
        solution = diffrax.diffeqsolve(
            vector_field,
            diffrax.Dopri5(),
            t0=t_save_jax[0],
            t1=t_save_jax[-1],
            dt0=0.01,
            y0=u0_batch,
            args=params_jax,
            saveat=diffrax.SaveAt(ts=t_save_jax),
            max_steps=16**5,
        )

        # Convert back to PyTorch
        trajectories_np = numpy.asarray(solution.ys).copy()
        trajectories = torch.from_numpy(trajectories_np).to(torch.float32)

        # Permute from (num_timepoints, num_samples*n_l, 2) to
        # (num_samples*n_l, 2, num_timepoints)
        trajectories = trajectories.permute(1, 2, 0)

        # Reshape to (num_samples, n_l, 2, num_timepoints)
        expected_shape = torch.Size(
            [num_samples * self.n_l, 2, int(self.days / self.saveat) + 1]
        )
        if trajectories.shape != expected_shape:
            trajectories = float("nan") * torch.ones(expected_shape)

        trajectories = trajectories.reshape(num_samples, self.n_l, 2, -1)

        return trajectories.float()

    def get_prior(self):
        """Get prior distribution.

        Returns a callable that samples from self.prior_dist using pyro.
        """

        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_prior_dist(self):
        """Get prior distribution object for likelihood computation."""
        return self.prior_dist

    def _get_transforms(self, automatic_transforms_enabled: bool = True, **kwargs: Any):
        """Get transforms for unconstrained <-> constrained space.

        Args:
            automatic_transforms_enabled: Whether to return transforms

        Returns:
            Dictionary with 'parameters' key containing the transform
        """
        return {"parameters": self.composite_transform.inv}

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls
                will result in SimulationBudgetExceeded exceptions. Defaults
                to None for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters: torch.Tensor) -> torch.Tensor:
            """Simulates Lotka-Volterra for hierarchical parameters.

            Parameters structure:
                - Global: [beta, gamma] (first 2 dims)
                - Local: [alpha_1, delta_1, alpha_2, delta_2, ...]
                  (next 2*n_l)

            Returns LogNormal-distributed observations.
            """
            num_samples = parameters.shape[0]

            # Solve ODE for all parameters and sites
            # Returns (num_samples, n_l, 2, num_timepoints)
            all_observations = self.solve_ode_trajectories(parameters)

            data = []
            for b in range(num_samples):
                context_data = []

                # For each local context (site)
                for i in range(self.n_l):
                    # Get trajectory for this site
                    u = all_observations[b, i, :, :]  # (2, timepoints)

                    # Check for NaN
                    if torch.isnan(u).any():
                        context_data.append(float("nan") * torch.ones(10))
                        continue

                    # Subsample every 21st time point
                    u_sub = u[:, ::21]  # (2, 5)

                    # Flatten to (10,)
                    u_flat = u_sub.flatten()

                    # Clamp to ensure valid log values
                    u_flat_clamped = u_flat.clamp(min=1e-10, max=10000.0)

                    # Sample from LogNormal distribution
                    lognormal_dist = pdist.LogNormal(
                        loc=torch.log(u_flat_clamped),
                        scale=0.1,
                    )
                    obs = lognormal_dist.sample()

                    context_data.append(obs)

                # Concatenate all contexts
                data.append(torch.cat(context_data))

            return torch.stack(data).float()

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _likelihood(
        self, parameters: torch.Tensor, data: torch.Tensor, log: bool = True
    ) -> torch.Tensor:
        """Compute likelihood of data given parameters.

        Uses LogNormal likelihood for observations. The likelihood is
        naturally bounded since LogNormal log-likelihood is always
        finite.

        Args:
            parameters: Parameter tensor with shape
                (num_samples, dim_parameters)
            data: Observation tensor with shape (num_samples, dim_data)
            log: If True, return log-likelihood; otherwise return
                likelihood

        Returns:
            (Log-)likelihood values with shape (num_samples,)
        """
        num_samples = parameters.shape[0]

        # Solve ODE for all parameters and sites
        all_observations = self.solve_ode_trajectories(parameters)

        log_likelihoods = []

        for b in range(num_samples):
            log_lik_sample = 0.0

            # For each local context (site)
            for i in range(self.n_l):
                # Get trajectory for this site
                u = all_observations[b, i, :, :]  # (2, timepoints)

                # Check for NaN
                if torch.isnan(u).any():
                    log_lik_sample = float("-inf")
                    break

                # Subsample every 21st time point
                u_sub = u[:, ::21]  # (2, 5)
                u_flat = u_sub.flatten()  # (10,)

                # Clamp to ensure valid log values
                u_flat_clamped = u_flat.clamp(min=1e-10, max=10000.0)

                # Get observed data for this context
                obs = data[b, i * 10 : (i + 1) * 10]

                # Compute LogNormal log-likelihood
                lognormal_dist = torch.distributions.LogNormal(
                    loc=torch.log(u_flat_clamped),
                    scale=0.1,
                )
                log_lik_context = lognormal_dist.log_prob(obs).sum()

                log_lik_sample += log_lik_context

            log_likelihoods.append(log_lik_sample)

        log_lik_tensor = torch.tensor(log_likelihoods)

        if log:
            return log_lik_tensor
        else:
            return torch.exp(log_lik_tensor)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data into multiple observations per site"""
        return data.reshape(-1, self.n_l, 10)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
    ) -> torch.Tensor:
        """Reference posteriors not available for hierarchical tasks

        Use reference-free metrics (reverse KL, LC2ST) instead.
        """
        raise NotImplementedError(
            "Reference posteriors are not available for hierarchical tasks. "
            "Use reference-free metrics (reverse KL, LC2ST) instead."
        )


if __name__ == "__main__":
    task = HierarchicalLotkaVolterra(n_l=5)
    task._setup(n_jobs=4, create_reference=False)

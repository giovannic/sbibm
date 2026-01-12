from pathlib import Path
from typing import Any, Optional

import diffrax
import jax
import jax.numpy as jnp
import numpy
import pyro
import torch
from pyro import distributions as pdist
from pyro.distributions import constraints
from pyro.distributions.transforms import biject_to

from sbibm.tasks.distributions import (
    BlockwiseDistribution,
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
        saveat: float = 5.,
        total_count: int = 100,
    ):
        """Hierarchical Lotka-Volterra model with partial pooling.

        Hierarchical extension of the Lotka-Volterra task implementing
        partial pooling with hyperpriors. Each observation consists of n_l
        local contexts (sites) where all four parameters [alpha, beta,
        gamma, delta] can vary by site, but are drawn from global
        hyperpriors that pool information across sites.

        Global parameters (8 total - hyperpriors):
            - mu_alpha, mu_beta, mu_gamma, mu_delta: Hyperprior means
              ~ Normal(0, 1) for each parameter
            - sigma_alpha, sigma_beta, sigma_gamma, sigma_delta:
              Hyperprior scales ~ HalfNormal(1)

        Local parameters (4*n_l total - site-specific):
            - alpha_i, beta_i, gamma_i, delta_i per site
            - Each site's parameters ~ LogNormal(mu_global, sigma_global)

        The Lotka-Volterra ODE model:
            dx/dt = alpha*x - beta*x*y
            dy/dt = -gamma*y + delta*x*y

        where x=prey population, y=predator population.

        Observations are LogNormal-distributed with the trajectory as the
        mean in log-space, ensuring bounded likelihood for numerical
        stability.

        Args:
            n_l: Number of local contexts/sites (default: 5)
            days: Number of days to simulate (default: 20.0)
            saveat: Time step for saving trajectory (default: 2.)
            total_count: Scaling factor for Poisson observations
                (default: 100)
        """
        self.n_l = n_l
        self.days = days
        self.saveat = saveat
        self.total_count = total_count

        # Calculate data dimensions:
        # Species * locales * observations
        dim_data = 2 * n_l * (int(days / saveat) + 1)

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
            dim_parameters=8 + 4 * n_l,  # 8 global + 4*n_l local
            dim_data=dim_data,
            name="hierarchical_lotka_volterra",
            name_display="Hierarchical Lotka-Volterra",
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            num_posterior_samples=10000,
            num_observations=10,
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        # Global distribution: hyperpriors for mean and scale of each
        # parameter
        # [mu_alpha, mu_beta, mu_gamma, mu_delta, sigma_alpha,
        # sigma_beta, sigma_gamma, sigma_delta]
        # Use smaller scale for hyperprior scales (0.5) to avoid extreme
        # parameter values that cause ODE solver failures
        global_components = [
            pdist.Independent(pdist.Normal(0.0, 0.1).expand([1]), 1),
            # mu_alpha
            pdist.Independent(pdist.Normal(0.0, 0.1).expand([1]), 1),
            # mu_beta
            pdist.Independent(pdist.Normal(0.0, 0.1).expand([1]), 1),
            # mu_gamma
            pdist.Independent(pdist.Normal(0.0, 0.1).expand([1]), 1),
            # mu_delta
            pdist.Independent(pdist.Exponential(1.).expand([1]), 1),
            # sigma_alpha
            pdist.Independent(pdist.Exponential(1.).expand([1]), 1),
            # sigma_beta
            pdist.Independent(pdist.Exponential(1.).expand([1]), 1),
            # sigma_gamma
            pdist.Independent(pdist.Exponential(1.).expand([1]), 1),
            # sigma_delta
        ]

        global_dist = BlockwiseDistribution(global_components)
        global_dist.set_default_validate_args(False)

        # Local distribution: site-specific parameters drawn from
        # hyperpriors
        # For each site: [alpha_i, beta_i, gamma_i, delta_i] ~
        # LogNormal(mu_global, sigma_global)
        def local_dist_fn(global_params, n_local_arg):
            # global_params shape: [..., 8]
            # Extract means and scales
            mu = global_params[..., :4]  # [..., 4] means
            sigma = global_params[..., 4:]  # [..., 4] scales

            # Create independent LogNormal for each site's parameters
            # Stack n_local_arg times with same hyperpriors
            loc_list = []
            scale_list = []
            for _ in range(n_local_arg):
                loc_list.append(mu)
                scale_list.append(sigma)

            # Concatenate across sites: [..., 4*n_local_arg]
            loc = torch.cat(loc_list, dim=-1)
            scale = torch.cat(scale_list, dim=-1)

            return pdist.Independent(pdist.LogNormal(loc, scale), 1)

        self.prior_dist = HierarchicalDistribution(
            global_dist,
            local_dist_fn,
            dim_global=8,
            dim_local=4,
            n_local=n_l,
        )
        self.prior_dist.set_default_validate_args(False)

        # Transforms will be built dynamically in _get_transforms

        # Initial conditions for ODE
        self.u0 = torch.tensor([30.0, 1.0])

    def _lotka_volterra_ode_func_single(
        self, t: jnp.ndarray, u: jnp.ndarray, args
    ) -> jnp.ndarray:
        """Lotka-Volterra ODE function for single (sample, site) pair.

        Args:
            t: Time (scalar)
            u: State vector shape (2,) with [prey, predator]
            args: Tuple of (alpha, beta, gamma, delta) scalars

        Returns:
            du/dt: State derivatives shape (2,)
        """
        x = u[0]
        y = u[1]
        alpha = args[0]
        beta = args[1]
        gamma = args[2]
        delta = args[3]

        dx = alpha * x - beta * x * y
        dy = -gamma * y + delta * x * y

        return jnp.array([dx, dy])

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

    def _solve_ode_single_site_jax(
        self,
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        gamma: jnp.ndarray,
        delta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve LV ODE for a single (sample, site) pair in JAX.

        Args:
            alpha, beta, gamma, delta: JAX scalars (Lotka-Volterra params)

        Returns:
            Trajectory array shape (2, num_timepoints) in JAX
            Returns NaN array if ODE solving fails
        """
        t_save = torch.linspace(0, self.days, int(self.days / self.saveat) + 1)
        t_save_jax = jnp.array(t_save.numpy())
        num_timepoints = len(t_save_jax)

        # Initial conditions
        u0 = jnp.array(self.u0.numpy(), dtype=jnp.float32)

        # ODE term
        vector_field = diffrax.ODETerm(self._lotka_volterra_ode_func_single)

        # Solve ODE
        solution = diffrax.diffeqsolve(
            vector_field,
            diffrax.Dopri5(),
            t0=t_save_jax[0],
            t1=t_save_jax[-1],
            dt0=0.01,
            y0=u0,
            args=jnp.array([alpha, beta, gamma, delta]),
            saveat=diffrax.SaveAt(ts=t_save_jax),
            max_steps=16**5,
        )

        # Permute to (2, timepoints)
        return solution.ys.T

    def _solve_ode_trajectories_jax(
        self,
        alpha_jax: jnp.ndarray,
        beta_jax: jnp.ndarray,
        gamma_jax: jnp.ndarray,
        delta_jax: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve hierarchical LV ODE using vmapped solver in JAX.

        All computation stays in JAX land. Caller converts torch to/from JAX.

        Args:
            alpha_jax, beta_jax, gamma_jax, delta_jax: JAX arrays
                shape (num_samples, n_l)

        Returns:
            Trajectories in JAX shape (num_samples, n_l, 2, num_timepoints)
        """
        # Create vmapped solver: vmap over sites (axis 1)
        solve_vmap_sites = jax.vmap(
            self._solve_ode_single_site_jax, in_axes=(0, 0, 0, 0)
        )

        # Create vmapped solver over samples
        solve_vmap_both = jax.vmap(
            solve_vmap_sites, in_axes=(0, 0, 0, 0)
        )

        # Solve all ODE systems (stays in JAX)
        trajectories_jax = solve_vmap_both(
            alpha_jax,
            beta_jax,
            gamma_jax,
            delta_jax
        )

        return trajectories_jax

    def solve_ode_trajectories(self, parameters: torch.Tensor) -> torch.Tensor:
        """Solve hierarchical LV ODE using vmapped solver.

        Solves ODE for each (sample, site) pair independently using
        JAX's vmap for vectorization.

        Parameters are structured as:
        - [:, 0:8]: global hyperprior params (not used for ODE solving)
        - [:, 8:]: local params [alpha_1, beta_1, gamma_1, delta_1,
          ..., alpha_n_l, beta_n_l, gamma_n_l, delta_n_l]

        Args:
            parameters: Shape (num_samples, 8 + 4*n_l)

        Returns:
            Trajectories shape (num_samples, n_l, 2, num_timepoints)
        """
        num_samples = parameters.shape[0]

        # Extract local parameters
        local_params = parameters[:, 8:]  # (num_samples, 4*n_l)

        # Infer n_l from local parameters shape
        n_l = local_params.shape[1] // 4

        # Reshape local params to (num_samples, n_l, 4)
        # where each site has [alpha_i, beta_i, gamma_i, delta_i]
        local_reshaped = local_params.reshape(num_samples, n_l, 4)

        # Convert to JAX arrays
        alpha = jnp.array(local_reshaped[:, :, 0].numpy())
        beta = jnp.array(local_reshaped[:, :, 1].numpy())
        gamma = jnp.array(local_reshaped[:, :, 2].numpy())
        delta = jnp.array(local_reshaped[:, :, 3].numpy())

        # Solve in JAX
        trajectories_jax = self._solve_ode_trajectories_jax(
            alpha, beta, gamma, delta
        )

        # Convert back to PyTorch
        trajectories_np = numpy.asarray(trajectories_jax).copy()
        trajectories = torch.from_numpy(trajectories_np).to(torch.float32)
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

    def _get_transforms(
        self, automatic_transforms_enabled: bool = True, n_l=None, **kwargs: Any
    ):
        """Get transforms for unconstrained <-> constrained space.

        Args:
            automatic_transforms_enabled: Whether to return transforms
            n_l: Number of local contexts (uses self.n_l if None)

        Returns:
            Dictionary with 'parameters' key containing the transform
        """
        if n_l is None:
            n_l = self.n_l

        # Build composite transform (constrained <-> unconstrained)
        # All parameters are positive (LogNormal in local) -> R
        transforms_list = []

        # Global hyperprior means: unbounded (Normal)
        for _ in range(4):
            transforms_list.append(torch.distributions.transforms.identity_transform)

        # Global hyperprior scales: positive (HalfNormal)
        for _ in range(4):
            transforms_list.append(biject_to(constraints.positive))

        # Local parameters: all positive (LogNormal)
        for _ in range(4 * n_l):
            transforms_list.append(biject_to(constraints.positive))

        # Use custom wrapper to ensure Jacobian is properly summed
        composite_transform = SummedStackTransform(transforms_list, dim=-1)
        return {"parameters": composite_transform.inv}

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
                - Global: [mu_alpha, mu_beta, mu_gamma, mu_delta,
                  sigma_alpha, sigma_beta, sigma_gamma, sigma_delta]
                  (first 8 dims)
                - Local: [alpha_1, beta_1, gamma_1, delta_1, ...,
                  alpha_n_l, beta_n_l, gamma_n_l, delta_n_l]
                  (next 4*n_l)

            Returns LogNormal-distributed observations.
            """
            # Solve ODE for all parameters and sites
            # Returns (num_samples, n_l, 2, num_timepoints)
            all_observations = self.solve_ode_trajectories(parameters)
            num_samples = all_observations.shape[0]
            dist = pdist.LogNormal(
                loc=torch.log(all_observations.clamp(1e-10, 10000.0)),
                scale=0.1,
            ).to_event(1)

            data = dist.sample()
            data = data.reshape((num_samples, -1))
            return data

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _likelihood(
        self, parameters: torch.Tensor, data: torch.Tensor, log: bool = True
    ) -> torch.Tensor:
        """Compute likelihood of data given parameters.

        Uses LogNormal likelihood for observations. The likelihood is
        naturally bounded since LogNormal log-likelihood is always
        finite.

        Parameters structure:
            - Global: [mu_alpha, mu_beta, mu_gamma, mu_delta,
              sigma_alpha, sigma_beta, sigma_gamma, sigma_delta]
            - Local: [alpha_1, beta_1, gamma_1, delta_1, ...,
              alpha_n_l, beta_n_l, gamma_n_l, delta_n_l]

        Args:
            parameters: Parameter tensor with shape
                (num_samples, 8 + 4*n_l)
            data: Observation tensor with shape (num_samples, dim_data)
            log: If True, return log-likelihood; otherwise return
                likelihood

        Returns:
            (Log-)likelihood values with shape (num_samples,)
        """
        num_samples = parameters.shape[0]

        # Solve ODE for all parameters and sites
        all_observations = self.solve_ode_trajectories(parameters)
        dist = pdist.LogNormal(
            loc=torch.log(all_observations.clamp(1e-10, 10000.0)).reshape(num_samples, -1),
            scale=0.1,
        ).to_event(1)

        log_lik = dist.log_prob(data)
        if log:
            return log_lik
        else:
            return torch.exp(log_lik)

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup hierarchical lotka volterra task"
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=5,
        help="Number of local contexts",
    )
    args = parser.parse_args()

    task = HierarchicalLotkaVolterra(n_l=args.n_l)
    task._setup(n_jobs=1, create_reference=False)

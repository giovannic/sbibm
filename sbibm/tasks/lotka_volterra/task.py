from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import diffrax
import jax
import jax.numpy as jnp
import numpy
import pyro
import torch
from pyro import distributions as pdist

import sbibm  # noqa -- needed for setting sysimage path
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class LotkaVolterra(Task):
    def __init__(
        self,
        days: float = 20.0,
        saveat: float = 0.1,
        summary: Optional[str] = "subsample",
    ):
        """Lotka-Volterra model

        Args:
            N: Total population
            I0: Initial number of infected individuals
            R0: Initial number of recovered individuals
            days: Number of days
            saveat: When to save during solving
            summary: Summaries to use

        References:
            [1]: https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html
        """
        self.dim_data_raw = int(2 * (days / saveat + 1))

        if summary is None:
            dim_data = self.dim_data_raw
        elif summary == "subsample":
            dim_data = 20
        else:
            raise NotImplementedError
        self.summary = summary

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000020,  # observation 1
            1000030,  # observation 2
            1000034,  # observation 3
            1000013,  # observation 4
            1000004,  # observation 5
            1000011,  # observation 6
            1000012,  # observation 7
            1000039,  # observation 8
            1000041,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=4,
            dim_data=dim_data,
            name=Path(__file__).parent.name,
            name_display="Lotka-Volterra",
            num_observations=len(observation_seeds),
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        # Prior
        mu_p1 = -0.125
        mu_p2 = -3.0
        sigma_p = 0.5
        self.prior_params = {
            "loc": torch.tensor([mu_p1, mu_p2, mu_p1, mu_p2]),
            "scale": torch.tensor([sigma_p, sigma_p, sigma_p, sigma_p]),
        }
        self.prior_dist = pdist.LogNormal(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.u0 = torch.tensor([30.0, 1.0])
        self.tspan = torch.tensor([0.0, days])
        self.days = days
        self.saveat = saveat

        # NOTE: For subsample statistic
        self.total_count = 1000  # TODO: Value?

    def _lotka_volterra_ode_func(
        self, t: jnp.ndarray, u: jnp.ndarray, args
    ) -> jnp.ndarray:
        """Vectorized Lotka-Volterra ODE function for batch solving.

        Args:
            t: Time (scalar)
            u: State vector shape (batch, 2) with [prey, predator]
            args: Tuple of (alpha, beta, gamma, delta) with
                shape (batch, 4)

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
        """Solve Lotka-Volterra ODE for batched parameters.

        Args:
            parameters: Shape (num_samples, 4) with
                [alpha, beta, gamma, delta] for each sample

        Returns:
            Trajectories shape (num_samples, 2, num_timepoints)
            with [prey, predator] populations over time
        """
        num_samples = parameters.shape[0]
        t_save = torch.linspace(0, self.days, int(self.days / self.saveat) + 1)

        # Convert to JAX arrays
        params_jax = jnp.array(parameters.numpy())
        u0_batch = jnp.tile(jnp.array(self.u0.numpy()), (num_samples, 1))
        t_save_jax = jnp.array(t_save.numpy())

        # Define ODE term
        vector_field = diffrax.ODETerm(self._lotka_volterra_ode_func)

        # Solve ODE for all samples in batch
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

        # Permute from (num_timepoints, num_samples, 2) to
        # (num_samples, 2, num_timepoints)
        trajectories = trajectories.permute(1, 2, 0)

        # Validate output shape
        expected_shape = torch.Size([num_samples, 2, int(self.dim_data_raw / 2)])
        if trajectories.shape != expected_shape:
            trajectories = float("nan") * torch.ones(expected_shape)

        return trajectories.float()

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$"]

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(
        self,
        max_calls: Optional[int] = None,
    ) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls
                will result in SimulationBudgetExceeded exceptions.
                Defaults to None for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            # Solve ODE for all parameters
            us = self.solve_ode_trajectories(parameters)

            idx_contains_nan = torch.where(
                torch.isnan(us.reshape(num_samples, -1)).any(axis=1)
            )[
                0
            ]  # noqa
            idx_contains_no_nan = torch.where(
                ~torch.isnan(us.reshape(num_samples, -1)).any(axis=1)
            )[
                0
            ]  # noqa

            if self.summary is None:
                return us

            elif self.summary == "subsample":
                data = float("nan") * torch.ones((num_samples, self.dim_data))
                if len(idx_contains_nan) == num_samples:
                    return data

                us = us[:, :, ::21].reshape(num_samples, -1)
                data[idx_contains_no_nan, :] = pyro.sample(
                    "data",
                    pdist.LogNormal(
                        loc=torch.log(us[idx_contains_no_nan, :].clamp(1e-10, 10000.0)),
                        scale=0.1,
                    ).to_event(1),
                )
                return data

            else:
                raise NotImplementedError

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data into multiple observations"""
        if self.summary is None:
            return data.reshape(-1, 2, int(self.dim_data / 2))
        else:
            return data.reshape(-1, self.dim_data)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Args:
            num_observation: Observation number
            num_samples: Number of samples to generate
            observation: Observed data, if None, will be loaded using `num_observation`
            kwargs: Passed to run_mcmc

        Returns:
            Samples from reference posterior
        """
        from sbibm.algorithms.pyro.mcmc import run as run_mcmc
        from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
        from sbibm.algorithms.pytorch.utils.proposal import get_proposal

        if num_observation is not None:
            initial_params = self.get_true_parameters(num_observation=num_observation)
        else:
            initial_params = None

        proposal_samples = run_mcmc(
            task=self,
            kernel="Slice",
            jit_compile=False,
            num_warmup=10_000,
            num_chains=1,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            initial_params=initial_params,
            automatic_transforms_enabled=True,
        )

        proposal_dist = get_proposal(
            task=self,
            samples=proposal_samples,
            prior_weight=0.1,
            bounded=True,
            density_estimator="flow",
            flow_model="nsf",
        )

        samples = run_rejection(
            task=self,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            batch_size=10_000,
            num_batches_without_new_max=1_000,
            multiplier_M=1.2,
            proposal_dist=proposal_dist,
        )

        return samples


if __name__ == "__main__":
    task = LotkaVolterra()
    task._setup(n_jobs=-1)

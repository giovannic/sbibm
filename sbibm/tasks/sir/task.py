import math
from pathlib import Path
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy
import pyro
import torch
from pyro import distributions as pdist

import diffrax

import sbibm  # noqa -- needed for setting sysimage path
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class SIR(Task):
    def __init__(
        self,
        N: float = 1000000.0,
        I0: float = 1.0,
        R0: float = 0.0,
        days: float = 160.0,
        saveat: float = 1.0,
        total_count: int = 1000,
        summary: Optional[str] = "subsample",
    ):
        """SIR epidemic model

        Inference is performed for two parameters:
        - Contact rate, beta
        - Mean recovery rate, gamma, (in 1/days)

        Args:
            N: Total population
            I0: Initial number of infected individuals
            R0: Initial number of recovered individuals
            days: Number of days
            saveat: When to save during solving
            summary: Summaries to use

        References:
            [1]: https://jrmihalj.github.io/estimating-transmission-by-fitting-mechanistic-models-in-Stan/
        """
        self.dim_data_raw = int(3 * (days / saveat + 1))

        if summary is None:
            dim_data = self.dim_data_raw
        elif summary == "subsample":
            dim_data = 10
        else:
            raise NotImplementedError
        self.summary = summary
        self.total_count = total_count

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000000,  # observation 1
            1000001,  # observation 2
            1000010,  # observation 3*
            1000011,  # observation 4*
            1000004,  # observation 5
            1000005,  # observation 6
            1000006,  # observation 7
            1000013,  # observation 8*
            1000008,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=2,
            dim_data=dim_data,
            name=Path(__file__).parent.name,
            name_display="SIR",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        # Prior over beta and gamma
        self.prior_params = {
            "loc": torch.tensor([math.log(0.4), math.log(0.125)]),
            "scale": torch.tensor([0.5, 0.2]),
        }
        self.prior_dist = pdist.LogNormal(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.u0 = torch.tensor([N - I0 - R0, I0, R0])
        self.tspan = torch.tensor([0.0, days])

        self.days = days
        self.saveat = saveat
        self.N = N

    def _sir_ode(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """SIR ODE right-hand side function

        Args:
            t: Time (scalar)
            u: State vector [S, I, R]

        Returns:
            du/dt: State derivatives [dS, dI, dR]
        """
        S, I, R = u[0], u[1], u[2]
        beta, gamma = self._current_params[0], self._current_params[1]

        dS = -beta * S * I / self.N
        dI = beta * S * I / self.N - gamma * I
        dR = gamma * I

        return torch.stack([dS, dI, dR])

    def _sir_ode_func(
        self, t: jnp.ndarray, u: jnp.ndarray, args
    ) -> jnp.ndarray:
        """Vectorized SIR ODE function for batch solving.

        Args:
            t: Time (scalar)
            u: State vector shape (batch, 3) with [S, I, R]
            args: Tuple of (beta, gamma) with shape (batch, 2)

        Returns:
            du/dt: State derivatives shape (batch, 3)
        """
        S = u[:, 0]
        I = u[:, 1]
        beta = args[:, 0]
        gamma = args[:, 1]

        dS = -beta * S * I / self.N
        dI = beta * S * I / self.N - gamma * I
        dR = gamma * I

        return jnp.stack([dS, dI, dR], axis=1)

    def solve_ode_trajectories(
        self, parameters: torch.Tensor
    ) -> torch.Tensor:
        """Solve SIR ODE for batched parameters (deterministic).

        Args:
            parameters: Shape (num_samples, 2) with [beta, gamma] for
                each sample

        Returns:
            Trajectories shape (num_samples, 3, num_timepoints) with
            [S, I, R] populations over time
        """
        num_samples = parameters.shape[0]
        t_save = torch.linspace(
            0, self.days, int(self.days / self.saveat) + 1
        )

        # Convert to JAX arrays
        params_jax = jnp.array(parameters.numpy())
        u0_batch = jnp.tile(
            jnp.array(self.u0.numpy()), (num_samples, 1)
        )
        t_save_jax = jnp.array(t_save.numpy())

        # Define ODE term
        vector_field = diffrax.ODETerm(self._sir_ode_func)

        # Solve ODE for all samples in batch
        solution = diffrax.diffeqsolve(
            vector_field,
            diffrax.Dopri5(),
            t0=t_save_jax[0],
            t1=t_save_jax[-1],
            dt0=0.1,
            y0=u0_batch,
            args=params_jax,
            saveat=diffrax.SaveAt(ts=t_save_jax),
            max_steps=16**5,
        )

        # Convert back to PyTorch
        trajectories_np = numpy.asarray(solution.ys).copy()
        trajectories = torch.from_numpy(
            trajectories_np
        ).to(torch.float32)

        # Permute from (num_timepoints, num_samples, 3) to
        # (num_samples, 3, num_timepoints)
        trajectories = trajectories.permute(1, 2, 0)

        # Validate output shape
        expected_shape = torch.Size(
            [num_samples, 3, int(self.dim_data_raw / 3)]
        )
        if trajectories.shape != expected_shape:
            trajectories = float("nan") * torch.ones(expected_shape)

        return trajectories.float()

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [r"$\beta$", r"$\gamma$"]

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
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

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

                us = us[:, 1, ::17].reshape(num_samples, -1)  # Only use I population
                data[idx_contains_no_nan, :] = pyro.sample(
                    "data",
                    pdist.Binomial(
                        total_count=self.total_count,
                        probs=(us[idx_contains_no_nan, :] / self.N).clamp(0.0, 1.0),
                    ).to_event(1),
                )
                return data

            else:
                raise NotImplementedError

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data into multiple observations"""
        if self.summary is None:
            return data.reshape(-1, 3, int(self.dim_data / 3))
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
    task = SIR()
    task._setup()

import math
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


class HierarchicalSIR(Task):
    def __init__(
        self,
        n_l: int = 5,
        N: float = 1000000.0,
        I0: float = 1.0,
        R0: float = 0.0,
        days: float = 160.0,
        saveat: float = 1.0,
        total_count: int = 1000,
        summary: Optional[str] = "subsample",
    ):
        """Hierarchical SIR epidemic model

        Hierarchical extension of the SIR task where each observation consists
        of n_l local contexts (e.g., different regions). Uses Strategy 1
        (natural global/local split): global parameter represents shared
        transmission rate (beta) across regions, while local parameters
        represent region-specific recovery rates (gamma).

        Global parameters (dim=1):
            - beta: Transmission rate shared across all regions
            - Prior: LogNormal(log(0.4), 0.5)

        Local parameters (dim=n_l):
            - gamma_i: Recovery rate per region
            - Prior: LogNormal(log(0.125), 0.2) for each region

        Args:
            n_l: Number of local contexts/regions (default: 5)
            N: Total population per region
            I0: Initial number of infected individuals per region
            R0: Initial number of recovered individuals per region
            days: Number of days
            saveat: When to save during solving
            total_count: Total count for Binomial sampling
            summary: Summaries to use ('subsample' or None)
        """
        self.n_l = n_l
        self.N = N
        self.I0 = I0
        self.R0 = R0
        self.days = days
        self.saveat = saveat
        self.total_count = total_count
        self.summary = summary

        self.dim_data_raw = int(3 * (days / saveat + 1))

        if summary is None:
            dim_data = self.dim_data_raw * n_l
        elif summary == "subsample":
            dim_data = 10 * n_l  # 10 subsampled points per region
        else:
            raise NotImplementedError

        # Observation seeds
        observation_seeds = [
            3000001,
            3000002,
            3000003,
            3000004,
            3000005,
            3000006,
            3000007,
            3000008,
            3000009,
            3000010,
        ]

        super().__init__(
            dim_parameters=1 + n_l,  # 1 global (beta) + n_l local (gamma)
            dim_data=dim_data,
            name="hierarchical_sir",
            name_display="Hierarchical SIR",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        # Define hierarchical prior distribution
        # Global parameter: beta (transmission rate)
        global_dist = pdist.LogNormal(
            loc=torch.tensor([math.log(0.4)]),
            scale=torch.tensor([0.5]),
        ).to_event(1)

        # Local parameters: gamma_i (recovery rates per region)
        def local_dist_fn(global_params):
            # Return LogNormal distribution for n_l recovery rates
            # Independent of global beta
            batch_shape = global_params.shape[:-1]
            return pdist.Independent(
                pdist.LogNormal(
                    loc=torch.tensor(math.log(0.125)),
                    scale=torch.tensor(0.2),
                ).expand(list(batch_shape) + [n_l]),
                1,
            )

        self.prior_dist = HierarchicalDistribution(
            global_dist, local_dist_fn, dim_global=1, dim_local=n_l
        )
        self.prior_dist.set_default_validate_args(False)

        # Build composite transform (constrained <-> unconstrained)
        # All parameters are log-scale (positive real) -> R
        transforms_list = []

        # Beta: LogNormal (R+) <-> R
        transforms_list.append(biject_to(constraints.positive))

        # Gammas: LogNormal (R+) <-> R
        for _ in range(n_l):
            transforms_list.append(biject_to(constraints.positive))

        # Use custom wrapper to ensure Jacobian is properly summed
        self.composite_transform = SummedStackTransform(transforms_list, dim=-1)

        # Initial conditions per region
        self.u0 = torch.tensor([N - I0 - R0, I0, R0])

    def _sir_ode_func(self, t: jnp.ndarray, u: jnp.ndarray, args) -> jnp.ndarray:
        """Vectorized SIR ODE function for batch solving.

        For hierarchical SIR, we batch across all (sample, region) pairs.

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

    def solve_ode_trajectories(self, parameters: torch.Tensor) -> torch.Tensor:
        """Solve hierarchical SIR ODE for batched parameters.

        For each sample, we have global beta and n_l local gammas.
        We flatten to (num_samples * n_l,) batch dimension for
        Diffrax vectorization.

        Args:
            parameters: Shape (num_samples, 1 + n_l) with
                [beta, gamma_1, ..., gamma_n_l]

        Returns:
            Trajectories shape (num_samples, n_l, 3, num_timepoints)
        """
        num_samples = parameters.shape[0]
        t_save = torch.linspace(0, self.days, int(self.days / self.saveat) + 1)

        # Extract beta and gamma for each sample
        beta = parameters[:, 0]  # (num_samples,)
        gamma = parameters[:, 1:]  # (num_samples, n_l)

        # Create flattened batch: (num_samples * n_l,)
        # Repeat beta for each region
        beta_expanded = beta.repeat_interleave(self.n_l)  # (num_samples * n_l,)
        gamma_flat = gamma.reshape(-1)  # (num_samples * n_l,)

        # Stack into args format: (num_samples * n_l, 2)
        params_jax = jnp.stack([beta_expanded.numpy(), gamma_flat.numpy()], axis=1)

        # Initial conditions for all batch elements
        u0_batch = jnp.tile(
            jnp.array(self.u0.numpy()),
            (num_samples * self.n_l, 1),
        )
        t_save_jax = jnp.array(t_save.numpy())

        # Define ODE term
        vector_field = diffrax.ODETerm(self._sir_ode_func)

        # Solve ODE for all (sample, region) pairs in batch
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
        trajectories = torch.from_numpy(trajectories_np).to(torch.float32)

        # Permute from (num_timepoints, num_samples*n_l, 3) to
        # (num_samples*n_l, 3, num_timepoints)
        trajectories = trajectories.permute(1, 2, 0)

        # Reshape to (num_samples, n_l, 3, num_timepoints)
        expected_shape = torch.Size(
            [num_samples * self.n_l, 3, int(self.dim_data_raw / 3)]
        )
        if trajectories.shape != expected_shape:
            trajectories = float("nan") * torch.ones(expected_shape)

        trajectories = trajectories.reshape(num_samples, self.n_l, 3, -1)

        return trajectories.float()

    def get_labels_parameters(self):
        """Get list containing parameter labels"""
        labels = [r"$\beta$"]  # Global parameter
        for i in range(self.n_l):
            labels.append(rf"$\gamma_{i}$")  # Local parameters
        return labels

    def get_prior(self):
        """Get prior distribution.

        Returns a callable that samples from self.prior_dist using pyro.
        """

        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get simulator function.

        For each local context (region), solves the SIR ODE with the global
        beta and region-specific gamma_i.

        Args:
            max_calls: Maximum number of simulator calls

        Returns:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            # Solve ODE for all parameters and regions
            # Vectorizes over both samples and local sites
            all_observations = self.solve_ode_trajectories(parameters)

            # Check for NaN values
            idx_contains_nan = torch.where(
                torch.isnan(all_observations.reshape(num_samples, -1)).any(axis=1)
            )[0]
            idx_contains_no_nan = torch.where(
                ~torch.isnan(all_observations.reshape(num_samples, -1)).any(axis=1)
            )[0]

            if self.summary is None:
                # Return raw observations: flatten to (num_samples, dim_data)
                return all_observations.reshape(num_samples, -1)

            elif self.summary == "subsample":
                data = float("nan") * torch.ones((num_samples, self.dim_data))
                if len(idx_contains_nan) == num_samples:
                    return data

                # Subsample infected population (I) every 17 time steps
                # all_observations[:, :, 1, ::17] -> (num_samples, n_l, 10)
                us_subsampled = all_observations[:, :, 1, ::17]

                # Apply Binomial sampling
                for region_idx in range(self.n_l):
                    us_region = us_subsampled[:, region_idx, :]
                    data_region = pyro.sample(
                        f"data_region_{region_idx}",
                        pdist.Binomial(
                            total_count=self.total_count,
                            probs=(us_region[idx_contains_no_nan, :] / self.N).clamp(
                                0.0, 1.0
                            ),
                        ).to_event(1),
                    )
                    # Place in correct position
                    data[
                        idx_contains_no_nan,
                        region_idx * 10 : (region_idx + 1) * 10,
                    ] = data_region

                return data

            else:
                raise NotImplementedError

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _likelihood(
        self, parameters: torch.Tensor, data: torch.Tensor, log: bool = True
    ) -> torch.Tensor:
        """Compute likelihood of data given parameters.

        Likelihood model: For each region and time point, observed
        counts follow Binomial(total_count, p=I(t)/N) where I(t) is
        the infected population from the ODE solution.

        Args:
            parameters: Parameter tensor (batch_size, dim_parameters)
            data: Data tensor (batch_size, dim_data)
            log: Whether to return log-likelihood (default: True)

        Returns:
            (Log-)likelihood values (batch_size,)
        """
        num_samples = parameters.shape[0]

        # Solve ODE for all parameters and regions
        all_observations = self.solve_ode_trajectories(parameters)

        log_likelihoods = []

        for num_sample in range(num_samples):
            data_sample = data[num_sample, :]
            sample_log_likelihood = 0.0

            # Check if this sample has NaN
            if torch.isnan(all_observations[num_sample].reshape(-1)).any():
                sample_log_likelihood = float("-inf")
            else:
                for region_idx in range(self.n_l):
                    # Get I population for this region
                    I_trajectory = all_observations[num_sample, region_idx, 1, :]

                    # Subsample every 17 time steps
                    I_subsampled = I_trajectory[::17]

                    # Get data for this region
                    data_region = data_sample[region_idx * 10 : (region_idx + 1) * 10]

                    # Compute Binomial log-likelihood
                    probs = (I_subsampled / self.N).clamp(0.0, 1.0)
                    binomial_dist = pdist.Binomial(
                        total_count=self.total_count,
                        probs=probs,
                    )

                    # Sum log-likelihood across time points
                    region_log_lik = binomial_dist.log_prob(data_region).sum()
                    sample_log_likelihood += region_log_lik

            log_likelihoods.append(sample_log_likelihood)

        log_likelihoods = torch.tensor(log_likelihoods)

        if log:
            return log_likelihoods
        else:
            return torch.exp(log_likelihoods)

    def get_prior_dist(self):
        """Get prior distribution object.

        Returns:
            Prior distribution
        """
        return self.prior_dist

    def _get_transforms(self, automatic_transforms_enabled: bool = True, **kwargs: Any):
        """Get transforms for unconstrained <-> constrained space.

        Args:
            automatic_transforms_enabled: Whether to return transforms

        Returns:
            Dictionary with 'parameters' key containing the transform
        """
        return {"parameters": self.composite_transform.inv}

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior - not implemented for hierarchical.

        Reference posteriors are not available for hierarchical tasks.
        Use reference-free metrics (reverse KL, LC2ST) instead.

        Args:
            num_samples: Number of samples to generate
            num_observation: Observation number
            observation: Observation tensor

        Raises:
            NotImplementedError: Reference posteriors not available
        """
        raise NotImplementedError(
            "Reference posteriors are not available for hierarchical tasks. "
            "Use reference-free metrics (reverse KL, LC2ST) instead."
        )


if __name__ == "__main__":
    task = HierarchicalSIR(n_l=5)
    task._setup(n_jobs=4, create_reference=False)

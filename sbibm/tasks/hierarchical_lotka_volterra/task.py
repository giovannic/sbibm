from pathlib import Path
from typing import Optional

import pyro
import torch
from pyro import distributions as pdist
from torchdiffeq import odeint

from sbibm.tasks.distributions import HierarchicalDistribution
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class HierarchicalLotkaVolterra(Task):
    def __init__(
        self, n_l: int = 5, days: float = 20.0, saveat: float = 0.2,
        total_count: int = 100
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

        Observations are Poisson-distributed counts with rate = trajectory /
        total_count, ensuring bounded likelihood for numerical stability.

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

        # Initial conditions for ODE
        self.u0 = torch.tensor([30.0, 1.0])

    def _lotka_volterra_ode(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Lotka-Volterra ODE right-hand side function

        Args:
            t: Time (scalar)
            u: State vector [prey, predator]

        Returns:
            du/dt: State derivatives [dx, dy]
        """
        x, y = u[0], u[1]
        alpha, beta, gamma, delta = (
            self._current_params[0],
            self._current_params[1],
            self._current_params[2],
            self._current_params[3],
        )

        dx = alpha * x - beta * x * y
        dy = -gamma * y + delta * x * y

        return torch.stack([dx, dy])

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
            """Simulates Lotka-Volterra for given hierarchical parameters

            Parameters structure:
                - Global: [beta, gamma] (first 2 dims)
                - Local: [alpha_1, delta_1, alpha_2, delta_2, ...] (next 2*n_l)

            Returns Poisson-distributed observations with rate proportional
            to ODE trajectory.
            """
            num_samples = parameters.shape[0]

            # Generate time points for ODE integration
            t = torch.linspace(0, self.days, int(self.days / self.saveat) + 1)

            # Split parameters into global and local
            global_params = parameters[:, :2]  # (num_samples, 2): [beta, gamma]
            local_params = parameters[:, 2:]  # (num_samples, 2*n_l)

            data = []
            for b in range(num_samples):
                context_data = []

                # For each local context (site), run ODE simulation
                for i in range(self.n_l):
                    # Extract local parameters for this site
                    alpha_i = local_params[b, 2 * i]
                    delta_i = local_params[b, 2 * i + 1]
                    beta = global_params[b, 0]
                    gamma = global_params[b, 1]

                    # Combine into full parameter vector [alpha, beta, gamma, delta]
                    self._current_params = torch.tensor(
                        [alpha_i, beta, gamma, delta_i]
                    )

                    # Solve ODE using torchdiffeq
                    try:
                        u_trajectory = odeint(
                            self._lotka_volterra_ode, self.u0, t, method="dopri5"
                        )
                        # Transpose to (state_dim, time_steps) to match format
                        u = u_trajectory.T  # (2, time_steps)

                        # Check for valid trajectory
                        if u.shape != torch.Size([2, int(self.days / self.saveat) + 1]):
                            # Invalid shape, return NaN
                            context_data.append(
                                float("nan") * torch.ones(10)
                            )
                            continue

                        if torch.isnan(u).any():
                            # NaN in trajectory
                            context_data.append(
                                float("nan") * torch.ones(10)
                            )
                            continue

                        # Subsample every 21st time point (0, 21, 42, 63, 84)
                        # This gives 5 time points
                        u_sub = u[:, ::21]  # (2, 5)

                        # Flatten to (10,) - [prey_t0, prey_t1, ..., predator_t0, ...]
                        u_flat = u_sub.flatten()

                        # Scale by total_count to get Poisson rates
                        rates = u_flat / self.total_count
                        rates = rates.clamp(min=1e-10)  # Ensure positive rates

                        # Sample from Poisson distribution (no pyro)
                        poisson_dist = torch.distributions.Poisson(rates)
                        obs = poisson_dist.sample()

                        context_data.append(obs)

                    except Exception:
                        # ODE solver failed, return NaN
                        context_data.append(
                            float("nan") * torch.ones(10)
                        )

                # Concatenate all contexts
                data.append(torch.cat(context_data))

            return torch.stack(data).float()

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _likelihood(
        self, parameters: torch.Tensor, data: torch.Tensor, log: bool = True
    ) -> torch.Tensor:
        """Compute likelihood of data given parameters

        Uses Poisson likelihood for count observations. The likelihood is
        naturally bounded since Poisson log-likelihood is always finite.

        Args:
            parameters: Parameter tensor with shape (num_samples, dim_parameters)
            data: Observation tensor with shape (num_samples, dim_data)
            log: If True, return log-likelihood; otherwise return likelihood

        Returns:
            (Log-)likelihood values with shape (num_samples,)
        """
        num_samples = parameters.shape[0]

        # Generate time points for ODE integration
        t = torch.linspace(0, self.days, int(self.days / self.saveat) + 1)

        # Split parameters into global and local
        global_params = parameters[:, :2]  # (num_samples, 2): [beta, gamma]
        local_params = parameters[:, 2:]  # (num_samples, 2*n_l)

        log_likelihoods = []

        for b in range(num_samples):
            log_lik_sample = 0.0

            # For each local context (site)
            for i in range(self.n_l):
                # Extract local parameters for this site
                alpha_i = local_params[b, 2 * i]
                delta_i = local_params[b, 2 * i + 1]
                beta = global_params[b, 0]
                gamma = global_params[b, 1]

                # Combine into full parameter vector [alpha, beta, gamma, delta]
                self._current_params = torch.tensor(
                    [alpha_i, beta, gamma, delta_i]
                )

                # Solve ODE
                try:
                    u_trajectory = odeint(
                        self._lotka_volterra_ode, self.u0, t, method="dopri5"
                    )
                    u = u_trajectory.T  # (2, time_steps)

                    # Check for valid trajectory
                    if u.shape != torch.Size([2, int(self.days / self.saveat) + 1]):
                        log_lik_sample = float("-inf")
                        break

                    if torch.isnan(u).any():
                        log_lik_sample = float("-inf")
                        break

                    # Subsample every 21st time point
                    u_sub = u[:, ::21]  # (2, 5)
                    u_flat = u_sub.flatten()  # (10,)

                    # Scale by total_count to get Poisson rates
                    rates = u_flat / self.total_count
                    rates = rates.clamp(min=1e-10)

                    # Get observed data for this context
                    obs = data[b, i * 10 : (i + 1) * 10]

                    # Compute Poisson log-likelihood
                    poisson_dist = torch.distributions.Poisson(rates)
                    log_lik_context = poisson_dist.log_prob(obs).sum()

                    log_lik_sample += log_lik_context

                except Exception:
                    # ODE solver failed
                    log_lik_sample = float("-inf")
                    break

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

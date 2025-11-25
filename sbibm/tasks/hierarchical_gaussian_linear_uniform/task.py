from pathlib import Path
from typing import Any, Optional

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


class HierarchicalGaussianLinearUniform(Task):
    def __init__(
        self,
        n_l: int = 5,
        dim: int = 26,
        prior_bound: float = 10.0,
        simulator_scale: float = 0.1,
    ):
        """Hierarchical Gaussian Linear Uniform

        Hierarchical extension of the Gaussian Linear Uniform task where
        each observation consists of n_l local contexts. Uses Strategy 1
        (natural global/local split): global parameters represent the shared
        noise scale, while local parameters represent context-specific means
        bounded by a uniform prior.
        This follows standard Bayesian regression where variance/scale is
        pooled globally and means/intercepts are estimated locally per group
        with bounded support.

        Total parameter space is divided as:
            - 1 global noise scale parameter
            - (dim - 1) / n_l local mean dimensions per context
            - (dim - 1) total local mean parameters across all n_l contexts

        Global parameters (dim_global=1):
            - Noise scale shared across all contexts
            - Prior: HalfNormal(simulator_scale)

        Local parameters (dim_local_per_context per context,
        dim_local_per_context * n_l total):
            - Context-specific mean structure per context
            - Prior: Uniform[-prior_bound, +prior_bound] for each dimension

        Args:
            n_l: Number of local contexts (default: 5)
            dim: Total dimensionality of parameter space (default: 26,
                which with default n_l=5 gives 1 + 5*5 = 26)
            prior_bound: Bound for uniform prior on local means
                (default: 10.0)
            simulator_scale: Scale parameter for HalfNormal prior on global
                noise scale (default: 0.1)

        Raises:
            ValueError: If (dim - 1) does not divide evenly by n_l
        """
        self.n_l = n_l
        self.prior_bound = prior_bound
        self.simulator_scale = simulator_scale

        dim = n_l + 1

        dim_global = 1
        dim_local_per_context = (dim - 1) // n_l
        dim_local_total = dim - 1

        self.dim_global = dim_global
        self.dim_local_per_context = dim_local_per_context
        self.dim_local_total = dim_local_total

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
            dim_parameters=dim,
            dim_data=dim_local_total,
            name="hierarchical_gaussian_linear_uniform",
            name_display="Hierarchical Gaussian Linear Uniform",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        # Define hierarchical prior distribution
        # Global parameters: shared noise scale (dim_global=1)
        global_dist = pdist.Independent(
            pdist.HalfNormal(simulator_scale).expand([1]), 1
        )

        # Local parameters: context-specific means bounded by uniform prior
        # (dim_local_total total across all contexts)
        def local_dist_fn(global_params, n_local_arg):
            # Return independent Uniform distributions for all local means.
            # dim_local_total = dim_local_per_context * n_local_arg
            # Independent of global_params
            batch_shape = global_params.shape[:-1]
            dim_local = dim_local_per_context * n_local_arg
            return pdist.Independent(
                pdist.Uniform(
                    low=-prior_bound * torch.ones(dim_local),
                    high=+prior_bound * torch.ones(dim_local),
                ).expand(list(batch_shape) + [dim_local]),
                1,
            )

        self.prior_dist = HierarchicalDistribution(
            global_dist,
            local_dist_fn,
            dim_global=dim_global,
            dim_local=dim_local_per_context,
            n_local=n_l,
        )
        self.prior_dist.set_default_validate_args(False)

        # Store parameters for dynamic transform building
        self.dim_local_per_context = dim_local_per_context
        self.prior_bound = prior_bound

    def get_prior(self):
        """Get prior distribution.

        Returns a callable that samples from self.prior_dist using pyro.
        """

        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get simulator function.

        For each local context, generates observations from a Gaussian
        distribution with context-specific mean and global noise scale.

        Args:
            max_calls: Maximum number of simulator calls

        Returns:
            Simulator callable
        """

        def simulator(parameters):
            # Split parameters into global and local
            # Global: [:, :dim_global] (shared noise scales)
            # Local: [:, dim_global:] (dim_local_per_context means per context)
            global_scale = parameters[:, : self.dim_global]  # noqa: E203
            local_means = parameters[:, self.dim_global :]  # noqa: E203
            n_l = local_means.shape[1]

            # For each local context, sample observations
            observations = []
            for i in range(n_l):
                # Extract mean for context i
                start_idx = i * self.dim_local_per_context
                end_idx = (i + 1) * self.dim_local_per_context
                mean_i = local_means[:, start_idx:end_idx]  # noqa: E203

                # Sample observations: Normal(mean_i, global_scale * I)
                # Broadcast global_scale across all dimensions
                obs_dist = pdist.Normal(
                    loc=mean_i,
                    scale=global_scale,
                )
                obs_i = obs_dist.sample()

                observations.append(obs_i)

            # Concatenate observations from all contexts
            observations = torch.cat(observations, dim=1)

            return observations

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_prior_dist(self):
        """Get prior distribution object.

        Returns:
            Prior distribution
        """
        return self.prior_dist

    def _get_transforms(
        self, automatic_transforms_enabled: bool = True, n_l=None, **kwargs: Any
    ):
        """Get transforms for converting between constrained and
        unconstrained space.

        Args:
            automatic_transforms_enabled: Whether to return transforms
            n_l: Number of local contexts (uses self.n_l if None)

        Returns:
            Dictionary with 'parameters' key containing the transform
        """
        if n_l is None:
            n_l = self.n_l

        # Build composite transform (constrained <-> unconstrained)
        transforms_list = []

        # global_scale: HalfNormal (R+) <-> R
        transforms_list.append(biject_to(constraints.positive))

        # local_means: Uniform (bounded) - use interval transform
        # dim_local_per_context means per context, n_l contexts
        for _ in range(self.dim_local_per_context * n_l):
            transforms_list.append(
                biject_to(constraints.interval(-self.prior_bound, +self.prior_bound))
            )

        # Use custom wrapper to ensure Jacobian is properly summed
        composite_transform = SummedStackTransform(transforms_list, dim=-1)
        return {"parameters": composite_transform.inv}

    def _likelihood(
        self,
        parameters: torch.Tensor,
        data: torch.Tensor,
        log: bool = True,
    ) -> torch.Tensor:
        """Compute likelihood of data given parameters.

        For hierarchical gaussian linear uniform, the likelihood is the
        product of independent Gaussian likelihoods for each local
        context.

        Args:
            parameters: Parameter tensor (batch_size, dim_parameters)
            data: Observation tensor (batch_size, dim_data)
            log: If True, return log-likelihood

        Returns:
            (Log-)likelihood values
        """
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, -1)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        assert parameters.shape[1] == self.dim_parameters
        assert data.shape[1] == self.dim_data

        batch_size = parameters.shape[0]

        # Split parameters: global scale [:, :dim_global],
        # local means [:, dim_global:]
        global_scale = parameters[:, : self.dim_global]  # noqa: E203
        local_means = parameters[:, self.dim_global :]  # noqa: E203

        # Split data into n_l contexts (each dim_local_per_context
        # observations)
        data_split = data.reshape(batch_size, self.n_l, self.dim_local_per_context)

        # Compute likelihood for each context
        log_likelihoods = []
        for i in range(self.n_l):
            # Extract data and mean for context i
            context_data = data_split[:, i, :]
            start_idx = i * self.dim_local_per_context
            end_idx = (i + 1) * self.dim_local_per_context
            mean_i = local_means[:, start_idx:end_idx]  # noqa: E203

            # Compute Gaussian log-likelihood:
            # N(x | mean_i, global_scale^2 * I)
            dist = pdist.Normal(
                loc=mean_i,
                scale=global_scale.expand(-1, self.dim_local_per_context),
            )
            log_lik_context = dist.log_prob(context_data).sum(dim=1)

            log_likelihoods.append(log_lik_context)

        # Sum log-likelihoods across contexts (product of likelihoods)
        total_log_likelihood = torch.stack(log_likelihoods, dim=0).sum(dim=0)

        return total_log_likelihood if log else torch.exp(total_log_likelihood)

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup hierarchical gaussian linear uniform task"
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=5,
        help="Number of local contexts",
    )
    args = parser.parse_args()

    task = HierarchicalGaussianLinearUniform(n_l=args.n_l)
    task._setup(n_jobs=1, create_reference=False)

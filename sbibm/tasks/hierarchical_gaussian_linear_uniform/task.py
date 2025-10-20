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
        dim: int = 10,
        prior_bound: float = 10.0,
        simulator_scale: float = 0.1,
    ):
        """Hierarchical Gaussian Linear Uniform

        Hierarchical extension of the Gaussian Linear Uniform task where
        each observation consists of n_l local contexts. Uses Strategy 1
        (natural global/local split): global parameters represent the shared
        mean structure with uniform prior, while local parameters represent
        context-specific noise scales.

        Global parameters (dim=10):
            - Mean structure shared across all contexts
            - Prior: Uniform(-prior_bound, +prior_bound) for each dimension

        Local parameters (dim=n_l):
            - Noise scale per context
            - Prior: HalfNormal(simulator_scale) for each context

        Args:
            n_l: Number of local contexts (default: 5)
            dim: Dimensionality of global mean parameters (default: 10)
            prior_bound: Bound for uniform prior on global mean
                (default: 10.0)
            simulator_scale: Scale parameter for HalfNormal prior on local
                noise scales (default: 0.1)
        """
        self.n_l = n_l
        self.dim = dim
        self.prior_bound = prior_bound
        self.simulator_scale = simulator_scale

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
            dim_parameters=dim + n_l,  # dim global + n_l local
            dim_data=dim * n_l,  # dim observations per context
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
        # Global parameters: mean structure (dim=10) with uniform prior
        global_dist = pdist.Independent(
            pdist.Uniform(
                low=-prior_bound * torch.ones(dim),
                high=+prior_bound * torch.ones(dim),
            ),
            1,
        )

        # Local parameters: noise scales (dim=n_l), independent of global
        def local_dist_fn(global_params):
            # Return HalfNormal distribution for n_l noise scales
            # Independent of global_params
            batch_shape = global_params.shape[:-1]
            return pdist.Independent(
                pdist.HalfNormal(simulator_scale).expand(
                    list(batch_shape) + [n_l]
                ),
                1,
            )

        self.prior_dist = HierarchicalDistribution(
            global_dist, local_dist_fn, dim_global=dim, dim_local=n_l
        )
        self.prior_dist.set_default_validate_args(False)

        # Build composite transform (constrained <-> unconstrained)
        transforms_list = []

        # global_mean: Uniform (bounded) - use biject_to(interval)
        for _ in range(dim):
            transforms_list.append(
                biject_to(
                    constraints.interval(-prior_bound, +prior_bound)
                )
            )

        # local_scales: HalfNormal (R+) <-> R
        for _ in range(n_l):
            transforms_list.append(biject_to(constraints.positive))

        # Use custom wrapper to ensure Jacobian is properly summed
        self.composite_transform = SummedStackTransform(
            transforms_list, dim=-1
        )

    def get_prior(self):
        """Get prior distribution.

        Returns a callable that samples from self.prior_dist using pyro.
        """

        def prior(num_samples=1):
            return pyro.sample(
                "parameters", self.prior_dist.expand_by([num_samples])
            )

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get simulator function.

        For each local context, generates observations from a Gaussian
        distribution with the global mean and context-specific noise scale.

        Args:
            max_calls: Maximum number of simulator calls

        Returns:
            Simulator callable
        """

        def simulator(parameters):
            # Split parameters into global and local
            # Global: [:, :dim] (mean structure)
            # Local: [:, dim:] (n_l noise scales)
            global_mean = parameters[:, : self.dim]  # noqa: E203
            local_scales = parameters[:, self.dim :]  # noqa: E203

            # For each local context, sample observations
            observations = []
            for i in range(self.n_l):
                # Extract noise scale for context i
                scale_i = local_scales[:, i : i + 1]  # noqa: E203

                # Sample observations: Normal(global_mean, scale_i * I)
                # Broadcast scale_i across all dimensions
                obs_dist = pdist.Normal(
                    loc=global_mean, scale=scale_i.expand(-1, self.dim)
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
        self, automatic_transforms_enabled: bool = True, **kwargs: Any
    ):
        """Get transforms for converting between constrained and
        unconstrained space.

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
    task = HierarchicalGaussianLinearUniform(n_l=5)
    task._setup(n_jobs=4, create_reference=False)

from pathlib import Path
from typing import Any, Optional

import pyro
import torch
from pyro import distributions as pdist
from pyro.distributions import constraints
from pyro.distributions.transforms import biject_to

import sbibm
from sbibm.tasks.distributions import (
    BlockwiseDistribution,
    HierarchicalDistribution,
    SummedStackTransform,
)
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class HierarchicalSLCP(Task):
    def __init__(self, n_l: int = 5):
        """Hierarchical SLCP

        Hierarchical extension of the SLCP task where each observation
        consists of n_l local contexts. Uses Strategy 1 (natural
        global/local split): covariance structure is global (shared),
        while means are local (context-specific).

        Global parameters (3 total):
            - s1, s2: Standard deviations (Uniform(0.5, 3.0))
            - rho: Correlation (Uniform(-3, 3))

        Local parameters (2 * n_l total):
            - For each context i: m0_i, m1_i ~ Uniform(-3, 3)

        Args:
            n_l: Number of local contexts (default: 5)
        """
        self.n_l = n_l
        self.num_data = 4  # 4 observations per context (as in original SLCP)

        # Observation seeds
        observation_seeds = [
            1000000,
            1000001,
            1000002,
            1000003,
            1000004,
            1000005,
            1000010,
            1000012,
            1000008,
            1000009,
        ]

        super().__init__(
            dim_parameters=3 + 2 * n_l,  # 3 global + 2*n_l local
            dim_data=8 * n_l,  # 8D observation per context (4 obs * 2D)
            name="hierarchical_slcp",
            name_display="Hierarchical SLCP",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        # Define hierarchical prior distribution
        # Global parameters: [s1, s2, rho]
        # s1, s2 ~ Uniform(0.5, 3.0), rho ~ Uniform(-3, 3)
        s_dist = pdist.Uniform(0.5, 3.0).expand([2]).to_event(1)
        rho_dist = pdist.Uniform(-3.0, 3.0).expand([1]).to_event(1)
        global_dist = BlockwiseDistribution([s_dist, rho_dist])

        # Local params distribution: means for each context
        def local_dist_fn(global_params, n_local_arg):
            # Each context has 2 means (m0, m1)
            # local_params: 2*n_local_arg dims, all ~ Uniform(-3, 3)
            batch_shape = global_params.shape[:-1]
            local_dist = (
                pdist.Uniform(-3.0, 3.0)
                .expand(list(batch_shape) + [2 * n_local_arg])
                .to_event(1)
            )
            return local_dist

        self.prior_dist = HierarchicalDistribution(
            global_dist,
            local_dist_fn,
            dim_global=3,
            dim_local=2,
            n_local=n_l,
        )
        self.prior_dist.set_default_validate_args(False)

        # Transforms will be built dynamically in _get_transforms

    def get_prior(self):
        """Get prior distribution.

        Returns a callable that samples from self.prior_dist using pyro.
        """

        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get simulator function.

        For each local context, generates observations from a bivariate
        normal with global covariance structure and local means.

        Args:
            max_calls: Maximum number of simulator calls

        Returns:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            # Split parameters into global and local
            # Global: [:, 0:3] (s1, s2, rho)
            # Local: [:, 3:] (2*n_l means)
            global_params = parameters[:, :3]
            local_params = parameters[:, 3:].reshape(num_samples, -1, 2)
            n_l = local_params.shape[1]

            # Extract global covariance parameters
            s1 = global_params[:, 0]
            s2 = global_params[:, 1]
            rho_param = global_params[:, 2]

            # Transform rho to correlation via tanh
            rho = torch.nn.Tanh()(rho_param)

            # Build covariance matrix for all samples
            S = torch.empty((num_samples, 2, 2))
            S[:, 0, 0] = s1**2
            S[:, 0, 1] = rho * s1 * s2
            S[:, 1, 0] = rho * s1 * s2
            S[:, 1, 1] = s2**2

            # Add eps to diagonal to ensure PSD
            eps = 0.000001
            S[:, 0, 0] += eps
            S[:, 1, 1] += eps

            # Vectorized sampling: create distribution for all contexts at
            # once Shape: (num_samples, n_l, num_data, 2)
            # Expand means and covariance for all contexts and observations
            m_expanded = (
                local_params.unsqueeze(2)
                .expand(num_samples, n_l, self.num_data, 2)
                .float()
            )
            S_expanded = (
                S.unsqueeze(1)
                .unsqueeze(2)
                .expand(num_samples, n_l, self.num_data, 2, 2)
                .float()
            )

            # Create batched MVN distribution
            # batch_shape: (num_samples, n_l, num_data)
            data_dist = pdist.MultivariateNormal(m_expanded, S_expanded)

            # Sample all at once (no loop, no pyro.sample)
            observations = data_dist.sample()

            # Reshape from (num_samples, n_l, num_data, 2) to
            # (num_samples, n_l * num_data * 2)
            observations = observations.reshape(num_samples, -1)

            return observations

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

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

        # Build composite transform for MCMC
        transforms_list = []

        # s1, s2: Uniform[0.5, 3.0] <-> R
        for _ in range(2):
            transforms_list.append(biject_to(constraints.interval(0.5, 3.0)))

        # rho: Uniform[-3, 3] <-> R
        transforms_list.append(biject_to(constraints.interval(-3.0, 3.0)))

        # Local means: Uniform[-3, 3] <-> R
        for _ in range(2 * n_l):
            transforms_list.append(biject_to(constraints.interval(-3.0, 3.0)))

        composite_transform = SummedStackTransform(transforms_list, dim=-1)
        return {"parameters": composite_transform.inv}

    def _likelihood(
        self,
        parameters: torch.Tensor,
        data: torch.Tensor,
        log: bool = True,
    ) -> torch.Tensor:
        """Compute likelihood of data given parameters.

        For hierarchical SLCP, the likelihood is the product of
        independent likelihoods for each local context.

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

        # Split parameters: global [:, 0:3], local [:, 3:]
        global_params = parameters[:, :3]
        local_params = parameters[:, 3:].reshape(batch_size, self.n_l, 2)

        # Extract global covariance parameters
        s1 = global_params[:, 0]
        s2 = global_params[:, 1]
        rho_param = global_params[:, 2]
        rho = torch.nn.Tanh()(rho_param)

        # Build covariance matrix
        S = torch.empty((batch_size, 2, 2))
        S[:, 0, 0] = s1**2
        S[:, 0, 1] = rho * s1 * s2
        S[:, 1, 0] = rho * s1 * s2
        S[:, 1, 1] = s2**2

        # Add eps to diagonal to ensure PSD
        eps = 0.000001
        S[:, 0, 0] += eps
        S[:, 1, 1] += eps

        # Split data into (batch_size, n_l, num_data, 2)
        data_split = data.reshape(batch_size, self.n_l, self.num_data, 2)

        # Vectorized likelihood computation
        # Expand means and covariance for all contexts
        m_expanded = (
            local_params.unsqueeze(2)
            .expand(batch_size, self.n_l, self.num_data, 2)
            .float()
        )
        S_expanded = (
            S.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, self.n_l, self.num_data, 2, 2)
            .float()
        )

        # Create batched MVN distribution
        dist = pdist.MultivariateNormal(m_expanded, S_expanded)

        # Compute log likelihood for all contexts and observations
        # Shape: (batch_size, n_l, num_data)
        log_probs = dist.log_prob(data_split.float())

        # Sum over observations and contexts
        total_log_likelihood = log_probs.sum(dim=[1, 2])

        return total_log_likelihood if log else torch.exp(total_log_likelihood)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior - not implemented for hierarchical.

        Hierarchical tasks do not provide reference posteriors due to
        computational cost. Use reference-free metrics (reverse KL, LC2ST)
        instead.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "Reference posteriors are not available for hierarchical tasks. "
            "Use reference-free metrics such as reverse KL or LC2ST instead."
        )


if __name__ == "__main__":
    task = HierarchicalSLCP(n_l=5)
    task._setup(n_jobs=4, create_reference=False)

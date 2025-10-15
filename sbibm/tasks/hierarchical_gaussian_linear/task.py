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


class HierarchicalGaussianLinear(Task):
    def __init__(
        self,
        n_l: int = 5,
        prior_scale: float = 1.0,
        simulator_scale: float = 1.0,
        dim_local_per_context: int = 1,
        device: str = 'cpu'
    ):
        """Hierarchical Gaussian Linear

        Hierarchical extension of the Gaussian Linear task where each
        observation consists of n_l local contexts. Uses Strategy 1
        (natural global/local split): global parameters represent the shared
        noise scale, while local parameters represent context-specific means.
        This follows standard Bayesian regression where variance/scale is
        pooled globally and means/intercepts are estimated locally per group.

        Parameters
        ----------
        The model has (1 + n_l * dim_local_per_context) parameters split into:

        **Global parameters** (dim=1):
            σ : scalar, σ > 0
                Shared noise scale across all contexts
                Prior: σ ~ HalfNormal(simulator_scale)

        **Local parameters** (dim=n_l * dim_local_per_context):
            μ_i : vector of length dim_local_per_context, μ_i ∈ ℝ (for each context i=1,...,n_l)
                Context-specific mean parameters
                Prior: μ_i ~ N(0, prior_scale²) independently for each context

        Parameter Layout
        ----------------
        Parameters are concatenated as: θ = [σ, μ_1, μ_2, ..., μ_{n_l}]

        For default dim_local_per_context=1 and n_l contexts:
        - θ[0]: Global noise scale σ
        - θ[1]: Local mean μ_1 for context 1
        - θ[2]: Local mean μ_2 for context 2
        - ...
        - θ[n_l]: Local mean μ_{n_l} for context n_l

        For dim_local_per_context > 1:
        - θ[0]: Global noise scale σ
        - θ[1:1+dim_local_per_context]: Local means μ_1 for context 1
        - θ[1+dim_local_per_context:1+2*dim_local_per_context]: Local means μ_2 for context 2
        - ...

        Simulator
        ---------
        For each local context i, the model generates observations according to:

            x_i ~ N(μ_i, σ² I_{dim_local_per_context})

        Where:
        - x_i ∈ ℝ^{dim_local_per_context} are observations from context i
        - μ_i ∈ ℝ^{dim_local_per_context} is the context-specific mean vector
        - σ > 0 is the shared noise scale (global parameter)
        - I_{dim_local_per_context} is the identity matrix of size dim_local_per_context

        The simulator loops over n_l contexts, generating independent observations
        from Gaussian distributions with context-specific means and shared noise scale.

        Likelihood
        ----------
        The likelihood factorizes across independent contexts:

            p(x | θ) = ∏_{i=1}^{n_l} N(x_i | μ_i, σ² I)

        Where x = [x_1, ..., x_{n_l}] represents observations from all contexts, and
        each context's observations are independent given the parameters.

        Args
        ----
        n_l : int, default=5
            Number of local contexts
        prior_scale : float, default=1.0
            Standard deviation of Normal prior on local means
        simulator_scale : float, default=1.0
            Scale parameter for HalfNormal prior on global noise scale
        dim_local_per_context : int, default=1
            Number of observations per local context

        Notes
        -----
        This model demonstrates Strategy 1 hierarchical modeling where there is a
        natural split between global (pooled) and local (context-specific) parameters
        based on the underlying statistical structure.

        The global noise scale σ is shared across all contexts, implementing partial
        pooling that borrows strength across groups while allowing context-specific
        means. This is equivalent to a random effects model in classical statistics.

        See Also
        --------
        hierarchical_gaussian_linear_uniform : Variant with bounded support on local means
        """
        self.n_l = n_l
        self.prior_scale = prior_scale
        self.simulator_scale = simulator_scale

        # Calculate dimensions: 1 global scale + n_l * dim_local_per_context locals
        dim = n_l * dim_local_per_context + 1

        dim_global = 1
        dim_local_total = dim - 1

        self.dim_global = dim_global
        self.dim_local_per_context = dim_local_per_context
        self.dim_local_total = dim_local_total

        # Observation seeds
        observation_seeds = [
            2000001,
            2000002,
            2000003,
            2000004,
            2000005,
            2000006,
            2000007,
            2000008,
            2000009,
            2000010,
        ]

        super().__init__(
            dim_parameters=dim,
            dim_data=dim_local_total,
            name="hierarchical_gaussian_linear",
            name_display="Hierarchical Gaussian Linear",
            num_observations=10,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
            generate_in_memory=True,
        )

        # Define hierarchical prior distribution
        # Global parameters: shared noise scale (dim_global=1)
        # Expand([1]) to get batch_shape=[], event_shape=[1], then use expand_by
        # to ensure proper 2D sampling
        global_dist = pdist.Independent(
            pdist.HalfNormal(
                torch.tensor(simulator_scale).to(device=device),
                validate_args=False
            ).expand([1]), 1
        )

        # Local parameters: context-specific means
        # Returns flat distribution over all local params across n_local groups
        def local_dist_fn(global_params, n_local):
            # Return independent Normal distributions for all local means.
            # dim_local_per_context dimensions per context
            # Total: n_local * dim_local_per_context
            batch_shape = global_params.shape[:-1]
            total_local_dim = n_local * dim_local_per_context
            return pdist.Independent(
                pdist.Normal(
                    loc=torch.zeros(total_local_dim).to(device=device),
                    scale=prior_scale * torch.ones(total_local_dim).to(device=device),
                ).expand(list(batch_shape) + [total_local_dim]),
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

            # Infer n_local from parameter shape
            # local_means has shape (batch_size, n_local * dim_local_per_context)
            n_local_total = local_means.shape[1]
            n_local = n_local_total // self.dim_local_per_context

            # For each local context, sample observations
            observations = []
            for i in range(n_local):
                # Extract mean for context i
                start_idx = i * self.dim_local_per_context
                end_idx = (i + 1) * self.dim_local_per_context
                mean_i = local_means[:, start_idx:end_idx]  # noqa: E203

                # Sample observations: Normal(mean_i, global_scale * I)
                # Broadcast global_scale across all dimensions
                obs_dist = pdist.Normal(
                    loc=mean_i,
                    scale=global_scale.expand(-1, self.dim_local_per_context),
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
        self,
        automatic_transforms_enabled: bool = True,
        n_l: Optional[int] = None,
        **kwargs: Any,
    ):
        """Get transforms for converting between constrained and unconstrained space.

        Args:
            automatic_transforms_enabled: Whether to return transforms
            n_l: Number of local contexts (defaults to self.n_l)

        Returns:
            Dictionary with 'parameters' key containing the transform
        """
        if n_l is None:
            n_l = self.n_l

        # Build composite transform (constrained <-> unconstrained)
        transforms_list = []

        # global_scale: HalfNormal (R+) <-> R
        transforms_list.append(biject_to(constraints.positive))

        # local_means: Normal (unbounded) - use identity transform
        # Total local dimensions: n_l * dim_local_per_context
        total_local_dim = n_l * self.dim_local_per_context
        for _ in range(total_local_dim):
            transforms_list.append(torch.distributions.transforms.identity_transform)

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

        For hierarchical gaussian linear, the likelihood is the product
        of independent Gaussian likelihoods for each local context.

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

        # Split data into n_l contexts (each dim_local_per_context observations)
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
        description="Setup hierarchical gaussian linear task"
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=5,
        help="Number of local contexts",
    )
    args = parser.parse_args()

    task = HierarchicalGaussianLinear(n_l=args.n_l)
    task._setup(n_jobs=1, create_reference=False)

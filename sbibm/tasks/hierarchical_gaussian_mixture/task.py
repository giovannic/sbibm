from pathlib import Path
from typing import Optional

import pyro
import torch
from pyro import distributions as pdist
from pyro.distributions import constraints
from pyro.distributions.transforms import biject_to

from sbibm.tasks.distributions import (
    BlockwiseDistribution,
    HierarchicalDistribution,
    SummedStackTransform,
    TruncatedNormal,
)
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class HierarchicalGaussianMixture(Task):
    def __init__(
        self,
        n_l: int = 5,
        dim: int = 1,
        prior_bound: float = 10.0,
        device: str = 'cpu'
    ):
        """Hierarchical Gaussian Mixture

        Hierarchical extension of the Gaussian Mixture task where each
        observation consists of n_l local contexts. Uses Strategy 2 (full
        local with pooling): all original parameters become local, with
        global hyperparameters controlling their hierarchical distribution.

        The simulator uses a two-component Gaussian mixture model where local
        parameters determine the mixture component means.

        Parameters
        ----------
        The model has (2*dim + dim*n_l) parameters split into:

        **Global parameters** (dim=2*dim):
            μ_global : vector of length dim, μ_global ∈ [-prior_bound, prior_bound]
                Global location hyperparameters (one per dimension)
                Prior: μ_global ~ U(-prior_bound, prior_bound)
            σ_global : vector of length dim, σ_global > 0
                Global scale hyperparameters (one per dimension)
                Prior: σ_global ~ HalfNormal(1.0)

        **Local parameters** (dim=dim*n_l):
            θ_i : vector of length dim, θ_i ∈ [-prior_bound, prior_bound] (for each context i=1,...,n_l)
                Context-specific parameters that determine mixture locations
                Prior: θ_i ~ TN(μ_global, σ_global, -prior_bound, prior_bound) | μ_global, σ_global

        Parameter Layout
        ----------------
        Parameters are concatenated as: θ = [μ_global, σ_global, θ_1, θ_2, ..., θ_{n_l}]

        For dim=1 and n_l contexts:
        - θ[0:1]: Global location μ_global
        - θ[1:2]: Global scale σ_global
        - θ[2]: Local parameter θ_1 for context 1
        - θ[3]: Local parameter θ_2 for context 2
        - ...
        - θ[2+n_l-1]: Local parameter θ_{n_l} for context n_l

        For general dim and n_l contexts:
        - θ[0:dim]: Global locations μ_global
        - θ[dim:2*dim]: Global scales σ_global
        - θ[2*dim:2*dim+dim]: Local parameters θ_1 for context 1
        - θ[2*dim+dim:2*dim+2*dim]: Local parameters θ_2 for context 2
        - ...
        - θ[2*dim+(n_l-1)*dim:2*dim+n_l*dim]: Local parameters θ_{n_l} for context n_l

        Simulator
        ---------
        For each local context i, the model generates observations from a two-component
        Gaussian mixture:

            k ~ Categorical([0.5, 0.5])
            x_i ~ N(factor[k] · θ_i, scale[k] · I_dim)

        Where:
        - k ∈ {0, 1} is the mixture component indicator
        - factor = [1.0, 1.0] (both components use θ_i directly as mean)
        - scale = [1.0, 0.1] (high-variance and low-variance components)
        - x_i ∈ ℝ^dim is the observation from context i

        Equivalently, the marginal observation distribution is:

            x_i ~ 0.5 · N(θ_i, 1.0 · I_dim) + 0.5 · N(θ_i, 0.01 · I_dim)

        Likelihood
        ----------
        The likelihood factorizes across independent contexts and marginalizes
        over mixture components:

            p(x | θ) = ∏_{i=1}^{n_l} [0.5 · N(x_i | θ_i, 1.0 · I) + 0.5 · N(x_i | θ_i, 0.01 · I)]

        Where x = [x_1, ..., x_{n_l}] represents observations from all contexts.

        Args
        ----
        n_l : int, default=5
            Number of local contexts
        dim : int, default=1
            Dimensionality of parameters and data per context
        prior_bound : float, default=10.0
            Bound for location parameters (support is [-prior_bound, prior_bound])

        Notes
        -----
        This model demonstrates Strategy 2 hierarchical modeling where the original
        task parameters become local (context-specific), and new global hyperparameters
        pool information across contexts via a hierarchical prior structure.

        The local parameters θ_i are drawn from a truncated normal distribution
        centered at global location μ_global with global scale σ_global, creating
        partial pooling across contexts. The truncation bounds ensure parameters
        remain in a reasonable support region.

        The two-component mixture with vastly different scales (1.0 vs 0.1) creates
        a challenging bimodal likelihood structure, making this task particularly
        difficult for inference methods.

        See Also
        --------
        gaussian_mixture : The non-hierarchical version
        """
        self.n_l = n_l
        self.dim = dim
        self.prior_bound = torch.tensor(prior_bound).to(device=device)

        # Observation seeds (same as original task)
        observation_seeds = [
            1000011,
            1000001,
            1000002,
            1000003,
            1000013,
            1000005,
            1000006,
            1000007,
            1000008,
            1000009,
        ]

        super().__init__(
            dim_parameters=2 * dim + dim * n_l,  # 2*dim global + dim*n_l local
            dim_data=dim * n_l,  # dim-D observation per context
            name="hierarchical_gaussian_mixture",
            name_display="Hierarchical Gaussian Mixture",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
            generate_in_memory=True,
        )

        # Store simulator params from original gaussian_mixture
        self.simulator_params = {
            "mixture_locs_factor": torch.tensor([1.0, 1.0]).to(device=device),
            "mixture_scales": torch.tensor([1.0, 0.1]).to(device=device),
            "mixture_weights": torch.tensor([0.5, 0.5]).to(device=device),
        }

        # Define hierarchical prior distribution
        # Global parameters: [loc_0, ..., loc_{dim-1}, scale_0, ...,
        # scale_{dim-1}]
        global_loc_dist = (
            pdist.Uniform(
                -self.prior_bound,
                self.prior_bound,
                validate_args=False
            ).expand([dim]).to_event(1)
        )
        global_scale_dist = pdist.HalfNormal(
            torch.tensor(1.0).to(device=device),
            validate_args=False
        ).expand([dim]).to_event(1)
        global_dist = BlockwiseDistribution([global_loc_dist, global_scale_dist])

        # Local params distribution conditioned on global
        def local_dist_fn(global_params, n_local_arg):
            # global_params shape: [..., 2*dim]
            # Extract locs and scales
            locs = global_params[..., :dim]  # [..., dim]
            scales = global_params[..., dim : 2 * dim]  # noqa: E203 [..., dim]

            # Create distribution for all local params (dim*n_local_arg dims)
            # Each local param (dim-D) is TruncatedNormal with bounded support
            # Replicate locs and scales for n_local_arg contexts
            batch_shape = global_params.shape[:-1]
            locs_expanded = (
                locs.unsqueeze(-2)
                .expand(list(batch_shape) + [n_local_arg, dim])
                .reshape(list(batch_shape) + [dim * n_local_arg])
            )
            scales_expanded = (
                scales.unsqueeze(-2)
                .expand(list(batch_shape) + [n_local_arg, dim])
                .reshape(list(batch_shape) + [dim * n_local_arg])
            )

            # Use TruncatedNormal with bounded support [-prior_bound, prior_bound]
            return pdist.Independent(
                TruncatedNormal(
                    loc=locs_expanded,
                    scale=scales_expanded,
                    low=-prior_bound,
                    high=+prior_bound,
                    validate_args=False
                ),
                1,
            )

        self.prior_dist = HierarchicalDistribution(
            global_dist,
            local_dist_fn,
            dim_global=2 * dim,
            dim_local=dim,
            n_local=n_l,
        )

        # Store parameters for dynamic transform building
        self.dim = dim
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

        For each local context, applies the original gaussian_mixture
        simulator to the corresponding local parameters.

        Args:
            max_calls: Maximum number of simulator calls

        Returns:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            # Split parameters into global and local
            # Global: [:, 0:2*dim] (dim locs + dim scales)
            # Local: [:, 2*dim:] (dim*n_l parameters)
            local_params = parameters[:, 2 * self.dim :]  # noqa: E203
            local_params = local_params.reshape(num_samples, -1, self.dim)
            n_l = local_params.shape[1]

            # Sample mixture indices for all contexts at once
            # Shape: (num_samples, n_l)
            idx = torch.distributions.Categorical(
                probs=self.simulator_params["mixture_weights"]
            ).sample((num_samples, n_l))

            # Expand for broadcasting: (num_samples, n_l, 1)
            idx_expanded = idx.unsqueeze(-1)

            # Select loc and scale factors based on mixture component
            # Shape: (num_samples, n_l, dim)
            loc = (
                self.simulator_params["mixture_locs_factor"][idx_expanded]
                * local_params
            )
            scale = self.simulator_params["mixture_scales"][idx_expanded]

            # Sample observations from Normal distribution
            # Shape: (num_samples, n_l, dim)
            observations = torch.distributions.Normal(loc, scale).sample()

            # Flatten to (num_samples, n_l * dim)
            observations = observations.reshape(num_samples, n_l * self.dim)

            return observations

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_prior_dist(self):
        """Get the prior distribution object.

        Returns:
            The hierarchical prior distribution
        """
        return self.prior_dist

    def _get_transforms(
        self, automatic_transforms_enabled: bool = True, n_l=None, **kwargs
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

        # global_loc: Uniform[-prior_bound, prior_bound] <-> R
        for _ in range(self.dim):
            transforms_list.append(
                biject_to(constraints.interval(-self.prior_bound, self.prior_bound))
            )

        # global_scale: HalfNormal (R+) <-> R
        for _ in range(self.dim):
            transforms_list.append(biject_to(constraints.positive))

        # local params: TruncatedNormal (bounded) - use interval transform
        for _ in range(self.dim * n_l):
            transforms_list.append(
                biject_to(constraints.interval(-self.prior_bound, self.prior_bound))
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

        For hierarchical gaussian mixture, we compute a marginalized
        likelihood over the mixture components. The likelihood for each
        context is a mixture of Gaussians.

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

        # Split parameters: local params [:, 2*dim:]
        local_params = parameters[:, 2 * self.dim :]  # noqa: E203
        local_params = local_params.reshape(batch_size, self.n_l, self.dim)

        # Split data into n_l contexts (each dim observations)
        data_split = data.reshape(batch_size, self.n_l, self.dim)

        # Compute likelihood for each context (mixture of Gaussians)
        log_likelihoods = []
        for i in range(self.n_l):
            # Extract local parameters and data for context i
            context_params = local_params[:, i, :]  # (batch_size, dim)
            context_data = data_split[:, i, :]  # (batch_size, dim)

            # Compute log-likelihood for each mixture component
            mixture_weights = self.simulator_params["mixture_weights"]
            num_components = len(mixture_weights)

            component_log_probs = []
            for k in range(num_components):
                loc = self.simulator_params["mixture_locs_factor"][k] * context_params
                scale = self.simulator_params["mixture_scales"][k]

                # Compute Normal log-likelihood
                dist = pdist.Normal(loc, scale)
                log_prob_k = dist.log_prob(context_data).sum(dim=1)

                # Add log of mixture weight
                log_prob_k = log_prob_k + torch.log(mixture_weights[k])

                component_log_probs.append(log_prob_k)

            # Log-sum-exp over mixture components
            component_log_probs = torch.stack(component_log_probs, dim=0)
            log_lik_context = torch.logsumexp(component_log_probs, dim=0)

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
        """Reference posteriors not available for hierarchical tasks.

        Hierarchical tasks use reference-free metrics (reverse KL, LC2ST)
        instead of MCMC-based reference posteriors due to computational
        complexity and convergence challenges.

        Args:
            num_samples: Number of samples requested
            num_observation: Observation number
            observation: Observation tensor

        Raises:
            NotImplementedError: Always raised for hierarchical tasks
        """
        raise NotImplementedError(
            "Reference posteriors are not available for hierarchical tasks. "
            "Please use reference-free metrics such as reverse KL divergence "
            "or LC2ST for evaluation."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup hierarchical gaussian mixture task"
    )
    parser.add_argument(
        "--n_l",
        type=int,
        default=5,
        help="Number of local contexts",
    )
    args = parser.parse_args()

    task = HierarchicalGaussianMixture(n_l=args.n_l)
    task._setup(n_jobs=1, create_reference=False)

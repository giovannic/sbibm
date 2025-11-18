import math
from pathlib import Path
from typing import Any, Optional

import pyro
import torch
from pyro import distributions as pdist
from pyro.distributions import constraints
from pyro.distributions.transforms import biject_to
from pyro.infer.mcmc import MCMC, NUTS

import sbibm
from sbibm.tasks.distributions import (
    BlockwiseDistribution,
    HierarchicalDistribution,
    SummedStackTransform,
    TruncatedNormal,
)
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.tasks.two_moons.task import TwoMoons
from sbibm.utils.io import save_convergence_stats


class HierarchicalTwoMoons(Task):
    def __init__(self, n_l: int = 5, invalid_log_prob=1e-10):
        """Hierarchical Two Moons

        Hierarchical extension of the Two Moons task where each observation
        consists of n_l local contexts. Uses Strategy 2 (full local with
        pooling): all original parameters become local, with global pooling
        parameters controlling their distribution.

        Global parameters (4 total):
            - global_loc_0, global_loc_1: Location parameters for the two
              dimensions (Uniform(-1, 1))
            - global_scale_0, global_scale_1: Scale parameters (HalfNormal(0.5))

        Local parameters (2 * n_l total):
            - For each context i: theta_i ~ Normal(global_loc, global_scale)

        Args:
            n_l: Number of local contexts (default: 5)
            invalid_log_prob: Log probability for invalid parameters
        """
        self.n_l = n_l
        self.invalid_log_prob = invalid_log_prob

        # Observation seeds
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
            dim_parameters=4 + 2 * n_l,  # 4 global + 2*n_l local
            dim_data=2 * n_l,  # 2D observation per context
            name="hierarchical_two_moons",
            name_display="Hierarchical Two Moons",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        # Store simulator params from original two_moons
        self.simulator_params = {
            "a_low": -math.pi / 2.0,
            "a_high": +math.pi / 2.0,
            "base_offset": 0.25,
            "r_loc": 0.1,
            "r_scale": 0.01,
        }

        # Define hierarchical prior distribution
        # Global parameters: [loc_0, loc_1, scale_0, scale_1]
        global_loc_dist = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)
        global_scale_dist = pdist.HalfNormal(0.5).expand([2]).to_event(1)
        global_dist = BlockwiseDistribution([global_loc_dist, global_scale_dist])

        # Local params distribution conditioned on global
        def local_dist_fn(global_params, n_local):
            # global_params shape: [..., 4]
            # n_local: number of local groups/contexts
            # Extract locs and scales
            locs = global_params[..., :2]  # [..., 2]
            scales = global_params[..., 2:4]  # [..., 2]

            # Create distribution for n_local groups
            # Each local context (2D) is TruncatedNormal(loc,
            # scale, -1, 1)
            # Replicate locs and scales for n_local contexts
            batch_shape = global_params.shape[:-1]
            locs_expanded = (
                locs.unsqueeze(-2)
                .expand(list(batch_shape) + [n_local, 2])
            )
            scales_expanded = (
                scales.unsqueeze(-2)
                .expand(list(batch_shape) + [n_local, 2])
            )

            return pdist.Independent(
                TruncatedNormal(locs_expanded, scales_expanded, -1.0, 1.0),
                2,
            )

        self.prior_dist = HierarchicalDistribution(
            global_dist,
            local_dist_fn,
            dim_global=4,
            dim_local=2,
            n_local=n_l,
        )
        self.prior_dist.set_default_validate_args(False)

        self.composite_transform = self._get_transforms()["parameters"].inv

    def get_prior(self):
        """Get prior distribution.

        Returns a callable that samples from self.prior_dist using pyro.
        """

        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get simulator function.

        For each local context, applies the original two_moons simulator to
        the corresponding local parameters.

        Args:
            max_calls: Maximum number of simulator calls

        Returns:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            # Split parameters into global and local
            # Global: [:, 0:4] (2 locs + 2 scales)
            # Local: [:, 4:] (2*n local parameters where n is
            # inferred from shape)
            n_local_dims = parameters.shape[1] - 4
            n_local_groups = n_local_dims // 2
            local_params = parameters[:, 4:].reshape(num_samples, n_local_groups, 2)

            # For each local context, run the two_moons
            # simulator
            observations = []
            for i in range(n_local_groups):
                # Extract local parameters for context i
                context_params = local_params[:, i, :]  # (num_samples, 2)

                # Sample noise components (a and r) for this context
                a_dist = (
                    pdist.Uniform(
                        low=self.simulator_params["a_low"],
                        high=self.simulator_params["a_high"],
                    )
                    .expand_by((num_samples, 1))
                    .to_event(1)
                )
                a = a_dist.sample()

                r_dist = (
                    pdist.Normal(
                        self.simulator_params["r_loc"],
                        self.simulator_params["r_scale"],
                    )
                    .expand_by((num_samples, 1))
                    .to_event(1)
                )
                r = r_dist.sample()

                p = torch.cat(
                    (
                        torch.cos(a) * r + self.simulator_params["base_offset"],
                        torch.sin(a) * r,
                    ),
                    dim=1,
                )

                # Apply two_moons mapping function
                obs_i = TwoMoons._map_fun(context_params, p)
                observations.append(obs_i)

            # Concatenate observations from all contexts
            observations = torch.cat(observations, dim=1)

            return observations

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _get_transforms(self, automatic_transforms_enabled: bool = True, n_l = None):
        if n_l is None:
            n_l = self.n_l

        # Build composite transform for MCMC (constrained <-> unconstrained)
        # This maps between constrained parameter space and unconstrained R^n
        transforms_list = []

        # global_loc: Uniform[-1, 1] <-> R
        for _ in range(2):
            transforms_list.append(biject_to(constraints.interval(-1.0, 1.0)))

        # global_scale: HalfNormal (R+) <-> R
        for _ in range(2):
            transforms_list.append(biject_to(constraints.positive))

        # local params: TruncatedNormal[-1, 1] <-> R
        for _ in range(2 * n_l):
            transforms_list.append(biject_to(constraints.interval(-1.0, 1.0)))

        composite_transform = SummedStackTransform(transforms_list, dim=-1)
        return {"parameters": composite_transform.inv}

    def _likelihood(
        self,
        parameters: torch.Tensor,
        data: torch.Tensor,
        log: bool = True,
    ) -> torch.Tensor:
        """Compute likelihood of data given parameters.

        For hierarchical two moons, the likelihood is the product of
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

        # Split parameters: global [:, 0:4], local [:, 4:]
        local_params = parameters[:, 4:].reshape(batch_size, self.n_l, 2)

        # Split data into n_l observations (each 2D)
        data_split = data.reshape(batch_size, self.n_l, 2)

        # Compute likelihood for each context and sum log-likelihoods
        log_likelihoods = []
        for i in range(self.n_l):
            # Extract local parameters and data for context i
            context_params = local_params[:, i, :]  # (batch_size, 2)
            context_data = data_split[:, i, :]  # (batch_size, 2)

            # Use original two_moons likelihood logic
            p = TwoMoons._map_fun_inv(context_params, context_data)
            if p.ndim == 1:
                p = p.reshape(1, -1)

            u = p[:, 0] - self.simulator_params["base_offset"]
            v = p[:, 1]

            r = torch.sqrt(u**2 + v**2)
            log_lik_context = -0.5 * (
                (r - self.simulator_params["r_loc"]) / self.simulator_params["r_scale"]
            ) ** 2 - 0.5 * torch.log(
                2 * torch.tensor([math.pi]) * self.simulator_params["r_scale"] ** 2
            )

            # Handle invalid region (u < 0)
            if len(torch.where(u < 0.0)[0]) > 0:
                log_lik_context[torch.where(u < 0.0)[0]] = self.invalid_log_prob

            log_likelihoods.append(log_lik_context)

        # Sum log-likelihoods across contexts (product of likelihoods)
        total_log_likelihood = torch.stack(log_likelihoods, dim=0).sum(dim=0)

        return total_log_likelihood if log else torch.exp(total_log_likelihood)

    def _get_potential_fn(
        self,
        observation: torch.Tensor,
    ):
        """Create potential function for NUTS with reparameterization.

        The potential function maps unconstrained parameters (R^n) to
        negative unnormalized log posterior, handling transforms and
        Jacobian corrections.

        Args:
            observation: Observed data

        Returns:
            Callable potential function
        """

        def potential_fn(z_unconstrained):
            """Compute negative unnormalized log posterior.

            Args:
                z_unconstrained: Dict with 'parameters' key, unconstrained

            Returns:
                Negative log posterior (potential energy)
            """
            z = z_unconstrained["parameters"]

            # Transform to constrained space
            theta_constrained = self.composite_transform(z)

            # Compute log prior in constrained space
            log_prior = self.prior_dist.log_prob(theta_constrained)

            # Compute log likelihood in constrained space
            log_likelihood = self._likelihood(theta_constrained, observation, log=True)

            # Compute Jacobian correction
            log_abs_det_jacobian = self.composite_transform.log_abs_det_jacobian(
                z, theta_constrained
            )

            # Potential = -(log_prior + log_likelihood + log|det J|)
            # The Jacobian accounts for change of variables
            log_posterior = log_prior + log_likelihood + log_abs_det_jacobian

            return -log_posterior.sum()

        return potential_fn

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        num_chains: int = 5,
        num_warmup: int = 5000,
        thinning: int = 1,
    ) -> torch.Tensor:
        """Sample reference posterior using MCMC.

        Uses Pyro's NUTS sampler with multiple chains to generate reference
        posterior samples. Computes and saves convergence diagnostics
        (R-hat, ESS).

        Args:
            num_samples: Number of samples to generate
            num_observation: Observation number to load
            observation: Observation tensor, alternative to num_observation
            num_chains: Number of MCMC chains (default: 5)
            num_warmup: Number of warmup steps (default: 5000)
            thinning: Thinning factor (default: 1)

        Returns:
            Samples from reference posterior (num_samples, dim_parameters)
        """
        log = sbibm.get_logger(__name__)

        # Run MCMC with NUTS sampler
        log.info(
            f"Running MCMC for observation {num_observation} " f"with n_l={self.n_l}"
        )

        # Get observation
        if observation is None:
            observation = self.get_observation(num_observation=num_observation)

        # Create potential function with reparameterization
        potential_fn = self._get_potential_fn(observation=observation)

        # Set up NUTS kernel with potential function
        # No need to pass transforms since potential_fn handles them internally
        kernel_parameters = {
            "jit_compile": False,
        }
        mcmc_kernel = NUTS(potential_fn=potential_fn, **kernel_parameters)

        # Initialize at prior samples (in unconstrained space)
        init_params_constrained = self.prior_dist.sample((num_chains,))
        init_params_unconstrained = self.composite_transform.inv(
            init_params_constrained
        )
        initial_params = {"parameters": init_params_unconstrained}

        mcmc_parameters = {
            "num_chains": num_chains,
            "num_samples": thinning * num_samples,
            "warmup_steps": num_warmup,
            "initial_params": initial_params,
        }

        mcmc = MCMC(mcmc_kernel, **mcmc_parameters)
        mcmc.run()

        log.info("MCMC complete, extracting samples and diagnostics")

        # Apply thinning
        mcmc._samples = {"parameters": mcmc._samples["parameters"][:, ::thinning, :]}

        # Get samples (these are in unconstrained space from NUTS)
        num_samples_available = (
            mcmc._samples["parameters"].shape[0] * mcmc._samples["parameters"].shape[1]
        )
        if num_samples_available < num_samples:
            log.warning("Some samples will be included multiple times")
            samples_unconstrained = mcmc.get_samples(
                num_samples=num_samples, group_by_chain=False
            )["parameters"].squeeze()
        else:
            samples_unconstrained = mcmc.get_samples(group_by_chain=False)[
                "parameters"
            ].squeeze()
            idx = torch.randperm(samples_unconstrained.shape[0])[:num_samples]
            samples_unconstrained = samples_unconstrained[idx, :]

        # Transform samples back to constrained space (unconstrained -> constrained)
        samples = self.composite_transform(samples_unconstrained)

        # Extract convergence diagnostics
        diagnostics = mcmc.diagnostics()
        r_hat = diagnostics["parameters"]["r_hat"].squeeze()
        n_eff = diagnostics["parameters"]["n_eff"].squeeze()

        # Log warnings for poor convergence
        if r_hat.max() > 1.01:
            log.warning(
                f"Some parameters have R-hat > 1.01. " f"Max R-hat: {r_hat.max():.4f}"
            )
            for i, r in enumerate(r_hat):
                if r > 1.01:
                    log.warning(f"  Parameter {i}: R-hat = {r:.4f}")

        # Save convergence statistics
        if num_observation is not None:
            stats_dict = {}
            for i in range(len(r_hat)):
                stats_dict[f"r_hat_param_{i}"] = float(r_hat[i])
                stats_dict[f"n_eff_param_{i}"] = float(n_eff[i])
            stats_dict["max_r_hat"] = float(r_hat.max())
            stats_dict["min_n_eff"] = float(n_eff.min())

            convergence_stats_path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "convergence_stats.csv"
            )
            save_convergence_stats(convergence_stats_path, stats_dict)
            log.info(f"Saved convergence stats to {convergence_stats_path}")

        return samples


if __name__ == "__main__":
    task = HierarchicalTwoMoons(n_l=5)
    task._setup(n_jobs=4, create_reference=True)

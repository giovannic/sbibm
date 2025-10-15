import math
from pathlib import Path
from typing import Optional

import torch
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.tasks.two_moons.task import TwoMoons


class HierarchicalTwoMoons(Task):
    def __init__(self, n_l: int = 5):
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
        """
        self.n_l = n_l

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

    def get_prior(self):
        """Get prior distribution.

        Returns a callable that samples from the hierarchical prior:
        - Global locs: Uniform(-1, 1) for each dimension
        - Global scales: HalfNormal(0.5) for each dimension
        - Local params: Normal(global_loc, global_scale) for each context
        """

        def prior(num_samples=1):
            # Sample global location parameters: Uniform(-1, 1)
            global_locs = pdist.Uniform(-1.0, 1.0).sample((num_samples, 2))

            # Sample global scale parameters: HalfNormal(0.5)
            global_scales = pdist.HalfNormal(0.5).sample((num_samples, 2))

            # Sample local parameters for each context
            # Each local param is 2D, drawn from Normal(global_loc, global_scale)
            local_params = []
            for i in range(self.n_l):
                # Sample 2D parameters for context i
                local_param = torch.normal(global_locs, global_scales)
                local_params.append(local_param)

            # Concatenate: [global_locs (2), global_scales (2), local_params
            # (2*n_l)]
            local_params_flat = torch.cat(local_params, dim=1)
            parameters = torch.cat(
                [global_locs, global_scales, local_params_flat], dim=1
            )

            return parameters

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
            # Local: [:, 4:] (2*n_l parameters)
            local_params = parameters[:, 4:].reshape(num_samples, self.n_l, 2)

            # For each local context, run the two_moons simulator
            observations = []
            for i in range(self.n_l):
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

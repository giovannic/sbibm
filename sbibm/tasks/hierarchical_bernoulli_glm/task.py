from pathlib import Path
from typing import Optional

import numpy as np
import pyro
import torch
from pyro import distributions as pdist

from sbibm.tasks.distributions import HierarchicalDistribution
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv
from sbibm.utils.torch import get_default_device


class HierarchicalBernoulliGLM(Task):
    def __init__(self, n_l: int = 5, summary="sufficient"):
        """Hierarchical Bernoulli GLM

        Hierarchical extension of the Bernoulli GLM task where each
        observation consists of n_l local contexts. Uses Strategy 1
        (natural global/local split): filter coefficients are global (shared
        across contexts), while offsets are local (context-specific).

        Global parameters (9 total):
            - Filter coefficients h (9 dims): Penalized 2nd order differences

        Local parameters (n_l total):
            - Offset b_i per context: Normal(0, 0.5)

        Args:
            n_l: Number of local contexts (default: 5)
            summary: Type of summary statistics ("sufficient" or "raw")
        """
        self.n_l = n_l
        self.summary = summary

        if self.summary == "sufficient":
            dim_data = 10 * n_l  # 10 summary stats per context
            name = "hierarchical_bernoulli_glm"
            name_display = "Hierarchical Bernoulli GLM"
            self.raw = False
        elif self.summary == "raw":
            dim_data = 100 * n_l  # 100 raw data points per context
            name = "hierarchical_bernoulli_glm_raw"
            name_display = "Hierarchical Bernoulli GLM Raw"
            self.raw = True
        else:
            raise NotImplementedError

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
            dim_parameters=9 + n_l,  # 9 global (filter) + n_l local (offsets)
            dim_data=dim_data,
            name=name,
            name_display=name_display,
            num_simulations=[1000, 10000, 100000, 1000000],
            num_posterior_samples=10000,
            num_observations=10,
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        self.stimulus = {
            "dt": 1,  # timestep
            "duration": 100,  # duration of input stimulus
            "seed": 42,  # seperate seed to freeze noise on input current
        }

        # Prior on filter (global parameters)
        # Smoothness in filter encouraged by penalizing 2nd order diffs
        M = 9  # Filter dimension (original dim_parameters - 1)
        D = torch.diag(torch.ones(M)) - torch.diag(torch.ones(M - 1), -1)
        F = torch.matmul(D, D) + torch.diag(1.0 * torch.arange(M) / (M)) ** 0.5
        Binv = torch.matmul(F.T, F)  # filter precision matrix

        # Global distribution: filter coefficients (9 dims)
        global_dist = pdist.MultivariateNormal(
            loc=torch.zeros((M,)), precision_matrix=Binv
        )
        global_dist.set_default_validate_args(False)

        # Local distribution: offsets per context (n_l dims)
        def local_dist_fn(global_params):
            # global_params shape: [..., 9]
            # Each local offset ~ Normal(0, 0.5)
            batch_shape = global_params.shape[:-1]
            loc = torch.zeros(list(batch_shape) + [n_l])
            scale = 0.5 * torch.ones(list(batch_shape) + [n_l])
            return pdist.Independent(pdist.Normal(loc, scale), 1)

        self.prior_dist = HierarchicalDistribution(
            global_dist, local_dist_fn, dim_global=9, dim_local=n_l
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
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls
                will result in SimulationBudgetExceeded exceptions. Defaults
                to None for infinite budget

        Return:
            Simulator callable
        """
        device = get_default_device()

        stimulus_I = torch.load(self.path / "files" / "stimulus_I.pt").to(device)
        design_matrix = torch.load(self.path / "files" / "design_matrix.pt").to(device)

        def simulator(
            parameters: torch.Tensor, return_both: bool = False
        ) -> torch.Tensor:
            """Simulates model for given parameters

            If `return_both` is True, will additionally return spike train
            not reduced to summary features
            """
            # Split parameters into global (filter) and local (offsets)
            # Global: [:, 0:9] (filter coefficients)
            # Local: [:, 9:] (offsets per context)
            batch_size = parameters.shape[0]
            global_filter = parameters[:, :9]  # (batch_size, 9)
            local_offsets = parameters[:, 9:]  # (batch_size, n_l)

            data = []
            data_raw = []

            for b in range(batch_size):
                context_data = []
                context_data_raw = []

                # For each local context, run the GLM simulation
                for i in range(self.n_l):
                    # Combine filter and offset for this context
                    # params_context: [offset, filter[0], ..., filter[8]]
                    params_context = torch.cat(
                        [
                            local_offsets[b : b + 1, i : i + 1],
                            global_filter[b : b + 1, :],
                        ],
                        dim=1,
                    ).squeeze()  # (10,)

                    # Simulate GLM
                    psi = torch.matmul(design_matrix, params_context)
                    z = 1 / (1 + torch.exp(-psi))
                    y = (torch.rand(design_matrix.shape[0]) < z).float()

                    # Calculate summary statistics
                    num_spikes = torch.sum(y).unsqueeze(0)
                    sta = torch.nn.functional.conv1d(
                        y.reshape(1, 1, -1),
                        stimulus_I.reshape(1, 1, -1),
                        padding=8,
                    ).squeeze()[-9:]
                    context_data.append(torch.cat((num_spikes, sta)))

                    if self.raw or return_both:
                        context_data_raw.append(y)

                # Concatenate data from all contexts
                if not self.raw:
                    data.append(torch.cat(context_data))
                else:
                    data.append(torch.cat(context_data_raw))

                if return_both:
                    data_raw.append(torch.cat(context_data_raw))

            if not return_both:
                return torch.stack(data)
            else:
                return torch.stack(data), torch.stack(data_raw)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Get observed data for a given observation number"""
        if not self.raw:
            path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "observation.csv"
            )
            return get_tensor_from_csv(path)
        else:
            path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "observation_raw.csv"
            )
            return get_tensor_from_csv(path)

    def flatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Flattens data

        Data returned by the simulator is always flattened into 2D Tensors
        """
        if type(data) == tuple:
            return data
        else:
            return data.reshape(-1, self.dim_data)

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

    def _setup(self, regenerate_stimulus=False):
        """Setup the task: generate observations and true parameters

        In most cases, you don't need to execute this method, since its
        results are stored to disk. Re-executing will overwrite existing
        files.

        Note: Reference posteriors are not generated for hierarchical tasks.
        Use reference-free metrics (reverse KL, LC2ST) for evaluation.
        """
        # Generate input stimulus (same across all observations)
        # Stimulus is Gaussian white noise ~N(0, 1)
        if regenerate_stimulus:
            stimulus_t = torch.arange(
                0,
                self.stimulus["duration"],
                self.stimulus["dt"],
                dtype=torch.float32,
            )
            path = self.path / "files" / "stimulus_t.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stimulus_t, path)
            stimulus_I = torch.from_numpy(
                np.random.RandomState(self.stimulus["seed"])
                .randn(len(stimulus_t))
                .reshape(-1)
                .astype(np.float32)
            )
            path = self.path / "files" / "stimulus_I.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stimulus_I, path)

            # Build design matrix X, such that X * params returns
            # convolution of x with filter plus offset
            # Including linear offset by first element
            design_matrix = torch.zeros(size=(len(stimulus_t), 9))  # 9 filter coeffs
            for j in range(9):
                design_matrix[j:, j] = stimulus_I[0 : len(stimulus_t) - j]
            design_matrix = torch.cat(
                (torch.ones(size=(len(stimulus_t), 1)), design_matrix), axis=1
            )
            path = self.path / "files" / "design_matrix.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(design_matrix, path)

        for num_observation, observation_seed in enumerate(
            self.observation_seeds, start=1
        ):
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            true_parameters = prior(num_samples=1)
            self._save_true_parameters(num_observation, true_parameters)

            simulator = self.get_simulator()
            observation, observation_raw = simulator(true_parameters, return_both=True)
            self._save_observation(num_observation, observation)

            # Save raw observation
            path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "observation_raw.csv"
            )
            old_dim_data = self.dim_data
            self.dim_data = 100 * self.n_l
            self.save_data(path, observation_raw)
            self.dim_data = old_dim_data


if __name__ == "__main__":
    task = HierarchicalBernoulliGLM(n_l=5)
    task._setup(regenerate_stimulus=True)

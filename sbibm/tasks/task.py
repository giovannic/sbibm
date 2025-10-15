from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pyro
import torch

from sbibm.utils.io import get_tensor_from_csv
from sbibm.utils.pyro import get_log_prob_fn


class Task:
    def __init__(
        self,
        dim_data: int,
        dim_parameters: int,
        name: str,
        num_observations: int,
        num_simulations: List[int],
        path: Path,
        name_display: Optional[str] = None,
        observation_seeds: Optional[List[int]] = None,
        generate_in_memory: bool = False,
    ):
        """Base class for tasks.

        Args:
            dim_data: Dimensionality of data.
            dim_parameters: Dimensionality of parameters.
            name: Name of task. Should be the name of the folder in which
                the task is stored. Used with `sbibm.get_task(name)`.
            num_observations: Number of different observations for this task.
            num_simulations: List containing number of different simulations to
                run this task for.
            path: Path to folder of task.
            name_display: Display name of task, with correct upper/lower-case
                spelling and spaces. Defaults to `name`.
            observation_seeds: List of observation seeds to use. Defaults to
                a sequence of length `num_observations`. Override to use specific
                seeds.
            generate_in_memory: If True, observations and true parameters are
                produced from `observation_seeds` on demand and cached in memory
                instead of being read from disk.
        """
        self.dim_data = dim_data
        self.dim_parameters = dim_parameters
        self.name = name
        self.num_observations = num_observations
        self.num_simulations = num_simulations
        self.path = path

        self.name_display = name_display if name_display is not None else name
        self.observation_seeds = (
            observation_seeds
            if observation_seeds is not None
            else [i + 1000000 for i in range(self.num_observations)]
        )
        self._generate_in_memory = generate_in_memory
        self._observation_cache = {}

    @abstractmethod
    def get_prior(self) -> Callable:
        """Get function returning parameters from prior"""
        raise NotImplementedError

    def get_prior_dist(self) -> torch.distributions.Distribution:
        """Get prior distribution"""
        return self.prior_dist

    def get_prior_params(self) -> Dict[str, torch.Tensor]:
        """Get parameters of prior distribution"""
        return self.prior_params

    def get_labels_data(self) -> List[str]:
        """Get list containing parameter labels"""
        return [f"data_{i+1}" for i in range(self.dim_data)]

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [f"parameter_{i+1}" for i in range(self.dim_parameters)]

    def _generate_observation(self, num_observation: int) -> None:
        """Generate observation and true parameters in-memory from seed.

        Results are cached so repeated calls are free. RNG state is
        saved and restored so this never perturbs the caller's random stream.
        """
        if num_observation in self._observation_cache:
            return

        observation_seed = self.observation_seeds[num_observation - 1]

        # Save RNG state
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = [
                torch.cuda.get_rng_state(d)
                for d in range(torch.cuda.device_count())
            ]

        try:
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)

            prior = self.get_prior()
            true_parameters = prior(num_samples=1)
            simulator = self.get_simulator()
            observation = simulator(true_parameters)

            self._observation_cache[num_observation] = (
                true_parameters.detach().cpu().clone(),
                self.flatten_data(observation.detach().cpu().clone()),
            )
        finally:
            # Restore RNG state
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                for d, state in enumerate(cuda_states):
                    torch.cuda.set_rng_state(state, d)

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Get observed data for a given observation number"""
        if self._generate_in_memory:
            self._generate_observation(num_observation)
            return self._observation_cache[num_observation][1]

        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "observation.csv"
        )
        return get_tensor_from_csv(path)

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        """Get reference posterior samples for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "reference_posterior_samples.csv.bz2"
        )
        return get_tensor_from_csv(path)

    @abstractmethod
    def get_simulator(self) -> Callable:
        """Get function returning parameters from prior"""
        raise NotImplementedError

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        """Get true parameters (parameters that generated the data) for a given observation number"""
        if self._generate_in_memory:
            self._generate_observation(num_observation)
            return self._observation_cache[num_observation][0]

        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        return get_tensor_from_csv(path)

    def flatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Flattens data

        Data returned by the simulator is always flattened into 2D Tensors
        """
        return data.reshape(-1, self.dim_data)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data

        Tasks that require more than 2 dimensions for output of the simulator (e.g.
        returning images) may override this method.
        """
        return data.reshape(-1, self.dim_data)

    def _get_transforms(
        self,
        automatic_transforms_enabled: bool = True,
        num_observation: Optional[int] = 1,
        observation: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Gets transforms

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            automatic_transforms_enabled: If True, will automatically construct
                transforms to unconstrained space

        Returns:
            Dict containing transforms
        """
        conditioned_model = self._get_pyro_model(
            num_observation=num_observation, observation=observation
        )

        _, transforms = get_log_prob_fn(
            conditioned_model,
            automatic_transform_enabled=automatic_transforms_enabled,
        )

        return transforms

    def _get_pyro_model(
        self,
        posterior: bool = True,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> Callable:
        """Get model function for use with Pyro

        If `num_observation` or `observation` is passed, the model is conditioned.

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            posterior: If False, will mask prior which will result in model useful
                for calculating log likelihoods instead of log posterior probabilities
        """
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        prior = self.get_prior()
        simulator = self.get_simulator()

        def model_fn():
            prior_ = pyro.poutine.mask(prior, torch.tensor(posterior))
            return simulator(prior_())

        if observation is not None:
            observation = self.unflatten_data(observation)
            return pyro.condition(model_fn, {"data": observation})
        else:
            return model_fn

    @abstractmethod
    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Args:
            num_samples: Number of samples
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Samples from reference posterior
        """
        raise NotImplementedError

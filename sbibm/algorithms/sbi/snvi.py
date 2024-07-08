import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import likelihood_nn

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "maf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    automatic_transforms_enabled: bool = True,
    q: str = "maf",
    vi_method: str = "fKL",
    vi_parameters: Dict[str, Any] = {
        "n_particles": 256,
        "learning_rate": 0.001,
        "gamma": 0.999,
        "max_num_iters": 2000,
        "min_num_iters": 10
    },
    z_score_x: str = "independent",
    z_score_theta: str = "independent",
    max_num_epochs: int = 2**31 - 1,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NVI from `sbi`

    Args:
        task: Task instance
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of maf / mdn / made / nsf
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        automatic_transforms_enabled: Whether to enable automatic transforms
        q: variational model
        vi_method: loss function for VI
        vi_parameters: VI parameters for training
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta
        max_num_epochs: Maximum number of epochs

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    if num_rounds == 1:
        log.info(f"Running NVI")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNVI")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)[
        "parameters"
    ]
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    density_estimator_fun = likelihood_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    inference_method = inference.SNLE_A(
        density_estimator=density_estimator_fun,
        prior=prior,
    )

    posteriors = []
    proposal = prior

    for r in range(num_rounds):
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )

        density_estimator = inference_method.append_simulations(
            theta, x, from_round=r
        ).train(
            training_batch_size=training_batch_size,
            retrain_from_scratch=False,
            discard_prior_samples=False,
            show_train_summary=True,
            max_num_epochs=max_num_epochs,
        )

        (
            potential_fn,
            theta_transform,
        ) = inference.likelihood_estimator_based_potential(
            density_estimator,
            prior,
            observation,
            # NOTE: disable transform if sbibm does it. will return IdentityTransform.
            enable_transform=not automatic_transforms_enabled,
        )
        posterior = inference.VIPosterior(
            potential_fn=potential_fn,
            prior=prior,
            q=q,
            theta_transform=theta_transform,
            vi_method=vi_method,
        ).train(**vi_parameters)
        # Change init_strategy to latest_sample after second round.

        q = posterior.set_default_x(observation)
        posteriors.append(posterior)

    posterior = wrap_posterior(posteriors[-1], transforms)

    assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    return samples, simulator.num_simulations, None

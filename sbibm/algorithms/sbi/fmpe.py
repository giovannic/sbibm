import logging
from typing import Optional, Tuple

import torch
from sbi import inference as inference

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
    vf_estimator: str = "mlp",
    simulation_batch_size: int = 1000,
    training_batch_size: int = 200,
    automatic_transforms_enabled: bool = False,
    max_num_epochs: Optional[int] = 2**31 - 1,
    device: str = "cpu",
    **kwargs,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], object]:
    """Runs FMPE (Flow Matching Posterior Estimation) from `sbi`

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        vf_estimator: Vector field estimator architecture, one of
            mlp / ada_mlp / transformer / transformer_cross_attn
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        automatic_transforms_enabled: Whether to enable automatic transforms
        max_num_epochs: Maximum number of epochs
        device: Device to use (cpu, cuda, cuda:0, etc.)

    Returns:
        Tuple of (samples, num_simulations, log_prob_true_params, posterior)
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)
    log.info("Running FMPE")

    if simulation_batch_size > num_simulations:
        simulation_batch_size = num_simulations
        log.warn("Reduced simulation_batch_size to num_simulations")

    if training_batch_size > num_simulations:
        training_batch_size = num_simulations
        log.warn("Reduced training_batch_size to num_simulations")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation).to(device=device)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]

    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    inference_method = inference.FMPE(
        prior, vf_estimator=vf_estimator, device=device
    )

    theta, x = inference.simulate_for_sbi(
        simulator,
        prior,
        num_simulations=num_simulations,
        simulation_batch_size=simulation_batch_size,
    )

    vector_field_estimator = inference_method.append_simulations(
        theta, x
    ).train(
        training_batch_size=training_batch_size,
        show_train_summary=True,
        max_num_epochs=max_num_epochs,
    )
    posterior = inference_method.build_posterior(vector_field_estimator)
    posterior.set_default_x(observation)

    posterior = wrap_posterior(posterior, transforms)

    assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    return samples, simulator.num_simulations, None, posterior

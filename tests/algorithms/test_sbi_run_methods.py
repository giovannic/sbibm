import pytest

import sbibm
from sbibm.algorithms.sbi import (
    mcabc,
    smcabc,
    snle,
    snvi,
    snpe,
    snre,
    sl
)

from typing import Dict, Any


# a fast test
@pytest.mark.parametrize(
    "run_method",
    (mcabc, smcabc, snle, snvi, snpe, snre, sl),
)
@pytest.mark.parametrize("task_name", ("gaussian_mixture",))
@pytest.mark.parametrize("num_observation", (1,))
def test_sbi_api(
    run_method,
    task_name,
    num_observation,
    num_simulations=2_000,
    num_samples=100,
):
    task = sbibm.get_task(task_name)
    num_rounds = 2

    kwargs: Dict[str, Any] = dict()

    if run_method not in (mcabc, smcabc, sl):  # neural algorithms
        kwargs = dict(
            num_rounds=num_rounds,
            training_batch_size=100,
            neural_net="mlp" if run_method == snre else "maf",
        )
    if run_method in (snle, snre):
        kwargs["mcmc_parameters"] = dict(
            num_chains=4,
            warmup_steps=100,
            thin=10,
            init_strategy="resample"
        )
    if run_method == snvi:
        kwargs["vi_parameters"] = dict(
            n_particles=8,
            max_num_iters=10
        )

    predicted, _, _ = run_method(
        task=task,
        num_observation=num_observation,
        num_simulations=num_simulations,
        num_samples=num_samples,
        **kwargs,
    )

    reference_samples = task.get_reference_posterior_samples(
        num_observation=num_observation
    )

    expected = reference_samples[:num_samples, :]

    assert expected.shape == predicted.shape

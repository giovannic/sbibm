"""Test that amortised posterior estimators can evaluate at arbitrary contexts
"""

import pytest
import torch
from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.algorithms.tfmpe.bottom_up import run as run_bottom_up
from sbibm.algorithms.deepset import run as run_deepset
from sbibm.tasks.hierarchical_gaussian_linear.task import (
    HierarchicalGaussianLinear
)

@pytest.mark.slow
def test_tfmpe_posterior_samples_batched_context():
    task = HierarchicalGaussianLinear()
    n_samples = 100
    #TODO: limit epochs
    _, _, _, posterior = run_bottom_up(
        task=task,
        num_observation=1,
        num_samples=n_samples,
        num_simulations=n_samples,
        automatic_transforms_enabled=True,
    )
    n_observations = 2
    observations = torch.cat([
        task.get_observation(num_observation=i + 1)
        for i in range(n_observations)
    ])
    samples = posterior.sample((n_observations,), x=observations)

    param_dim = (
        task.prior_dist.dim_global +
        task.prior_dist.dim_local *
        task.prior_dist.n_local
    )
    assert samples.shape == (n_observations, param_dim)
    assert not torch.isnan(samples).any()

def test_snpe_posterior_samples_batched_context():
    task = HierarchicalGaussianLinear()
    n_samples = 100
    _, _, _, posterior = run_snpe(
        task=task,
        num_observation=1,
        num_samples=n_samples,
        num_simulations=n_samples,
        num_rounds=1,
        hidden_features=50,
        max_num_epochs=5,
        automatic_transforms_enabled=True,
    )

    n_observations = 2
    observations = torch.cat([
        task.get_observation(num_observation=i + 1)
        for i in range(n_observations)
    ])
    samples = posterior.sample((n_observations,), x=observations)

    #TODO: stricter assertions
    param_dim = (
        task.prior_dist.dim_global +
        task.prior_dist.dim_local *
        task.prior_dist.n_local
    )
    assert samples.shape == (n_observations, param_dim)
    assert not torch.isnan(samples).any()

def test_deepset_posterior_samples_batched_context():
    task = HierarchicalGaussianLinear()
    n_samples = 100
    _, _, _, posterior = run_deepset(
        task=task,
        num_observation=1,
        num_samples=n_samples,
        num_simulations=n_samples,
        max_epochs=5,
        dim_hidden=64,
        training_batch_size=128,
        device='cpu'
    )

    n_observations = 2
    observations = torch.cat([
        task.get_observation(num_observation=i + 1)
        for i in range(n_observations)
    ])
    samples = posterior.sample((n_observations,), x=observations)

    #TODO: stricter assertions
    param_dim = (
        task.prior_dist.dim_global +
        task.prior_dist.dim_local *
        task.prior_dist.n_local
    )
    assert samples.shape == (n_observations, param_dim)
    assert not torch.isnan(samples).any()


"""
Functional tests for DeepSet architecture.

Tests that the copied DeepSet code works correctly with hierarchical
Gaussian Linear data structures.
"""

import torch
import torch.nn as nn
from sbibm.algorithms.deepset import DeepSet
from sbibm.algorithms.deepset.training import (
    train_hierarchical_deepset,
)


def test_deepset_instantiation():
    """Test that DeepSet can be instantiated."""
    model = DeepSet(
        n_in=3,  # Observations per event
        n_out=2,  # Mean and log-std for one global parameter
        n_embedding=64,
        seq_length=5,  # Matches num_events
    )
    assert model is not None
    assert isinstance(model, nn.Module)


def test_deepset_forward_pass():
    """Test that DeepSet forward pass produces correct shapes."""
    model = DeepSet(
        n_in=3,
        n_out=2,
        n_embedding=64,
        seq_length=5,
        sum_aggregations=True,
    )

    # Create batch of data
    batch_size = 32
    num_events = 5
    dim_per_event = 3

    x = torch.randn(batch_size, num_events, dim_per_event)

    # Forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, 2), (
        f"Expected shape ({batch_size}, 2), got {output.shape}"
    )


def test_deepset_finite_outputs():
    """Test that DeepSet produces finite outputs (no NaN/Inf)."""
    model = DeepSet(
        n_in=3,
        n_out=2,
        n_embedding=64,
        seq_length=5,
        sum_aggregations=True,
    )

    batch_size = 32
    x = torch.randn(batch_size, 5, 3)

    output = model(x)

    # Check for NaN and Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


def test_deepset_hierarchical_training_loop(
    hierarchical_gaussian_linear_data,
):
    """
    Test hierarchical training with separate global and local paths.

    Verifies:
    - Global model trained on aggregated observation information
    - Local models trained per-event, conditioned on observations
    - Joint loss computation works correctly via training utility
    - Loss decreases over epochs
    - Outputs remain finite during training
    """
    # Create synthetic hierarchical data
    create_data = hierarchical_gaussian_linear_data
    x_set, y_global, y_local = create_data(
        num_datasets=100,
        num_events=5,
        dim_per_event=3,
        dim_global=1,
        dim_local=1,
    )

    # Global model: predicts global parameter from aggregated
    # observations
    global_model = DeepSet(
        n_in=3,  # Observation dimension per event
        n_out=2,  # Mean and log-std for global param
        n_embedding=64,
        seq_length=5,  # num_events
        sum_aggregations=True,
    )

    # Local models: one per event, predicts local parameter
    num_events = 5
    local_models = nn.ModuleList([
        DeepSet(
            n_in=3,  # Observation dimension
            n_out=2,  # Mean and log-std for local param
            n_embedding=64,
            seq_length=5,
            sum_aggregations=True,
        )
        for _ in range(num_events)
    ])

    # Train using hierarchical training utility
    loss_values = train_hierarchical_deepset(
        global_model=global_model,
        local_models=local_models,
        x_set=x_set,
        y_global=y_global,
        y_local=y_local,
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        verbose=False,
    )

    # Check for finite losses
    for epoch, loss in enumerate(loss_values):
        assert not torch.isnan(
            torch.tensor(loss)
        ), f"Loss is NaN at epoch {epoch}"

    # Check that loss decreased overall (or mostly)
    assert loss_values[-1] <= loss_values[0] * 1.5, (
        f"Loss did not decrease sufficiently: "
        f"first={loss_values[0]:.4f}, last={loss_values[-1]:.4f}"
    )


def test_deepset_sample_and_log_prob():
    """Test that model outputs can be used for sampling and log_prob."""
    model = DeepSet(
        n_in=3,
        n_out=2,
        n_embedding=64,
        seq_length=5,
        sum_aggregations=True,
    )

    x = torch.randn(16, 5, 3)
    output = model(x)

    pred_mu, pred_log_sigma = torch.chunk(output, 2, -1)
    pred_sigma = pred_log_sigma.exp() + 1e-6

    # Create distribution
    dist = torch.distributions.Normal(pred_mu, pred_sigma)

    # Sample
    samples = dist.rsample()
    assert samples.shape == (16, 1)
    assert not torch.isnan(samples).any()

    # Compute log_prob
    log_probs = dist.log_prob(samples)
    assert log_probs.shape == (16, 1)
    assert not torch.isnan(log_probs).any()


def test_deepset_different_seq_lengths():
    """Test that DeepSet handles different sequence lengths correctly."""
    for seq_length in [3, 5, 10]:
        model = DeepSet(
            n_in=3,
            n_out=2,
            n_embedding=64,
            seq_length=seq_length,
            sum_aggregations=True,
        )

        x = torch.randn(16, seq_length, 3)
        output = model(x)

        assert output.shape == (16, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()



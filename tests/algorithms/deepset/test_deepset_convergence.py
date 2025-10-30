"""
Convergence tests for DeepSet on hierarchical Gaussian Linear task.

Tests that DeepSet actually learns to recover global and local
parameters on a controlled synthetic task.
"""

import torch
import torch.nn as nn
from sbibm.algorithms.deepset import DeepSet
from sbibm.algorithms.deepset.training import (
    train_hierarchical_deepset,
)


def test_deepset_global_parameter_recovery(
    hierarchical_gaussian_linear_data,
):
    """
    Test that DeepSet recovers global parameter (σ) accurately.

    Creates training data, trains global model, and verifies
    MSE on test set is reasonable.
    """
    create_data = hierarchical_gaussian_linear_data

    # Create training data
    x_train, y_global_train, _ = create_data(
        num_datasets=1000,
        num_events=5,
        dim_per_event=3,
        seed=42,
    )

    # Create test data
    x_test, y_global_test, _ = create_data(
        num_datasets=100,
        num_events=5,
        dim_per_event=3,
        seed=43,
    )

    # Create global model
    global_model = DeepSet(
        n_in=3,
        n_out=2,
        n_embedding=64,
        seq_length=5,
        sum_aggregations=True,
    )

    # Create dummy local models (not trained for this test)
    local_models = nn.ModuleList([
        DeepSet(
            n_in=3, n_out=2, n_embedding=64, seq_length=5,
            sum_aggregations=True
        )
        for _ in range(5)
    ])

    # Train
    train_hierarchical_deepset(
        global_model=global_model,
        local_models=local_models,
        x_set=x_train,
        y_global=y_global_train,
        y_local=torch.randn(x_train.shape[0], 5, 1),
        num_epochs=20,
        batch_size=64,
        learning_rate=1e-3,
    )

    # Evaluate on test set
    with torch.no_grad():
        global_pred = global_model(x_test)
        pred_mu, pred_log_sigma = torch.chunk(global_pred, 2, -1)
        pred_mu = pred_mu.squeeze(-1)

    # Compute MSE
    mse = ((pred_mu - y_global_test) ** 2).mean().item()

    # For Gamma(2, 2) prior (mean ≈ 1, variance ≈ 0.5),
    # we expect reasonable prediction error
    assert mse < 0.3, (
        f"Global parameter MSE too high: {mse:.4f} (expected < 0.3)"
    )


def test_deepset_local_parameter_recovery(
    hierarchical_gaussian_linear_data,
):
    """
    Test that DeepSet recovers local parameters (μ_i) accurately.

    Creates training data, trains local models, and verifies
    MSE on test set is reasonable.
    """
    create_data = hierarchical_gaussian_linear_data

    # Create training data
    x_train, _, y_local_train = create_data(
        num_datasets=1000,
        num_events=5,
        dim_per_event=3,
        seed=42,
    )

    # Create test data
    x_test, _, y_local_test = create_data(
        num_datasets=100,
        num_events=5,
        dim_per_event=3,
        seed=43,
    )

    # Create dummy global model (not trained for this test)
    global_model = DeepSet(
        n_in=3, n_out=2, n_embedding=64, seq_length=5,
        sum_aggregations=True
    )

    # Create local models
    num_events = 5
    local_models = nn.ModuleList([
        DeepSet(
            n_in=3, n_out=2, n_embedding=64, seq_length=5,
            sum_aggregations=True
        )
        for _ in range(num_events)
    ])

    # Train
    train_hierarchical_deepset(
        global_model=global_model,
        local_models=local_models,
        x_set=x_train,
        y_global=torch.randn(x_train.shape[0], 1),
        y_local=y_local_train,
        num_epochs=20,
        batch_size=64,
        learning_rate=1e-3,
    )

    # Evaluate on test set
    seq_length = 5
    mse_per_event = []

    with torch.no_grad():
        for event_idx in range(num_events):
            x_event = x_test[:, event_idx:event_idx + 1, :]
            x_event_padded = torch.zeros(
                x_event.shape[0], seq_length, x_event.shape[2]
            )
            x_event_padded[:, 0, :] = x_event.squeeze(1)

            local_pred = local_models[event_idx](x_event_padded)
            pred_mu, _ = torch.chunk(local_pred, 2, -1)
            pred_mu = pred_mu.squeeze(-1)

            event_mse = (
                (pred_mu - y_local_test[:, event_idx]) ** 2
            ).mean().item()
            mse_per_event.append(event_mse)

    # Average MSE across events
    mean_mse = sum(mse_per_event) / len(mse_per_event)

    # For N(0, 1) prior, expect reasonable prediction error
    assert mean_mse < 0.5, (
        f"Local parameter MSE too high: {mean_mse:.4f} "
        f"(expected < 0.5)"
    )

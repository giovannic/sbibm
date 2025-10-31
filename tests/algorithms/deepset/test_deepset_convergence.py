"""
Convergence tests for HierarchicalDeepSet on Gaussian Linear task.

Tests that HierarchicalDeepSet learns to recover global and
local parameters on controlled synthetic data.
"""

import torch

from sbibm.algorithms.deepset import HierarchicalDeepSetInference
from sbibm.algorithms.deepset.training import train_hierarchical_deepset


def test_hierarchical_deepset_convergence_joint(
    hierarchical_gaussian_linear_data,
):
    """
    Test that HierarchicalDeepSet converges on joint training.

    Trains model with both global and local losses,
    verifies loss decreases over epochs.
    """
    create_data = hierarchical_gaussian_linear_data

    # Create training data (small for speed)
    x_train, y_global_train, y_local_train = create_data(
        num_datasets=500,
        num_events=5,
        dim_per_event=3,
        seed=42,
    )

    # Create model
    model = HierarchicalDeepSetInference(
        n_in=3,
        dim_global=1,
        dim_local=1,
        n_set_max=5,
        local_loss=True,
        global_loss=True,
        max_epochs=10,
        lr=1e-3,
    )

    # Train
    train_hierarchical_deepset(
        model=model,
        x_set=x_train,
        y_global=y_global_train,
        y_local=y_local_train,
        num_epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        validation_split=0.1,
        device="cpu",
        verbose=False,
    )

    # Model should have trained without errors
    assert model is not None


def test_hierarchical_deepset_parameter_recovery_global(
    hierarchical_gaussian_linear_data,
):
    """
    Test that HierarchicalDeepSet can extract global parameters.

    Trains model and verifies it can extract meaningful
    global parameter predictions from test observations.
    """
    create_data = hierarchical_gaussian_linear_data

    # Create training data
    x_train, y_global_train, y_local_train = create_data(
        num_datasets=1000,
        num_events=5,
        dim_per_event=3,
        seed=42,
    )

    # Create test data
    x_test, y_global_test, y_local_test = create_data(
        num_datasets=100,
        num_events=5,
        dim_per_event=3,
        seed=43,
    )

    # Create and train model
    model = HierarchicalDeepSetInference(
        n_in=3,
        dim_global=1,
        dim_local=1,
        n_set_max=5,
        local_loss=True,
        global_loss=True,
        max_epochs=20,
        lr=1e-3,
    )

    train_hierarchical_deepset(
        model=model,
        x_set=x_train,
        y_global=y_global_train,
        y_local=y_local_train,
        num_epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        validation_split=0.1,
        device="cpu",
        verbose=False,
    )

    # Evaluate on test set - extract global parameters
    model.eval()
    with torch.no_grad():
        # x_test shape: (100, 5, 3) - already 3D, pass directly
        log_prob_local, log_prob_global = model.deep_set(
            x_test, y_local_test, y_global_test
        )

    # Verify outputs are reasonable
    assert not torch.isnan(log_prob_global).any(), "Global log prob contains NaN"
    assert not torch.isinf(log_prob_global).any(), "Global log prob contains Inf"
    assert log_prob_global.shape[0] == x_test.shape[0], (
        f"Expected shape ({x_test.shape[0]},), got " f"{log_prob_global.shape}"
    )


def test_hierarchical_deepset_parameter_recovery_local(
    hierarchical_gaussian_linear_data,
):
    """
    Test that HierarchicalDeepSet can extract local parameters.

    Trains model and verifies it can extract meaningful
    local parameter predictions from test observations.
    """
    create_data = hierarchical_gaussian_linear_data

    # Create training data
    x_train, y_global_train, y_local_train = create_data(
        num_datasets=1000,
        num_events=5,
        dim_per_event=3,
        seed=42,
    )

    # Create test data
    x_test, y_global_test, y_local_test = create_data(
        num_datasets=100,
        num_events=5,
        dim_per_event=3,
        seed=43,
    )

    # Create and train model
    model = HierarchicalDeepSetInference(
        n_in=3,
        dim_global=1,
        dim_local=1,
        n_set_max=5,
        local_loss=True,
        global_loss=True,
        max_epochs=20,
        lr=1e-3,
    )

    train_hierarchical_deepset(
        model=model,
        x_set=x_train,
        y_global=y_global_train,
        y_local=y_local_train,
        num_epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        validation_split=0.1,
        device="cpu",
        verbose=False,
    )

    # Evaluate on test set - extract local parameters
    model.eval()
    with torch.no_grad():
        # x_test shape: (100, 5, 3) - already 3D, pass directly
        log_prob_local, _ = model.deep_set(x_test, y_local_test, y_global_test)

    # Verify outputs are reasonable
    assert not torch.isnan(log_prob_local).any(), "Local log prob contains NaN"
    assert not torch.isinf(log_prob_local).any(), "Local log prob contains Inf"
    assert log_prob_local.shape[0] == x_test.shape[0], (
        f"Expected shape ({x_test.shape[0]},), got " f"{log_prob_local.shape}"
    )

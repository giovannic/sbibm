"""
Functional tests for HierarchicalDeepSet architecture.

Tests that the extracted HierarchicalDeepSet code works correctly with
hierarchical Gaussian Linear data structures.
"""

import pytest
import torch
import torch.nn as nn

from sbibm.algorithms.deepset import (
    HierarchicalDeepSet,
    HierarchicalDeepSetInference,
)


def test_hierarchical_deepset_instantiation():
    """Test that HierarchicalDeepSet can be instantiated."""
    model = HierarchicalDeepSet(
        dim_hidden=64,
        condition_local_on_global=True,
        n_set_max=5,
    )
    assert model is not None
    assert isinstance(model, nn.Module)


def test_hierarchical_deepset_forward_pass():
    """Test that HierarchicalDeepSet forward pass produces correct shapes."""
    model = HierarchicalDeepSet(
        dim_hidden=64,
        condition_local_on_global=True,
        n_set_max=5,
    )

    # Create batch of data
    batch_size = 4
    n_set = 5
    h, w = 64, 64  # Image dimensions
    dim_global = 2  # Global parameters
    dim_local = 4  # Local parameters

    x = torch.randn(batch_size, n_set, h, w)
    y_local = torch.randn(batch_size, n_set, dim_local)
    y_global = torch.randn(batch_size, dim_global)

    # Forward pass
    log_prob_local, log_prob_global = model(x, y_local, y_global)

    # Check output shapes
    assert log_prob_local.shape == (batch_size,), (
        f"Expected log_prob_local shape ({batch_size},), "
        f"got {log_prob_local.shape}"
    )
    assert log_prob_global.shape == (batch_size,), (
        f"Expected log_prob_global shape ({batch_size},), "
        f"got {log_prob_global.shape}"
    )


def test_hierarchical_deepset_finite_outputs():
    """Test that HierarchicalDeepSet produces finite outputs (no NaN/Inf)."""
    model = HierarchicalDeepSet(
        dim_hidden=64,
        condition_local_on_global=True,
        n_set_max=5,
    )

    batch_size = 4
    x = torch.randn(batch_size, 5, 64, 64)
    y_local = torch.randn(batch_size, 5, 4)
    y_global = torch.randn(batch_size, 2)

    log_prob_local, log_prob_global = model(x, y_local, y_global)

    # Check for NaN and Inf
    assert not torch.isnan(log_prob_local).any(), "log_prob_local contains NaN"
    assert not torch.isinf(log_prob_local).any(), "log_prob_local contains Inf"
    assert not torch.isnan(
        log_prob_global
    ).any(), "log_prob_global contains NaN"
    assert not torch.isinf(
        log_prob_global
    ).any(), "log_prob_global contains Inf"


def test_hierarchical_deepset_loss_computation():
    """
    Test loss computation with joint global and local losses.

    Verifies:
    - HierarchicalDeepSetInference instantiates correctly
    - Forward pass computes both local and global log-probs
    - Loss computation works with combined local+global losses
    - Outputs remain finite
    """
    # Create small synthetic hierarchical data
    batch_size = 4
    n_set = 5
    h, w = 64, 64
    dim_global = 2
    dim_local = 4

    x = torch.randn(batch_size, n_set, h, w)
    y_local = torch.randn(batch_size, n_set, dim_local)
    y_global = torch.randn(batch_size, dim_global)

    # Create Lightning module for hierarchical inference
    model = HierarchicalDeepSetInference(
        n_set_max=n_set,
        local_loss=True,
        global_loss=True,
        max_epochs=5,
        lr=1e-3,
    )

    # Forward pass
    log_prob_local, log_prob_global = model(x, y_local, y_global)

    # Compute loss as done in training_step
    log_prob = torch.zeros_like(log_prob_local).to(log_prob_local.device)
    log_prob += log_prob_local
    log_prob += log_prob_global
    loss = -log_prob.mean()

    # Check for finite loss
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert loss.item() > 0, "Loss should be positive"


@pytest.mark.parametrize("n_set_max", [3, 5, 10])
def test_hierarchical_deepset_different_set_sizes(n_set_max):
    """Test that HierarchicalDeepSet handles different set sizes."""
    model = HierarchicalDeepSet(
        dim_hidden=64,
        condition_local_on_global=True,
        n_set_max=n_set_max,
    )

    batch_size = 2
    x = torch.randn(batch_size, n_set_max, 64, 64)
    y_local = torch.randn(batch_size, n_set_max, 4)
    y_global = torch.randn(batch_size, 2)

    log_prob_local, log_prob_global = model(x, y_local, y_global)

    assert log_prob_local.shape == (batch_size,)
    assert log_prob_global.shape == (batch_size,)
    assert not torch.isnan(log_prob_local).any()
    assert not torch.isinf(log_prob_local).any()
    assert not torch.isnan(log_prob_global).any()
    assert not torch.isinf(log_prob_global).any()

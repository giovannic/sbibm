"""Tests for hierarchical task utilities."""

import pytest
import torch
from pyro import distributions as pdist

from sbibm.tasks.hierarchical_utilities import (
    BlockwiseDistribution,
    HierarchicalDistribution,
)


def test_blockwise_distribution_construction():
    """Test that BlockwiseDistribution can be constructed."""
    # Create mixed distributions: 2D Uniform + 2D HalfNormal
    dist1 = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)
    dist2 = pdist.HalfNormal(0.5).expand([2]).to_event(1)

    blockwise_dist = BlockwiseDistribution([dist1, dist2])

    assert blockwise_dist.event_shape == torch.Size([4])


def test_blockwise_distribution_sample_shape():
    """Test that BlockwiseDistribution sampling returns correct shapes."""
    dist1 = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)
    dist2 = pdist.HalfNormal(0.5).expand([3]).to_event(1)

    blockwise_dist = BlockwiseDistribution([dist1, dist2])

    # Sample without sample_shape
    sample = blockwise_dist.sample()
    assert sample.shape == torch.Size([5])

    # Sample with sample_shape
    samples = blockwise_dist.sample(torch.Size([100]))
    assert samples.shape == torch.Size([100, 5])


def test_blockwise_distribution_log_prob():
    """Test log_prob computation for BlockwiseDistribution."""
    dist1 = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)
    dist2 = pdist.Normal(0.0, 1.0).expand([2]).to_event(1)

    blockwise_dist = BlockwiseDistribution([dist1, dist2])

    # Test value: [0.5, -0.5, 0.0, 0.0]
    value = torch.tensor([0.5, -0.5, 0.0, 0.0])

    log_p = blockwise_dist.log_prob(value)

    # Manually compute expected
    log_p1 = dist1.log_prob(value[:2])
    log_p2 = dist2.log_prob(value[2:])
    expected = log_p1 + log_p2

    assert torch.allclose(log_p, expected)


def test_blockwise_distribution_sample_no_nan():
    """Test that BlockwiseDistribution samples contain no NaN."""
    dist1 = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)
    dist2 = pdist.HalfNormal(0.5).expand([2]).to_event(1)

    blockwise_dist = BlockwiseDistribution([dist1, dist2])

    samples = blockwise_dist.sample(torch.Size([50]))
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


def test_hierarchical_distribution_construction():
    """Test that HierarchicalDistribution can be constructed."""
    # Global: 2D Uniform
    global_dist = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)

    # Local: depends on global, 4D Normal
    def local_dist_fn(global_params):
        batch_shape = global_params.shape[:-1]
        return pdist.Normal(0.0, 1.0).expand(list(batch_shape) + [4]).to_event(1)

    hier_dist = HierarchicalDistribution(
        global_dist, local_dist_fn, dim_global=2, dim_local=4
    )

    assert hier_dist.event_shape == torch.Size([6])


def test_hierarchical_distribution_sample_shape():
    """Test that sampling returns correct shapes."""
    # Global: 2D Uniform
    global_dist = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)

    # Local: 4D Normal conditioned on global
    def local_dist_fn(global_params):
        batch_shape = global_params.shape[:-1]
        return pdist.Normal(0.0, 1.0).expand(list(batch_shape) + [4]).to_event(1)

    hier_dist = HierarchicalDistribution(
        global_dist, local_dist_fn, dim_global=2, dim_local=4
    )

    # Sample without sample_shape
    sample = hier_dist.sample()
    assert sample.shape == torch.Size([6])

    # Sample with sample_shape
    samples = hier_dist.sample(torch.Size([100]))
    assert samples.shape == torch.Size([100, 6])


def test_hierarchical_distribution_sample_no_nan():
    """Test that samples contain no NaN values."""
    global_dist = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)

    def local_dist_fn(global_params):
        batch_shape = global_params.shape[:-1]
        return pdist.Normal(0.0, 1.0).expand(list(batch_shape) + [4]).to_event(1)

    hier_dist = HierarchicalDistribution(
        global_dist, local_dist_fn, dim_global=2, dim_local=4
    )

    samples = hier_dist.sample(torch.Size([50]))
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


def test_hierarchical_distribution_log_prob():
    """Test log_prob computation."""
    # Global: 2D Uniform(-1, 1)
    global_dist = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)

    # Local: 3D Normal(0, 1)
    def local_dist_fn(global_params):
        batch_shape = global_params.shape[:-1]
        return pdist.Normal(0.0, 1.0).expand(list(batch_shape) + [3]).to_event(1)

    hier_dist = HierarchicalDistribution(
        global_dist, local_dist_fn, dim_global=2, dim_local=3
    )

    # Create test values
    value = torch.tensor([0.5, -0.5, 0.0, 0.0, 0.0])  # 2 global + 3 local

    # Compute log prob
    log_p = hier_dist.log_prob(value)

    # Manually compute expected log prob
    global_params = value[:2]
    local_params = value[2:]

    log_p_global = global_dist.log_prob(global_params)
    log_p_local = pdist.Normal(0.0, 1.0).expand([3]).to_event(1).log_prob(local_params)
    expected_log_p = log_p_global + log_p_local

    assert torch.allclose(log_p, expected_log_p)


def test_hierarchical_distribution_log_prob_batch():
    """Test log_prob with batched values."""
    global_dist = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)

    def local_dist_fn(global_params):
        batch_shape = global_params.shape[:-1]
        return pdist.Normal(0.0, 1.0).expand(list(batch_shape) + [3]).to_event(1)

    hier_dist = HierarchicalDistribution(
        global_dist, local_dist_fn, dim_global=2, dim_local=3
    )

    # Batch of values
    values = torch.randn(10, 5)  # 10 samples, 5 dims (2 global + 3 local)

    # Ensure global params are in valid range
    values[:, :2] = torch.clamp(values[:, :2], -0.9, 0.9)

    log_probs = hier_dist.log_prob(values)

    assert log_probs.shape == torch.Size([10])
    assert not torch.isnan(log_probs).any()


def test_hierarchical_distribution_expand():
    """Test expand method."""
    global_dist = pdist.Uniform(-1.0, 1.0).expand([2]).to_event(1)

    def local_dist_fn(global_params):
        batch_shape = global_params.shape[:-1]
        return pdist.Normal(0.0, 1.0).expand(list(batch_shape) + [3]).to_event(1)

    hier_dist = HierarchicalDistribution(
        global_dist, local_dist_fn, dim_global=2, dim_local=3
    )

    # Expand batch shape
    expanded = hier_dist.expand(torch.Size([10]))

    assert expanded.batch_shape == torch.Size([10])
    assert expanded.event_shape == torch.Size([5])

    # Sample from expanded
    samples = expanded.sample()
    assert samples.shape == torch.Size([10, 5])

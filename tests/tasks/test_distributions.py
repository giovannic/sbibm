"""Tests for custom distributions and hierarchical task utilities."""

import pytest
import torch
from pyro import distributions as pdist

from sbibm.tasks.distributions import (
    BlockwiseDistribution,
    HierarchicalDistribution,
    TruncatedNormal,
)

# ============================================================================
# TruncatedNormal Tests
# ============================================================================


def test_truncated_normal_construction():
    """Test that TruncatedNormal can be constructed."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    dist = TruncatedNormal(loc, scale, low=-1.0, high=1.0)

    assert dist is not None
    assert dist.low == -1.0
    assert dist.high == 1.0


def test_truncated_normal_invalid_bounds():
    """Test that construction fails when low >= high."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)

    # Should raise ValueError when low >= high
    with pytest.raises(ValueError, match="Lower bound must be less than"):
        TruncatedNormal(loc, scale, low=1.0, high=-1.0)

    with pytest.raises(ValueError, match="Lower bound must be less than"):
        TruncatedNormal(loc, scale, low=0.0, high=0.0)


def test_truncated_normal_sample_shape():
    """Test that TruncatedNormal sampling returns correct shapes."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    dist = TruncatedNormal(loc, scale, low=-2.0, high=2.0)

    # Sample without sample_shape
    sample = dist.sample()
    assert sample.shape == torch.Size([])

    # Sample with sample_shape
    samples = dist.sample(torch.Size([100]))
    assert samples.shape == torch.Size([100])


def test_truncated_normal_samples_in_bounds():
    """Test that all samples are within [low, high]."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    low = -1.5
    high = 2.0
    dist = TruncatedNormal(loc, scale, low=low, high=high)

    samples = dist.sample(torch.Size([1000]))

    assert samples.min() >= low
    assert samples.max() <= high
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


def test_truncated_normal_log_prob():
    """Test log_prob has correct ordering (center more likely than tails)."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    dist = TruncatedNormal(loc, scale, low=-1.0, high=1.0)

    # Test values at different quartiles
    # Center should be more likely than edges
    center = torch.tensor(0.0)
    q1 = torch.tensor(-0.5)
    q3 = torch.tensor(0.5)

    log_p_center = dist.log_prob(center)
    log_p_q1 = dist.log_prob(q1)
    log_p_q3 = dist.log_prob(q3)

    # Center should have higher log prob than quartiles
    assert log_p_center > log_p_q1
    assert log_p_center > log_p_q3

    # All should be finite
    assert torch.isfinite(log_p_center)
    assert torch.isfinite(log_p_q1)
    assert torch.isfinite(log_p_q3)


def test_truncated_normal_log_prob_out_of_bounds():
    """Test log_prob returns -inf for values outside bounds."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    dist = TruncatedNormal(loc, scale, low=-1.0, high=1.0)

    # Values outside bounds
    value_low = torch.tensor(-2.0)
    value_high = torch.tensor(2.0)

    log_p_low = dist.log_prob(value_low)
    log_p_high = dist.log_prob(value_high)

    assert torch.isneginf(log_p_low)
    assert torch.isneginf(log_p_high)


def test_truncated_normal_normalization():
    """Test that distribution approximately integrates to 1."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    dist = TruncatedNormal(loc, scale, low=-1.0, high=1.0)

    # Numerical integration using many samples
    # Sum of exp(log_prob) * dx should be close to 1
    x = torch.linspace(-1.0, 1.0, 1000)
    dx = (x[1] - x[0]).item()

    log_probs = dist.log_prob(x)
    integral = torch.exp(log_probs).sum() * dx

    # Should be close to 1 (within 1% tolerance)
    assert torch.abs(integral - 1.0) < 0.01


def test_truncated_normal_batch_mode():
    """Test TruncatedNormal with batched loc and scale."""
    # Batch of 3 different loc/scale pairs
    loc = torch.tensor([0.0, 0.5, -0.5])
    scale = torch.tensor([1.0, 0.5, 1.5])
    dist = TruncatedNormal(loc, scale, low=-1.0, high=1.0)

    assert dist.batch_shape == torch.Size([3])

    # Sample should have batch dimension
    sample = dist.sample()
    assert sample.shape == torch.Size([3])

    # All samples should be in bounds
    assert (sample >= -1.0).all()
    assert (sample <= 1.0).all()

    # log_prob should work on batched values
    values = torch.tensor([0.0, 0.5, -0.5])
    log_probs = dist.log_prob(values)
    assert log_probs.shape == torch.Size([3])
    assert torch.isfinite(log_probs).all()


def test_truncated_normal_no_nan():
    """Test that samples contain no NaN or Inf values."""
    loc = torch.tensor([0.0, 1.0, -1.0])
    scale = torch.tensor([0.5, 1.0, 0.3])
    dist = TruncatedNormal(loc, scale, low=-2.0, high=2.0)

    samples = dist.sample(torch.Size([100]))

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


def test_truncated_normal_expand():
    """Test expand method."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    dist = TruncatedNormal(loc, scale, low=-1.0, high=1.0)

    # Expand to batch shape [5]
    expanded = dist.expand(torch.Size([5]))

    assert expanded.batch_shape == torch.Size([5])

    # Sample from expanded distribution
    samples = expanded.sample()
    assert samples.shape == torch.Size([5])
    assert (samples >= -1.0).all()
    assert (samples <= 1.0).all()


def test_truncated_normal_edge_cases():
    """Test edge cases like tight bounds."""
    loc = torch.tensor(0.0)
    scale = torch.tensor(0.1)  # Small scale
    dist = TruncatedNormal(loc, scale, low=-0.05, high=0.05)

    samples = dist.sample(torch.Size([100]))

    # All samples should be tightly bounded
    assert samples.min() >= -0.05
    assert samples.max() <= 0.05

    # Should have reasonable mean near loc
    assert torch.abs(samples.mean()) < 0.05


def test_truncated_normal_bijection_respects_constraints():
    """Test that biject_to respects TruncatedNormal support bounds.

    When we apply biject_to(dist.support) and then its inverse, samples
    should remain within the distribution's support bounds.
    """
    from pyro.distributions.transforms import biject_to

    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    low = -1.0
    high = 1.0
    dist = TruncatedNormal(loc, scale, low=low, high=high)

    # Get the bijection for this distribution's support
    transform = biject_to(dist.support)

    # Sample from the distribution (should be in [low, high])
    samples_constrained = dist.sample(torch.Size([100]))

    # Verify samples are within bounds
    assert (samples_constrained >= low).all()
    assert (samples_constrained <= high).all()

    # Transform to unconstrained space
    samples_unconstrained = transform(samples_constrained)

    # Transform back to constrained space
    samples_back = transform.inv(samples_unconstrained)

    # Samples should still be within bounds after round-trip
    assert (samples_back >= low).all(), \
        f"Samples below bound: min={samples_back.min()}, low={low}"
    assert (samples_back <= high).all(), \
        f"Samples above bound: max={samples_back.max()}, high={high}"

    # Should be close to original (within numerical precision)
    assert torch.allclose(samples_constrained, samples_back, atol=1e-5)


# ============================================================================
# BlockwiseDistribution Tests
# ============================================================================


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


# ============================================================================
# HierarchicalDistribution Tests
# ============================================================================


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

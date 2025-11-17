"""Utilities for hierarchical task implementations and custom distributions."""

from typing import Callable, List

import torch
from pyro.distributions.torch_distribution import TorchDistributionMixin


class SummedStackTransform(torch.distributions.transforms.StackTransform):
    """StackTransform that sums Jacobians across dimensions.

    The base StackTransform returns per-dimension Jacobians, but some
    code (like FlowWrapper) expects a scalar Jacobian per batch element.
    This is useful for hierarchical tasks that need composite transforms.
    """

    def log_abs_det_jacobian(self, x, y):
        """Compute log abs det Jacobian, summing across dimensions."""
        jac_per_dim = super().log_abs_det_jacobian(x, y)
        return jac_per_dim.sum(dim=-1)


class TruncatedNormal(torch.distributions.Distribution, TorchDistributionMixin):
    """Truncated Normal distribution using CDF-based methods.

    Implements a Normal distribution truncated to [low, high] using:
    - Inverse CDF sampling for sample()
    - CDF normalization for log_prob()

    Args:
        loc: Mean of the underlying Normal distribution
        scale: Standard deviation of the underlying Normal distribution
        low: Lower bound of truncation
        high: Upper bound of truncation
    """

    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
        "low": torch.distributions.constraints.real,
        "high": torch.distributions.constraints.real,
    }
    has_rsample = False

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float,
        high: float,
        validate_args=None,
    ):
        """Initialize truncated normal distribution.

        Args:
            loc: Mean parameter (can be batched)
            scale: Scale parameter (can be batched)
            low: Lower truncation bound (scalar)
            high: Upper truncation bound (scalar)
            validate_args: Whether to validate arguments

        Raises:
            ValueError: If low >= high
        """
        if low >= high:
            raise ValueError(
                f"Lower bound must be less than upper bound, "
                f"got low={low}, high={high}"
            )

        self.loc = loc
        self.scale = scale
        self.low = torch.tensor(low, dtype=loc.dtype, device=loc.device)
        self.high = torch.tensor(high, dtype=loc.dtype, device=loc.device)

        # Base normal distribution
        self.base_dist = torch.distributions.Normal(loc, scale)

        # Compute CDF values at bounds for normalization
        self._alpha = self.base_dist.cdf(self.low)  # CDF(low)
        self._beta = self.base_dist.cdf(self.high)  # CDF(high)
        self._Z = self._beta - self._alpha  # Normalization constant

        # Determine batch and event shape
        batch_shape = torch.broadcast_shapes(
            torch.as_tensor(loc).shape, torch.as_tensor(scale).shape
        )

        super().__init__(batch_shape, torch.Size(), validate_args=validate_args)

    @property
    def support(self):
        """Return the support of the distribution."""
        return torch.distributions.constraints.interval(
            self.low.item(), self.high.item()
        )

    def sample(self, sample_shape=torch.Size()):
        """Sample using inverse CDF method.

        Sample u ~ Uniform(0, 1), then compute:
        x = ICDF(u * (CDF(high) - CDF(low)) + CDF(low))

        Args:
            sample_shape: Shape of samples to generate

        Returns:
            Samples from truncated normal
        """
        # Sample from uniform
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)

        # Transform to CDF range [alpha, beta]
        cdf_values = u * self._Z + self._alpha

        # Apply inverse CDF (ICDF = mean + std * sqrt(2) * erf_inv(2*p - 1))
        # For Normal: ICDF(p) = loc + scale * sqrt(2) * erfinv(2*p - 1)
        samples = self.loc + self.scale * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(
            2 * cdf_values - 1
        )

        return samples

    def log_prob(self, value):
        """Compute log probability using normalization.

        log p(x) = log N(x | loc, scale) - log Z
        where Z = CDF(high) - CDF(low)

        Args:
            value: Values to compute log probability for

        Returns:
            Log probabilities (or -inf for out-of-bounds values)
        """
        # Check bounds
        in_bounds = (value >= self.low) & (value <= self.high)

        # Compute base log prob
        log_prob_base = self.base_dist.log_prob(value)

        # Subtract normalization constant
        log_prob_normalized = log_prob_base - torch.log(self._Z)

        # Set out-of-bounds values to -inf
        result = torch.where(
            in_bounds, log_prob_normalized, torch.tensor(float("-inf"))
        )

        return result

    def expand(self, batch_shape, _instance=None):
        """Expand the distribution batch shape.

        Args:
            batch_shape: New batch shape
            _instance: Instance to use for expansion (internal)

        Returns:
            Expanded distribution
        """
        new = self._get_checked_instance(TruncatedNormal, _instance)
        batch_shape = torch.Size(batch_shape)

        # Expand loc and scale
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.low = self.low
        new.high = self.high

        # Recreate base dist and normalization constants
        new.base_dist = torch.distributions.Normal(new.loc, new.scale)
        new._alpha = new.base_dist.cdf(new.low)
        new._beta = new.base_dist.cdf(new.high)
        new._Z = new._beta - new._alpha

        super(TruncatedNormal, new).__init__(
            batch_shape, torch.Size(), validate_args=False
        )
        return new


class BlockwiseDistribution(torch.distributions.Distribution, TorchDistributionMixin):
    """Distribution composed of independent blocks with different distributions.

    Concatenates samples and sums log probabilities from multiple independent
    distributions. Useful when different parameter dimensions have different
    distribution types (e.g., some Uniform, some HalfNormal).

    Args:
        distributions: List of independent distributions to concatenate
    """

    def __init__(self, distributions: List[torch.distributions.Distribution]):
        """Initialize blockwise distribution.

        Args:
            distributions: List of distributions, each with event_shape [dim_i]
        """
        self.distributions = distributions

        # Compute total event shape
        total_dim = sum(d.event_shape[0] for d in distributions)
        event_shape = torch.Size([total_dim])

        # Use batch shape from first distribution (assume all match)
        batch_shape = distributions[0].batch_shape

        super().__init__(batch_shape, event_shape, validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        """Sample from each distribution and concatenate.

        Args:
            sample_shape: Shape of samples to generate

        Returns:
            Concatenated samples with shape sample_shape + batch_shape +
            [total_dim]
        """
        samples = [d.sample(sample_shape) for d in self.distributions]
        return torch.cat(samples, dim=-1)

    def log_prob(self, value):
        """Compute log probability as sum of independent log probabilities.

        Args:
            value: Parameter tensor with shape [..., total_dim]

        Returns:
            Log probability with shape [...]
        """
        log_probs = []
        offset = 0

        for dist in self.distributions:
            dim = dist.event_shape[0]
            block_value = value[..., offset:offset + dim]
            log_probs.append(dist.log_prob(block_value))
            offset += dim

        return torch.stack(log_probs, dim=0).sum(dim=0)

    def expand(self, batch_shape, _instance=None):
        """Expand the distribution batch shape.

        Args:
            batch_shape: New batch shape
            _instance: Instance to use for expansion (internal)

        Returns:
            Expanded distribution
        """
        new = self._get_checked_instance(BlockwiseDistribution, _instance)
        new.distributions = [d.expand(batch_shape) for d in self.distributions]
        super(BlockwiseDistribution, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        return new


class HierarchicalDistribution(
    torch.distributions.Distribution, TorchDistributionMixin
):
    """Distribution for hierarchical models with dependent parameters.

    This distribution handles cases where some parameters (local) depend on
    other parameters (global). It properly implements both sampling and
    log_prob computation for the joint distribution.

    The joint distribution is: p(global, local) = p(global) * p(local|global)

    Args:
        global_dist: Distribution over global parameters
        local_dist_fn: Function taking global params (shape [..., dim_global]),
            returning distribution over local params
        dim_global: Dimensionality of global parameters
        dim_local: Dimensionality of local parameters
    """

    def __init__(
        self,
        global_dist: torch.distributions.Distribution,
        local_dist_fn: Callable[[torch.Tensor, int], torch.distributions.Distribution],
        dim_global: int,
        dim_local: int,
        default_n_local: int,
    ):
        """Initialize hierarchical distribution.

        Args:
            global_dist: Distribution over global parameters
            local_dist_fn: Function taking global params and n_local,
                returning distribution over local params
            dim_global: Dimensionality of global parameters
            dim_local: Dimensionality of local parameters when sampled
                with default n_local
            default_n_local: Default number of local groups to use for
                sample() when n_local is not specified
        """
        self.global_dist = global_dist
        self.local_dist_fn = local_dist_fn
        self.dim_global = dim_global
        self.dim_local = dim_local
        self.default_n_local = default_n_local

        batch_shape = global_dist.batch_shape
        event_shape = torch.Size([dim_global + dim_local])
        super().__init__(batch_shape, event_shape, validate_args=False)

    def sample_n_local(
        self, n_local, sample_shape=torch.Size()
    ):
        """Sample from the hierarchical distribution with variable local
        dimensions.

        First samples global parameters, then samples local
        parameters conditioned on the global parameters, where the
        number of local groups is determined by n_local.

        Args:
            n_local: Number of local groups to sample
            sample_shape: Shape of samples to generate (default: empty)

        Returns:
            Samples with shape sample_shape + batch_shape +
            [dim_global + local_event_dim]
            where local_event_dim depends on the local_dist_fn output
        """
        # Sample global parameters
        global_params = self.global_dist.sample(sample_shape)

        # Sample local parameters conditioned on global
        # Pass n_local to local_dist_fn
        local_dist = self.local_dist_fn(global_params, n_local)
        local_params = local_dist.sample()

        # Concatenate global and local parameters
        return torch.cat([global_params, local_params], dim=-1)

    def sample(self, sample_shape=torch.Size()):
        """Sample from the hierarchical distribution with default n_local.

        First samples global parameters, then samples local
        parameters conditioned on the global parameters, using the
        default n_local value specified at initialization.

        Args:
            sample_shape: Shape of samples to generate (default: empty)

        Returns:
            Samples with shape sample_shape + batch_shape +
            [dim_global + dim_local]
        """
        # Use default_n_local for standard sampling interface
        return self.sample_n_local(self.default_n_local, sample_shape)

    def log_prob(self, value):
        """Compute log probability of the joint distribution.

        log p(global, local) = log p(global) + log p(local |
        global)

        Args:
            value: Parameter tensor with shape [..., dim_global +
                dim_local]

        Returns:
            Log probability with shape [...]
        """
        # Split value into global and local components
        global_params = value[..., :self.dim_global]
        local_params = value[..., self.dim_global:]

        # Compute log p(global)
        log_prob_global = self.global_dist.log_prob(global_params)

        # Compute log p(local | global)
        local_dist = self.local_dist_fn(
            global_params, self.default_n_local
        )
        log_prob_local = local_dist.log_prob(local_params)

        # Return joint log probability
        return log_prob_global + log_prob_local

    def expand(self, batch_shape, _instance=None):
        """Expand the distribution batch shape.

        Args:
            batch_shape: New batch shape
            _instance: Instance to use for expansion (internal)

        Returns:
            Expanded distribution
        """
        new = self._get_checked_instance(HierarchicalDistribution, _instance)
        new.global_dist = self.global_dist.expand(batch_shape)
        new.local_dist_fn = self.local_dist_fn
        new.dim_global = self.dim_global
        new.dim_local = self.dim_local
        super(HierarchicalDistribution, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        return new

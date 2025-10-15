"""Utilities for hierarchical task implementations."""

from typing import Callable, List

import torch
from pyro.distributions.torch_distribution import TorchDistributionMixin


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
            block_value = value[..., offset : offset + dim]
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
        local_dist_fn: Callable[[torch.Tensor], torch.distributions.Distribution],
        dim_global: int,
        dim_local: int,
    ):
        """Initialize hierarchical distribution.

        Args:
            global_dist: Distribution over global parameters
            local_dist_fn: Function taking global params, returning
                distribution over local params
            dim_global: Dimensionality of global parameters
            dim_local: Dimensionality of local parameters
        """
        self.global_dist = global_dist
        self.local_dist_fn = local_dist_fn
        self.dim_global = dim_global
        self.dim_local = dim_local

        batch_shape = global_dist.batch_shape
        event_shape = torch.Size([dim_global + dim_local])
        super().__init__(batch_shape, event_shape, validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        """Sample from the hierarchical distribution.

        First samples global parameters, then samples local parameters
        conditioned on the global parameters.

        Args:
            sample_shape: Shape of samples to generate

        Returns:
            Samples with shape sample_shape + batch_shape + [dim_global +
            dim_local]
        """
        # Sample global parameters
        global_params = self.global_dist.sample(sample_shape)

        # Sample local parameters conditioned on global
        local_dist = self.local_dist_fn(global_params)
        local_params = local_dist.sample()

        # Concatenate global and local parameters
        return torch.cat([global_params, local_params], dim=-1)

    def log_prob(self, value):
        """Compute log probability of the joint distribution.

        log p(global, local) = log p(global) + log p(local | global)

        Args:
            value: Parameter tensor with shape [..., dim_global + dim_local]

        Returns:
            Log probability with shape [...]
        """
        # Split value into global and local components
        global_params = value[..., : self.dim_global]
        local_params = value[..., self.dim_global :]

        # Compute log p(global)
        log_prob_global = self.global_dist.log_prob(global_params)

        # Compute log p(local | global)
        local_dist = self.local_dist_fn(global_params)
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

"""
Shared fixtures for deepset algorithm tests.
"""

import pytest
import torch


@pytest.fixture
def hierarchical_gaussian_linear_data():
    """
    Create synthetic hierarchical Gaussian Linear data.

    Returns a function that generates data with specified parameters.
    """
    def _create_data(
        num_datasets: int = 100,
        num_events: int = 5,
        dim_per_event: int = 3,
        seed: int = 42,
    ):
        """
        Create synthetic hierarchical Gaussian Linear data.

        Model: x_ij ~ N(μ_i, σ) where σ is global and μ_i are local.
        - Global parameters: σ (1-dim) - noise standard deviation
        - Local parameters: μ_i (1-dim) - per-event means

        Returns:
            x_set: shape (num_datasets, num_events, dim_per_event)
            y_global: shape (num_datasets, 1) - noise std
            y_local: shape (num_datasets, num_events, 1) - means
        """
        torch.manual_seed(seed)

        # Global parameter: σ ~ Gamma(2, 2) via exp(N(0.5, 0.5))
        y_global = torch.randn(num_datasets, 1) * 0.5 + 0.5
        y_global = torch.exp(y_global)  # Ensure positive

        # Local parameters: μ_i ~ N(0, 1) (1-dim per event)
        y_local = torch.randn(num_datasets, num_events, 1)

        # Observations: x_ij ~ N(μ_i, σ)
        noise = torch.randn(num_datasets, num_events, dim_per_event) * (
            y_global.view(num_datasets, 1, 1)
        )
        y_local_expanded = y_local.expand(
            num_datasets, num_events, dim_per_event
        )
        x_set = y_local_expanded + noise

        return x_set, y_global, y_local

    return _create_data

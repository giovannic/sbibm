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
        dim_global: int = 1,
        dim_local: int = 1,
        seed: int = 42,
    ):
        """
        Create synthetic hierarchical Gaussian Linear data.

        Returns:
            x_set: shape (num_datasets, num_events, dim_per_event)
            y_global: shape (num_datasets, dim_global)
            y_local: shape (num_datasets, num_events, dim_local)
        """
        torch.manual_seed(seed)

        # Global parameters: σ ~ Gamma(2, 2)
        # (Using exponential of normal for stability)
        y_global = torch.randn(num_datasets, dim_global) * 0.5 + 0.5
        y_global = torch.exp(y_global)  # Ensure positive

        # Local parameters: μ_i ~ N(0, 1)
        y_local = torch.randn(num_datasets, num_events, dim_local)

        # Observations: x_ij ~ N(μ_i, σ) for j in [1..dim_per_event]
        # For each event i and measurement j: x_ij ~ N(μ_i, σ)
        # Shape: (num_datasets, num_events, dim_per_event)
        noise = torch.randn(num_datasets, num_events, dim_per_event) * (
            y_global.view(num_datasets, 1, 1)
        )
        # y_local has shape (num_datasets, num_events, 1)
        # Expand to (num_datasets, num_events, dim_per_event) for broadcasting
        y_local_expanded = y_local.expand(
            num_datasets, num_events, dim_per_event
        )
        x_set = y_local_expanded + noise

        return x_set, y_global, y_local

    return _create_data

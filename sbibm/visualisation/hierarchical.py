"""Matplotlib-based visualization utilities for hierarchical tasks.

This module provides visualization functions that avoid altair/deneb
dependencies, using pure matplotlib for hierarchical posterior plots.
"""
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_hierarchical_labels(
    task,
    max_local_contexts: int,
) -> List[str]:
    """Generate custom parameter labels for hierarchical tasks.

    Args:
        task: Hierarchical task with task.prior_dist containing dim_global,
              dim_local, and n_local
        max_local_contexts: Max local contexts to show in visualization

    Returns:
        List of parameter labels (e.g., ["global₀", "local₀_θ₀", ...])
    """
    numbers_unicode = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    labels = []

    # Get dimensions from task.prior_dist
    dim_global = task.prior_dist.dim_global
    dim_local = task.prior_dist.dim_local
    n_l = task.prior_dist.n_local

    # Generate global parameter labels
    for i in range(dim_global):
        idx_str = (
            "".join(numbers_unicode[int(d)] for d in str(i))
            if i < 10
            else str(i)
        )
        labels.append(f"global{idx_str}")

    # Generate local parameter labels
    num_local_to_show = min(max_local_contexts, n_l)
    for ctx in range(num_local_to_show):
        ctx_str = numbers_unicode[ctx] if ctx < 10 else str(ctx)
        for i in range(dim_local):
            idx_str = numbers_unicode[i] if i < 10 else str(i)
            labels.append(f"local{ctx_str}_θ{idx_str}")

    return labels


def generate_observation_labels(
    task,
    max_local_contexts: int,
) -> List[str]:
    """Generate observation labels for hierarchical tasks.

    Args:
        task: Hierarchical task with task.dim_data and task.n_l
        max_local_contexts: Max local contexts to show in visualization

    Returns:
        List of observation labels (e.g., ["x₀[0]", "x₀[1]", "x₁[0]", ...])
    """
    numbers_unicode = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    labels = []

    # Infer dimensions
    n_l = task.n_l
    dim_per_context = task.dim_data // n_l

    # Generate observation labels per context
    num_contexts_to_show = min(max_local_contexts, n_l)
    for ctx in range(num_contexts_to_show):
        ctx_str = numbers_unicode[ctx] if ctx < 10 else str(ctx)
        for i in range(dim_per_context):
            idx_str = numbers_unicode[i] if i < 10 else str(i)
            labels.append(f"x{ctx_str}[{idx_str}]")

    return labels


def plot_observation_grid(
    samples: torch.Tensor,
    n_l: int,
    dim_per_context: int,
    output_path: Path,
    true_observation: Optional[np.ndarray] = None,
    max_local_contexts: int = 5,
    figsize_scale: float = 3.0,
    dpi: int = 300,
) -> None:
    """Create grid of plots showing prior predictive observations per context.

    For tasks with 1D or 2D observations per context, creates a grid of plots
    where each subplot shows samples from one local context.

    Args:
        samples: Prior predictive samples (shape: [num_samples, dim_data])
        n_l: Number of local contexts in the samples
        dim_per_context: Dimensionality per context (1 or 2)
        output_path: Path to save the figure
        true_observation: Optional true observation to overlay
        max_local_contexts: Maximum contexts to display
        figsize_scale: Scale factor for figure size
        dpi: Resolution for saved figure
    """
    # Convert samples to numpy
    samples_np = samples.cpu().numpy() if samples.is_cuda else samples.numpy()

    # Limit contexts to show
    num_contexts = min(max_local_contexts, n_l)

    # Reshape to (num_samples, num_contexts, dim_per_context)
    samples_reshaped = samples_np[:, : num_contexts * dim_per_context].reshape(
        -1, num_contexts, dim_per_context
    )

    # Reshape true observation if provided
    true_obs_reshaped: Optional[np.ndarray] = None
    if true_observation is not None:
        true_obs_reshaped = true_observation[: num_contexts * dim_per_context].reshape(
            num_contexts, dim_per_context
        )

    # Determine grid layout
    ncols = min(3, num_contexts)
    nrows = (num_contexts + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_scale * ncols, figsize_scale * nrows),
        squeeze=False,
    )

    for ctx in range(num_contexts):
        row, col = ctx // ncols, ctx % ncols
        ax = axes[row, col]

        if dim_per_context == 2:
            # 2D scatter plot
            ax.scatter(
                samples_reshaped[:, ctx, 0],
                samples_reshaped[:, ctx, 1],
                s=3,
                alpha=0.3,
                color="#0035FD",
            )
            if true_obs_reshaped is not None:
                ax.scatter(
                    true_obs_reshaped[ctx, 0],
                    true_obs_reshaped[ctx, 1],
                    s=100,
                    color="#f92700",
                    marker="x",
                    linewidths=2,
                    zorder=10,
                )
            ax.set_xlabel(f"x[0]", fontsize=10)
            ax.set_ylabel(f"x[1]", fontsize=10)
        else:
            # 1D histogram
            ax.hist(
                samples_reshaped[:, ctx, 0],
                bins=30,
                color="#0035FD",
                alpha=0.6,
                density=True,
            )
            if true_obs_reshaped is not None:
                ax.axvline(
                    true_obs_reshaped[ctx, 0],
                    color="#f92700",
                    linestyle="--",
                    linewidth=2,
                )
            ax.set_xlabel(f"x", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)

        ax.set_title(f"Context {ctx + 1}", fontsize=12)

    # Hide unused subplots
    for idx in range(num_contexts, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis("off")

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_hierarchical_posterior(
    samples: torch.Tensor,
    labels: List[str],
    output_path: Path,
    true_params: Optional[np.ndarray] = None,
    figsize_scale: float = 2.0,
    dpi: int = 300,
) -> None:
    """Create and save a pairplot visualization of hierarchical posterior.

    Args:
        samples: Posterior samples to plot (shape: [num_samples, num_dims])
        labels: Parameter labels for axes
        output_path: Path to save the figure
        true_params: True parameter values (default: None)
        figsize_scale: Scale factor for figure size (default: 2.0)
        dpi: Resolution for saved figure (default: 300)
    """
    # Convert samples to numpy (handles GPU tensors)
    samples_np = samples.cpu().numpy() if samples.is_cuda else samples.numpy()

    num_dims = samples_np.shape[1]

    # Create figure with subplots
    fig, axes = plt.subplots(
        num_dims,
        num_dims,
        figsize=(figsize_scale * num_dims, figsize_scale * num_dims),
    )

    # Handle case where num_dims is 1 (axes won't be 2D array)
    if num_dims == 1:
        axes = np.array([[axes]])

    # Plot pairwise comparisons
    for i in range(num_dims):
        for j in range(num_dims):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                ax.hist(
                    samples_np[:, i],
                    bins=30,
                    color="#0035FD",
                    alpha=0.6,
                    density=True,
                )
                if true_params is not None:
                    ax.axvline(
                        true_params[i],
                        color="#f92700",
                        linestyle="--",
                        linewidth=2,
                        label="True",
                    )
                ax.set_yticks([])
            elif i > j:
                # Lower triangle: scatter plot
                ax.scatter(
                    samples_np[:, j],
                    samples_np[:, i],
                    s=1,
                    alpha=0.3,
                    color="#0035FD",
                )
                if true_params is not None:
                    ax.scatter(
                        true_params[j],
                        true_params[i],
                        s=50,
                        color="#f92700",
                        marker="x",
                        linewidths=2,
                    )
            else:
                # Upper triangle: hide
                ax.axis("off")

            # Add labels on edges
            if i == num_dims - 1:
                ax.set_xlabel(labels[j], fontsize=10)
            else:
                ax.set_xticks([])

            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=10)
            else:
                ax.set_yticks([])

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close()

"""
Utility functions for hierarchical inference.

Source: https://github.com/smsharma/hierarchical-inference/blob/main/models/flows.py
Extracted from hierarchical-inference repository.
"""

import torch
import torch.nn as nn
from sbibm.tasks.task import Task


def build_mlp(input_dim, hidden_dim, output_dim, layers):
    """Create a MLP from the configurations"""

    activation = nn.GELU

    seq = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

def reshape_observations(task: Task, x: torch.Tensor):
    """
    Reshape flat observations to hierarchical structure.

    Args:
        task: Task instance
        x: Observations, shape (batch, dim_data)

    Returns:
        x_set: Hierarchical observations,
               shape (batch, num_events, dim_per_event)
    """
    batch_size = x.shape[0]
    n_events = task.n_l #type: ignore
    dim_per_event = x.shape[-1] // n_events

    x_set = x.reshape(batch_size, n_events, dim_per_event)
    return x_set

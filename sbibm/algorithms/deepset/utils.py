"""
Utility functions for hierarchical inference.

Source: https://github.com/smsharma/hierarchical-inference/blob/main/models/flows.py
Extracted from hierarchical-inference repository.
"""

import torch.nn as nn


def build_mlp(input_dim, hidden_dim, output_dim, layers):
    """Create a MLP from the configurations"""

    activation = nn.GELU

    seq = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

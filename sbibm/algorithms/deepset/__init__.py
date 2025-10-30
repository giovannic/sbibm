"""
DeepSet algorithm integration for sbibm.

Provides hierarchical inference through permutation-invariant set encoding.
Source: https://github.com/smsharma/hierarchical-inference
"""

from .flows import build_maf, build_nsf
from .hierarchical_deepset import (
    HierarchicalDeepSet,
    HierarchicalDeepSetInference,
)
from .resnet import ResNetEstimator
from .utils import build_mlp

__all__ = [
    "HierarchicalDeepSet",
    "HierarchicalDeepSetInference",
    "build_maf",
    "build_nsf",
    "build_mlp",
    "ResNetEstimator",
]

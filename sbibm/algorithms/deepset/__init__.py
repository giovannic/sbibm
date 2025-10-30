"""
DeepSet algorithm integration for sbibm.

Provides hierarchical inference through permutation-invariant set encoding.
Source: https://github.com/smsharma/hierarchical-inference
"""

from .hierarchical_deepset import HierarchicalDeepSet, HierarchicalDeepSetInference
from .flows import build_maf, build_nsf
from .utils import build_mlp
from .resnet import ResNetEstimator

__all__ = [
    "HierarchicalDeepSet",
    "HierarchicalDeepSetInference",
    "build_maf",
    "build_nsf",
    "build_mlp",
    "ResNetEstimator",
]

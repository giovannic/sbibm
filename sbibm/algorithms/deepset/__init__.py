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
from .posterior import HierarchicalPosterior
from .run import run
from .utils import build_mlp

__all__ = [
    "HierarchicalDeepSet",
    "HierarchicalDeepSetInference",
    "HierarchicalPosterior",
    "build_maf",
    "build_nsf",
    "build_mlp",
    "run",
]

"""Baseline algorithms for shipyard scheduling.

This package contains heuristic and optimization approaches used as
comparisons to the reinforcement learning agent. Each baseline exposes
a `decide` method that takes the environment instance and returns an
action dictionary.
"""

from .rule_based import RuleBasedScheduler
from .myopic_rl import MyopicRLScheduler
from .siloed_opt import SiloedOptimizationScheduler

__all__ = [
    "RuleBasedScheduler",
    "MyopicRLScheduler",
    "SiloedOptimizationScheduler",
]
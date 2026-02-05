"""Baseline algorithms for shipyard scheduling.

This package contains heuristic and optimization approaches used as
comparisons to the reinforcement learning agent. Each baseline exposes
a `decide` method that takes the environment instance and returns an
action dictionary.

Baselines included:
- Rule-based: FIFO, EDD, Slack heuristics
- Myopic RL: Single-step reward optimization
- Siloed optimization: Independent equipment scheduling
- MIP: Mixed Integer Programming optimal scheduler
- CP: Constraint Programming scheduler
"""

from .rule_based import RuleBasedScheduler
from .myopic_rl import MyopicRLScheduler
from .siloed_opt import SiloedOptimizationScheduler

# Optional OR imports (may not be available)
try:
    from .mip_scheduler import MIPScheduler, MIPSolution
    _MIP_AVAILABLE = True
except ImportError:
    _MIP_AVAILABLE = False

try:
    from .cp_scheduler import CPScheduler, CPSolution, JobShopCPScheduler
    _CP_AVAILABLE = True
except ImportError:
    _CP_AVAILABLE = False

__all__ = [
    "RuleBasedScheduler",
    "MyopicRLScheduler",
    "SiloedOptimizationScheduler",
]

if _MIP_AVAILABLE:
    __all__.extend(["MIPScheduler", "MIPSolution"])

if _CP_AVAILABLE:
    __all__.extend(["CPScheduler", "CPSolution", "JobShopCPScheduler"])

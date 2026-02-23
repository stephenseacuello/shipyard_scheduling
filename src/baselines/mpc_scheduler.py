"""Rolling Horizon Model Predictive Control (MPC) scheduler.

Implements a receding horizon approach using optimization
(CP-SAT or MIP) for local planning with re-planning at each
control step.

This provides a strong baseline that combines online planning
with optimization-based scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from copy import deepcopy

if TYPE_CHECKING:
    from simulation.environment import ShipyardEnv

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


@dataclass
class MPCConfig:
    """Configuration for MPC scheduler."""
    prediction_horizon: int = 50  # Steps to look ahead
    control_horizon: int = 10     # Steps to optimize for execution
    replanning_frequency: int = 5  # Steps between replanning
    solver_time_limit: float = 5.0  # Seconds per solve
    objective_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.objective_weights is None:
            self.objective_weights = {
                "makespan": 1.0,
                "tardiness": 10.0,
                "utilization": 0.5,
            }


class RollingHorizonMPC:
    """Model Predictive Control with rolling optimization window.

    At each control step:
    1. Solve optimization problem for prediction horizon
    2. Execute first control_horizon actions
    3. Re-solve after replanning_frequency steps

    Args:
        config: MPC configuration.
    """

    def __init__(self, config: Optional[MPCConfig] = None):
        self.config = config or MPCConfig()
        self._current_plan: List[Dict] = []
        self._plan_index: int = 0
        self._steps_since_replan: int = 0

    def solve_window(
        self,
        env: "ShipyardEnv",
        horizon: int,
    ) -> List[Dict]:
        """Solve optimization for current planning window.

        Args:
            env: Current environment state.
            horizon: Number of steps to plan.

        Returns:
            List of action dictionaries.
        """
        if not ORTOOLS_AVAILABLE:
            return self._fallback_greedy(env, horizon)

        # Create CP-SAT model
        model = cp_model.CpModel()

        # Get current state
        blocks = list(env.entities.get("blocks", []))
        spmts = list(env.entities.get("spmts", []))
        n_blocks = len(blocks)
        n_spmts = len(spmts)

        if n_blocks == 0 or n_spmts == 0:
            return [{"action_type": 3}] * horizon  # Hold

        # Decision variables
        # x[b, s, t] = 1 if block b is transported by SPMT s at time t
        x = {}
        for b in range(min(n_blocks, horizon)):
            for s in range(n_spmts):
                for t in range(horizon):
                    x[b, s, t] = model.NewBoolVar(f"x_{b}_{s}_{t}")

        # Constraints
        # Each block transported at most once in horizon
        for b in range(min(n_blocks, horizon)):
            model.Add(sum(x[b, s, t] for s in range(n_spmts) for t in range(horizon)) <= 1)

        # SPMT can only transport one block at a time
        for s in range(n_spmts):
            for t in range(horizon):
                model.Add(sum(x[b, s, t] for b in range(min(n_blocks, horizon))) <= 1)

        # Simple objective: minimize completion times
        completion_times = []
        for b in range(min(n_blocks, horizon)):
            for s in range(n_spmts):
                for t in range(horizon):
                    # Penalize later transport times
                    completion_times.append(t * x[b, s, t])

        if completion_times:
            model.Minimize(sum(completion_times))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.solver_time_limit
        status = solver.Solve(model)

        # Extract solution
        schedule = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for t in range(horizon):
                action = {"action_type": 3}  # Default: hold

                for b in range(min(n_blocks, horizon)):
                    for s in range(n_spmts):
                        if solver.Value(x[b, s, t]) == 1:
                            action = {
                                "action_type": 0,
                                "spmt": s,
                                "request": b % max(1, len(getattr(env, "transport_requests", [1]))),
                                "crane": 0,
                                "lift": 0,
                                "equipment": 0,
                            }
                            break
                    if action["action_type"] == 0:
                        break

                schedule.append(action)
        else:
            # Fallback to greedy
            schedule = self._fallback_greedy(env, horizon)

        return schedule

    def _fallback_greedy(
        self,
        env: "ShipyardEnv",
        horizon: int,
    ) -> List[Dict]:
        """Greedy fallback when optimization fails."""
        schedule = []
        spmts = list(env.entities.get("spmts", []))

        for t in range(horizon):
            # Cycle through SPMTs
            spmt_idx = t % max(1, len(spmts))
            action = {
                "action_type": 0 if t % 3 != 2 else 3,  # Dispatch or hold
                "spmt": spmt_idx,
                "request": t % max(1, len(getattr(env, "transport_requests", [1]))),
                "crane": 0,
                "lift": 0,
                "equipment": 0,
            }
            schedule.append(action)

        return schedule

    def step(self, env: "ShipyardEnv") -> Dict[str, Any]:
        """Get next action from MPC.

        Args:
            env: Current environment state.

        Returns:
            Action dictionary.
        """
        # Check if replanning needed
        if (not self._current_plan or
            self._plan_index >= len(self._current_plan) or
            self._steps_since_replan >= self.config.replanning_frequency):

            # Solve new plan
            self._current_plan = self.solve_window(
                env, self.config.prediction_horizon
            )
            self._plan_index = 0
            self._steps_since_replan = 0

        # Get next action
        if self._plan_index < len(self._current_plan):
            action = self._current_plan[self._plan_index]
            self._plan_index += 1
        else:
            action = {"action_type": 3}  # Hold

        self._steps_since_replan += 1

        return action

    def reset(self):
        """Reset planner state."""
        self._current_plan = []
        self._plan_index = 0
        self._steps_since_replan = 0


def run_mpc_episode(
    env: "ShipyardEnv",
    config: Optional[MPCConfig] = None,
    max_steps: int = 10000,
) -> Dict[str, Any]:
    """Run a full episode using MPC.

    Args:
        env: Environment.
        config: MPC configuration.
        max_steps: Maximum steps.

    Returns:
        Episode statistics.
    """
    mpc = RollingHorizonMPC(config)

    obs, info = env.reset()
    mpc.reset()

    total_reward = 0.0
    steps = 0
    solve_times = []

    for step in range(max_steps):
        import time
        start = time.time()
        action = mpc.step(env)
        solve_time = time.time() - start
        solve_times.append(solve_time)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "blocks_completed": env.metrics.get("blocks_completed", 0),
        "throughput": env.metrics.get("blocks_completed", 0) / max(env.sim_time, 1),
        "avg_solve_time": np.mean(solve_times) if solve_times else 0.0,
        "max_solve_time": np.max(solve_times) if solve_times else 0.0,
    }

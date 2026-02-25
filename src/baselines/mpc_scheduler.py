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

    def _make_hold_action(self) -> Dict[str, Any]:
        """Return a correctly-keyed hold action."""
        return {
            "action_type": 3,
            "spmt_idx": 0, "request_idx": 0,
            "crane_idx": 0, "lift_idx": 0, "erection_idx": 0,
            "equipment_idx": 0,
        }

    def _edd_dispatch(self, env: "ShipyardEnv") -> Dict[str, Any]:
        """Fast EDD-style dispatch for reactive decisions.

        Prioritizes crane erection when available (critical path),
        then SPMT transport, then hold.
        """
        # Priority 1: Erect blocks if crane requests available
        erection_reqs = getattr(env, "erection_requests", [])
        if erection_reqs:
            cranes = env.entities.get("goliath_cranes", env.entities.get("cranes", []))
            for ci, crane in enumerate(cranes):
                if crane.status.name == "IDLE":
                    # Find earliest-due erection request
                    best_idx = 0
                    best_due = float("inf")
                    for ri, req in enumerate(erection_reqs):
                        block = req if hasattr(req, "due_date") else None
                        if block and block.due_date < best_due:
                            best_due = block.due_date
                            best_idx = ri
                    return {
                        "action_type": 1,
                        "crane_idx": ci, "erection_idx": best_idx,
                        "spmt_idx": 0, "request_idx": 0,
                        "lift_idx": best_idx, "equipment_idx": 0,
                    }

        # Priority 2: Transport blocks if requests available
        trans_reqs = getattr(env, "transport_requests", [])
        if trans_reqs:
            spmts = env.entities.get("spmts", [])
            for si, spmt in enumerate(spmts):
                if spmt.status.name == "IDLE":
                    best_idx = 0
                    best_due = float("inf")
                    for ri, req in enumerate(trans_reqs):
                        block = req if hasattr(req, "due_date") else None
                        if block and block.due_date < best_due:
                            best_due = block.due_date
                            best_idx = ri
                    return {
                        "action_type": 0,
                        "spmt_idx": si, "request_idx": best_idx,
                        "crane_idx": 0, "lift_idx": 0, "erection_idx": 0,
                        "equipment_idx": 0,
                    }

        return self._make_hold_action()

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
        trans_reqs = getattr(env, "transport_requests", [])
        n_blocks = len(blocks)
        n_spmts = len(spmts)
        n_reqs = len(trans_reqs)

        if n_blocks == 0 or n_spmts == 0 or n_reqs == 0:
            return [self._make_hold_action()] * horizon

        # Decision variables
        # x[r, s, t] = 1 if request r is dispatched to SPMT s at time t
        n_plan_reqs = min(n_reqs, horizon)
        x = {}
        for r in range(n_plan_reqs):
            for s in range(n_spmts):
                for t in range(horizon):
                    x[r, s, t] = model.NewBoolVar(f"x_{r}_{s}_{t}")

        # Constraints: each request dispatched at most once
        for r in range(n_plan_reqs):
            model.Add(sum(x[r, s, t] for s in range(n_spmts) for t in range(horizon)) <= 1)

        # SPMT handles one request per timestep
        for s in range(n_spmts):
            for t in range(horizon):
                model.Add(sum(x[r, s, t] for r in range(n_plan_reqs)) <= 1)

        # Objective: dispatch early, prefer earliest-due requests
        obj_terms = []
        for r in range(n_plan_reqs):
            req = trans_reqs[r]
            due_weight = 1.0
            if hasattr(req, "due_date"):
                # Lower due_date = higher urgency = lower weight multiplier
                due_weight = max(0.1, req.due_date - getattr(env, "sim_time", 0)) / 1000.0
            for s in range(n_spmts):
                for t in range(horizon):
                    obj_terms.append(int((t + 1) * due_weight * 10) * x[r, s, t])

        # Bonus for actually dispatching (negative cost = encouragement)
        for r in range(n_plan_reqs):
            for s in range(n_spmts):
                for t in range(horizon):
                    obj_terms.append(-100 * x[r, s, t])

        if obj_terms:
            model.Minimize(sum(obj_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.solver_time_limit
        status = solver.Solve(model)

        # Extract solution
        schedule = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for t in range(horizon):
                action = self._make_hold_action()

                for r in range(n_plan_reqs):
                    for s in range(n_spmts):
                        if solver.Value(x[r, s, t]) == 1:
                            action = {
                                "action_type": 0,
                                "spmt_idx": s,
                                "request_idx": r % max(1, n_reqs),
                                "crane_idx": 0, "lift_idx": 0,
                                "erection_idx": 0, "equipment_idx": 0,
                            }
                            break
                    if action["action_type"] == 0:
                        break

                schedule.append(action)
        else:
            schedule = self._fallback_greedy(env, horizon)

        return schedule

    def _fallback_greedy(
        self,
        env: "ShipyardEnv",
        horizon: int,
    ) -> List[Dict]:
        """Greedy fallback when optimization fails."""
        schedule = []
        for _ in range(horizon):
            schedule.append(self._edd_dispatch(env))
        return schedule

    def step(self, env: "ShipyardEnv") -> Dict[str, Any]:
        """Get next action from MPC.

        Uses a hybrid approach: MPC optimizes transport scheduling,
        but crane erection requests are handled reactively via EDD
        (same approach that works for PuLP).

        Args:
            env: Current environment state.

        Returns:
            Action dictionary.
        """
        # Priority 1: Handle erection requests immediately (critical path)
        erection_reqs = getattr(env, "erection_requests", [])
        if erection_reqs:
            edd_action = self._edd_dispatch(env)
            if edd_action.get("action_type") == 1:
                self._steps_since_replan += 1
                return edd_action

        # Check if replanning needed
        n_reqs = len(getattr(env, "transport_requests", []))
        replan_needed = (
            not self._current_plan
            or self._plan_index >= len(self._current_plan)
            or self._steps_since_replan >= self.config.replanning_frequency
        )

        if replan_needed:
            self._current_plan = self.solve_window(
                env, self.config.prediction_horizon
            )
            self._plan_index = 0
            self._steps_since_replan = 0

        # Get next action from plan
        if self._plan_index < len(self._current_plan):
            action = self._current_plan[self._plan_index]
            self._plan_index += 1
        else:
            action = self._make_hold_action()

        # If the planned action is a transport but there are no requests, use EDD
        trans_reqs = getattr(env, "transport_requests", [])
        if action.get("action_type") == 0 and not trans_reqs:
            action = self._edd_dispatch(env)

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

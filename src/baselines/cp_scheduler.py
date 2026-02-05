"""Constraint Programming scheduler using Google OR-Tools CP-SAT.

This module provides a constraint programming baseline using the CP-SAT solver
from Google OR-Tools. CP is particularly well-suited for scheduling problems
with complex constraints.

Advantages over MIP:
- Better handling of disjunctive constraints (job shop structure)
- Native interval variables for time-based scheduling
- Efficient propagation for cumulative resources

Features:
- Interval variables for block processing
- No-overlap constraints for equipment
- Precedence constraint propagation
- Optional activities for conditional scheduling
- Cumulative constraints for resource limits

Reference:
- Google OR-Tools CP-SAT: https://developers.google.com/optimization/cp/cp_solver
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import time

import numpy as np


@dataclass
class CPSolution:
    """Solution from the CP-SAT solver."""
    schedule: List[Dict[str, int]]
    makespan: int
    total_tardiness: int
    objective_value: int
    solve_time: float
    status: str
    num_branches: int
    num_conflicts: int


def _try_import_cpsat():
    """Try to import OR-Tools CP-SAT."""
    try:
        from ortools.sat.python import cp_model
        return cp_model
    except ImportError:
        return None


class CPScheduler:
    """Constraint Programming scheduler using OR-Tools CP-SAT.

    Provides optimal/near-optimal schedules for comparison with RL agents.
    CP-SAT is particularly efficient for scheduling with:
    - Precedence constraints
    - Resource constraints (no-overlap, cumulative)
    - Time windows

    Args:
        time_limit: Maximum solve time in seconds.
        num_workers: Number of parallel workers for search.
        verbose: Print solver statistics.
    """

    def __init__(
        self,
        time_limit: float = 60.0,
        num_workers: int = 4,
        verbose: bool = False,
    ):
        self.time_limit = time_limit
        self.num_workers = num_workers
        self.verbose = verbose

        self.cp_model = _try_import_cpsat()
        if self.cp_model is None:
            raise ImportError(
                "OR-Tools CP-SAT not available. Install with: pip install ortools"
            )

    def solve(self, env) -> CPSolution:
        """Solve the scheduling problem using CP-SAT.

        Args:
            env: Shipyard environment instance.

        Returns:
            CPSolution with schedule and metrics.
        """
        cp_model = self.cp_model

        start_time = time.time()

        # Extract problem data
        n_blocks = len(env.blocks)
        n_spmts = len(env.spmts)
        n_cranes = len(env.cranes)

        # Get problem parameters
        horizon = int(env.max_time * 1.5)  # Allow some slack

        # Create model
        model = cp_model.CpModel()

        # ========== Variables ==========

        # Interval variables for each block's processing
        block_intervals = []
        block_starts = []
        block_ends = []
        block_durations = []

        for i, block in enumerate(env.blocks):
            duration = int(getattr(block, "processing_time", 10))
            start_var = model.NewIntVar(0, horizon, f"start_{i}")
            end_var = model.NewIntVar(0, horizon, f"end_{i}")
            interval = model.NewIntervalVar(start_var, duration, end_var, f"interval_{i}")

            block_starts.append(start_var)
            block_ends.append(end_var)
            block_durations.append(duration)
            block_intervals.append(interval)

        # Assignment variables: which SPMT processes each block
        spmt_assignment = []
        for i in range(n_blocks):
            assignment = [
                model.NewBoolVar(f"block_{i}_spmt_{s}")
                for s in range(n_spmts)
            ]
            spmt_assignment.append(assignment)
            # Each block assigned to exactly one SPMT
            model.AddExactlyOne(assignment)

        # Assignment variables: which crane lifts each block
        crane_assignment = []
        for i in range(n_blocks):
            assignment = [
                model.NewBoolVar(f"block_{i}_crane_{c}")
                for c in range(n_cranes)
            ]
            crane_assignment.append(assignment)
            # Each block assigned to exactly one crane
            model.AddExactlyOne(assignment)

        # Optional intervals for each block-SPMT combination
        spmt_intervals = []  # spmt_intervals[s] = list of intervals on SPMT s
        for s in range(n_spmts):
            intervals_on_spmt = []
            for i in range(n_blocks):
                # Optional interval: only active if block assigned to this SPMT
                opt_interval = model.NewOptionalIntervalVar(
                    block_starts[i],
                    block_durations[i],
                    block_ends[i],
                    spmt_assignment[i][s],
                    f"block_{i}_on_spmt_{s}"
                )
                intervals_on_spmt.append(opt_interval)
            spmt_intervals.append(intervals_on_spmt)

        # Tardiness variables
        tardiness = []
        for i, block in enumerate(env.blocks):
            due_date = int(getattr(block, "due_date", horizon))
            tard = model.NewIntVar(0, horizon, f"tardiness_{i}")
            # tardiness = max(0, end - due_date)
            model.AddMaxEquality(tard, [0, block_ends[i] - due_date])
            tardiness.append(tard)

        # Makespan variable
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, block_ends)

        # ========== Constraints ==========

        # 1. Precedence constraints
        if hasattr(env, "precedence_graph"):
            for i in range(n_blocks):
                for pred_id in env.precedence_graph.predecessors(i):
                    model.Add(block_starts[i] >= block_ends[pred_id])

        # 2. No-overlap constraints for SPMTs (each SPMT can process one block at a time)
        for s in range(n_spmts):
            model.AddNoOverlap(spmt_intervals[s])

        # 3. Crane capacity constraints (simplified: cranes can lift one block at a time)
        # Use cumulative constraint with demand=1 for each block and capacity=1 for cranes
        crane_intervals = []
        for c in range(n_cranes):
            intervals_on_crane = []
            for i in range(n_blocks):
                opt_interval = model.NewOptionalIntervalVar(
                    block_starts[i],
                    block_durations[i],
                    block_ends[i],
                    crane_assignment[i][c],
                    f"block_{i}_on_crane_{c}"
                )
                intervals_on_crane.append(opt_interval)
            crane_intervals.append(intervals_on_crane)
            model.AddNoOverlap(intervals_on_crane)

        # 4. Health-aware constraints (optional)
        # Prefer healthy equipment by adding soft constraints
        for i in range(n_blocks):
            for s, spmt in enumerate(env.spmts):
                health = getattr(spmt, "health", 1.0)
                if health < 0.3:  # Critical health - avoid if possible
                    # Add penalty (will be minimized in objective)
                    pass  # Handled in objective function below

        # ========== Objective ==========

        # Multi-objective: minimize weighted combination of makespan and tardiness
        # Also add health preferences
        alpha = 500  # Weight for tardiness vs makespan

        total_weighted_tardiness = sum(
            int(getattr(env.blocks[i], "priority", 1)) * tardiness[i]
            for i in range(n_blocks)
        )

        # Health penalty: prefer assigning to healthy equipment
        health_penalty = 0
        for i in range(n_blocks):
            for s, spmt in enumerate(env.spmts):
                health = getattr(spmt, "health", 1.0)
                penalty = int((1 - health) * 100)  # Higher penalty for lower health
                health_penalty += penalty * spmt_assignment[i][s]

        model.Minimize(makespan + alpha * total_weighted_tardiness + health_penalty)

        # ========== Solve ==========

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = self.num_workers

        if self.verbose:
            solver.parameters.log_search_progress = True

        status = solver.Solve(model)
        solve_time = time.time() - start_time

        # ========== Extract Solution ==========

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = self._extract_schedule(
                solver, env, n_blocks, n_spmts, n_cranes,
                block_starts, spmt_assignment, crane_assignment
            )

            status_str = "optimal" if status == cp_model.OPTIMAL else "feasible"

            return CPSolution(
                schedule=schedule,
                makespan=solver.Value(makespan),
                total_tardiness=sum(solver.Value(t) for t in tardiness),
                objective_value=int(solver.ObjectiveValue()),
                solve_time=solve_time,
                status=status_str,
                num_branches=solver.NumBranches(),
                num_conflicts=solver.NumConflicts(),
            )
        else:
            status_map = {
                cp_model.INFEASIBLE: "infeasible",
                cp_model.MODEL_INVALID: "invalid",
                cp_model.UNKNOWN: "unknown/timeout",
            }
            return CPSolution(
                schedule=[],
                makespan=0,
                total_tardiness=0,
                objective_value=0,
                solve_time=solve_time,
                status=status_map.get(status, "error"),
                num_branches=solver.NumBranches(),
                num_conflicts=solver.NumConflicts(),
            )

    def _extract_schedule(
        self,
        solver,
        env,
        n_blocks: int,
        n_spmts: int,
        n_cranes: int,
        block_starts,
        spmt_assignment,
        crane_assignment,
    ) -> List[Dict[str, int]]:
        """Extract action sequence from CP-SAT solution."""
        schedule = []

        # Get solution values
        block_data = []
        for i in range(n_blocks):
            start_time = solver.Value(block_starts[i])

            # Find assigned SPMT
            spmt_id = 0
            for s in range(n_spmts):
                if solver.Value(spmt_assignment[i][s]):
                    spmt_id = s
                    break

            # Find assigned crane
            crane_id = 0
            for c in range(n_cranes):
                if solver.Value(crane_assignment[i][c]):
                    crane_id = c
                    break

            block_data.append({
                "block_id": i,
                "start_time": start_time,
                "spmt_id": spmt_id,
                "crane_id": crane_id,
            })

        # Sort by start time
        block_data.sort(key=lambda x: x["start_time"])

        # Convert to action format
        for data in block_data:
            schedule.append({
                "action_type": 0,  # Transport action
                "spmt_idx": data["spmt_id"],
                "crane_idx": data["crane_id"],
                "request_idx": data["block_id"],
            })

        return schedule

    def get_schedule_as_actions(self, solution: CPSolution) -> List[Dict[str, int]]:
        """Convert CP solution to environment action sequence."""
        return solution.schedule

    def get_solver_statistics(self, solution: CPSolution) -> Dict[str, Any]:
        """Get detailed solver statistics."""
        return {
            "solve_time": solution.solve_time,
            "status": solution.status,
            "num_branches": solution.num_branches,
            "num_conflicts": solution.num_conflicts,
            "makespan": solution.makespan,
            "total_tardiness": solution.total_tardiness,
            "objective_value": solution.objective_value,
        }


class JobShopCPScheduler(CPScheduler):
    """Specialized CP scheduler for job-shop style problems.

    In job-shop scheduling, each job (block) must visit multiple machines
    (facilities) in a specific order. This is more complex than the basic
    parallel machine problem.
    """

    def solve_job_shop(
        self,
        jobs: List[List[Tuple[int, int]]],  # jobs[j] = [(machine, duration), ...]
        n_machines: int,
        horizon: int,
    ) -> CPSolution:
        """Solve a pure job-shop problem.

        Args:
            jobs: List of jobs, each job is a list of (machine, duration) tuples.
            n_machines: Number of machines.
            horizon: Time horizon.

        Returns:
            CPSolution with the schedule.
        """
        cp_model = self.cp_model

        start_time = time.time()
        model = cp_model.CpModel()

        # Variables
        all_tasks = {}  # all_tasks[job_id, task_id] = (start, end, interval)
        machine_to_intervals = {m: [] for m in range(n_machines)}

        for job_id, job in enumerate(jobs):
            for task_id, (machine, duration) in enumerate(job):
                start_var = model.NewIntVar(0, horizon, f"start_{job_id}_{task_id}")
                end_var = model.NewIntVar(0, horizon, f"end_{job_id}_{task_id}")
                interval = model.NewIntervalVar(
                    start_var, duration, end_var, f"interval_{job_id}_{task_id}"
                )
                all_tasks[job_id, task_id] = (start_var, end_var, interval)
                machine_to_intervals[machine].append(interval)

        # Constraints
        # 1. Precedence within jobs
        for job_id, job in enumerate(jobs):
            for task_id in range(len(job) - 1):
                model.Add(
                    all_tasks[job_id, task_id + 1][0] >= all_tasks[job_id, task_id][1]
                )

        # 2. No overlap on machines
        for machine in range(n_machines):
            model.AddNoOverlap(machine_to_intervals[machine])

        # Objective: minimize makespan
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(
            makespan,
            [all_tasks[job_id, len(job) - 1][1] for job_id, job in enumerate(jobs)]
        )
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = self.num_workers

        status = solver.Solve(model)
        solve_time = time.time() - start_time

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract schedule
            schedule = []
            for job_id, job in enumerate(jobs):
                for task_id, (machine, duration) in enumerate(job):
                    start, end, _ = all_tasks[job_id, task_id]
                    schedule.append({
                        "job_id": job_id,
                        "task_id": task_id,
                        "machine": machine,
                        "start": solver.Value(start),
                        "end": solver.Value(end),
                    })

            schedule.sort(key=lambda x: x["start"])

            return CPSolution(
                schedule=schedule,
                makespan=solver.Value(makespan),
                total_tardiness=0,
                objective_value=solver.Value(makespan),
                solve_time=solve_time,
                status="optimal" if status == cp_model.OPTIMAL else "feasible",
                num_branches=solver.NumBranches(),
                num_conflicts=solver.NumConflicts(),
            )
        else:
            return CPSolution(
                schedule=[],
                makespan=0,
                total_tardiness=0,
                objective_value=0,
                solve_time=solve_time,
                status="infeasible",
                num_branches=solver.NumBranches(),
                num_conflicts=solver.NumConflicts(),
            )

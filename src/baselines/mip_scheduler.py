"""Mixed Integer Programming (MIP) scheduler for shipyard block scheduling.

This module provides an optimal baseline using mathematical programming.
It formulates the block scheduling problem as a MIP and solves it using
either Gurobi (if available) or Google OR-Tools (open source fallback).

The MIP formulation includes:
- Block assignment to equipment (SPMTs, cranes)
- Sequencing with precedence constraints
- Resource capacity constraints
- Time window constraints (due dates)
- Optional: Maintenance scheduling

Note: MIP is only tractable for small-medium instances (≤50 blocks).
For larger instances, use the CP-SAT scheduler or RL agents.

Reference:
- Pinedo, M. "Scheduling: Theory, Algorithms, and Systems"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import time

import numpy as np


@dataclass
class BlockData:
    """Data for a block in the scheduling problem."""
    id: int
    processing_time: float
    due_date: float
    weight: float  # Priority weight
    predecessors: List[int]  # IDs of predecessor blocks
    facility: int  # Current/target facility


@dataclass
class EquipmentData:
    """Data for equipment (SPMT or crane)."""
    id: int
    capacity: int
    available_time: float
    health: float  # Current health level


@dataclass
class MIPSolution:
    """Solution from the MIP solver."""
    schedule: List[Dict[str, int]]  # List of actions
    makespan: float
    total_tardiness: float
    objective_value: float
    solve_time: float
    status: str  # 'optimal', 'feasible', 'infeasible', 'timeout'
    gap: float  # Optimality gap (if applicable)


def _try_import_gurobi():
    """Try to import Gurobi, return None if not available."""
    try:
        import gurobipy as gp
        return gp
    except ImportError:
        return None


def _try_import_ortools():
    """Try to import OR-Tools, return None if not available."""
    try:
        from ortools.linear_solver import pywraplp
        return pywraplp
    except ImportError:
        return None


class MIPScheduler:
    """MIP-based scheduler for block scheduling.

    Provides an optimal (or near-optimal) baseline for comparison with RL.

    Args:
        time_limit: Maximum solve time in seconds.
        gap_tolerance: Acceptable optimality gap (0.0 = optimal).
        solver: 'auto', 'gurobi', or 'ortools'.
        verbose: Print solver output.
    """

    def __init__(
        self,
        time_limit: float = 60.0,
        gap_tolerance: float = 0.01,
        solver: str = "auto",
        verbose: bool = False,
    ):
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.verbose = verbose

        # Select solver
        self.gp = _try_import_gurobi()
        self.ortools = _try_import_ortools()

        if solver == "gurobi" and self.gp is None:
            raise ImportError("Gurobi not available")
        elif solver == "ortools" and self.ortools is None:
            raise ImportError("OR-Tools not available")
        elif solver == "auto":
            if self.gp is not None:
                self.solver_type = "gurobi"
            elif self.ortools is not None:
                self.solver_type = "ortools"
            else:
                raise ImportError("No MIP solver available. Install gurobipy or ortools.")
        else:
            self.solver_type = solver

    def extract_problem_data(self, env) -> Tuple[List[BlockData], List[EquipmentData], List[EquipmentData]]:
        """Extract scheduling problem data from environment.

        Args:
            env: Shipyard environment instance.

        Returns:
            Tuple of (blocks, spmts, cranes) data.
        """
        blocks = []
        for i, block in enumerate(env.blocks):
            # Get predecessors from precedence graph
            predecessors = []
            if hasattr(env, "precedence_graph"):
                for pred_id in env.precedence_graph.predecessors(i):
                    predecessors.append(pred_id)

            blocks.append(BlockData(
                id=i,
                processing_time=getattr(block, "processing_time", 10.0),
                due_date=getattr(block, "due_date", env.max_time),
                weight=getattr(block, "priority", 1.0),
                predecessors=predecessors,
                facility=getattr(block, "facility_idx", 0),
            ))

        spmts = []
        for i, spmt in enumerate(env.spmts):
            spmts.append(EquipmentData(
                id=i,
                capacity=1,
                available_time=0.0,
                health=getattr(spmt, "health", 1.0),
            ))

        cranes = []
        for i, crane in enumerate(env.cranes):
            cranes.append(EquipmentData(
                id=i,
                capacity=1,
                available_time=0.0,
                health=getattr(crane, "health", 1.0),
            ))

        return blocks, spmts, cranes

    def solve(self, env) -> MIPSolution:
        """Solve the scheduling problem.

        Args:
            env: Shipyard environment instance.

        Returns:
            MIPSolution with schedule and metrics.
        """
        blocks, spmts, cranes = self.extract_problem_data(env)

        if self.solver_type == "gurobi":
            return self._solve_gurobi(blocks, spmts, cranes, env)
        else:
            return self._solve_ortools(blocks, spmts, cranes, env)

    def _solve_gurobi(
        self,
        blocks: List[BlockData],
        spmts: List[EquipmentData],
        cranes: List[EquipmentData],
        env,
    ) -> MIPSolution:
        """Solve using Gurobi."""
        gp = self.gp

        start_time = time.time()
        n_blocks = len(blocks)
        n_spmts = len(spmts)
        n_cranes = len(cranes)

        try:
            # Create model
            model = gp.Model("shipyard_scheduling")
            model.setParam("TimeLimit", self.time_limit)
            model.setParam("MIPGap", self.gap_tolerance)
            if not self.verbose:
                model.setParam("OutputFlag", 0)

            # Big-M for disjunctive constraints
            M = env.max_time * 2

            # Decision variables
            # x[i,s] = 1 if block i assigned to SPMT s
            x = model.addVars(n_blocks, n_spmts, vtype=gp.GRB.BINARY, name="x")
            # y[i,c] = 1 if block i assigned to crane c
            y = model.addVars(n_blocks, n_cranes, vtype=gp.GRB.BINARY, name="y")
            # start[i] = start time of block i
            start = model.addVars(n_blocks, lb=0, ub=M, name="start")
            # completion[i] = completion time of block i
            completion = model.addVars(n_blocks, lb=0, ub=M, name="completion")
            # tardiness[i] = max(0, completion[i] - due_date[i])
            tardiness = model.addVars(n_blocks, lb=0, name="tardiness")
            # z[i,j,s] = 1 if block i precedes block j on SPMT s
            z = model.addVars(n_blocks, n_blocks, n_spmts, vtype=gp.GRB.BINARY, name="z")

            # Makespan
            makespan = model.addVar(lb=0, name="makespan")

            # Constraints
            # 1. Each block assigned to exactly one SPMT
            for i in range(n_blocks):
                model.addConstr(gp.quicksum(x[i, s] for s in range(n_spmts)) == 1)

            # 2. Each block assigned to exactly one crane
            for i in range(n_blocks):
                model.addConstr(gp.quicksum(y[i, c] for c in range(n_cranes)) == 1)

            # 3. Completion time = start time + processing time
            for i, block in enumerate(blocks):
                model.addConstr(completion[i] == start[i] + block.processing_time)

            # 4. Tardiness definition
            for i, block in enumerate(blocks):
                model.addConstr(tardiness[i] >= completion[i] - block.due_date)

            # 5. Precedence constraints
            for i, block in enumerate(blocks):
                for pred_id in block.predecessors:
                    model.addConstr(start[i] >= completion[pred_id])

            # 6. Disjunctive constraints for SPMT sequencing
            for i in range(n_blocks):
                for j in range(i + 1, n_blocks):
                    for s in range(n_spmts):
                        # If both i and j on same SPMT, one must precede other
                        model.addConstr(
                            start[j] >= completion[i] - M * (1 - z[i, j, s]) - M * (2 - x[i, s] - x[j, s])
                        )
                        model.addConstr(
                            start[i] >= completion[j] - M * z[i, j, s] - M * (2 - x[i, s] - x[j, s])
                        )

            # 7. Makespan definition
            for i in range(n_blocks):
                model.addConstr(makespan >= completion[i])

            # Objective: Weighted combination of makespan and tardiness
            alpha = 0.5  # Weight for makespan vs tardiness
            model.setObjective(
                alpha * makespan + (1 - alpha) * gp.quicksum(
                    blocks[i].weight * tardiness[i] for i in range(n_blocks)
                ),
                gp.GRB.MINIMIZE
            )

            # Solve
            model.optimize()

            solve_time = time.time() - start_time

            # Extract solution
            if model.Status == gp.GRB.OPTIMAL or model.Status == gp.GRB.TIME_LIMIT:
                schedule = self._extract_gurobi_schedule(model, blocks, spmts, cranes, x, y, start)
                status = "optimal" if model.Status == gp.GRB.OPTIMAL else "feasible"
                return MIPSolution(
                    schedule=schedule,
                    makespan=makespan.X,
                    total_tardiness=sum(tardiness[i].X for i in range(n_blocks)),
                    objective_value=model.ObjVal,
                    solve_time=solve_time,
                    status=status,
                    gap=model.MIPGap if hasattr(model, "MIPGap") else 0.0,
                )
            else:
                return MIPSolution(
                    schedule=[],
                    makespan=float("inf"),
                    total_tardiness=float("inf"),
                    objective_value=float("inf"),
                    solve_time=solve_time,
                    status="infeasible",
                    gap=1.0,
                )

        except Exception as e:
            return MIPSolution(
                schedule=[],
                makespan=float("inf"),
                total_tardiness=float("inf"),
                objective_value=float("inf"),
                solve_time=time.time() - start_time,
                status=f"error: {str(e)}",
                gap=1.0,
            )

    def _extract_gurobi_schedule(self, model, blocks, spmts, cranes, x, y, start) -> List[Dict[str, int]]:
        """Extract action sequence from Gurobi solution."""
        schedule = []

        # Sort blocks by start time
        block_order = sorted(range(len(blocks)), key=lambda i: start[i].X)

        for block_id in block_order:
            # Find assigned SPMT
            spmt_id = 0
            for s in range(len(spmts)):
                if x[block_id, s].X > 0.5:
                    spmt_id = s
                    break

            # Find assigned crane
            crane_id = 0
            for c in range(len(cranes)):
                if y[block_id, c].X > 0.5:
                    crane_id = c
                    break

            # Create action (simplified - actual env may need different format)
            schedule.append({
                "action_type": 0,  # Move/transport action
                "spmt_idx": spmt_id,
                "crane_idx": crane_id,
                "request_idx": block_id,
            })

        return schedule

    def _solve_ortools(
        self,
        blocks: List[BlockData],
        spmts: List[EquipmentData],
        cranes: List[EquipmentData],
        env,
    ) -> MIPSolution:
        """Solve using OR-Tools MIP solver (CBC/SCIP)."""
        pywraplp = self.ortools

        start_time = time.time()
        n_blocks = len(blocks)
        n_spmts = len(spmts)
        n_cranes = len(cranes)

        # Create solver
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            return MIPSolution(
                schedule=[],
                makespan=float("inf"),
                total_tardiness=float("inf"),
                objective_value=float("inf"),
                solve_time=0,
                status="error: no solver backend",
                gap=1.0,
            )

        solver.SetTimeLimit(int(self.time_limit * 1000))  # milliseconds

        M = env.max_time * 2

        # Decision variables
        x = {}  # x[i,s] = 1 if block i on SPMT s
        for i in range(n_blocks):
            for s in range(n_spmts):
                x[i, s] = solver.BoolVar(f"x_{i}_{s}")

        y = {}  # y[i,c] = 1 if block i on crane c
        for i in range(n_blocks):
            for c in range(n_cranes):
                y[i, c] = solver.BoolVar(f"y_{i}_{c}")

        # Start times
        start_vars = [solver.NumVar(0, M, f"start_{i}") for i in range(n_blocks)]
        completion = [solver.NumVar(0, M, f"completion_{i}") for i in range(n_blocks)]
        tardiness = [solver.NumVar(0, M, f"tardiness_{i}") for i in range(n_blocks)]
        makespan = solver.NumVar(0, M, "makespan")

        # Sequencing variables
        z = {}
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                for s in range(n_spmts):
                    z[i, j, s] = solver.BoolVar(f"z_{i}_{j}_{s}")

        # Constraints
        # 1. Each block to exactly one SPMT
        for i in range(n_blocks):
            solver.Add(sum(x[i, s] for s in range(n_spmts)) == 1)

        # 2. Each block to exactly one crane
        for i in range(n_blocks):
            solver.Add(sum(y[i, c] for c in range(n_cranes)) == 1)

        # 3. Completion time
        for i, block in enumerate(blocks):
            solver.Add(completion[i] == start_vars[i] + block.processing_time)

        # 4. Tardiness
        for i, block in enumerate(blocks):
            solver.Add(tardiness[i] >= completion[i] - block.due_date)

        # 5. Precedence
        for i, block in enumerate(blocks):
            for pred_id in block.predecessors:
                solver.Add(start_vars[i] >= completion[pred_id])

        # 6. Disjunctive (simplified - only for blocks on same SPMT)
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                for s in range(n_spmts):
                    solver.Add(
                        start_vars[j] >= completion[i] - M * (1 - z[i, j, s]) - M * (2 - x[i, s] - x[j, s])
                    )
                    solver.Add(
                        start_vars[i] >= completion[j] - M * z[i, j, s] - M * (2 - x[i, s] - x[j, s])
                    )

        # 7. Makespan
        for i in range(n_blocks):
            solver.Add(makespan >= completion[i])

        # Objective
        alpha = 0.5
        solver.Minimize(
            alpha * makespan + (1 - alpha) * sum(blocks[i].weight * tardiness[i] for i in range(n_blocks))
        )

        # Solve
        status = solver.Solve()
        solve_time = time.time() - start_time

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            schedule = self._extract_ortools_schedule(blocks, spmts, cranes, x, y, start_vars)
            status_str = "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible"
            return MIPSolution(
                schedule=schedule,
                makespan=makespan.solution_value(),
                total_tardiness=sum(tardiness[i].solution_value() for i in range(n_blocks)),
                objective_value=solver.Objective().Value(),
                solve_time=solve_time,
                status=status_str,
                gap=0.0,  # OR-Tools doesn't report gap easily
            )
        else:
            return MIPSolution(
                schedule=[],
                makespan=float("inf"),
                total_tardiness=float("inf"),
                objective_value=float("inf"),
                solve_time=solve_time,
                status="infeasible",
                gap=1.0,
            )

    def _extract_ortools_schedule(self, blocks, spmts, cranes, x, y, start_vars) -> List[Dict[str, int]]:
        """Extract action sequence from OR-Tools solution."""
        schedule = []

        # Sort by start time
        block_order = sorted(range(len(blocks)), key=lambda i: start_vars[i].solution_value())

        for block_id in block_order:
            spmt_id = 0
            for s in range(len(spmts)):
                if x[block_id, s].solution_value() > 0.5:
                    spmt_id = s
                    break

            crane_id = 0
            for c in range(len(cranes)):
                if y[block_id, c].solution_value() > 0.5:
                    crane_id = c
                    break

            schedule.append({
                "action_type": 0,
                "spmt_idx": spmt_id,
                "crane_idx": crane_id,
                "request_idx": block_id,
            })

        return schedule

    def get_schedule_as_actions(self, solution: MIPSolution) -> List[Dict[str, int]]:
        """Convert MIP solution to environment action sequence."""
        return solution.schedule

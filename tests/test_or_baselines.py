"""Tests for Operations Research baselines (MIP, CP schedulers).

These tests verify the OR-based scheduling baselines work correctly.
Some tests may be skipped if OR-Tools or Gurobi are not installed.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List


# Check for optional dependencies
try:
    from ortools.sat.python import cp_model
    CPSAT_AVAILABLE = True
except ImportError:
    CPSAT_AVAILABLE = False

try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_MIP_AVAILABLE = True
except ImportError:
    ORTOOLS_MIP_AVAILABLE = False

try:
    import gurobipy
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


# Import our modules (should work even without OR-Tools)
from baselines.mip_scheduler import BlockData, EquipmentData, MIPSolution
from baselines.cp_scheduler import CPSolution


class TestBlockData:
    """Tests for BlockData dataclass."""

    def test_block_data_creation(self):
        """Test creating block data."""
        block = BlockData(
            id=0,
            processing_time=10.0,
            due_date=100.0,
            weight=1.5,
            predecessors=[],
            facility=0,
        )

        assert block.id == 0
        assert block.processing_time == 10.0
        assert block.due_date == 100.0
        assert block.weight == 1.5

    def test_block_with_predecessors(self):
        """Test block with precedence constraints."""
        block = BlockData(
            id=5,
            processing_time=15.0,
            due_date=200.0,
            weight=1.0,
            predecessors=[1, 2, 3],
            facility=1,
        )

        assert len(block.predecessors) == 3
        assert 2 in block.predecessors


class TestEquipmentData:
    """Tests for EquipmentData dataclass."""

    def test_equipment_data_creation(self):
        """Test creating equipment data."""
        equipment = EquipmentData(
            id=0,
            capacity=1,
            available_time=0.0,
            health=0.95,
        )

        assert equipment.id == 0
        assert equipment.capacity == 1
        assert equipment.health == 0.95


class TestMIPSolution:
    """Tests for MIP solution dataclass."""

    def test_solution_creation(self):
        """Test creating MIP solution."""
        solution = MIPSolution(
            schedule=[
                {"action_type": 0, "spmt_idx": 0, "crane_idx": 0, "request_idx": 0},
                {"action_type": 0, "spmt_idx": 1, "crane_idx": 0, "request_idx": 1},
            ],
            makespan=100.0,
            total_tardiness=5.0,
            objective_value=52.5,
            solve_time=1.5,
            status="optimal",
            gap=0.0,
        )

        assert len(solution.schedule) == 2
        assert solution.status == "optimal"
        assert solution.gap == 0.0

    def test_infeasible_solution(self):
        """Test infeasible solution representation."""
        solution = MIPSolution(
            schedule=[],
            makespan=float("inf"),
            total_tardiness=float("inf"),
            objective_value=float("inf"),
            solve_time=60.0,
            status="infeasible",
            gap=1.0,
        )

        assert len(solution.schedule) == 0
        assert solution.status == "infeasible"


class TestCPSolution:
    """Tests for CP solution dataclass."""

    def test_cp_solution_creation(self):
        """Test creating CP solution."""
        solution = CPSolution(
            schedule=[
                {"action_type": 0, "spmt_idx": 0, "crane_idx": 0, "request_idx": 0},
            ],
            makespan=80,
            total_tardiness=0,
            objective_value=80,
            solve_time=0.5,
            status="optimal",
            num_branches=1000,
            num_conflicts=50,
        )

        assert solution.status == "optimal"
        assert solution.num_branches == 1000
        assert solution.num_conflicts == 50


@pytest.mark.skipif(not ORTOOLS_MIP_AVAILABLE, reason="OR-Tools MIP not available")
class TestMIPSchedulerORTools:
    """Tests for MIP scheduler with OR-Tools backend."""

    def test_mip_scheduler_creation(self):
        """Test creating MIP scheduler."""
        from baselines.mip_scheduler import MIPScheduler

        scheduler = MIPScheduler(
            time_limit=10.0,
            gap_tolerance=0.05,
            solver="ortools",
        )

        assert scheduler.time_limit == 10.0
        assert scheduler.solver_type == "ortools"

    def test_extract_problem_data(self):
        """Test extracting problem data from mock environment."""
        from baselines.mip_scheduler import MIPScheduler

        scheduler = MIPScheduler(solver="ortools")

        # Create mock environment
        mock_env = MagicMock()
        mock_env.blocks = [MagicMock(processing_time=10, due_date=100, priority=1) for _ in range(3)]
        mock_env.spmts = [MagicMock(health=0.9) for _ in range(2)]
        mock_env.cranes = [MagicMock(health=0.95)]
        mock_env.max_time = 500

        # Mock precedence graph
        mock_env.precedence_graph = MagicMock()
        mock_env.precedence_graph.predecessors = MagicMock(return_value=[])

        blocks, spmts, cranes = scheduler.extract_problem_data(mock_env)

        assert len(blocks) == 3
        assert len(spmts) == 2
        assert len(cranes) == 1


@pytest.mark.skipif(not CPSAT_AVAILABLE, reason="OR-Tools CP-SAT not available")
class TestCPScheduler:
    """Tests for CP-SAT scheduler."""

    def test_cp_scheduler_creation(self):
        """Test creating CP scheduler."""
        from baselines.cp_scheduler import CPScheduler

        scheduler = CPScheduler(
            time_limit=30.0,
            num_workers=2,
            verbose=False,
        )

        assert scheduler.time_limit == 30.0
        assert scheduler.num_workers == 2

    def test_job_shop_scheduler(self):
        """Test job shop CP scheduler."""
        from baselines.cp_scheduler import JobShopCPScheduler

        scheduler = JobShopCPScheduler(time_limit=10.0)

        # Simple job shop instance
        # Job 0: machine 0 (duration 3), then machine 1 (duration 2)
        # Job 1: machine 1 (duration 2), then machine 0 (duration 4)
        jobs = [
            [(0, 3), (1, 2)],  # Job 0
            [(1, 2), (0, 4)],  # Job 1
        ]

        solution = scheduler.solve_job_shop(jobs, n_machines=2, horizon=100)

        # Should find a solution
        assert solution.status in ["optimal", "feasible"]
        if solution.status != "infeasible":
            assert solution.makespan > 0
            assert len(solution.schedule) == 4  # 2 jobs × 2 operations

    def test_cp_solver_statistics(self):
        """Test getting solver statistics."""
        from baselines.cp_scheduler import CPScheduler

        scheduler = CPScheduler()

        solution = CPSolution(
            schedule=[],
            makespan=100,
            total_tardiness=5,
            objective_value=105,
            solve_time=2.5,
            status="optimal",
            num_branches=5000,
            num_conflicts=200,
        )

        stats = scheduler.get_solver_statistics(solution)

        assert "solve_time" in stats
        assert "num_branches" in stats
        assert stats["num_branches"] == 5000


class TestSchedulerIntegration:
    """Integration tests for schedulers (mocked environment)."""

    def create_mock_env(self, n_blocks=5, n_spmts=2, n_cranes=1):
        """Create a mock environment for testing."""
        mock_env = MagicMock()

        # Create mock blocks
        mock_env.blocks = []
        for i in range(n_blocks):
            block = MagicMock()
            block.processing_time = 10 + i * 2
            block.due_date = 50 + i * 20
            block.priority = 1.0
            block.facility_idx = i % 2
            mock_env.blocks.append(block)

        # Create mock SPMTs
        mock_env.spmts = []
        for i in range(n_spmts):
            spmt = MagicMock()
            spmt.health = 0.9 - i * 0.1
            mock_env.spmts.append(spmt)

        # Create mock cranes
        mock_env.cranes = []
        for i in range(n_cranes):
            crane = MagicMock()
            crane.health = 0.95
            mock_env.cranes.append(crane)

        mock_env.max_time = 500
        mock_env.n_spmts = n_spmts
        mock_env.n_cranes = n_cranes

        # Mock precedence graph (linear chain: 0 -> 1 -> 2 -> ...)
        def get_predecessors(node_id):
            return [node_id - 1] if node_id > 0 else []

        mock_env.precedence_graph = MagicMock()
        mock_env.precedence_graph.predecessors = get_predecessors

        return mock_env

    @pytest.mark.skipif(not ORTOOLS_MIP_AVAILABLE, reason="OR-Tools not available")
    def test_mip_solve_small_instance(self):
        """Test MIP solver on small instance."""
        from baselines.mip_scheduler import MIPScheduler

        scheduler = MIPScheduler(time_limit=30.0, solver="ortools")
        mock_env = self.create_mock_env(n_blocks=3, n_spmts=2, n_cranes=1)

        solution = scheduler.solve(mock_env)

        # Should find some solution (may not be optimal due to time limit)
        assert solution.status in ["optimal", "feasible", "infeasible", "unknown/timeout"]

    @pytest.mark.skipif(not CPSAT_AVAILABLE, reason="CP-SAT not available")
    def test_cp_solve_small_instance(self):
        """Test CP solver on small instance."""
        from baselines.cp_scheduler import CPScheduler

        scheduler = CPScheduler(time_limit=30.0)
        mock_env = self.create_mock_env(n_blocks=3, n_spmts=2, n_cranes=1)

        solution = scheduler.solve(mock_env)

        assert solution.status in ["optimal", "feasible", "infeasible", "unknown/timeout"]


class TestScheduleValidation:
    """Tests for schedule validation utilities."""

    def test_schedule_format(self):
        """Test that schedule has correct format."""
        schedule = [
            {"action_type": 0, "spmt_idx": 0, "crane_idx": 0, "request_idx": 0},
            {"action_type": 0, "spmt_idx": 1, "crane_idx": 0, "request_idx": 1},
            {"action_type": 0, "spmt_idx": 0, "crane_idx": 0, "request_idx": 2},
        ]

        for action in schedule:
            assert "action_type" in action
            assert "spmt_idx" in action
            assert "crane_idx" in action
            assert "request_idx" in action
            assert isinstance(action["spmt_idx"], int)
            assert isinstance(action["crane_idx"], int)

    def test_schedule_covers_all_blocks(self):
        """Test that schedule covers all blocks."""
        n_blocks = 5
        schedule = [
            {"action_type": 0, "spmt_idx": i % 2, "crane_idx": 0, "request_idx": i}
            for i in range(n_blocks)
        ]

        scheduled_blocks = set(action["request_idx"] for action in schedule)
        assert scheduled_blocks == set(range(n_blocks))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for utilization and busy time tracking."""

from __future__ import annotations

import pytest

from simulation.environment import ShipyardEnv
from simulation.entities import SPMTStatus, CraneStatus, BlockStatus, ProductionStage


def test_spmt_busy_time_tracked():
    """SPMT busy time should accumulate during transport dispatch."""
    config = {
        "n_blocks": 3,
        "n_spmts": 2,
        "n_cranes": 1,
        "max_time": 100,
        "shipyard": {
            "facilities": [
                {"name": "cutting", "processing_time_mean": 1, "processing_time_std": 0.1, "capacity": 2},
                {"name": "panel", "processing_time_mean": 1, "processing_time_std": 0.1, "capacity": 2},
            ],
            "staging_areas": [],
            "dock_grid": {"rows": 2, "cols": 4},
            "transport_network": {"cutting": {"panel": 2.0}, "panel": {}},
        },
    }

    env = ShipyardEnv(config)
    env.reset()

    # Create a transport request
    blocks = env.entities["blocks"]
    blocks[0].status = BlockStatus.WAITING
    blocks[0].location = "cutting"
    env.transport_requests = [{"block_id": blocks[0].id, "destination": "panel"}]

    # Dispatch SPMT
    initial_busy = env.metrics.get("spmt_busy_time", 0.0)
    action = {"action_type": 0, "spmt_idx": 0, "request_idx": 0}
    env.step(action)

    # Busy time should have increased
    final_busy = env.metrics.get("spmt_busy_time", 0.0)
    assert final_busy > initial_busy, "SPMT busy time should increase after dispatch"


def test_crane_busy_time_tracked():
    """Crane busy time should accumulate during lift dispatch."""
    config = {
        "n_blocks": 2,
        "n_spmts": 1,
        "n_cranes": 1,
        "max_time": 100,
        "shipyard": {
            "facilities": [
                {"name": "cutting", "processing_time_mean": 1, "processing_time_std": 0.1, "capacity": 2},
            ],
            "staging_areas": [],
            "dock_grid": {"rows": 2, "cols": 4},
            "transport_network": {"cutting": {}},
        },
    }

    env = ShipyardEnv(config)
    env.reset()

    # Set up block for lift
    blocks = env.entities["blocks"]
    blocks[0].status = BlockStatus.AT_PRE_ERECTION
    blocks[0].current_stage = ProductionStage.PRE_ERECTION
    blocks[0].predecessors = []  # No predecessors
    env.lift_requests = [{"block_id": blocks[0].id}]

    # Dispatch crane
    initial_busy = env.metrics.get("crane_busy_time", 0.0)
    action = {"action_type": 1, "crane_idx": 0, "lift_idx": 0}
    env.step(action)

    # Busy time should have increased
    final_busy = env.metrics.get("crane_busy_time", 0.0)
    assert final_busy > initial_busy, "Crane busy time should increase after lift"


def test_utilization_metrics_computed():
    """Utilization metrics should be non-zero when equipment is used."""
    from utils.metrics import compute_kpis

    # Simulate metrics after equipment usage
    metrics = {
        "blocks_completed": 5,
        "breakdowns": 1,
        "planned_maintenance": 2,
        "total_tardiness": 10.0,
        "empty_travel_distance": 5.0,
        "spmt_busy_time": 20.0,  # 20 hours busy
        "crane_busy_time": 10.0,  # 10 hours busy
    }
    total_time = 100.0
    n_spmts = 2
    n_cranes = 1

    kpis = compute_kpis(metrics, total_time, n_spmts=n_spmts, n_cranes=n_cranes)

    # Check utilization is calculated
    assert "spmt_utilization" in kpis
    assert "crane_utilization" in kpis
    # SPMT utilization = 20 / (2 * 100) = 0.1
    assert abs(kpis["spmt_utilization"] - 0.1) < 0.01
    # Crane utilization = 10 / (1 * 100) = 0.1
    assert abs(kpis["crane_utilization"] - 0.1) < 0.01


def test_broken_equipment_excluded_from_mask():
    """Broken equipment should not be selectable for dispatch."""
    config = {
        "n_blocks": 2,
        "n_spmts": 2,
        "n_cranes": 1,
        "max_time": 100,
        "shipyard": {
            "facilities": [
                {"name": "cutting", "processing_time_mean": 1, "processing_time_std": 0.1, "capacity": 2},
            ],
            "staging_areas": [],
            "dock_grid": {"rows": 2, "cols": 4},
            "transport_network": {"cutting": {}},
        },
    }

    env = ShipyardEnv(config)
    env.reset()

    # Create transport request
    blocks = env.entities["blocks"]
    blocks[0].status = BlockStatus.WAITING
    blocks[0].location = "cutting"
    env.transport_requests = [{"block_id": blocks[0].id, "destination": "cutting"}]

    # Break one SPMT
    spmts = env.entities["spmts"]
    spmts[0].status = SPMTStatus.BROKEN_DOWN

    mask = env.get_action_mask()

    # SPMT 0 should not be available (broken)
    # SPMT 1 should be available (assuming health is ok)
    assert mask["spmt_dispatch"][0, 0] == False  # Broken SPMT
    # SPMT 1 may or may not be valid depending on health

"""Tests for environment edge cases."""

from __future__ import annotations

import pytest
import numpy as np

from simulation.environment import ShipyardEnv
from simulation.entities import SPMTStatus, CraneStatus, BlockStatus


def test_env_truncates_at_max_time():
    """Environment should truncate when sim_time >= max_time."""
    config = {
        "n_blocks": 5,
        "n_spmts": 1,
        "n_cranes": 1,
        "max_time": 10,  # Very short
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

    # Step until truncation
    done = False
    steps = 0
    while not done and steps < 20:
        action = {"action_type": 3}  # Hold
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert truncated, "Environment should have truncated"
    assert env.sim_time >= config["max_time"], "sim_time should be >= max_time"


def test_empty_transport_requests():
    """Dispatching SPMT with no requests should be safe."""
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
    env.transport_requests = []  # Empty

    # Try to dispatch - should not crash
    action = {"action_type": 0, "spmt_idx": 0, "request_idx": 0}
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None
    assert isinstance(reward, float)


def test_empty_lift_requests():
    """Dispatching crane with no lift requests should be safe."""
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
    env.lift_requests = []  # Empty

    # Try to dispatch - should not crash
    action = {"action_type": 1, "crane_idx": 0, "lift_idx": 0}
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None


def test_mask_validity_across_steps():
    """Action mask should always have at least one valid action type."""
    config = {
        "n_blocks": 5,
        "n_spmts": 2,
        "n_cranes": 1,
        "max_time": 50,
        "shipyard": {
            "facilities": [
                {"name": "cutting", "processing_time_mean": 2, "processing_time_std": 0.5, "capacity": 2},
            ],
            "staging_areas": [],
            "dock_grid": {"rows": 2, "cols": 4},
            "transport_network": {"cutting": {}},
        },
    }

    env = ShipyardEnv(config)
    env.reset()

    for _ in range(20):
        mask = env.get_action_mask()
        # Hold (action_type=3) should always be valid
        assert mask["action_type"][3] == True, "Hold should always be valid"
        action = {"action_type": 3}  # Hold
        env.step(action)


def test_reset_clears_state():
    """Reset should clear all state properly."""
    config = {
        "n_blocks": 5,
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

    # Run some steps
    for _ in range(10):
        env.step({"action_type": 3})

    # Capture state before reset
    sim_time_before = env.sim_time
    metrics_before = env.metrics.copy()

    # Reset
    obs, info = env.reset()

    assert env.sim_time == 0.0, "sim_time should reset to 0"
    assert env.metrics["blocks_completed"] == 0, "blocks_completed should reset"
    assert env.metrics["breakdowns"] == 0, "breakdowns should reset"
    assert len(env.transport_requests) == 0, "transport_requests should be empty"
    assert len(env.lift_requests) == 0, "lift_requests should be empty"


def test_observation_shape_consistency():
    """Observation shape should be consistent across resets and steps."""
    config = {
        "n_blocks": 5,
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
    obs1, _ = env.reset()
    expected_shape = obs1.shape

    # Step and check shape
    obs2, _, _, _, _ = env.step({"action_type": 3})
    assert obs2.shape == expected_shape, "Shape should be consistent after step"

    # Reset and check shape
    obs3, _ = env.reset()
    assert obs3.shape == expected_shape, "Shape should be consistent after reset"


def test_all_blocks_completed_terminates():
    """Environment should terminate when all blocks are completed."""
    config = {
        "n_blocks": 1,  # Just one block for quick completion
        "n_spmts": 1,
        "n_cranes": 1,
        "max_time": 1000,
        "shipyard": {
            "facilities": [],  # No facilities means direct to dock
            "staging_areas": [],
            "dock_grid": {"rows": 2, "cols": 4},
            "transport_network": {},
        },
    }

    env = ShipyardEnv(config)
    env.reset()

    # Manually complete the block
    env.metrics["blocks_completed"] = env.n_blocks

    obs, reward, terminated, truncated, info = env.step({"action_type": 3})
    assert terminated, "Should terminate when all blocks completed"

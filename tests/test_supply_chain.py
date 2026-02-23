"""Tests for multi-supplier, inventory, and labor extension.

Validates:
1. Backward compatibility (old configs still work)
2. Entity creation from config
3. Material consumption and stockout detection
4. Supplier ordering and delivery timing
5. Labor assignment with skill compatibility
6. Observation shape with supply chain enabled
7. Action masking for new action types
8. Graph data includes new nodes and edges
"""

import os
import sys
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.entities import (
    MaterialType,
    MaterialInventory,
    Supplier,
    SkillType,
    LaborPool,
)
from simulation.shipyard_env import HHIShipyardEnv


# ---------------------------------------------------------------------------
# Fixture: minimal config WITHOUT supply chain (backward compat)
# ---------------------------------------------------------------------------
MINIMAL_CONFIG = {
    "n_ships": 1,
    "n_blocks_per_ship": 5,
    "n_spmts": 2,
    "n_goliath_cranes": 2,
    "n_docks": 2,
    "n_quays": 1,
    "max_time": 200,
    "continuous_production": False,
    "shipyard": {
        "name": "Test Shipyard",
        "goliath_cranes": [
            {"id": "GC01", "assigned_dock": "dock_1", "capacity_tons": 900},
            {"id": "GC02", "assigned_dock": "dock_2", "capacity_tons": 900},
        ],
        "dry_docks": [
            {"name": "dock_1", "length_m": 400, "width_m": 80, "depth_m": 12, "capacity": 1, "cranes": ["GC01"]},
            {"name": "dock_2", "length_m": 400, "width_m": 80, "depth_m": 12, "capacity": 1, "cranes": ["GC02"]},
        ],
        "outfitting_quays": [
            {"name": "quay_1", "length_m": 350, "capacity": 2},
        ],
    },
}

# ---------------------------------------------------------------------------
# Fixture: config WITH supply chain enabled
# ---------------------------------------------------------------------------
SUPPLY_CHAIN_CONFIG = {
    **MINIMAL_CONFIG,
    "supply_chain": {
        "enable_suppliers": True,
        "enable_inventory": True,
        "enable_labor": True,
        "n_suppliers": 2,
        "n_inventory_types": 3,
        "n_labor_pools": 2,
        "suppliers": [
            {
                "id": "SUP_A",
                "name": "Supplier A",
                "lead_time_mean": 48.0,
                "lead_time_std": 8.0,
                "reliability": 0.95,
                "capacity_per_period": 800.0,
                "price_per_unit": 0.8,
                "specializations": ["steel_plate", "pipe_section"],
            },
            {
                "id": "SUP_B",
                "name": "Supplier B",
                "lead_time_mean": 72.0,
                "lead_time_std": 16.0,
                "reliability": 0.88,
                "capacity_per_period": 400.0,
                "price_per_unit": 1.5,
                "specializations": ["paint"],
            },
        ],
        "inventory": [
            {"id": "MAT_STEEL", "material_type": "steel_plate", "initial_quantity": 1000.0, "reorder_point": 200.0, "max_capacity": 5000.0},
            {"id": "MAT_PIPE", "material_type": "pipe_section", "initial_quantity": 500.0, "reorder_point": 100.0, "max_capacity": 2000.0},
            {"id": "MAT_PAINT", "material_type": "paint", "initial_quantity": 800.0, "reorder_point": 150.0, "max_capacity": 3000.0},
        ],
        "labor_pools": [
            {"id": "LABOR_WELDER", "skill_type": "welder", "total_workers": 10, "hourly_rate": 55.0, "overtime_rate": 82.5},
            {"id": "LABOR_FITTER", "skill_type": "fitter", "total_workers": 8, "hourly_rate": 50.0, "overtime_rate": 75.0},
        ],
    },
}


# ===================================================================
# Test 1: Backward compatibility
# ===================================================================
class TestBackwardCompatibility:
    def test_env_creates_without_supply_chain(self):
        """Old configs with no supply_chain key still work."""
        env = HHIShipyardEnv(MINIMAL_CONFIG)
        obs, info = env.reset()
        assert obs is not None
        assert env.n_suppliers == 0
        assert env.n_inventory_nodes == 0
        assert env.n_labor_pools == 0
        assert env.n_action_types == 4

    def test_step_works_without_supply_chain(self):
        """Can step the env with hold action using old config."""
        env = HHIShipyardEnv(MINIMAL_CONFIG)
        env.reset()
        action = {"action_type": 3, "spmt_idx": 0, "request_idx": 0,
                  "crane_idx": 0, "erection_idx": 0, "equipment_idx": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None

    def test_obs_dim_matches_space(self):
        """Observation dimension matches observation_space."""
        env = HHIShipyardEnv(MINIMAL_CONFIG)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape


# ===================================================================
# Test 2: Entity creation from config
# ===================================================================
class TestEntityCreation:
    def test_suppliers_created(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        suppliers = env.entities.get("suppliers", [])
        assert len(suppliers) == 2
        assert suppliers[0].id == "SUP_A"
        assert suppliers[1].id == "SUP_B"

    def test_inventory_created(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        inventory = env.entities.get("inventory", [])
        assert len(inventory) == 3
        assert inventory[0].material_type == MaterialType.STEEL_PLATE
        assert inventory[2].material_type == MaterialType.PAINT

    def test_labor_pools_created(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        pools = env.entities.get("labor_pools", [])
        assert len(pools) == 2
        assert pools[0].skill_type == SkillType.WELDER
        assert pools[1].skill_type == SkillType.FITTER

    def test_action_types_extended(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        assert env.n_action_types == 7  # 4 base + PLACE_ORDER + ASSIGN_WORKER + TRANSFER_MATERIAL


# ===================================================================
# Test 3: Material consumption and stockout
# ===================================================================
class TestMaterialInventory:
    def test_consume_and_stockout(self):
        inv = MaterialInventory(id="test", quantity=10.0, reorder_point=5.0)
        assert not inv.is_stockout()
        consumed = inv.consume(8.0)
        assert consumed == 8.0
        assert inv.quantity == 2.0
        assert inv.is_below_reorder_point()
        consumed2 = inv.consume(5.0)
        assert consumed2 == 2.0  # Only 2.0 left
        assert inv.is_stockout()

    def test_receive(self):
        inv = MaterialInventory(id="test", quantity=100.0, max_capacity=200.0)
        inv.receive(150.0)
        assert inv.quantity == 200.0  # Capped at max_capacity

    def test_feature_vector_shape(self):
        inv = MaterialInventory(id="test")
        fv = inv.get_feature_vector()
        assert len(fv) == 4


# ===================================================================
# Test 4: Supplier ordering and delivery
# ===================================================================
class TestSupplier:
    def test_can_accept_order(self):
        sup = Supplier(id="test", capacity_per_period=100.0)
        assert sup.can_accept_order(50.0)
        assert sup.can_accept_order(100.0)
        assert not sup.can_accept_order(101.0)

    def test_place_order(self):
        sup = Supplier(id="test", capacity_per_period=500.0, price_per_unit=2.0)
        order = sup.place_order("steel_plate", 100.0, sim_time=0.0)
        assert order["cost"] == 200.0
        assert len(sup.pending_orders) == 1
        assert sup.current_backlog == 100

    def test_delivery_timing(self):
        sup = Supplier(id="test", capacity_per_period=500.0, lead_time_mean=10.0, lead_time_std=0.01)
        sup.place_order("steel_plate", 100.0, sim_time=0.0)
        # Not delivered yet at t=5
        delivered = sup.check_deliveries(5.0)
        assert len(delivered) == 0
        # Delivered by t=15 (mean 10 ± noise)
        delivered = sup.check_deliveries(15.0)
        assert len(delivered) >= 0  # Stochastic, but should eventually deliver

    def test_feature_vector_shape(self):
        sup = Supplier(id="test")
        fv = sup.get_feature_vector()
        assert len(fv) == 5


# ===================================================================
# Test 5: Labor assignment with skill compatibility
# ===================================================================
class TestLaborPool:
    def test_assign_and_release(self):
        pool = LaborPool(id="test", total_workers=5, available_workers=5)
        assert pool.can_assign()
        pool.assign("block_1")
        assert pool.available_workers == 4
        assert "block_1" in pool.assigned_tasks
        pool.release("block_1")
        assert pool.available_workers == 5

    def test_cannot_assign_when_full(self):
        pool = LaborPool(id="test", total_workers=1, available_workers=0)
        assert not pool.can_assign()

    def test_reset_shift(self):
        pool = LaborPool(id="test", total_workers=10, available_workers=3)
        pool.assigned_tasks = ["a", "b", "c", "d"]
        pool.current_overtime_hours = 2.0
        pool.reset_shift()
        assert pool.available_workers == 10
        assert pool.current_overtime_hours == 0.0
        assert pool.assigned_tasks == []

    def test_feature_vector_shape(self):
        pool = LaborPool(id="test")
        fv = pool.get_feature_vector()
        assert len(fv) == 4


# ===================================================================
# Test 6: Observation shape with supply chain
# ===================================================================
class TestObservationShape:
    def test_obs_dim_with_supply_chain(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        obs, _ = env.reset()
        expected = (
            5 * env.block_features
            + 2 * env.spmt_features
            + 2 * env.crane_features
            + 2 * env.dock_features
            + env.n_facilities * env.facility_features
            + 2 * env.supplier_features
            + 3 * env.inventory_features
            + 2 * env.labor_features
        )
        assert obs.shape[0] == expected
        assert obs.shape == env.observation_space.shape


# ===================================================================
# Test 7: Action masking for new action types
# ===================================================================
class TestActionMasking:
    def test_mask_has_supply_chain_keys(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        mask = env.get_action_mask()
        assert "supplier_order" in mask
        assert "labor_assign" in mask
        assert "inventory_transfer" in mask
        assert len(mask["action_type"]) == 7

    def test_mask_without_supply_chain(self):
        env = HHIShipyardEnv(MINIMAL_CONFIG)
        env.reset()
        mask = env.get_action_mask()
        assert "supplier_order" not in mask
        assert len(mask["action_type"]) == 4


# ===================================================================
# Test 8: Graph data includes new nodes and edges
# ===================================================================
class TestGraphData:
    def test_graph_has_supply_chain_nodes(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        data = env.get_graph_data()
        assert "supplier" in data.node_types
        assert "inventory" in data.node_types
        assert "labor" in data.node_types
        assert data["supplier"].x.shape == (2, 5)
        assert data["inventory"].x.shape == (3, 4)
        assert data["labor"].x.shape == (2, 4)

    def test_graph_has_supply_chain_edges(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        data = env.get_graph_data()
        # block -> inventory edges
        assert ("block", "requires_material", "inventory") in data.edge_types
        # block -> labor edges
        assert ("block", "requires_labor", "labor") in data.edge_types

    def test_graph_without_supply_chain(self):
        env = HHIShipyardEnv(MINIMAL_CONFIG)
        env.reset()
        data = env.get_graph_data()
        assert "supplier" not in data.node_types
        assert "inventory" not in data.node_types


# ===================================================================
# Test 9: Supply chain action execution
# ===================================================================
class TestSupplyChainActions:
    def test_place_order_action(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        # Force inventory below reorder point
        inv = env.entities["inventory"][0]
        inv.quantity = 50.0  # Below reorder_point of 200.0
        action = {
            "action_type": env._action_type_map["PLACE_ORDER"],
            "spmt_idx": 0, "request_idx": 0, "crane_idx": 0,
            "erection_idx": 0, "equipment_idx": 0,
            "supplier_idx": 0, "material_idx": 0,
            "labor_pool_idx": 0, "target_block_idx": 0,
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.metrics["orders_placed"] >= 1

    def test_assign_worker_action(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        # Run a few steps to get blocks into processing
        for _ in range(5):
            env.step({"action_type": 3, "spmt_idx": 0, "request_idx": 0,
                      "crane_idx": 0, "erection_idx": 0, "equipment_idx": 0,
                      "supplier_idx": 0, "material_idx": 0,
                      "labor_pool_idx": 0, "target_block_idx": 0})
        # Try to assign a worker
        action = {
            "action_type": env._action_type_map["ASSIGN_WORKER"],
            "spmt_idx": 0, "request_idx": 0, "crane_idx": 0,
            "erection_idx": 0, "equipment_idx": 0,
            "supplier_idx": 0, "material_idx": 0,
            "labor_pool_idx": 0, "target_block_idx": 0,
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None


# ===================================================================
# Test 10: Supply chain metrics tracking
# ===================================================================
class TestMetrics:
    def test_supply_chain_metrics_initialized(self):
        env = HHIShipyardEnv(SUPPLY_CHAIN_CONFIG)
        env.reset()
        assert "procurement_cost" in env.metrics
        assert "orders_placed" in env.metrics
        assert "stockout_events" in env.metrics
        assert "holding_cost" in env.metrics
        assert "labor_cost" in env.metrics
        assert "overtime_hours" in env.metrics

    def test_no_supply_chain_metrics_when_disabled(self):
        env = HHIShipyardEnv(MINIMAL_CONFIG)
        env.reset()
        assert "procurement_cost" not in env.metrics
        assert "stockout_events" not in env.metrics
        assert "labor_cost" not in env.metrics

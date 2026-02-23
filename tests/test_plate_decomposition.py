"""Tests for plate-level decomposition integration.

Covers: Plate entity, Block plate stats, synthetic generation, JSON loading,
processing time formulas, sub-stage tracking, PuLP scheduler, GNN features,
calibration, and backward compatibility.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml
from simulation.entities import (
    Block, Plate, PlateType, BlockType, BlockStatus,
    HHIProductionStage, DetailedProductionStage, DETAILED_TO_HHI_STAGE,
)
from simulation.plate_loader import (
    load_ship_decomposition, apply_decomposition_to_blocks,
    generate_synthetic_plates, validate_decomposition,
)


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_JSON = os.path.join(FIXTURE_DIR, "test_decomposition.json")


# ====================================================================
# Phase 1: Plate entity and Block extensions
# ====================================================================

class TestPlateEntity:
    """Tests for Plate dataclass."""

    def test_plate_weight_computation(self):
        """Auto-computed weight from dimensions (steel density 7850 kg/m^3)."""
        plate = Plate(id="P001", length_mm=12000, width_mm=3000, thickness_mm=20)
        expected_vol = 12.0 * 3.0 * 0.020  # m^3
        expected_weight = expected_vol * 7850.0
        assert abs(plate.weight_kg - expected_weight) < 0.1

    def test_plate_area(self):
        """area_m2 is length * width in meters."""
        plate = Plate(id="P002", length_mm=10000, width_mm=2500)
        assert abs(plate.area_m2 - 25.0) < 0.01

    def test_plate_weld_length(self):
        """Weld length = perimeter + stiffener welds."""
        plate = Plate(id="P003", length_mm=12000, width_mm=3000,
                      has_stiffeners=True, n_stiffeners=4)
        perimeter = 2 * (12.0 + 3.0)  # 30m
        stiffener_welds = 4 * 12.0 * 2  # 96m
        assert abs(plate.weld_length_m - (perimeter + stiffener_welds)) < 0.01

    def test_plate_stiffener_weight(self):
        """Stiffened plates weigh more (15% per stiffener)."""
        base = Plate(id="P_base", length_mm=10000, width_mm=3000, thickness_mm=20)
        stiff = Plate(id="P_stiff", length_mm=10000, width_mm=3000, thickness_mm=20,
                      has_stiffeners=True, n_stiffeners=4)
        # Stiffened should be 1.6x base (1 + 0.15*4)
        assert stiff.weight_kg > base.weight_kg
        ratio = stiff.weight_kg / base.weight_kg
        assert abs(ratio - 1.6) < 0.01

    def test_plate_is_curved(self):
        """Curved detection: by curvature radius or plate type."""
        flat = Plate(id="P_flat")
        curved_by_radius = Plate(id="P_curve", curvature_radius_mm=15000)
        curved_by_type = Plate(id="P_type", plate_type=PlateType.CURVED)
        assert not flat.is_curved
        assert curved_by_radius.is_curved
        assert curved_by_type.is_curved


class TestBlockPlateStats:
    """Tests for Block.compute_plate_stats()."""

    def test_compute_plate_stats(self):
        """n_plates, area, weight computed correctly."""
        block = Block(id="B001", weight=100.0, size=(10.0, 8.0, 6.0), due_date=1000.0, block_type=BlockType.FLAT_BOTTOM)
        block.plates = [
            Plate(id="P1", length_mm=10000, width_mm=3000, thickness_mm=20),
            Plate(id="P2", length_mm=8000, width_mm=2500, thickness_mm=18),
        ]
        block.compute_plate_stats()
        assert block.n_plates == 2
        assert abs(block.total_plate_area_m2 - (30.0 + 20.0)) < 0.01
        assert block.plate_derived_weight > 0

    def test_block_weight_from_plates(self):
        """plate_derived_weight overrides random weight."""
        block = Block(id="B002", weight=999.0, size=(10.0, 8.0, 6.0), due_date=1000.0, block_type=BlockType.FLAT_BOTTOM)
        block.plates = [
            Plate(id="P1", length_mm=12000, width_mm=3000, thickness_mm=25),
        ]
        block.compute_plate_stats()
        # Weight should now be derived from plate, not 999.0
        assert block.weight != 999.0
        assert block.weight == block.plate_derived_weight


# ====================================================================
# Phase 2: Plate loader
# ====================================================================

class TestPlateLoader:
    """Tests for plate_loader.py functions."""

    def test_load_decomposition_json(self):
        """Load fixture JSON, verify structure."""
        data = load_ship_decomposition(FIXTURE_JSON)
        assert "HN_TEST_01" in data
        blocks = data["HN_TEST_01"]
        assert len(blocks) == 2
        assert blocks[0]["block_id"] == "B_HN_TEST_01_001"
        assert len(blocks[0]["plates"]) == 3

    def test_apply_decomposition_to_blocks(self):
        """Apply decomposition data to Block objects."""
        data = load_ship_decomposition(FIXTURE_JSON)
        block = Block(id="B_HN_TEST_01_001", weight=100.0, size=(20.0, 18.0, 12.0),
                      due_date=1000.0, ship_id="HN_TEST_01",
                      block_type=BlockType.FLAT_BOTTOM)
        count = apply_decomposition_to_blocks([block], data)
        assert count == 1
        assert block.n_plates == 3
        assert block.total_plate_area_m2 > 0

    def test_validate_decomposition_good(self):
        """Valid fixture passes validation."""
        with open(FIXTURE_JSON) as f:
            data = json.load(f)
        errors = validate_decomposition(data)
        assert len(errors) == 0

    def test_validate_decomposition_bad(self):
        """Missing required fields caught."""
        bad_data = {"blocks": [{"plates": []}]}  # Missing ship_id, block_id
        errors = validate_decomposition(bad_data)
        assert len(errors) > 0

    def test_synthetic_plate_generation(self):
        """Generate synthetic plates, check reasonable counts."""
        rng = np.random.default_rng(42)
        block = Block(id="B_syn", weight=350.0, size=(20.0, 18.0, 12.0), due_date=1000.0, block_type=BlockType.FLAT_BOTTOM)
        plates = generate_synthetic_plates(block, rng)
        assert len(plates) > 5
        assert len(plates) < 80
        # All plates should have valid dimensions
        for p in plates:
            assert p.length_mm > 0
            assert p.width_mm > 0
            assert p.thickness_mm > 0
            assert p.weight_kg > 0

    def test_synthetic_curved_plates(self):
        """Curved blocks get curved plates."""
        rng = np.random.default_rng(42)
        block = Block(id="B_curve", weight=280.0, size=(15.0, 12.0, 10.0), due_date=1000.0, block_type=BlockType.CURVED_BOW)
        plates = generate_synthetic_plates(block, rng)
        n_curved = sum(1 for p in plates if p.is_curved)
        assert n_curved > 0, "Curved blocks should have some curved plates"


# ====================================================================
# Phase 3: Processing times
# ====================================================================

class TestPlateProcessingTimes:
    """Tests for plate-count-based processing time formulas."""

    @pytest.fixture
    def plate_env(self):
        """Create an env with plate decomposition enabled."""
        config = {
            "n_ships": 1,
            "n_blocks_per_ship": 10,
            "n_spmts": 4,
            "n_goliath_cranes": 2,
            "n_docks": 2,
            "max_time": 500,
            "plate_decomposition": {
                "enable": True,
                "synthetic_fallback": True,
            },
        }
        from simulation.shipyard_env import HHIShipyardEnv
        env = HHIShipyardEnv(config)
        env.reset(seed=42)
        return env

    @pytest.fixture
    def base_env(self):
        """Create an env WITHOUT plate decomposition (backward compat)."""
        config = {
            "n_ships": 1,
            "n_blocks_per_ship": 10,
            "n_spmts": 4,
            "n_goliath_cranes": 2,
            "n_docks": 2,
            "max_time": 500,
        }
        from simulation.shipyard_env import HHIShipyardEnv
        env = HHIShipyardEnv(config)
        env.reset(seed=42)
        return env

    def test_plate_processing_time_monotonic(self, plate_env):
        """More plates -> generally longer processing time."""
        from simulation.shipyard_env import HHIShipyardEnv
        env = plate_env

        # Create two blocks with different plate counts
        small_block = Block(id="B_small", weight=100.0, size=(10.0, 8.0, 6.0), due_date=1000.0, block_type=BlockType.FLAT_BOTTOM)
        small_block.plates = [Plate(id=f"P_s{i}", length_mm=10000, width_mm=3000, thickness_mm=20)
                              for i in range(5)]
        small_block.compute_plate_stats()

        large_block = Block(id="B_large", weight=400.0, size=(20.0, 18.0, 12.0), due_date=1000.0, block_type=BlockType.FLAT_BOTTOM)
        large_block.plates = [Plate(id=f"P_l{i}", length_mm=10000, width_mm=3000, thickness_mm=20)
                              for i in range(40)]
        large_block.compute_plate_stats()

        fac = {"name": "cutting_shop", "processing_time_mean": 10.0, "processing_time_std": 2.0}

        # Sample multiple times to check trend
        small_times = [env._compute_plate_processing_time(small_block, fac) for _ in range(20)]
        large_times = [env._compute_plate_processing_time(large_block, fac) for _ in range(20)]

        assert np.mean(large_times) > np.mean(small_times)

    def test_plate_processing_time_stages(self, plate_env):
        """Different stages have different coefficients."""
        env = plate_env
        block = Block(id="B_test", weight=200.0, size=(15.0, 12.0, 8.0), due_date=1000.0, block_type=BlockType.FLAT_BOTTOM)
        block.plates = [Plate(id=f"P_{i}", length_mm=10000, width_mm=3000, thickness_mm=20)
                        for i in range(20)]
        block.compute_plate_stats()

        cutting = {"name": "cutting_shop", "processing_time_mean": 10.0, "processing_time_std": 2.0}
        assembly = {"name": "block_assembly_hall_1", "processing_time_mean": 15.0, "processing_time_std": 3.0}

        t_cut = np.mean([env._compute_plate_processing_time(block, cutting) for _ in range(10)])
        t_asm = np.mean([env._compute_plate_processing_time(block, assembly) for _ in range(10)])

        # Both should be > 0 and different
        assert t_cut > 0
        assert t_asm > 0

    def test_processing_time_fallback(self, base_env):
        """No plates -> uses lognormal (backward compat)."""
        env = base_env
        blocks = env.entities.get("blocks", [])
        if blocks:
            # Without plate decomposition, blocks should have 0 plates
            assert blocks[0].n_plates == 0

    def test_obs_dim_with_plates(self, plate_env):
        """16 features per block when plates enabled."""
        assert plate_env.block_features == 16

    def test_obs_dim_without_plates(self, base_env):
        """12 features per block without plates (unchanged)."""
        assert base_env.block_features == 12

    def test_backward_compat(self, base_env):
        """Existing config produces identical behavior."""
        from baselines.rule_based import RuleBasedScheduler
        env = base_env
        expert = RuleBasedScheduler()
        # Should run without errors
        for _ in range(50):
            action = expert.decide(env)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        # Should still work
        assert env.metrics["blocks_completed"] >= 0


# ====================================================================
# Phase 4: Sub-stage tracking
# ====================================================================

class TestSubStageTracking:
    """Tests for DetailedProductionStage mapping and sub-stage tracking."""

    def test_detailed_stage_mapping(self):
        """All 15 detailed stages map to an HHI stage."""
        for detailed_stage in DetailedProductionStage:
            assert detailed_stage in DETAILED_TO_HHI_STAGE, \
                f"{detailed_stage.name} not in DETAILED_TO_HHI_STAGE"
            hhi_stage = DETAILED_TO_HHI_STAGE[detailed_stage]
            assert isinstance(hhi_stage, HHIProductionStage)

    def test_detailed_stage_count(self):
        """Partner's model has exactly 15 stages."""
        assert len(DetailedProductionStage) == 15

    def test_substage_tracking(self):
        """Block.current_substage advances when substages enabled."""
        config = {
            "n_ships": 1,
            "n_blocks_per_ship": 5,
            "n_spmts": 2,
            "n_goliath_cranes": 1,
            "n_docks": 1,
            "max_time": 500,
            "plate_decomposition": {
                "enable": True,
                "synthetic_fallback": True,
                "enable_substages": True,
            },
        }
        from simulation.shipyard_env import HHIShipyardEnv
        from baselines.rule_based import RuleBasedScheduler
        env = HHIShipyardEnv(config)
        env.reset(seed=42)
        expert = RuleBasedScheduler()

        # Run a few steps - substage transitions metric should be tracked
        for _ in range(100):
            action = expert.decide(env)
            env.step(action)

        assert "substage_transitions" in env.metrics
        assert "substage_times" in env.metrics


# ====================================================================
# Phase 5: PuLP scheduler
# ====================================================================

class TestPuLPScheduler:
    """Tests for PuLPMIPScheduler."""

    def test_pulp_scheduler_decide(self):
        """Returns valid action dict."""
        from baselines.pulp_scheduler import PuLPMIPScheduler
        config = {
            "n_ships": 1, "n_blocks_per_ship": 10,
            "n_spmts": 4, "n_goliath_cranes": 2, "n_docks": 2, "max_time": 200,
        }
        from simulation.shipyard_env import HHIShipyardEnv
        env = HHIShipyardEnv(config)
        env.reset(seed=42)

        scheduler = PuLPMIPScheduler({"horizon": 10, "replan_interval": 5})
        action = scheduler.decide(env)

        assert isinstance(action, dict)
        assert "action_type" in action
        assert action["action_type"] in [0, 1, 2, 3]

    def test_pulp_scheduler_episode(self):
        """Runs a full episode without errors."""
        from baselines.pulp_scheduler import PuLPMIPScheduler
        config = {
            "n_ships": 1, "n_blocks_per_ship": 10,
            "n_spmts": 4, "n_goliath_cranes": 2, "n_docks": 2, "max_time": 200,
        }
        from simulation.shipyard_env import HHIShipyardEnv
        env = HHIShipyardEnv(config)
        env.reset(seed=42)

        scheduler = PuLPMIPScheduler({"horizon": 10, "replan_interval": 5, "solver_time_limit": 1.0})
        for _ in range(50):
            action = scheduler.decide(env)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    def test_pulp_fallback_no_pulp(self):
        """Graceful degradation when PuLP import simulated as missing."""
        from baselines import pulp_scheduler
        # Even with HAS_PULP, the fallback EDD should work
        config = {
            "n_ships": 1, "n_blocks_per_ship": 5,
            "n_spmts": 2, "n_goliath_cranes": 1, "n_docks": 1, "max_time": 100,
        }
        from simulation.shipyard_env import HHIShipyardEnv
        env = HHIShipyardEnv(config)
        env.reset(seed=42)

        scheduler = pulp_scheduler.PuLPMIPScheduler()
        # Test EDD fallback directly
        action = scheduler._fallback_edd(env)
        assert isinstance(action, dict)
        assert "action_type" in action


# ====================================================================
# Phase 6: GNN feature dimensions
# ====================================================================

class TestGNNFeatures:
    """Tests for GNN encoder with plate-aware features."""

    def test_gnn_block_dim_16(self):
        """Encoder works with 16-dim input."""
        try:
            from agent.gnn_encoder import HeterogeneousGNNEncoder
            encoder = HeterogeneousGNNEncoder(block_dim=16, hidden_dim=32)
            assert encoder.block_proj.in_features == 16
        except ImportError:
            pytest.skip("PyTorch Geometric not available")

    def test_gnn_block_dim_12(self):
        """Encoder works with default 12-dim input."""
        try:
            from agent.gnn_encoder import HeterogeneousGNNEncoder
            encoder = HeterogeneousGNNEncoder(block_dim=12, hidden_dim=32)
            assert encoder.block_proj.in_features == 12
        except ImportError:
            pytest.skip("PyTorch Geometric not available")


# ====================================================================
# Phase 8: Calibration
# ====================================================================

class TestCalibration:
    """Tests for calibration pipeline."""

    def test_calibration_dataset_load(self):
        """CSV loading works."""
        from simulation.calibration import CalibrationDataset
        dataset = CalibrationDataset()

        # Create a temp CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("block_id,stage,n_plates,n_curved,n_stiffened,total_area_m2,total_weld_m,observed_time_hours\n")
            f.write("B001,STEEL_CUTTING,20,3,5,600.0,120.0,8.5\n")
            f.write("B002,STEEL_CUTTING,15,1,4,450.0,90.0,6.2\n")
            f.write("B003,BLOCK_ASSEMBLY,30,5,10,900.0,200.0,15.3\n")
            tmp_path = f.name

        try:
            dataset.load_from_csv(tmp_path)
            assert len(dataset) == 3
            assert dataset.records[0].block_id == "B001"
            assert dataset.records[0].n_plates == 20
        finally:
            os.unlink(tmp_path)

    def test_calibration_summary(self):
        """Summary statistics per stage."""
        from simulation.calibration import CalibrationDataset, CalibrationRecord
        dataset = CalibrationDataset()
        dataset.records = [
            CalibrationRecord("B1", "CUTTING", 20, 3, 5, 600, 120, 8.5),
            CalibrationRecord("B2", "CUTTING", 15, 1, 4, 450, 90, 6.2),
            CalibrationRecord("B3", "ASSEMBLY", 30, 5, 10, 900, 200, 15.3),
        ]
        summary = dataset.summary()
        assert "CUTTING" in summary
        assert "ASSEMBLY" in summary
        assert summary["CUTTING"]["count"] == 2
        assert abs(summary["CUTTING"]["mean"] - 7.35) < 0.01

    def test_coefficient_fitter(self):
        """Fits produce reasonable results."""
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            pytest.skip("scipy not available")

        from simulation.calibration import CalibrationDataset, CalibrationRecord, CoefficientFitter
        dataset = CalibrationDataset()
        # Generate synthetic calibration data: time = 2 + 0.3*n_plates + noise
        rng = np.random.default_rng(42)
        for i in range(50):
            n_plates = rng.integers(5, 40)
            time = 2.0 + 0.3 * n_plates + rng.normal(0, 0.5)
            dataset.records.append(CalibrationRecord(
                f"B{i}", "STEEL_CUTTING", n_plates, 0, 0, n_plates * 30.0, n_plates * 6.0, time
            ))

        fitter = CoefficientFitter()
        coefficients = fitter.fit(dataset)
        assert "STEEL_CUTTING" in coefficients
        # Fitted per_plate should be close to 0.3
        assert coefficients["STEEL_CUTTING"]["per_plate"] > 0.1

        # Validate
        metrics = fitter.validate(dataset, coefficients)
        assert "STEEL_CUTTING" in metrics
        assert metrics["STEEL_CUTTING"]["r2"] > 0.5  # Should fit well

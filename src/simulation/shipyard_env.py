"""Single-yard Gymnasium environment for HD Hyundai Heavy Industries shipyard scheduling.

This environment models LNG carrier production at HHI Ulsan shipyard in South Korea.
The world's largest shipyard with 10 dry docks and 9 Goliath cranes.

The agent manages:
- Block production through 11 stages (steel cutting to delivery)
- SPMT dispatching for internal transport
- Goliath crane operations for block erection
- Equipment health and maintenance
- Multiple ships in concurrent production
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .shipyard import HHIShipyardGraph, HHI_ULSAN_DEFAULT_CONFIG
from .entities import (
    Block,
    SPMT,
    GoliathCrane,
    LNGCarrier,
    DryDock,
    OutfittingQuay,
    HHIProductionStage,
    BlockStatus,
    BlockType,
    SPMTStatus,
    GoliathCraneStatus,
    ShipStatus,
    # Supply chain entities
    Supplier,
    MaterialInventory,
    MaterialType,
    LaborPool,
    SkillType,
    # Plate decomposition
    DetailedProductionStage,
)
from .degradation import WienerDegradationModel
from .precedence import is_predecessor_complete


class HHIShipyardEnv(gym.Env):
    """Gymnasium environment for HHI Ulsan shipyard scheduling.

    Models LNG carrier production:
    - Steel cutting -> Panel assembly -> Block assembly -> Outfitting -> Painting
    - Pre-erection -> Erection (Goliath cranes) -> Quay outfitting -> Sea trials -> Delivery
    """

    metadata = {"render_modes": ["human", "none"]}

    # Plate-count processing time coefficients per production stage
    _PLATE_TIME_COEFFICIENTS = {
        HHIProductionStage.STEEL_CUTTING: {
            "per_plate": 0.3, "per_curved": 0.6, "base_hours": 2.0,
        },
        HHIProductionStage.PART_FABRICATION: {
            "per_plate": 0.4, "per_curved": 0.8, "base_hours": 3.0,
        },
        HHIProductionStage.PANEL_ASSEMBLY: {
            "per_stiffened": 0.5, "per_flat": 0.2, "base_hours": 4.0,
        },
        HHIProductionStage.BLOCK_ASSEMBLY: {
            "per_plate": 0.15, "per_area_m2": 0.01, "per_weld_m": 0.05, "base_hours": 8.0,
        },
        HHIProductionStage.BLOCK_OUTFITTING: {
            "per_plate": 0.1, "base_hours": 10.0,
        },
        HHIProductionStage.PAINTING: {
            "per_area_m2": 0.01, "base_hours": 4.0,
        },
        HHIProductionStage.PRE_ERECTION: {
            "per_plate": 0.05, "base_hours": 6.0,
        },
    }

    # Map facility names to production stages
    _STAGE_MAP = {
        "steel_stockyard": HHIProductionStage.STEEL_CUTTING,
        "cutting_shop": HHIProductionStage.STEEL_CUTTING,
        "part_fabrication": HHIProductionStage.PART_FABRICATION,
        "flat_panel_line_1": HHIProductionStage.PANEL_ASSEMBLY,
        "flat_panel_line_2": HHIProductionStage.PANEL_ASSEMBLY,
        "curved_block_shop": HHIProductionStage.PANEL_ASSEMBLY,
        "block_assembly_hall_1": HHIProductionStage.BLOCK_ASSEMBLY,
        "block_assembly_hall_2": HHIProductionStage.BLOCK_ASSEMBLY,
        "block_assembly_hall_3": HHIProductionStage.BLOCK_ASSEMBLY,
        "outfitting_shop": HHIProductionStage.BLOCK_OUTFITTING,
        "paint_shop": HHIProductionStage.PAINTING,
        "grand_block_staging_north": HHIProductionStage.PRE_ERECTION,
        "grand_block_staging_south": HHIProductionStage.PRE_ERECTION,
    }

    # Sub-stage mapping: facility -> list of detailed stages (partner's 15-stage model)
    _SUBSTAGE_FACILITIES = {
        "steel_stockyard": [DetailedProductionStage.RAW_MATERIAL_STORAGE],
        "cutting_shop": [
            DetailedProductionStage.PLATE_TRANSPORT_TO_FAB,
            DetailedProductionStage.FAB_CUTTING,
        ],
        "part_fabrication": [
            DetailedProductionStage.FAB_ROLLING,
            DetailedProductionStage.CUT_ROLL_STORAGE,
        ],
        "flat_panel_line_1": [DetailedProductionStage.PANEL_STIFFENER_WELDING],
        "flat_panel_line_2": [DetailedProductionStage.PANEL_STIFFENER_WELDING],
        "curved_block_shop": [DetailedProductionStage.PANEL_STIFFENER_WELDING],
        "block_assembly_hall_1": [
            DetailedProductionStage.BLOCK_ASSEMBLY_STORAGE,
            DetailedProductionStage.WELD_PLATE_TO_BLOCK,
            DetailedProductionStage.PLATES_ON_PARTIAL_BLOCK,
        ],
        "block_assembly_hall_2": [
            DetailedProductionStage.BLOCK_ASSEMBLY_STORAGE,
            DetailedProductionStage.WELD_PLATE_TO_BLOCK,
            DetailedProductionStage.PLATES_ON_PARTIAL_BLOCK,
        ],
        "block_assembly_hall_3": [
            DetailedProductionStage.BLOCK_ASSEMBLY_STORAGE,
            DetailedProductionStage.WELD_PLATE_TO_BLOCK,
            DetailedProductionStage.PLATES_ON_PARTIAL_BLOCK,
        ],
        "grand_block_staging_north": [DetailedProductionStage.TRANSPORT_TO_SHIP_ERECTION],
        "grand_block_staging_south": [DetailedProductionStage.TRANSPORT_TO_SHIP_ERECTION],
    }

    def __init__(self, config: dict, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.db_logging_enabled = False

        # Shipyard graph
        yard_config = config.get("shipyard", HHI_ULSAN_DEFAULT_CONFIG)
        self.shipyard = HHIShipyardGraph(yard_config)

        # Entity counts
        self.n_ships = int(config.get("n_ships", 8))
        self.n_blocks_per_ship = int(config.get("n_blocks_per_ship", 200))
        self.n_blocks = self.n_ships * self.n_blocks_per_ship
        self.n_spmts = int(config.get("n_spmts", 24))
        self.n_goliath_cranes = int(config.get("n_goliath_cranes", 9))
        self.n_docks = int(config.get("n_docks", 10))
        self.n_quays = int(config.get("n_quays", 3))

        # Continuous production mode
        self.continuous_production = config.get("continuous_production", False)
        self.total_ships_created = 0  # Counter for unique ship IDs

        # Count facilities
        self.n_facilities = len(self._get_all_facilities())

        # Plate decomposition configuration (must be parsed before feature dims)
        plate_cfg = config.get("plate_decomposition", {})
        self.enable_plate_decomposition = plate_cfg.get("enable", False)
        self._plate_data_dir = plate_cfg.get("data_dir", "")
        self._plate_synthetic_fallback = plate_cfg.get("synthetic_fallback", True)
        self._plate_enable_substages = plate_cfg.get("enable_substages", False)
        self._decomposition_data: Dict = {}

        # Feature dimensions (plate decomposition adds 4 extra block features)
        self.block_features = 16 if self.enable_plate_decomposition else 12
        self.spmt_features = 9
        self.crane_features = 11  # dock_norm + pos + status(4) + health(3) + load + capacity
        self.dock_features = 6
        self.facility_features = 4

        # Supply chain configuration (backward-compatible: disabled by default)
        # Must be parsed BEFORE observation/action space creation
        sc = config.get("supply_chain", {})
        self.enable_suppliers = sc.get("enable_suppliers", False)
        self.enable_inventory = sc.get("enable_inventory", False)
        self.enable_labor = sc.get("enable_labor", False)

        self.n_suppliers = int(sc.get("n_suppliers", 0)) if self.enable_suppliers else 0
        self.n_inventory_nodes = int(sc.get("n_inventory_types", 0)) if self.enable_inventory else 0
        self.n_labor_pools = int(sc.get("n_labor_pools", 0)) if self.enable_labor else 0

        self.supplier_features = 5
        self.inventory_features = 4
        self.labor_features = 4

        # Dynamic action type mapping
        self._action_type_map: Dict[str, int] = {
            "DISPATCH_SPMT": 0, "DISPATCH_CRANE": 1, "MAINTENANCE": 2, "HOLD": 3,
        }
        next_idx = 4
        if self.enable_suppliers:
            self._action_type_map["PLACE_ORDER"] = next_idx; next_idx += 1
        if self.enable_labor:
            self._action_type_map["ASSIGN_WORKER"] = next_idx; next_idx += 1
        if self.enable_inventory:
            self._action_type_map["TRANSFER_MATERIAL"] = next_idx; next_idx += 1
        self.n_action_types = next_idx

        # Supply chain reward weights
        self.w_stockout = float(sc.get("reward_stockout", 20.0))
        self.w_holding = float(sc.get("reward_holding_cost", 0.01))
        self.w_procurement = float(sc.get("reward_procurement_cost", 0.1))
        self.w_labor_cost = float(sc.get("reward_labor_cost", 0.05))

        # Observation space (flat vector)
        obs_dim = (
            self.n_blocks * self.block_features
            + self.n_spmts * self.spmt_features
            + self.n_goliath_cranes * self.crane_features
            + self.n_docks * self.dock_features
            + self.n_facilities * self.facility_features
            + self.n_suppliers * self.supplier_features
            + self.n_inventory_nodes * self.inventory_features
            + self.n_labor_pools * self.labor_features
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space (simplified: no barge actions)
        max_requests = min(self.n_blocks, 500)  # Cap for memory
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(self.n_action_types),
            "spmt_idx": spaces.Discrete(self.n_spmts),
            "request_idx": spaces.Discrete(max_requests),
            "crane_idx": spaces.Discrete(self.n_goliath_cranes),
            "erection_idx": spaces.Discrete(max_requests),
            "equipment_idx": spaces.Discrete(self.n_spmts + self.n_goliath_cranes),
            # Supply chain sub-action heads
            "supplier_idx": spaces.Discrete(max(self.n_suppliers, 1)),
            "material_idx": spaces.Discrete(max(self.n_inventory_nodes, 1)),
            "labor_pool_idx": spaces.Discrete(max(self.n_labor_pools, 1)),
            "target_block_idx": spaces.Discrete(max(self.n_blocks, 1)),
        })

        # Reward weights
        self.w_tardy = float(config.get("reward_tardy", 10.0))
        self.w_empty = float(config.get("reward_empty_travel", 0.1))
        self.w_breakdown = float(config.get("reward_breakdown", 100.0))
        self.w_maintenance = float(config.get("reward_maintenance", 5.0))
        self.w_completion = float(config.get("reward_completion", 1.0))
        self.w_erection = float(config.get("reward_erection", 10.0))
        self.w_ship_delivery = float(config.get("reward_ship_delivery", 500.0))

        # Simulation state
        self.sim_time: float = 0.0
        self.entities: Dict[str, List[Any]] = {}
        self.transport_requests: List[Dict[str, Any]] = []
        self.erection_requests: List[Dict[str, Any]] = []
        self.facility_queues: Dict[str, List[str]] = {}
        self.facility_processing: Dict[str, List[str]] = {}
        self.facility_remaining_time: Dict[str, Dict[str, float]] = {}
        # Initialize degradation model with config values
        deg_config = config.get("degradation", {})
        self.degradation_model = WienerDegradationModel(
            base_drift=deg_config.get("base_drift", 0.02),
            load_drift_factor=deg_config.get("load_drift_factor", 0.03),
            volatility=deg_config.get("volatility", 0.5),
            failure_threshold=deg_config.get("failure_threshold", 20.0),
        )
        self.max_time = float(config.get("max_time", 50000))

    # ------------------------------------------------------------------
    # Facility helpers
    # ------------------------------------------------------------------
    def _get_all_facilities(self) -> List[Dict]:
        """Get all facility configs (supports zone-grouped and flat formats)."""
        facilities = []
        yard_config = self.config.get("shipyard", {})

        # Try zone-grouped format first (HHI style)
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_facs = yard_config.get(zone, {}).get("facilities", [])
            facilities.extend(zone_facs)

        # Fallback to flat format (default.yaml style)
        if not facilities:
            facilities = yard_config.get("facilities", [])

        # Final fallback to HHI default config
        if not facilities:
            for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
                zone_facs = HHI_ULSAN_DEFAULT_CONFIG.get(zone, {}).get("facilities", [])
                facilities.extend(zone_facs)

        return facilities

    def _get_facility_config(self, fac_name: str) -> Optional[Dict]:
        """Get config for a specific facility."""
        for fac in self._get_all_facilities():
            if fac["name"] == fac_name:
                return fac
        return None

    def _get_first_facility(self) -> Optional[str]:
        """Get the name of the first facility in the production flow."""
        yard_config = self.config.get("shipyard", {})

        # Try zone-grouped format first
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_facilities = yard_config.get(zone, {}).get("facilities", [])
            if zone_facilities:
                return zone_facilities[0]["name"]

        # Fallback to flat format
        facilities = yard_config.get("facilities", [])
        if facilities:
            return facilities[0]["name"]

        # Final fallback to HHI default
        steel_facs = HHI_ULSAN_DEFAULT_CONFIG.get("steel_processing", {}).get("facilities", [])
        if steel_facs:
            return steel_facs[0]["name"]

        return None

    # ------------------------------------------------------------------
    # Entity creation
    # ------------------------------------------------------------------
    def _create_ships(self) -> None:
        """Create LNG carriers under construction."""
        ships: List[LNGCarrier] = []
        for i in range(self.n_ships):
            ship = LNGCarrier(
                id=f"HN{2900 + i}",
                hull_number=f"S{2900 + i}",
                total_blocks=self.n_blocks_per_ship,
                target_delivery_date=float(5000 + 2000 * i),
            )
            ships.append(ship)
        self.entities["ships"] = ships

    def _create_blocks(self) -> None:
        """Create blocks for all ships."""
        blocks: List[Block] = []
        rng = np.random.default_rng(seed=42)

        block_types = list(BlockType)
        type_weights = [0.125, 0.15, 0.125, 0.15, 0.075, 0.125, 0.15, 0.10]

        for ship_idx in range(self.n_ships):
            ship_id = f"HN{2900 + ship_idx}"
            base_due = 5000 + 2000 * ship_idx

            for block_idx in range(self.n_blocks_per_ship):
                # Assign block type based on weights
                block_type = rng.choice(block_types, p=type_weights)

                # Weight varies by type
                base_weight = {
                    BlockType.FLAT_BOTTOM: 350.0,
                    BlockType.FLAT_SIDE: 280.0,
                    BlockType.DECK: 250.0,
                    BlockType.CARGO_TANK_SUPPORT: 320.0,
                    BlockType.ENGINE_ROOM: 400.0,
                    BlockType.CURVED_BOW: 280.0,
                    BlockType.CURVED_STERN: 300.0,
                    BlockType.ACCOMMODATION: 200.0,
                }.get(block_type, 300.0)

                weight = float(rng.uniform(base_weight * 0.8, base_weight * 1.2))
                size = (
                    float(rng.uniform(15.0, 25.0)),
                    float(rng.uniform(15.0, 25.0)),
                    float(rng.uniform(8.0, 15.0)),
                )
                due_date = float(base_due + 10 * block_idx)

                block = Block(
                    id=f"B{ship_idx:02d}{block_idx:03d}",
                    weight=weight,
                    size=size,
                    due_date=due_date,
                    block_type=block_type,
                    current_stage=HHIProductionStage.STEEL_CUTTING,
                    ship_id=ship_id,
                    erection_sequence=block_idx + 1,
                )
                # Apply plate decomposition data if available
                if self.enable_plate_decomposition:
                    from .plate_loader import apply_decomposition_to_blocks, generate_synthetic_plates
                    if self._decomposition_data and ship_id in self._decomposition_data:
                        apply_decomposition_to_blocks([block], self._decomposition_data)
                    elif self._plate_synthetic_fallback:
                        block.plates = generate_synthetic_plates(block, rng)
                        block.compute_plate_stats()

                blocks.append(block)

                # Add to ship's block list
                self.entities["ships"][ship_idx].blocks.append(block.id)

        self.entities["blocks"] = blocks

    def _create_spmts(self) -> None:
        """Create SPMTs for internal transport."""
        spmts: List[SPMT] = []
        for i in range(self.n_spmts):
            spmt = SPMT(
                id=f"SPMT{i:02d}",
                capacity=500.0 if i < 16 else 1200.0,  # Some DCTs with higher capacity
                current_location="spmt_depot",
            )
            spmts.append(spmt)
        self.entities["spmts"] = spmts

    def _create_goliath_cranes(self) -> None:
        """Create Goliath cranes for block erection."""
        cranes: List[GoliathCrane] = []
        crane_configs = self.config.get("shipyard", {}).get("goliath_cranes", [])

        for i, crane_cfg in enumerate(crane_configs[:self.n_goliath_cranes]):
            crane = GoliathCrane(
                id=crane_cfg.get("id", f"GC{i+1:02d}"),
                assigned_dock=crane_cfg.get("assigned_dock", f"dock_{i+1}"),
                capacity_tons=crane_cfg.get("capacity_tons", 900.0),
                height_m=crane_cfg.get("height_m", 109.0),
                span_m=crane_cfg.get("span_m", 150.0),
                rail_length_m=crane_cfg.get("rail_length_m", 500.0),
            )
            cranes.append(crane)

        # Fill remaining with defaults if config is short
        for i in range(len(cranes), self.n_goliath_cranes):
            crane = GoliathCrane(
                id=f"GC{i+1:02d}",
                assigned_dock=f"dock_{(i % 10) + 1}",
            )
            cranes.append(crane)

        self.entities["goliath_cranes"] = cranes

    def _create_docks_and_quays(self) -> None:
        """Create dry docks and outfitting quays."""
        dock_configs = self.config.get("shipyard", {}).get("dry_docks", [])
        docks: List[DryDock] = []

        for i, dock_cfg in enumerate(dock_configs[:self.n_docks]):
            dock = DryDock(
                id=f"dock_{i+1}",
                name=dock_cfg.get("name", f"Dock {i+1}"),
                length_m=dock_cfg.get("length_m", 400.0),
                width_m=dock_cfg.get("width_m", 80.0),
                depth_m=dock_cfg.get("depth_m", 12.0),
                assigned_cranes=dock_cfg.get("cranes", []),
            )
            docks.append(dock)

        # Fill remaining with defaults
        for i in range(len(docks), self.n_docks):
            dock = DryDock(
                id=f"dock_{i+1}",
                name=f"Dock {i+1}",
            )
            docks.append(dock)

        self.entities["docks"] = docks

        # Outfitting quays
        quay_configs = self.config.get("shipyard", {}).get("outfitting_quays", [])
        quays: List[OutfittingQuay] = []

        for i, quay_cfg in enumerate(quay_configs[:self.n_quays]):
            quay = OutfittingQuay(
                id=f"quay_{i+1}",
                name=quay_cfg.get("name", f"Quay {i+1}"),
                length_m=quay_cfg.get("length_m", 350.0),
                capacity=quay_cfg.get("capacity", 2),
            )
            quays.append(quay)

        for i in range(len(quays), self.n_quays):
            quay = OutfittingQuay(
                id=f"quay_{i+1}",
                name=f"Quay {i+1}",
            )
            quays.append(quay)

        self.entities["quays"] = quays

    def _initialize_facilities(self) -> None:
        """Initialize facility queues."""
        for fac in self._get_all_facilities():
            name = fac["name"]
            self.facility_queues[name] = []
            self.facility_processing[name] = []
            self.facility_remaining_time[name] = {}

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.sim_time = 0.0
        self.entities.clear()
        self.transport_requests.clear()
        self.erection_requests.clear()
        self.facility_queues.clear()
        self.facility_processing.clear()
        self.facility_remaining_time.clear()

        # Load plate decomposition data if configured
        if self.enable_plate_decomposition and self._plate_data_dir:
            from pathlib import Path as _Path
            from .plate_loader import load_ship_decomposition
            self._decomposition_data = {}
            data_path = _Path(self._plate_data_dir)
            if data_path.exists():
                for json_file in data_path.glob("*.json"):
                    data = load_ship_decomposition(str(json_file))
                    self._decomposition_data.update(data)

        # Create entities
        self._create_ships()
        self._create_blocks()
        self._create_spmts()
        self._create_goliath_cranes()
        self._create_docks_and_quays()
        self._initialize_facilities()

        # Supply chain entities
        if self.enable_suppliers:
            self._create_suppliers()
        if self.enable_inventory:
            self._create_inventory()
        if self.enable_labor:
            self._create_labor_pools()

        # Track total ships created (for continuous production unique IDs)
        self.total_ships_created = self.n_ships

        # Place blocks in initial queue
        first_fac = self._get_first_facility()
        if first_fac and first_fac in self.facility_queues:
            for block in self.entities["blocks"]:
                self.facility_queues[first_fac].append(block.id)
                block.location = f"queue_{first_fac}"

        # Reset equipment status
        for spmt in self.entities.get("spmts", []):
            spmt.status = SPMTStatus.IDLE
            spmt.current_load = None
            spmt.health_hydraulic = 100.0
            spmt.health_tires = 100.0
            spmt.health_engine = 100.0

        for crane in self.entities.get("goliath_cranes", []):
            crane.status = GoliathCraneStatus.IDLE
            crane.current_block = None
            crane.health_hoist = 100.0
            crane.health_trolley = 100.0
            crane.health_gantry = 100.0

        # Assign initial ships to docks
        for i, ship in enumerate(self.entities.get("ships", [])[:self.n_docks]):
            if i < len(self.entities.get("docks", [])):
                dock = self.entities["docks"][i]
                dock.assign_ship(ship.id, 0.0)
                ship.assigned_dock = dock.id
                ship.status = ShipStatus.IN_ERECTION
                self._log_ship_event(ship.id, "initial", "IN_ERECTION", dock.id)

        # Reset metrics
        self.metrics = {
            "blocks_completed": 0,
            "blocks_erected": 0,
            "ships_delivered": 0,
            "ships_spawned": 0,  # For continuous production tracking
            "ships_in_outfitting": 0,
            "ships_in_trials": 0,
            "breakdowns": 0,
            "planned_maintenance": 0,
            "total_tardiness": 0.0,
            "empty_travel_distance": 0.0,
            "average_ship_cycle_time": 0.0,
        }
        # Supply chain metrics
        if self.enable_suppliers:
            self.metrics["procurement_cost"] = 0.0
            self.metrics["orders_placed"] = 0
            self.metrics["deliveries_received"] = 0
        if self.enable_inventory:
            self.metrics["stockout_events"] = 0
            self.metrics["holding_cost"] = 0.0
        if self.enable_labor:
            self.metrics["labor_cost"] = 0.0
            self.metrics["overtime_hours"] = 0.0

        # Ship delivery tracking for reward calculation
        self._ships_delivered_before = 0

        # Plate decomposition / sub-stage metrics
        if self.enable_plate_decomposition:
            self.metrics["plate_throughput"] = 0  # plates processed per step
        if self._plate_enable_substages:
            self.metrics["substage_times"] = {}  # {stage_name: [durations]}
            self.metrics["substage_transitions"] = 0

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Simulation update methods
    # ------------------------------------------------------------------
    def _compute_plate_processing_time(self, block: "Block", facility: Dict) -> float:
        """Compute plate-count-based processing time for a block at a facility.

        Uses _PLATE_TIME_COEFFICIENTS for the block's current stage.
        Falls back to lognormal for stages not in the coefficient table.
        """
        stage = self._STAGE_MAP.get(facility["name"])
        coeffs = self._PLATE_TIME_COEFFICIENTS.get(stage)
        if coeffs is None:
            # Fallback to lognormal for unmapped stages
            mean = facility["processing_time_mean"]
            std = facility["processing_time_std"]
            multiplier = block.get_processing_multiplier()
            return float(np.random.lognormal(
                mean=np.log(mean * multiplier), sigma=std / mean
            ))

        t = coeffs.get("base_hours", 0.0)
        n_curved = sum(1 for p in block.plates if p.is_curved)
        n_stiffened = sum(1 for p in block.plates if p.has_stiffeners)
        n_flat = max(0, block.n_plates - n_curved - n_stiffened)
        total_area = block.total_plate_area_m2
        total_weld = sum(p.weld_length_m for p in block.plates)

        t += coeffs.get("per_plate", 0.0) * block.n_plates
        t += coeffs.get("per_curved", 0.0) * n_curved
        t += coeffs.get("per_stiffened", 0.0) * n_stiffened
        t += coeffs.get("per_flat", 0.0) * n_flat
        t += coeffs.get("per_area_m2", 0.0) * total_area
        t += coeffs.get("per_weld_m", 0.0) * total_weld

        # Add stochastic noise (5% coefficient of variation)
        noise = float(np.random.normal(1.0, 0.05))
        return max(1.0, t * noise)

    def _advance_substage(self, block: "Block", fac_name: str, progress_frac: float) -> None:
        """Advance block through detailed sub-stages within a facility.

        Each facility maps to 1+ DetailedProductionStages. As the block's
        processing progresses (0→1), it advances through those sub-stages.
        When all sub-stages for a facility are complete, the block advances
        to the next facility as normal via _handle_block_completion.

        Args:
            block: The block being processed.
            fac_name: Current facility name.
            progress_frac: Fraction of total facility processing completed (0-1).
        """
        substages = self._SUBSTAGE_FACILITIES.get(fac_name)
        if not substages or block.current_substage is None:
            return

        n_substages = len(substages)
        # Determine which sub-stage this progress fraction maps to
        substage_idx = min(int(progress_frac * n_substages), n_substages - 1)
        new_substage = substages[substage_idx]

        # Track transition
        if new_substage != block.current_substage:
            old_name = block.current_substage.name
            block.current_substage = new_substage
            self.metrics["substage_transitions"] = self.metrics.get("substage_transitions", 0) + 1

            # Record time spent in previous substage
            if "substage_times" in self.metrics:
                if old_name not in self.metrics["substage_times"]:
                    self.metrics["substage_times"][old_name] = []
                self.metrics["substage_times"][old_name].append(block.substage_progress)

            block.substage_progress = 0.0

        # Update progress within current sub-stage
        if n_substages > 1:
            local_start = substage_idx / n_substages
            local_end = (substage_idx + 1) / n_substages
            local_range = local_end - local_start
            block.substage_progress = (progress_frac - local_start) / local_range if local_range > 0 else 1.0
        else:
            block.substage_progress = progress_frac

    def _assign_blocks_to_facilities(self) -> None:
        """Assign blocks from queues to processing."""
        for fac in self._get_all_facilities():
            fac_name = fac["name"]
            capacity = fac.get("capacity", 1)
            queue = self.facility_queues.get(fac_name, [])

            while queue and len(self.facility_processing.get(fac_name, [])) < capacity:
                block_id = queue.pop(0)
                self.facility_processing[fac_name].append(block_id)

                # Processing time: plate-count formula or lognormal fallback
                block = self._get_block(block_id)
                if self.enable_plate_decomposition and block.n_plates > 0:
                    proc_time = self._compute_plate_processing_time(block, fac)
                else:
                    mean = fac["processing_time_mean"]
                    std = fac["processing_time_std"]
                    multiplier = block.get_processing_multiplier()
                    proc_time = float(np.random.lognormal(
                        mean=np.log(mean * multiplier),
                        sigma=std / mean
                    ))
                self.facility_remaining_time[fac_name][block_id] = proc_time

                # Update block
                block.status = BlockStatus.IN_PROCESS
                block.location = fac_name

                # Set stage
                if fac_name in self._STAGE_MAP:
                    block.current_stage = self._STAGE_MAP[fac_name]

                # Set sub-stage if enabled
                if self._plate_enable_substages and fac_name in self._SUBSTAGE_FACILITIES:
                    substages = self._SUBSTAGE_FACILITIES[fac_name]
                    if substages:
                        block.current_substage = substages[0]
                        block.substage_progress = 0.0

    def _update_processing(self, dt: float) -> None:
        """Advance processing in all facilities."""
        completed_blocks: List[Tuple[str, str]] = []

        for fac_name, remaining_times in list(self.facility_remaining_time.items()):
            finished_ids: List[str] = []
            for block_id, remaining in list(remaining_times.items()):
                new_remaining = remaining - dt
                self.facility_remaining_time[fac_name][block_id] = new_remaining

                # Advance sub-stage tracking if enabled
                if self._plate_enable_substages and new_remaining > 0:
                    total_time = remaining  # remaining before this tick
                    elapsed = total_time - new_remaining
                    # We need the original total time, estimate from elapsed fraction
                    # total = remaining_at_start; progress = 1 - (new_remaining / remaining_at_start)
                    # We only have current remaining, so compute from dt/remaining ratio
                    progress_frac = max(0.0, min(1.0, 1.0 - (new_remaining / max(remaining, dt))))
                    block = self._get_block(block_id)
                    if block.current_substage is not None:
                        self._advance_substage(block, fac_name, progress_frac)

                if new_remaining <= 0.0:
                    finished_ids.append(block_id)

            for block_id in finished_ids:
                completed_blocks.append((fac_name, block_id))
                if block_id in self.facility_processing.get(fac_name, []):
                    self.facility_processing[fac_name].remove(block_id)
                if block_id in self.facility_remaining_time.get(fac_name, {}):
                    del self.facility_remaining_time[fac_name][block_id]

        # Handle completions
        for fac_name, block_id in completed_blocks:
            # Record final sub-stage time on completion
            if self._plate_enable_substages:
                block = self._get_block(block_id)
                if block.current_substage is not None:
                    stage_name = block.current_substage.name
                    if "substage_times" in self.metrics:
                        if stage_name not in self.metrics["substage_times"]:
                            self.metrics["substage_times"][stage_name] = []
                        self.metrics["substage_times"][stage_name].append(block.substage_progress)
                    block.current_substage = None
                    block.substage_progress = 0.0
                # Track plate throughput
                if self.enable_plate_decomposition and block.n_plates > 0:
                    self.metrics["plate_throughput"] = self.metrics.get("plate_throughput", 0) + block.n_plates
            self._handle_block_completion(fac_name, block_id)

    def _get_next_facility(self, current_fac: str, block: Block) -> Optional[str]:
        """Determine the next facility for a block based on its stage and type."""
        # HHI zone-grouped facility progression
        hhi_progression = {
            "steel_stockyard": "cutting_shop",
            "cutting_shop": "part_fabrication",
            "part_fabrication": lambda b: "curved_block_shop" if b.is_curved() else "flat_panel_line_1",
            "flat_panel_line_1": "block_assembly_hall_1",
            "flat_panel_line_2": "block_assembly_hall_2",
            "curved_block_shop": "block_assembly_hall_3",
            "block_assembly_hall_1": "outfitting_shop",
            "block_assembly_hall_2": "outfitting_shop",
            "block_assembly_hall_3": "outfitting_shop",
            "outfitting_shop": "paint_shop",
            "paint_shop": "grand_block_staging_north",
            "grand_block_staging_north": None,  # Ready for erection
            "grand_block_staging_south": None,
        }

        # Flat format facility progression (default.yaml style)
        flat_progression = {
            "cutting": "panel",
            "panel": "assembly",
            "assembly": "outfitting",
            "outfitting": "paint",
            "paint": None,  # Ready for erection after paint
        }

        # Try HHI progression first
        if current_fac in hhi_progression:
            next_fac = hhi_progression.get(current_fac)
            if callable(next_fac):
                return next_fac(block)
            return next_fac

        # Fallback to flat progression
        return flat_progression.get(current_fac)

    def _handle_block_completion(self, fac_name: str, block_id: str) -> None:
        """Handle a block completing processing at a facility."""
        block = self._get_block(block_id)
        block.status = BlockStatus.WAITING
        self._log_block_event(block_id, "stage_complete", block.current_stage.name, fac_name)

        next_fac = self._get_next_facility(fac_name, block)

        if next_fac is None:
            # Block is ready for erection (at pre-erection area)
            block.status = BlockStatus.AT_PRE_ERECTION
            block.current_stage = HHIProductionStage.PRE_ERECTION
            self.erection_requests.append({
                "block_id": block_id,
                "ship_id": block.ship_id,
            })
            self._log_block_event(block_id, "ready_for_erection", "PRE_ERECTION", fac_name)
        else:
            # Create transport request to next facility
            self.transport_requests.append({
                "block_id": block_id,
                "destination": next_fac,
            })
            block.location = f"waiting_transport_to_{next_fac}"

    def _degrade_equipment(self, dt: float) -> None:
        """Apply degradation to all equipment."""
        # Degrade SPMTs
        for spmt in self.entities.get("spmts", []):
            operating = spmt.status in {
                SPMTStatus.TRAVELING_EMPTY,
                SPMTStatus.TRAVELING_LOADED,
                SPMTStatus.LOADING,
                SPMTStatus.UNLOADING,
            }
            load_ratio = 0.0
            if spmt.current_load:
                block = self._get_block(spmt.current_load)
                load_ratio = block.weight / spmt.capacity

            spmt.health_hydraulic, fail_h = self.degradation_model.step(
                spmt.health_hydraulic, dt, load_ratio, operating
            )
            spmt.health_tires, fail_t = self.degradation_model.step(
                spmt.health_tires, dt, load_ratio, operating
            )
            spmt.health_engine, fail_e = self.degradation_model.step(
                spmt.health_engine, dt, load_ratio, operating
            )

            if fail_h or fail_t or fail_e:
                spmt.status = SPMTStatus.BROKEN_DOWN
                self.metrics["breakdowns"] += 1
                self._log_equipment_event(spmt.id, "spmt", "breakdown")

        # Degrade Goliath cranes
        for crane in self.entities.get("goliath_cranes", []):
            operating = crane.status in {
                GoliathCraneStatus.LIFTING,
                GoliathCraneStatus.TRAVELING,
                GoliathCraneStatus.POSITIONING,
                GoliathCraneStatus.LOWERING,
            }

            crane.health_hoist, fail_h = self.degradation_model.step(
                crane.health_hoist, dt, load_ratio=0.3, operating=operating
            )
            crane.health_trolley, fail_t = self.degradation_model.step(
                crane.health_trolley, dt, load_ratio=0.2, operating=operating
            )
            crane.health_gantry, fail_g = self.degradation_model.step(
                crane.health_gantry, dt, load_ratio=0.1, operating=operating
            )

            if fail_h or fail_t or fail_g:
                crane.status = GoliathCraneStatus.BROKEN_DOWN
                self.metrics["breakdowns"] += 1
                self._log_equipment_event(crane.id, "goliath_crane", "breakdown")

    def _advance_simulation(self, dt: float = 1.0) -> None:
        """Advance simulation by dt hours."""
        self._assign_blocks_to_facilities()
        self._update_processing(dt)
        self._update_ship_processing(dt)  # Ship-level stages 8-10
        self._degrade_equipment(dt)
        # Supply chain subsystem updates
        self._update_suppliers(dt)
        self._update_inventory(dt)
        self._update_labor(dt)
        self.sim_time += dt

        # Tardiness accumulation - only count the INCREMENT per step (dt for each tardy block)
        # This prevents exponential blowup of cumulative tardiness
        for block in self.entities.get("blocks", []):
            if block.status not in {BlockStatus.ERECTED, BlockStatus.COMPLETE}:
                if self.sim_time > block.due_date:
                    # Block is tardy - add dt (1 unit of additional tardiness per step)
                    self.metrics["total_tardiness"] += dt

    def _update_ship_processing(self, dt: float) -> None:
        """Process ships through stages 8-10: quay outfitting → sea trials → delivery.

        After all blocks are erected and ship is launched (AFLOAT), the ship
        progresses through:
        - QUAY_OUTFITTING: ~200 hours of systems installation
        - SEA_TRIALS: ~168 hours (1 week) of testing at sea
        - DELIVERED: Handover to owner
        """
        for ship in self.entities.get("ships", []):
            # Initialize processing time if ship just entered a new stage
            if not hasattr(ship, 'stage_remaining_time'):
                ship.stage_remaining_time = 0.0
            if not hasattr(ship, 'sea_position'):
                ship.sea_position = 0.0  # For animation: 0 = at quay, 1 = at sea

            if ship.status == ShipStatus.AFLOAT:
                # Ship just launched - assign to quay and start outfitting
                if ship.assigned_quay:
                    ship.status = ShipStatus.IN_QUAY_OUTFITTING
                    # Outfitting takes 180-220 hours (mean 200)
                    ship.stage_remaining_time = float(np.random.lognormal(
                        mean=np.log(200.0), sigma=0.1
                    ))
                    ship.log_event(self.sim_time, "Started quay outfitting")
                    self._log_ship_event(ship.id, "status_change", "IN_QUAY_OUTFITTING", ship.assigned_quay)
                    self.metrics["ships_in_outfitting"] = self.metrics.get("ships_in_outfitting", 0) + 1

            elif ship.status == ShipStatus.IN_QUAY_OUTFITTING:
                ship.stage_remaining_time -= dt
                # Update completion percentage
                initial_time = 200.0  # Approximate initial time
                ship.completion_pct = max(0, min(100,
                    70 + 20 * (1 - ship.stage_remaining_time / initial_time)))

                if ship.stage_remaining_time <= 0:
                    # Move to sea trials
                    ship.status = ShipStatus.IN_SEA_TRIALS
                    # Sea trials: 144-192 hours (mean 168, ~1 week)
                    ship.stage_remaining_time = float(np.random.lognormal(
                        mean=np.log(168.0), sigma=0.1
                    ))
                    ship.log_event(self.sim_time, "Started sea trials")
                    self._log_ship_event(ship.id, "status_change", "IN_SEA_TRIALS", "")
                    # Release quay
                    if ship.assigned_quay:
                        for quay in self.entities.get("quays", []):
                            if quay.id == ship.assigned_quay:
                                quay.release_ship(ship.id)
                                break
                        ship.assigned_quay = None
                    self.metrics["ships_in_trials"] = self.metrics.get("ships_in_trials", 0) + 1

            elif ship.status == ShipStatus.IN_SEA_TRIALS:
                ship.stage_remaining_time -= dt
                # Animate ship moving out to sea
                initial_time = 168.0
                ship.sea_position = min(1.0, 1.0 - ship.stage_remaining_time / initial_time)
                ship.completion_pct = max(0, min(100,
                    90 + 10 * (1 - ship.stage_remaining_time / initial_time)))

                if ship.stage_remaining_time <= 0:
                    # Ship delivered!
                    ship.status = ShipStatus.DELIVERED
                    ship.completion_pct = 100.0
                    ship.sea_position = 1.0  # Fully at sea (departed)
                    ship.log_event(self.sim_time, "DELIVERED to owner!")
                    self._log_ship_event(ship.id, "status_change", "DELIVERED", "")
                    self.metrics["ships_delivered"] = self.metrics.get("ships_delivered", 0) + 1

                    # Calculate cycle time
                    cycle_time = self.sim_time  # Time from start to delivery
                    total_delivered = self.metrics["ships_delivered"]
                    avg_cycle = self.metrics.get("average_ship_cycle_time", 0.0)
                    self.metrics["average_ship_cycle_time"] = (
                        (avg_cycle * (total_delivered - 1) + cycle_time) / total_delivered
                    )

                    # Spawn new ship to replace delivered one (continuous production)
                    if self.continuous_production:
                        self._spawn_new_ship(ship.id)

    def _spawn_new_ship(self, delivered_ship_id: str) -> None:
        """Spawn a new ship to replace a delivered one with fresh blocks.

        This enables continuous production where new orders come in as
        ships are delivered, simulating realistic shipyard operations.
        """
        self.total_ships_created += 1
        new_ship_id = f"ship_{self.total_ships_created:03d}"

        # Create new ship
        new_ship = LNGCarrier(
            id=new_ship_id,
            hull_number=f"HN-{self.total_ships_created:04d}",
            capacity_cbm=174000,
            total_blocks=self.n_blocks_per_ship,
            target_delivery_date=self.sim_time + 5000,  # ~5000 hours from now
        )

        # Create blocks for this ship
        block_types = list(BlockType)
        type_weights = [0.125, 0.15, 0.125, 0.15, 0.075, 0.125, 0.15, 0.10]
        rng = np.random.default_rng()

        new_blocks = []
        for i in range(self.n_blocks_per_ship):
            block_id = f"{new_ship_id}_block_{i:03d}"
            block_type = rng.choice(block_types, p=type_weights)

            # Weight varies by type
            base_weight = {
                BlockType.FLAT_BOTTOM: 350.0,
                BlockType.FLAT_SIDE: 280.0,
                BlockType.DECK: 250.0,
                BlockType.CARGO_TANK_SUPPORT: 320.0,
                BlockType.ENGINE_ROOM: 400.0,
                BlockType.CURVED_BOW: 280.0,
                BlockType.CURVED_STERN: 300.0,
                BlockType.ACCOMMODATION: 200.0,
            }.get(block_type, 300.0)

            weight = float(rng.uniform(base_weight * 0.8, base_weight * 1.2))
            size = (
                float(rng.uniform(15.0, 25.0)),
                float(rng.uniform(20.0, 32.0)),
                float(rng.uniform(10.0, 18.0)),
            )

            block = Block(
                id=block_id,
                ship_id=new_ship_id,
                weight=weight,
                size=size,
                block_type=block_type,
                due_date=self.sim_time + rng.uniform(500, 4500),
                erection_sequence=i,
            )
            new_blocks.append(block)

        # Add to entities
        self.entities["ships"].append(new_ship)
        self.entities["blocks"].extend(new_blocks)

        # Find and assign to the dock that was freed by delivered ship
        for dock in self.entities.get("docks", []):
            if dock.current_ship == delivered_ship_id or dock.is_available():
                dock.assign_ship(new_ship.id, self.sim_time)
                new_ship.assigned_dock = dock.id
                new_ship.status = ShipStatus.IN_ERECTION
                self._log_ship_event(new_ship.id, "spawned", "IN_ERECTION", dock.id)
                break

        # Start blocks in production (assign to steel cutting facility)
        first_fac = self._get_first_facility()
        for block in new_blocks:
            block.current_stage = HHIProductionStage.STEEL_CUTTING
            # Assign to first facility queue
            if first_fac and first_fac in self.facility_queues:
                self.facility_queues[first_fac].append(block.id)
                block.location = f"queue_{first_fac}"

        # Update metrics
        self.metrics["ships_spawned"] = self.metrics.get("ships_spawned", 0) + 1
        self.n_blocks += self.n_blocks_per_ship  # Increase total block count

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _dispatch_spmt(self, spmt_idx: int, request_idx: int) -> float:
        """Dispatch an SPMT for internal transport."""
        spmts = self.entities.get("spmts", [])
        if spmt_idx >= len(spmts):
            return 0.0

        spmt = spmts[spmt_idx]

        if request_idx >= len(self.transport_requests):
            return 0.0

        request = self.transport_requests.pop(request_idx)
        block_id = request["block_id"]
        destination = request["destination"]
        block = self._get_block(block_id)

        if spmt.status != SPMTStatus.IDLE or spmt.current_load is not None:
            return 0.0

        # Calculate travel (simplified)
        travel_to_block = self.shipyard.get_travel_time(spmt.current_location, block.location)
        travel_to_dest = self.shipyard.get_travel_time(block.location, destination)
        empty_distance = travel_to_block

        self.metrics["empty_travel_distance"] += empty_distance

        # Execute transport
        spmt.status = SPMTStatus.TRAVELING_LOADED
        spmt.current_load = block_id
        spmt.current_location = destination

        # Complete transport
        block.location = f"queue_{destination}"
        self.facility_queues[destination].append(block_id)
        spmt.current_load = None
        spmt.status = SPMTStatus.IDLE

        self._log_block_event(block_id, "transport_arrival", block.current_stage.name, destination)
        return -self.w_empty * empty_distance

    def _dispatch_goliath_crane(self, crane_idx: int, erection_idx: int) -> float:
        """Dispatch a Goliath crane for block erection."""
        cranes = self.entities.get("goliath_cranes", [])
        if crane_idx >= len(cranes):
            return 0.0

        crane = cranes[crane_idx]

        if erection_idx >= len(self.erection_requests):
            return 0.0

        if crane.status != GoliathCraneStatus.IDLE:
            return 0.0

        # Validate the request BEFORE popping it from the list
        request = self.erection_requests[erection_idx]
        block_id = request["block_id"]
        ship_id = request["ship_id"]
        block = self._get_block(block_id)
        ship = self._get_ship(ship_id)

        if not ship or not ship.assigned_dock:
            return 0.0

        # Check if crane serves the ship's dock
        if crane.assigned_dock != ship.assigned_dock:
            return 0.0

        # Check weight capacity
        if not crane.can_lift(block.weight):
            return 0.0

        # All checks passed - now remove the request from the list
        self.erection_requests.pop(erection_idx)

        # Execute erection
        crane.status = GoliathCraneStatus.LIFTING
        crane.current_block = block_id

        block.status = BlockStatus.IN_ERECTION
        block.current_stage = HHIProductionStage.ERECTION

        # Complete erection
        block.status = BlockStatus.ERECTED
        block.location = f"dock_{ship.assigned_dock}"

        # Update ship progress
        ship.blocks_erected += 1
        ship.update_erection_progress()

        # Add block to dock
        dock = self._get_dock(ship.assigned_dock)
        if dock:
            dock.blocks_in_dock.append(block_id)

        crane.status = GoliathCraneStatus.IDLE
        crane.current_block = None

        self.metrics["blocks_erected"] += 1
        self.metrics["blocks_completed"] += 1  # Keep both metrics in sync for compatibility
        self._log_block_event(block_id, "erected", "ERECTION", ship.assigned_dock)

        # Check if ship erection is complete
        if ship.is_erection_complete():
            self._launch_ship(ship)

        return self.w_erection

    def _launch_ship(self, ship: LNGCarrier) -> None:
        """Launch a ship from dry dock to outfitting quay."""
        if not ship.assigned_dock:
            return

        dock = self._get_dock(ship.assigned_dock)
        if dock:
            dock.release_ship(self.sim_time)

        # Find available quay
        for quay in self.entities.get("quays", []):
            if quay.is_available():
                quay.assign_ship(ship.id)
                ship.assigned_quay = quay.id
                ship.status = ShipStatus.AFLOAT
                ship.log_event(self.sim_time, f"Launched to {quay.id}")
                self._log_ship_event(ship.id, "status_change", "AFLOAT", quay.id)
                break

        ship.assigned_dock = None

    def _trigger_maintenance(self, equipment_idx: int) -> float:
        """Trigger maintenance on equipment."""
        n_spmts = len(self.entities.get("spmts", []))

        if equipment_idx < n_spmts:
            spmt = self.entities["spmts"][equipment_idx]
            if spmt.status == SPMTStatus.IDLE:
                spmt.status = SPMTStatus.IN_MAINTENANCE
                spmt.health_hydraulic = self.degradation_model.perform_maintenance()
                spmt.health_tires = self.degradation_model.perform_maintenance()
                spmt.health_engine = self.degradation_model.perform_maintenance()
                spmt.status = SPMTStatus.IDLE
                self.metrics["planned_maintenance"] += 1
                self._log_equipment_event(spmt.id, "spmt", "maintenance_complete")
                return -self.w_maintenance
        else:
            crane_idx = equipment_idx - n_spmts
            cranes = self.entities.get("goliath_cranes", [])
            if crane_idx < len(cranes):
                crane = cranes[crane_idx]
                if crane.status == GoliathCraneStatus.IDLE:
                    crane.status = GoliathCraneStatus.IN_MAINTENANCE
                    crane.health_hoist = self.degradation_model.perform_maintenance()
                    crane.health_trolley = self.degradation_model.perform_maintenance()
                    crane.health_gantry = self.degradation_model.perform_maintenance()
                    crane.status = GoliathCraneStatus.IDLE
                    self.metrics["planned_maintenance"] += 1
                    self._log_equipment_event(crane.id, "goliath_crane", "maintenance_complete")
                    return -self.w_maintenance
        return 0.0

    # ------------------------------------------------------------------
    # Supply chain: entity creation
    # ------------------------------------------------------------------
    def _create_suppliers(self) -> None:
        """Create suppliers from config."""
        sc = self.config.get("supply_chain", {})
        suppliers_config = sc.get("suppliers", [])
        suppliers = []
        for i, sup_cfg in enumerate(suppliers_config[:self.n_suppliers]):
            supplier = Supplier(
                id=sup_cfg.get("id", f"SUP{i:02d}"),
                name=sup_cfg.get("name", f"Supplier {i}"),
                lead_time_mean=sup_cfg.get("lead_time_mean", 72.0),
                lead_time_std=sup_cfg.get("lead_time_std", 12.0),
                reliability=sup_cfg.get("reliability", 0.9),
                capacity_per_period=sup_cfg.get("capacity_per_period", 500.0),
                price_per_unit=sup_cfg.get("price_per_unit", 1.0),
                specializations=sup_cfg.get("specializations", ["steel_plate"]),
            )
            suppliers.append(supplier)
        for i in range(len(suppliers), self.n_suppliers):
            suppliers.append(Supplier(id=f"SUP{i:02d}"))
        self.entities["suppliers"] = suppliers

    def _create_inventory(self) -> None:
        """Create material inventory nodes from config."""
        sc = self.config.get("supply_chain", {})
        inv_config = sc.get("inventory", [])
        inventory = []
        for i, inv_cfg in enumerate(inv_config[:self.n_inventory_nodes]):
            mat_type = MaterialType(inv_cfg.get("material_type", "steel_plate"))
            inv = MaterialInventory(
                id=inv_cfg.get("id", f"MAT_{mat_type.value}_{i}"),
                material_type=mat_type,
                quantity=inv_cfg.get("initial_quantity", 1000.0),
                location=inv_cfg.get("location", "steel_stockyard"),
                reorder_point=inv_cfg.get("reorder_point", 200.0),
                max_capacity=inv_cfg.get("max_capacity", 5000.0),
                consumption_rate=inv_cfg.get("consumption_rate", 0.5),
            )
            inventory.append(inv)
        for i in range(len(inventory), self.n_inventory_nodes):
            inventory.append(MaterialInventory(id=f"MAT_{i}"))
        self.entities["inventory"] = inventory

    def _create_labor_pools(self) -> None:
        """Create labor pools from config."""
        sc = self.config.get("supply_chain", {})
        labor_config = sc.get("labor_pools", [])
        pools = []
        for i, pool_cfg in enumerate(labor_config[:self.n_labor_pools]):
            skill = SkillType(pool_cfg.get("skill_type", "welder"))
            pool = LaborPool(
                id=pool_cfg.get("id", f"LABOR_{skill.value}"),
                skill_type=skill,
                total_workers=pool_cfg.get("total_workers", 20),
                available_workers=pool_cfg.get("total_workers", 20),
                hourly_rate=pool_cfg.get("hourly_rate", 50.0),
                overtime_rate=pool_cfg.get("overtime_rate", 75.0),
            )
            pools.append(pool)
        for i in range(len(pools), self.n_labor_pools):
            pools.append(LaborPool(id=f"LABOR_{i}"))
        self.entities["labor_pools"] = pools

    # ------------------------------------------------------------------
    # Supply chain: simulation updates
    # ------------------------------------------------------------------
    _STAGE_MATERIALS: Dict[HHIProductionStage, Dict[str, float]] = {
        HHIProductionStage.STEEL_CUTTING: {"steel_plate": 1.0},
        HHIProductionStage.PART_FABRICATION: {"steel_plate": 0.5, "welding_consumable": 0.2},
        HHIProductionStage.PANEL_ASSEMBLY: {"steel_plate": 0.3, "welding_consumable": 0.3},
        HHIProductionStage.BLOCK_ASSEMBLY: {"welding_consumable": 0.5, "pipe_section": 0.1},
        HHIProductionStage.BLOCK_OUTFITTING: {"pipe_section": 0.3, "electrical_cable": 0.2, "insulation": 0.1},
        HHIProductionStage.PAINTING: {"paint": 0.5},
    }

    _STAGE_SKILLS: Dict[HHIProductionStage, List[SkillType]] = {
        HHIProductionStage.STEEL_CUTTING: [SkillType.FITTER],
        HHIProductionStage.PANEL_ASSEMBLY: [SkillType.WELDER, SkillType.FITTER],
        HHIProductionStage.BLOCK_ASSEMBLY: [SkillType.WELDER, SkillType.FITTER],
        HHIProductionStage.BLOCK_OUTFITTING: [SkillType.ELECTRICIAN, SkillType.FITTER],
        HHIProductionStage.PAINTING: [SkillType.PAINTER],
        HHIProductionStage.ERECTION: [SkillType.CRANE_OPERATOR],
    }

    def _update_suppliers(self, dt: float) -> None:
        """Check for supplier deliveries and route to inventory."""
        if not self.enable_suppliers:
            return
        for supplier in self.entities.get("suppliers", []):
            delivered = supplier.check_deliveries(self.sim_time)
            for order in delivered:
                for inv in self.entities.get("inventory", []):
                    if inv.material_type.value == order["material_type"]:
                        inv.receive(order["quantity"])
                        self.metrics["deliveries_received"] = (
                            self.metrics.get("deliveries_received", 0) + 1
                        )
                        break

    def _update_inventory(self, dt: float) -> None:
        """Consume materials as blocks are processed; track holding cost."""
        if not self.enable_inventory:
            return
        for fac_name, block_ids in self.facility_processing.items():
            for block_id in block_ids:
                block = self._get_block(block_id)
                required = self._STAGE_MATERIALS.get(block.current_stage, {})
                for mat_type, rate in required.items():
                    for inv in self.entities.get("inventory", []):
                        if inv.material_type.value == mat_type:
                            consumed = inv.consume(rate * dt)
                            if consumed < rate * dt:
                                self.metrics["stockout_events"] = (
                                    self.metrics.get("stockout_events", 0) + 1
                                )
                            break
        for inv in self.entities.get("inventory", []):
            self.metrics["holding_cost"] = (
                self.metrics.get("holding_cost", 0.0)
                + inv.holding_cost_per_unit * inv.quantity * dt
            )

    def _update_labor(self, dt: float) -> None:
        """Accumulate labor costs and handle shift boundaries."""
        if not self.enable_labor:
            return
        for pool in self.entities.get("labor_pools", []):
            assigned = len(pool.assigned_tasks)
            cost = assigned * pool.hourly_rate * dt
            if pool.current_overtime_hours > 0:
                cost += assigned * (pool.overtime_rate - pool.hourly_rate) * dt
                self.metrics["overtime_hours"] = (
                    self.metrics.get("overtime_hours", 0.0) + dt * assigned
                )
            self.metrics["labor_cost"] = self.metrics.get("labor_cost", 0.0) + cost
            if self.sim_time > 0 and self.sim_time % pool.shift_hours < dt:
                pool.reset_shift()

    # ------------------------------------------------------------------
    # Supply chain: action execution
    # ------------------------------------------------------------------
    def _place_order(self, supplier_idx: int, material_idx: int) -> float:
        """Place a procurement order with a supplier."""
        suppliers = self.entities.get("suppliers", [])
        inventory = self.entities.get("inventory", [])
        if supplier_idx >= len(suppliers) or material_idx >= len(inventory):
            return -0.01
        supplier = suppliers[supplier_idx]
        inv = inventory[material_idx]
        if inv.material_type.value not in supplier.specializations:
            return -0.01
        # Order up to supplier capacity or inventory shortfall, whichever is smaller
        shortfall = max(inv.max_capacity - inv.quantity, 0.0)
        if shortfall <= 0:
            return -0.01
        order_qty = min(shortfall, supplier.capacity_per_period - supplier.current_backlog)
        if order_qty <= 0:
            return -0.01
        order = supplier.place_order(inv.material_type.value, order_qty, self.sim_time)
        self.metrics["procurement_cost"] = (
            self.metrics.get("procurement_cost", 0.0) + order["cost"]
        )
        self.metrics["orders_placed"] = self.metrics.get("orders_placed", 0) + 1
        return -self.w_procurement * order["cost"]

    def _assign_worker(self, labor_pool_idx: int, target_block_idx: int) -> float:
        """Assign workers from a labor pool to a block in processing."""
        pools = self.entities.get("labor_pools", [])
        blocks = self.entities.get("blocks", [])
        if labor_pool_idx >= len(pools) or target_block_idx >= len(blocks):
            return -0.01
        pool = pools[labor_pool_idx]
        block = blocks[target_block_idx]
        if block.status != BlockStatus.IN_PROCESS:
            return -0.01
        if not pool.can_assign():
            return -0.01
        required = self._STAGE_SKILLS.get(block.current_stage, [])
        if pool.skill_type not in required:
            return -0.01
        pool.assign(block.id)
        for fac_name, remaining in self.facility_remaining_time.items():
            if block.id in remaining:
                remaining[block.id] *= 0.9
                break
        return 0.1

    def _transfer_material(self, source_idx: int, dest_idx: int) -> float:
        """Transfer material between inventory locations."""
        inventory = self.entities.get("inventory", [])
        if source_idx >= len(inventory) or dest_idx >= len(inventory):
            return -0.01
        if source_idx == dest_idx:
            return -0.01
        source = inventory[source_idx]
        dest = inventory[dest_idx]
        if source.material_type != dest.material_type:
            return -0.01
        transfer_amount = min(source.quantity * 0.5, dest.max_capacity - dest.quantity)
        if transfer_amount <= 0:
            return -0.01
        source.quantity -= transfer_amount
        dest.receive(transfer_amount)
        return 0.05

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep."""
        reward = 0.0
        breakdowns_before = self.metrics["breakdowns"]
        tardiness_before = self.metrics.get("total_tardiness", 0.0)

        action_type = int(action.get("action_type", 3))  # default hold

        if action_type == 0:  # SPMT dispatch
            spmt_idx = int(action.get("spmt_idx", 0))
            req_idx = int(action.get("request_idx", 0))
            reward += self._dispatch_spmt(spmt_idx, req_idx)

        elif action_type == 1:  # Goliath crane dispatch
            crane_idx = int(action.get("crane_idx", 0))
            erection_idx = int(action.get("erection_idx", 0))
            reward += self._dispatch_goliath_crane(crane_idx, erection_idx)

        elif action_type == 2:  # Maintenance
            equip_idx = int(action.get("equipment_idx", 0))
            reward += self._trigger_maintenance(equip_idx)

        elif action_type == 3:  # Hold
            pass

        elif action_type == self._action_type_map.get("PLACE_ORDER"):
            sup_idx = int(action.get("supplier_idx", 0))
            mat_idx = int(action.get("material_idx", 0))
            reward += self._place_order(sup_idx, mat_idx)

        elif action_type == self._action_type_map.get("ASSIGN_WORKER"):
            pool_idx = int(action.get("labor_pool_idx", 0))
            block_idx = int(action.get("target_block_idx", 0))
            reward += self._assign_worker(pool_idx, block_idx)

        elif action_type == self._action_type_map.get("TRANSFER_MATERIAL"):
            src_idx = int(action.get("material_idx", 0))
            dst_idx = int(action.get("target_block_idx", 0))
            reward += self._transfer_material(src_idx, dst_idx)

        # Advance simulation
        self._advance_simulation(dt=1.0)

        # Penalties
        new_breakdowns = self.metrics["breakdowns"] - breakdowns_before
        reward -= self.w_breakdown * new_breakdowns

        tardiness_delta = self.metrics.get("total_tardiness", 0.0) - tardiness_before
        reward -= self.w_tardy * tardiness_delta * 0.01

        # Logging
        if self.db_logging_enabled and int(self.sim_time) % 50 == 0:
            self.log_state_to_db()

        # Ship delivery reward — awarded when a ship transitions to DELIVERED
        ships_now = sum(1 for s in self.entities.get("ships", [])
                        if s.status == ShipStatus.DELIVERED)
        ships_before = getattr(self, '_ships_delivered_before', 0)
        new_deliveries = ships_now - ships_before
        if new_deliveries > 0:
            reward += self.w_ship_delivery * new_deliveries
        self._ships_delivered_before = ships_now

        # Check termination
        if self.continuous_production:
            # In continuous mode, never terminate early - only truncate at max_time
            terminated = False
        else:
            # Standard mode: terminate when all ships are delivered.
            # Do NOT terminate on blocks_erected alone — ships still need
            # quay outfitting (~200h) and sea trials (~168h) after erection.
            all_ships_delivered = ships_now >= self.n_ships
            terminated = all_ships_delivered

        truncated = self.sim_time >= self.max_time

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _encode_block(self, block: Block) -> List[float]:
        stage = block.current_stage.value
        block_type = block.block_type.value if hasattr(block.block_type, 'value') else 0
        type_enc = list(BlockType).index(block.block_type) / 7.0 if isinstance(block.block_type, BlockType) else 0.0
        completion = block.completion_pct / 100.0
        time_to_due = max(0.0, (block.due_date - self.sim_time)) / 1000.0
        weight = block.weight / 500.0
        in_transit = 1.0 if block.status == BlockStatus.IN_TRANSIT else 0.0
        waiting = 1.0 if block.status == BlockStatus.WAITING else 0.0
        in_erection = 1.0 if block.status == BlockStatus.IN_ERECTION else 0.0
        erected = 1.0 if block.status == BlockStatus.ERECTED else 0.0
        is_curved = 1.0 if block.is_curved() else 0.0
        erection_seq = block.erection_sequence / 200.0
        features = [stage / 10.0, type_enc, completion, time_to_due, weight, in_transit,
                    waiting, in_erection, erected, is_curved, erection_seq, 0.0]
        if self.enable_plate_decomposition:
            n_plates_norm = block.n_plates / 50.0  # Normalize assuming max ~50 plates
            area_norm = block.total_plate_area_m2 / 1000.0  # Normalize area
            pct_curved = (sum(1 for p in block.plates if p.is_curved) / max(1, block.n_plates))
            pct_stiffened = (sum(1 for p in block.plates if p.has_stiffeners) / max(1, block.n_plates))
            features.extend([n_plates_norm, area_norm, pct_curved, pct_stiffened])
        return features

    def _encode_spmt(self, spmt: SPMT) -> List[float]:
        status_enc = [0.0] * 4
        status_map = {SPMTStatus.IDLE: 0, SPMTStatus.TRAVELING_EMPTY: 1,
                      SPMTStatus.TRAVELING_LOADED: 2, SPMTStatus.IN_MAINTENANCE: 3}
        idx = status_map.get(spmt.status, 0)
        status_enc[idx] = 1.0
        load_ratio = 0.0
        if spmt.current_load:
            block = self._get_block(spmt.current_load)
            load_ratio = block.weight / spmt.capacity
        health = spmt.get_health_vector().tolist()
        capacity_norm = spmt.capacity / 1200.0
        return status_enc + [load_ratio] + health + [capacity_norm]

    def _encode_goliath_crane(self, crane: GoliathCrane) -> List[float]:
        dock_idx = int(crane.assigned_dock.split("_")[-1]) if crane.assigned_dock else 0
        dock_norm = dock_idx / 10.0
        pos_norm = crane.position_on_rail / 100.0
        status_enc = [0.0] * 4
        status_map = {GoliathCraneStatus.IDLE: 0, GoliathCraneStatus.LIFTING: 1,
                      GoliathCraneStatus.TRAVELING: 2, GoliathCraneStatus.POSITIONING: 3}
        idx = status_map.get(crane.status, 0)
        status_enc[idx] = 1.0
        health = crane.get_health_vector().tolist()
        current_load = 1.0 if crane.current_block else 0.0
        capacity_norm = crane.capacity_tons / 900.0
        return [dock_norm, pos_norm] + status_enc + health + [current_load, capacity_norm]

    def _encode_dock(self, dock: DryDock) -> List[float]:
        occupied = 1.0 if dock.current_ship else 0.0
        blocks_count = len(dock.blocks_in_dock) / 200.0
        length_norm = dock.length_m / 500.0
        width_norm = dock.width_m / 120.0
        crane_count = len(dock.assigned_cranes) / 2.0
        status_enc = 0.0 if dock.status == "idle" else 0.5 if dock.status == "in_use" else 1.0
        return [occupied, blocks_count, length_norm, width_norm, crane_count, status_enc]

    def _encode_facility(self, fac_name: str) -> List[float]:
        queue_len = len(self.facility_queues.get(fac_name, []))
        processing_count = len(self.facility_processing.get(fac_name, []))
        fac_cfg = self._get_facility_config(fac_name)
        capacity = fac_cfg.get("capacity", 1) if fac_cfg else 1
        utilization = processing_count / max(1, capacity)
        return [queue_len / 50.0, processing_count / 10.0, utilization, 0.0]

    def _get_observation(self) -> np.ndarray:
        obs: List[float] = []

        for block in self.entities.get("blocks", []):
            obs.extend(self._encode_block(block))

        for spmt in self.entities.get("spmts", []):
            obs.extend(self._encode_spmt(spmt))

        for crane in self.entities.get("goliath_cranes", []):
            obs.extend(self._encode_goliath_crane(crane))

        for dock in self.entities.get("docks", []):
            obs.extend(self._encode_dock(dock))

        for fac in self._get_all_facilities():
            obs.extend(self._encode_facility(fac["name"]))

        # Supply chain features
        for sup in self.entities.get("suppliers", []):
            obs.extend(sup.get_feature_vector())
        for inv in self.entities.get("inventory", []):
            obs.extend(inv.get_feature_vector())
        for pool in self.entities.get("labor_pools", []):
            obs.extend(pool.get_feature_vector())

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        return {
            "sim_time": self.sim_time,
            "metrics": self.metrics.copy(),
            "transport_requests": len(self.transport_requests),
            "erection_requests": len(self.erection_requests),
            "ships_in_erection": sum(1 for s in self.entities.get("ships", [])
                                      if s.status == ShipStatus.IN_ERECTION),
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _get_block(self, block_id: str) -> Block:
        for b in self.entities.get("blocks", []):
            if b.id == block_id:
                return b
        raise KeyError(f"Block {block_id} not found")

    def _get_ship(self, ship_id: str) -> Optional[LNGCarrier]:
        for s in self.entities.get("ships", []):
            if s.id == ship_id:
                return s
        return None

    def _get_dock(self, dock_id: str) -> Optional[DryDock]:
        for d in self.entities.get("docks", []):
            if d.id == dock_id:
                return d
        return None

    def _log_block_event(self, block_id: str, event_type: str, stage: str, location: str) -> None:
        if not self.db_logging_enabled:
            return
        try:
            from mes.database import log_block_event
            log_block_event(block_id, self.sim_time, event_type, stage, location)
        except Exception:
            pass

    def _log_equipment_event(self, equipment_id: str, equipment_type: str, event_type: str) -> None:
        if not self.db_logging_enabled:
            return
        try:
            from mes.database import log_block_event
            log_block_event(equipment_id, self.sim_time, event_type, equipment_type, "equipment")
        except Exception:
            pass

    def _log_ship_event(self, ship_id: str, event_type: str, status: str, dock_id: str = "") -> None:
        """Log a ship lifecycle event for Gantt chart visualization."""
        if not self.db_logging_enabled:
            return
        try:
            from mes.database import log_ship_event
            log_ship_event(ship_id, self.sim_time, event_type, status, dock_id)
        except Exception:
            pass

    def log_state_to_db(self) -> None:
        """Log full simulation state to the MES database."""
        if not self.db_logging_enabled:
            return
        try:
            from mes.database import (
                log_metrics, log_entities, log_health_snapshot,
                log_queue_depths, log_position_snapshot,
            )

            log_metrics(self.sim_time, self.metrics)
            log_health_snapshot(
                self.sim_time,
                self.entities.get("spmts", []),
                self.entities.get("goliath_cranes", []),
            )
            log_queue_depths(self.sim_time, self.facility_queues, self.facility_processing)
        except Exception:
            pass

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """Return a mask of valid actions for the current state."""
        n_spmts = self.n_spmts
        n_cranes = self.n_goliath_cranes
        n_transport = len(self.transport_requests)
        n_erection = len(self.erection_requests)

        mask: Dict[str, Any] = {
            "action_type": np.ones(self.n_action_types, dtype=bool),
            "spmt_dispatch": np.zeros((n_spmts, max(1, n_transport)), dtype=bool),
            "crane_dispatch": np.zeros((n_cranes, max(1, n_erection)), dtype=bool),
            "maintenance": np.zeros(n_spmts + n_cranes, dtype=bool),
        }

        # SPMT dispatch mask
        for i, spmt in enumerate(self.entities.get("spmts", [])):
            if spmt.status == SPMTStatus.IDLE and spmt.status != SPMTStatus.BROKEN_DOWN:
                for j, req in enumerate(self.transport_requests):
                    block = self._get_block(req["block_id"])
                    if block.weight <= spmt.capacity and spmt.get_min_health() > 20.0:
                        mask["spmt_dispatch"][i, j] = True

        # Crane dispatch mask
        for i, crane in enumerate(self.entities.get("goliath_cranes", [])):
            if crane.status == GoliathCraneStatus.IDLE and crane.status != GoliathCraneStatus.BROKEN_DOWN:
                for j, req in enumerate(self.erection_requests):
                    block = self._get_block(req["block_id"])
                    ship = self._get_ship(req["ship_id"])
                    if (ship and ship.assigned_dock == crane.assigned_dock
                            and crane.can_lift(block.weight)
                            and crane.get_min_health() > 20.0):
                        mask["crane_dispatch"][i, j] = True

        # Maintenance mask
        for i, spmt in enumerate(self.entities.get("spmts", [])):
            if (spmt.status == SPMTStatus.IDLE
                    and spmt.status != SPMTStatus.BROKEN_DOWN
                    and spmt.get_min_health() < 60.0):
                mask["maintenance"][i] = True
        for i, crane in enumerate(self.entities.get("goliath_cranes", [])):
            idx = n_spmts + i
            if (crane.status == GoliathCraneStatus.IDLE
                    and crane.status != GoliathCraneStatus.BROKEN_DOWN
                    and crane.get_min_health() < 60.0):
                mask["maintenance"][idx] = True

        # Disable action types if no valid options
        if not mask["spmt_dispatch"].any():
            mask["action_type"][0] = False
        if not mask["crane_dispatch"].any():
            mask["action_type"][1] = False
        if not mask["maintenance"].any():
            mask["action_type"][2] = False

        # Supply chain masks
        if self.enable_suppliers:
            n_sup = max(self.n_suppliers, 1)
            n_inv = max(self.n_inventory_nodes, 1)
            mask["supplier_order"] = np.zeros((n_sup, n_inv), dtype=bool)
            for i, sup in enumerate(self.entities.get("suppliers", [])):
                for j, inv in enumerate(self.entities.get("inventory", [])):
                    if (inv.is_below_reorder_point()
                            and sup.can_accept_order(inv.max_capacity - inv.quantity)
                            and inv.material_type.value in sup.specializations):
                        mask["supplier_order"][i, j] = True
            place_idx = self._action_type_map.get("PLACE_ORDER")
            if place_idx is not None and not mask["supplier_order"].any():
                mask["action_type"][place_idx] = False

        if self.enable_labor:
            n_pools = max(self.n_labor_pools, 1)
            mask["labor_assign"] = np.zeros(n_pools, dtype=bool)
            for i, pool in enumerate(self.entities.get("labor_pools", [])):
                if pool.can_assign():
                    mask["labor_assign"][i] = True
            assign_idx = self._action_type_map.get("ASSIGN_WORKER")
            if assign_idx is not None and not mask["labor_assign"].any():
                mask["action_type"][assign_idx] = False

        if self.enable_inventory:
            n_inv = max(self.n_inventory_nodes, 1)
            mask["inventory_transfer"] = np.zeros(n_inv, dtype=bool)
            for i, inv in enumerate(self.entities.get("inventory", [])):
                if inv.quantity > inv.reorder_point * 1.5:
                    mask["inventory_transfer"][i] = True
            xfer_idx = self._action_type_map.get("TRANSFER_MATERIAL")
            if xfer_idx is not None and not mask["inventory_transfer"].any():
                mask["action_type"][xfer_idx] = False

        return mask

    def get_graph_data(self):
        """Construct a PyG heterogeneous graph representation of the state.

        Returns a HeteroData object with node features and edge indices for:
        - block, spmt, crane (goliath_cranes), and facility nodes
        - edges representing transport needs, lift needs, and locations
        """
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()

        # Node features
        block_x, spmt_x, crane_x, fac_x = [], [], [], []
        block_batch, spmt_batch, crane_batch, fac_batch = [], [], [], []

        for b in self.entities.get("blocks", []):
            block_x.append(torch.tensor(self._encode_block(b), dtype=torch.float))
            block_batch.append(0)

        for s in self.entities.get("spmts", []):
            spmt_x.append(torch.tensor(self._encode_spmt(s), dtype=torch.float))
            spmt_batch.append(0)

        for c in self.entities.get("goliath_cranes", []):
            crane_x.append(torch.tensor(self._encode_goliath_crane(c), dtype=torch.float))
            crane_batch.append(0)

        # Get facilities
        fac_names = []
        for fac in self._get_all_facilities():
            fac_name = fac["name"]
            fac_names.append(fac_name)
            fac_x.append(torch.tensor(self._encode_facility(fac_name), dtype=torch.float))
            fac_batch.append(0)

        # Assign node features
        data["block"].x = torch.stack(block_x) if block_x else torch.empty((0, self.block_features))
        data["block"].batch = torch.tensor(block_batch, dtype=torch.long)
        data["spmt"].x = torch.stack(spmt_x) if spmt_x else torch.empty((0, self.spmt_features))
        data["spmt"].batch = torch.tensor(spmt_batch, dtype=torch.long)
        data["crane"].x = torch.stack(crane_x) if crane_x else torch.empty((0, self.crane_features))
        data["crane"].batch = torch.tensor(crane_batch, dtype=torch.long)
        data["facility"].x = torch.stack(fac_x) if fac_x else torch.empty((0, self.facility_features))
        data["facility"].batch = torch.tensor(fac_batch, dtype=torch.long)

        # Edge indices
        n_blocks = len(self.entities.get("blocks", []))
        n_spmts = len(self.entities.get("spmts", []))
        n_cranes = len(self.entities.get("goliath_cranes", []))

        # Block <-> SPMT edges (all blocks can potentially use any SPMT)
        if n_blocks > 0 and n_spmts > 0:
            block_to_spmt_src, block_to_spmt_dst = [], []
            for i in range(n_blocks):
                for j in range(n_spmts):
                    block_to_spmt_src.append(i)
                    block_to_spmt_dst.append(j)
            data["block", "needs_transport", "spmt"].edge_index = torch.tensor(
                [block_to_spmt_src, block_to_spmt_dst], dtype=torch.long
            )
            data["spmt", "can_transport", "block"].edge_index = torch.tensor(
                [block_to_spmt_dst, block_to_spmt_src], dtype=torch.long
            )

        # Block <-> Crane edges
        if n_blocks > 0 and n_cranes > 0:
            block_to_crane_src, block_to_crane_dst = [], []
            for i in range(n_blocks):
                for j in range(n_cranes):
                    block_to_crane_src.append(i)
                    block_to_crane_dst.append(j)
            data["block", "needs_lift", "crane"].edge_index = torch.tensor(
                [block_to_crane_src, block_to_crane_dst], dtype=torch.long
            )
            data["crane", "can_lift", "block"].edge_index = torch.tensor(
                [block_to_crane_dst, block_to_crane_src], dtype=torch.long
            )

        # Block -> Facility edges (current location)
        if n_blocks > 0 and fac_names:
            block_to_fac_src, block_to_fac_dst = [], []
            for i, b in enumerate(self.entities.get("blocks", [])):
                fac_idx = 0
                for idx, name in enumerate(fac_names):
                    if name in b.location:
                        fac_idx = idx
                        break
                block_to_fac_src.append(i)
                block_to_fac_dst.append(fac_idx)
            data["block", "at", "facility"].edge_index = torch.tensor(
                [block_to_fac_src, block_to_fac_dst], dtype=torch.long
            )

        # Block -> Block precedence edges (erection sequence)
        if n_blocks > 1:
            precedes_src, precedes_dst = [], []
            blocks = self.entities.get("blocks", [])
            for i, b in enumerate(blocks):
                # Blocks with lower erection_sequence precede blocks with higher
                for j, other in enumerate(blocks):
                    if b.ship_id == other.ship_id and b.erection_sequence < other.erection_sequence:
                        # Only add edge if sequences are adjacent
                        if other.erection_sequence - b.erection_sequence == 1:
                            precedes_src.append(i)
                            precedes_dst.append(j)
            if precedes_src:
                data["block", "precedes", "block"].edge_index = torch.tensor(
                    [precedes_src, precedes_dst], dtype=torch.long
                )

        # SPMT -> Facility edges
        if n_spmts > 0 and fac_names:
            spmt_fac_src, spmt_fac_dst = [], []
            for j, s in enumerate(self.entities.get("spmts", [])):
                fac_idx = 0
                for idx, name in enumerate(fac_names):
                    if name in s.current_location:
                        fac_idx = idx
                        break
                spmt_fac_src.append(j)
                spmt_fac_dst.append(fac_idx)
            data["spmt", "at", "facility"].edge_index = torch.tensor(
                [spmt_fac_src, spmt_fac_dst], dtype=torch.long
            )

        # Crane -> Facility edges (cranes are at docks, use last facility index)
        if n_cranes > 0 and fac_names:
            crane_fac_src, crane_fac_dst = [], []
            dock_fac_idx = len(fac_names) - 1
            for k in range(n_cranes):
                crane_fac_src.append(k)
                crane_fac_dst.append(dock_fac_idx)
            data["crane", "at", "facility"].edge_index = torch.tensor(
                [crane_fac_src, crane_fac_dst], dtype=torch.long
            )

        # --- Supply chain nodes and edges ---
        suppliers = self.entities.get("suppliers", [])
        inventory = self.entities.get("inventory", [])
        labor_pools = self.entities.get("labor_pools", [])

        if suppliers:
            sup_x = [torch.tensor(s.get_feature_vector(), dtype=torch.float) for s in suppliers]
            data["supplier"].x = torch.stack(sup_x)
            data["supplier"].batch = torch.zeros(len(suppliers), dtype=torch.long)

        if inventory:
            inv_x = [torch.tensor(inv.get_feature_vector(), dtype=torch.float) for inv in inventory]
            data["inventory"].x = torch.stack(inv_x)
            data["inventory"].batch = torch.zeros(len(inventory), dtype=torch.long)

        if labor_pools:
            lab_x = [torch.tensor(lp.get_feature_vector(), dtype=torch.float) for lp in labor_pools]
            data["labor"].x = torch.stack(lab_x)
            data["labor"].batch = torch.zeros(len(labor_pools), dtype=torch.long)

        n_inv = len(inventory)
        n_sup = len(suppliers)
        n_lab = len(labor_pools)

        # block -> inventory edges (requires_material): each block links to all inventory
        if n_blocks > 0 and n_inv > 0:
            src, dst = [], []
            for i in range(n_blocks):
                for j in range(n_inv):
                    src.append(i)
                    dst.append(j)
            data["block", "requires_material", "inventory"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        # inventory -> supplier edges (supplied_by)
        if n_inv > 0 and n_sup > 0:
            src, dst = [], []
            for i in range(n_inv):
                mat_type = inventory[i].material_type.value
                for j in range(n_sup):
                    if mat_type in suppliers[j].specializations:
                        src.append(i)
                        dst.append(j)
            if src:
                data["inventory", "supplied_by", "supplier"].edge_index = torch.tensor(
                    [src, dst], dtype=torch.long
                )

        # supplier -> facility edges (delivers_to): each supplier connects to first facility
        if n_sup > 0 and fac_names:
            src, dst = [], []
            for j in range(n_sup):
                src.append(j)
                dst.append(0)  # Delivers to first facility (steel_stockyard)
            data["supplier", "delivers_to", "facility"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        # block -> labor edges (requires_labor)
        if n_blocks > 0 and n_lab > 0:
            src, dst = [], []
            for i in range(n_blocks):
                for j in range(n_lab):
                    src.append(i)
                    dst.append(j)
            data["block", "requires_labor", "labor"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        # labor -> facility edges (works_at)
        if n_lab > 0 and fac_names:
            src, dst = [], []
            for j in range(n_lab):
                src.append(j)
                dst.append(0)
            data["labor", "works_at", "facility"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        # inventory -> facility edges (stored_at)
        if n_inv > 0 and fac_names:
            src, dst = [], []
            for j in range(n_inv):
                src.append(j)
                dst.append(0)
            data["inventory", "stored_at", "facility"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        return data

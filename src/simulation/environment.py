"""Gymnasium environment for shipyard scheduling.

This environment encapsulates a simplified discrete‑event simulation of a
shipyard. It models block progression through production stages, transport by
SPMTs, lifts by cranes, and equipment degradation. The agent interacts
with the environment by dispatching vehicles, triggering maintenance, or
holding. The state is encoded as a heterogeneous graph suitable for a
graph neural network (GNN) and flattened into a vector for simple
policies. Rewards penalize tardiness, empty travel, and breakdowns while
encouraging block completion and preventive maintenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .shipyard import ShipyardGraph
from .entities import (
    Block,
    SPMT,
    GoliathCrane,
    HHIProductionStage,
    BlockStatus,
    SPMTStatus,
    GoliathCraneStatus,
)

# Backward compatibility aliases
Crane = GoliathCrane
CraneStatus = GoliathCraneStatus
ProductionStage = HHIProductionStage
from .degradation import WienerDegradationModel
from .precedence import is_predecessor_complete


class ShipyardEnv(gym.Env):
    """OpenAI Gymnasium environment for health‑aware shipyard scheduling."""

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, config: dict, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.db_logging_enabled = False

        # Shipyard graph
        self.shipyard = ShipyardGraph(config.get("shipyard", {}))

        # Counts - support both flat config and HHI nested config
        self.n_blocks = int(config.get("n_blocks", 200))

        # SPMTs: check transporters.n_spmts first (HHI config), then n_spmts
        transporters = config.get("transporters", {})
        self.n_spmts = int(transporters.get("n_spmts", config.get("n_spmts", 12)))

        # Cranes: check goliath_cranes list first (HHI config), then n_cranes
        goliath_cranes_list = config.get("goliath_cranes", [])
        self.n_cranes = len(goliath_cranes_list) if goliath_cranes_list else int(config.get("n_cranes", 3))

        # Facilities: count from all zones (HHI config) or flat list
        facility_count = 0
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_config = config.get(zone, {})
            facility_count += len(zone_config.get("facilities", []))
        self.n_facilities = facility_count if facility_count > 0 else len(config.get("shipyard", {}).get("facilities", []))

        # Feature dimensions (matching the agent architecture)
        self.block_features = 8
        self.spmt_features = 9
        self.crane_features = 8  # GoliathCrane has 3 health components (hoist, trolley, gantry)
        self.facility_features = 3

        # Define observation space as a flat vector. A GNN encoder will reshape.
        obs_dim = (
            self.n_blocks * self.block_features
            + self.n_spmts * self.spmt_features
            + self.n_cranes * self.crane_features
            + self.n_facilities * self.facility_features
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space definition (hierarchical). See the policy for details.
        max_requests = max(self.n_blocks, 1)  # upper bound on pending requests
        max_equipment = max(self.n_spmts + self.n_cranes, 1)
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(4),  # dispatch spmt, dispatch crane, maint, hold
                "spmt_idx": spaces.Discrete(max(self.n_spmts, 1)),
                "request_idx": spaces.Discrete(max_requests),
                "crane_idx": spaces.Discrete(max(self.n_cranes, 1)),
                "lift_idx": spaces.Discrete(max_requests),
                "equipment_idx": spaces.Discrete(max_equipment),
            }
        )

        # Reward weights
        self.w_tardy = float(config.get("reward_tardy", 10.0))
        self.w_empty = float(config.get("reward_empty_travel", 0.1))
        self.w_breakdown = float(config.get("reward_breakdown", 100.0))
        self.w_maintenance = float(config.get("reward_maintenance", 5.0))
        self.w_completion = float(config.get("reward_completion", 1.0))

        # Internal simulation state
        self.sim_time: float = 0.0
        self.entities: Dict[str, List[Any]] = {}
        self.transport_requests: List[Dict[str, Any]] = []
        self.lift_requests: List[Dict[str, Any]] = []
        self.facility_queues: Dict[str, List[str]] = {}
        self.facility_processing: Dict[str, List[str]] = {}
        self.facility_remaining_time: Dict[str, Dict[str, float]] = {}
        self.degradation_model = WienerDegradationModel()
        self.max_time = float(config.get("max_time", 10000))

        # Initialize ships list
        self.ships: List[Any] = []

    # ------------------------------------------------------------------
    # Properties for entity access
    # ------------------------------------------------------------------
    @property
    def blocks(self) -> List[Block]:
        """Get all blocks."""
        return self.entities.get("blocks", [])

    @property
    def spmts(self) -> List[SPMT]:
        """Get all SPMTs."""
        return self.entities.get("spmts", [])

    # goliath_cranes is set in _create_cranes()

    # ------------------------------------------------------------------
    # Creation and reset methods
    # ------------------------------------------------------------------
    def _create_blocks(self) -> None:
        """Initialize blocks with random attributes and due dates."""
        blocks: List[Block] = []
        rng = np.random.default_rng(seed=42)
        for i in range(self.n_blocks):
            weight = float(rng.uniform(50.0, 300.0))  # tons
            size = (float(rng.uniform(10.0, 20.0)), float(rng.uniform(10.0, 20.0)))
            due_date = float(100 + 5 * i)  # staggered due dates
            block = Block(
                id=f"B{i}",
                weight=weight,
                size=size,
                due_date=due_date,
            )
            blocks.append(block)
        self.entities["blocks"] = blocks

    def _create_spmts(self) -> None:
        """Create SPMTs from config."""
        spmts: List[SPMT] = []
        transporters = self.config.get("transporters", {})
        spmt_capacity = float(transporters.get("spmt_capacity_tons", 500.0))
        for i in range(self.n_spmts):
            spmt = SPMT(id=f"SPMT-{i:02d}", capacity=spmt_capacity, current_location="spmt_depot")
            spmts.append(spmt)
        self.entities["spmts"] = spmts

    def _create_cranes(self) -> None:
        """Create Goliath cranes from config."""
        cranes: List[GoliathCrane] = []
        goliath_cranes_list = self.config.get("goliath_cranes", [])

        if goliath_cranes_list:
            # HHI config with explicit crane definitions
            for crane_cfg in goliath_cranes_list:
                crane = GoliathCrane(
                    id=crane_cfg.get("id", f"GC{len(cranes):02d}"),
                    assigned_dock=crane_cfg.get("assigned_dock", ""),
                    capacity_tons=float(crane_cfg.get("capacity_tons", 900.0)),
                    height_m=float(crane_cfg.get("height_m", 109.0)),
                )
                cranes.append(crane)
        else:
            # Fallback: generate cranes
            for i in range(self.n_cranes):
                crane = GoliathCrane(
                    id=f"GC{i:02d}",
                    assigned_dock=f"dock_{(i % 10) + 1}",
                    capacity_tons=900.0,
                )
                cranes.append(crane)

        self.entities["cranes"] = cranes
        # Also store as goliath_cranes for compatibility
        self.goliath_cranes = cranes

    def _initialize_facilities(self) -> None:
        """Create queues and processing lists for each facility."""
        # Try HHI zone structure first
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_config = self.config.get(zone, {})
            for f in zone_config.get("facilities", []):
                name = f["name"]
                self.facility_queues[name] = []
                self.facility_processing[name] = []
                self.facility_remaining_time[name] = {}

        # Fallback to flat facilities list
        if not self.facility_queues:
            facilities = self.config.get("shipyard", {}).get("facilities", [])
            for f in facilities:
                name = f["name"]
                self.facility_queues[name] = []
                self.facility_processing[name] = []
                self.facility_remaining_time[name] = {}

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        # Reset time and event lists
        self.sim_time = 0.0
        self.transport_requests.clear()
        self.lift_requests.clear()
        self.entities.clear()
        self.facility_queues.clear()
        self.facility_processing.clear()
        self.facility_remaining_time.clear()
        # Create entities
        self._create_blocks()
        self._create_spmts()
        self._create_cranes()
        self._initialize_facilities()
        # Assign initial queues: all blocks wait at first facility queue
        first_fac = None
        # Try HHI zone structure first
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_config = self.config.get(zone, {})
            zone_facilities = zone_config.get("facilities", [])
            if zone_facilities:
                first_fac = zone_facilities[0]["name"]
                break
        # Fallback to flat facilities list
        if not first_fac:
            facilities = self.config.get("shipyard", {}).get("facilities", [])
            if facilities:
                first_fac = facilities[0]["name"]
        if first_fac:
            for block in self.entities["blocks"]:
                self.facility_queues[first_fac].append(block.id)
                block.location = f"queue_{first_fac}"
        # Clear equipment statuses
        for spmt in self.entities["spmts"]:
            spmt.status = SPMTStatus.IDLE
            spmt.current_load = None
            spmt.health_hydraulic = 100.0
            spmt.health_tires = 100.0
            spmt.health_engine = 100.0
        for crane in self.entities["cranes"]:
            crane.status = CraneStatus.IDLE
            crane.current_block = None
            crane.health_hoist = 100.0
            crane.health_trolley = 100.0
            crane.health_gantry = 100.0
        # Reset metrics
        self.metrics = {
            "blocks_completed": 0,
            "breakdowns": 0,
            "planned_maintenance": 0,
            "total_tardiness": 0.0,
            "empty_travel_distance": 0.0,
        }
        # Warm-up: advance simulation to create initial transport requests
        # This ensures SPMT dispatch is available from the start of training
        warmup_steps = 10
        for _ in range(warmup_steps):
            self._advance_simulation(dt=1.0)
        # Reset sim_time to 0 after warmup (tests expect reset to clear time)
        self.sim_time = 0.0
        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Simulation update helpers
    # ------------------------------------------------------------------
    # Map config facility names to HHI ProductionStage enum members
    _FAC_NAME_TO_STAGE = {
        # Steel Processing Zone (HHI style)
        "cutting_shop": ProductionStage.STEEL_CUTTING,
        "steel_stockyard": ProductionStage.STEEL_CUTTING,
        "part_fabrication": ProductionStage.PART_FABRICATION,
        # Panel Assembly Zone (HHI style)
        "flat_panel_line": ProductionStage.PANEL_ASSEMBLY,
        "flat_panel_line_1": ProductionStage.PANEL_ASSEMBLY,
        "flat_panel_line_2": ProductionStage.PANEL_ASSEMBLY,
        "curved_block_shop": ProductionStage.PANEL_ASSEMBLY,
        # Block Assembly Zone (HHI style)
        "block_assembly": ProductionStage.BLOCK_ASSEMBLY,
        "block_assembly_hall_1": ProductionStage.BLOCK_ASSEMBLY,
        "block_assembly_hall_2": ProductionStage.BLOCK_ASSEMBLY,
        "block_assembly_hall_3": ProductionStage.BLOCK_ASSEMBLY,
        "outfitting_shop": ProductionStage.BLOCK_OUTFITTING,
        "paint_shop": ProductionStage.PAINTING,
        # Pre-Erection Zone (HHI style)
        "grand_block_staging": ProductionStage.PRE_ERECTION,
        "grand_block_staging_north": ProductionStage.PRE_ERECTION,
        "grand_block_staging_south": ProductionStage.PRE_ERECTION,
        # Dry Docks (HHI has 10 dry docks)
        "dock": ProductionStage.ERECTION,
        "dock_1": ProductionStage.ERECTION,
        "dock_2": ProductionStage.ERECTION,
        "dock_3": ProductionStage.ERECTION,
        "dock_4": ProductionStage.ERECTION,
        "dock_5": ProductionStage.ERECTION,
        "dock_6": ProductionStage.ERECTION,
        "dock_7": ProductionStage.ERECTION,
        "dock_8": ProductionStage.ERECTION,
        "dock_9": ProductionStage.ERECTION,
        "dock_10": ProductionStage.ERECTION,
        # Default/flat config facility names (for default.yaml)
        "cutting": ProductionStage.STEEL_CUTTING,
        "panel": ProductionStage.PANEL_ASSEMBLY,
        "assembly": ProductionStage.BLOCK_ASSEMBLY,
        "outfitting": ProductionStage.BLOCK_OUTFITTING,
        "paint": ProductionStage.PAINTING,
    }

    # Map ProductionStage to list of possible facility names (order = priority)
    # Includes both HHI and default/flat config names
    _STAGE_TO_FACILITIES = {
        ProductionStage.STEEL_CUTTING: ["steel_stockyard", "cutting_shop", "cutting"],
        ProductionStage.PART_FABRICATION: ["part_fabrication", "cutting"],  # fallback to cutting
        ProductionStage.PANEL_ASSEMBLY: ["flat_panel_line_1", "flat_panel_line_2", "curved_block_shop", "panel"],
        ProductionStage.BLOCK_ASSEMBLY: ["block_assembly_hall_1", "block_assembly_hall_2", "block_assembly_hall_3", "assembly"],
        ProductionStage.BLOCK_OUTFITTING: ["outfitting_shop", "outfitting"],
        ProductionStage.PAINTING: ["paint_shop", "paint"],
        ProductionStage.PRE_ERECTION: ["grand_block_staging_north", "grand_block_staging_south"],
        ProductionStage.ERECTION: None,  # Handled by crane lift, not SPMT transport
    }

    def _get_facility_for_stage(self, stage: ProductionStage, block: "Block") -> Optional[str]:
        """Get the target facility for a given production stage.

        Returns None if the stage requires crane lift (ERECTION) or no facility mapping.
        Uses load balancing: picks facility with shortest queue from available facilities.
        """
        facilities = self._STAGE_TO_FACILITIES.get(stage)
        if facilities is None:
            return None

        # Filter to only facilities that actually exist in our queues
        available_facilities = [f for f in facilities if f in self.facility_queues]
        if not available_facilities:
            return None

        # Load balance: pick facility with shortest queue
        min_queue_len = float("inf")
        best_fac = available_facilities[0]
        for fac in available_facilities:
            queue_len = len(self.facility_queues.get(fac, []))
            if queue_len < min_queue_len:
                min_queue_len = queue_len
                best_fac = fac
        return best_fac

    def _assign_blocks_to_facilities(self) -> None:
        """Assign blocks from facility queues to processing if capacity allows."""
        # Build facility info from HHI zone structure or flat config
        fac_info = {}
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_config = self.config.get(zone, {})
            for f in zone_config.get("facilities", []):
                fac_info[f["name"]] = f
        # Fallback to flat config
        if not fac_info:
            facilities_cfg = self.config.get("shipyard", {}).get("facilities", [])
            fac_info = {f["name"]: f for f in facilities_cfg}

        for fac_name, queue in self.facility_queues.items():
            fac = fac_info.get(fac_name, {})
            capacity = fac.get("capacity", 1)
            while queue and len(self.facility_processing[fac_name]) < capacity:
                block_id = queue.pop(0)
                self.facility_processing[fac_name].append(block_id)
                # Determine processing time (log‑normal)
                mean = fac.get("processing_time_mean", 10.0)
                std = fac.get("processing_time_std", 2.0)
                proc_time = float(np.random.lognormal(mean=np.log(mean), sigma=std / mean))
                self.facility_remaining_time[fac_name][block_id] = proc_time
                # Update block status
                block = self._get_block(block_id)
                block.status = BlockStatus.IN_PROCESS
                # NOTE: Do NOT reset block.current_stage here!
                # The stage is set when block advances after completing a facility.
                # Resetting here causes blocks to oscillate between stages when
                # a facility handles multiple stages (e.g., "cutting" for both
                # STEEL_CUTTING and PART_FABRICATION fallback).
                block.location = fac_name

    def _update_processing(self, dt: float) -> None:
        """Advance processing on all facilities by dt hours."""
        completed_blocks: List[Tuple[str, str]] = []  # (fac_name, block_id)
        for fac_name, remaining_times in list(self.facility_remaining_time.items()):
            finished_ids: List[str] = []
            for block_id, remaining in list(remaining_times.items()):
                new_remaining = remaining - dt
                self.facility_remaining_time[fac_name][block_id] = new_remaining
                if new_remaining <= 0.0:
                    finished_ids.append(block_id)
            # Move finished blocks to completed list
            for block_id in finished_ids:
                completed_blocks.append((fac_name, block_id))
                self.facility_processing[fac_name].remove(block_id)
                del self.facility_remaining_time[fac_name][block_id]

        # Handle completion: create transport or lift requests depending on stage
        # Build facilities_list from zone structure (matches HHI config)
        facilities_list = []
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_config = self.config.get(zone, {})
            facilities_list.extend(zone_config.get("facilities", []))
        # Fallback to flat config if zones not present
        if not facilities_list:
            facilities_list = self.config.get("shipyard", {}).get("facilities", [])
        for fac_name, block_id in completed_blocks:
            block = self._get_block(block_id)
            block.status = BlockStatus.WAITING
            # Determine next stage
            next_stage_index = block.current_stage.value + 1
            if next_stage_index >= len(ProductionStage):
                # Completed all stages
                block.status = BlockStatus.ERECTED
                self.metrics["blocks_completed"] += 1
                self._log_block_event(block_id, "completed", "DOCK", "dock")
                continue
            next_stage = ProductionStage(next_stage_index)
            block.current_stage = next_stage
            # Track stage advances for reward computation
            self.metrics["stage_advances"] = self.metrics.get("stage_advances", 0) + 1
            self._log_block_event(block_id, "stage_complete", fac_name, block.location)

            # Map stage to facility using _STAGE_TO_FACILITIES
            target_fac = self._get_facility_for_stage(next_stage, block)
            if target_fac is None:
                # PRE_ERECTION / ERECTION — create lift request
                self.lift_requests.append({"block_id": block.id})
                block.location = "pre_erection"
                block.status = BlockStatus.AT_PRE_ERECTION
            else:
                # Create transport request to next facility queue
                self.transport_requests.append(
                    {
                        "block_id": block.id,
                        "destination": target_fac,
                    }
                )
                block.location = f"waiting_transport_to_{target_fac}"
                block.status = BlockStatus.WAITING

    def _degrade_equipment(self, dt: float) -> None:
        """Apply degradation to SPMTs and cranes based on operation status."""
        for spmt in self.entities.get("spmts", []):
            operating = spmt.status in {
                SPMTStatus.TRAVELING_EMPTY,
                SPMTStatus.TRAVELING_LOADED,
                SPMTStatus.LOADING,
                SPMTStatus.UNLOADING,
            }
            load_ratio = 0.0
            if spmt.current_load:
                # approximate load ratio based on block weight
                block = self._get_block(spmt.current_load)
                load_ratio = block.weight / spmt.capacity
            # Degrade each component separately
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
                # Mark as broken down
                spmt.status = SPMTStatus.BROKEN_DOWN
                self.metrics["breakdowns"] += 1
                self._log_equipment_event(spmt.id, "spmt", "breakdown")
        for crane in self.entities.get("cranes", []):
            operating = crane.status in {CraneStatus.LIFTING, CraneStatus.POSITIONING}
            crane.health_hoist, fail_hoist = self.degradation_model.step(
                crane.health_hoist, dt, load_ratio=0.2, operating=operating
            )
            crane.health_trolley, fail_trolley = self.degradation_model.step(
                crane.health_trolley, dt, load_ratio=0.2, operating=operating
            )
            crane.health_gantry, fail_gantry = self.degradation_model.step(
                crane.health_gantry, dt, load_ratio=0.2, operating=operating
            )
            if fail_hoist or fail_trolley or fail_gantry:
                crane.status = CraneStatus.BROKEN_DOWN
                self.metrics["breakdowns"] += 1
                self._log_equipment_event(crane.id, "crane", "breakdown")

    def _advance_simulation(self, dt: float = 1.0) -> None:
        """Advance simulation by dt hours.

        This helper function updates processing, degrades equipment and
        increments the simulation clock. It is called after each agent action.
        """
        # Assign blocks from queues to facilities
        self._assign_blocks_to_facilities()
        # Update processing times
        self._update_processing(dt)
        # Degrade equipment
        self._degrade_equipment(dt)
        # Increase time
        self.sim_time += dt
        # Tardiness accumulation - only count the INCREMENT per step (dt for each tardy block)
        # This prevents exponential blowup of cumulative tardiness
        for block in self.entities.get("blocks", []):
            if block.status != BlockStatus.ERECTED:
                if self.sim_time > block.due_date:
                    # Block is tardy - add dt (1 unit of additional tardiness per step)
                    self.metrics["total_tardiness"] += dt

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _dispatch_spmt(self, spmt_idx: int, request_idx: int) -> float:
        """Dispatch an SPMT to fulfill a transport request.

        Returns a reward component (e.g. negative empty travel distance).
        Tracks SPMT busy time for utilization metrics.
        """
        if request_idx >= len(self.transport_requests):
            return -0.01  # Penalty for invalid dispatch (no such request)
        request = self.transport_requests.pop(request_idx)
        block_id = request["block_id"]
        destination = request["destination"]
        spmt = self.entities["spmts"][spmt_idx]
        block = self._get_block(block_id)
        # If SPMT is already carrying a load or not idle, penalize
        if spmt.status != SPMTStatus.IDLE or spmt.current_load is not None:
            # Put the request back (it wasn't fulfilled)
            self.transport_requests.insert(request_idx, request)
            return -0.01  # Penalty for invalid dispatch (SPMT busy)
        # Compute travel time (distance) from current location to block
        travel_to_block = self.shipyard.get_travel_time(spmt.current_location, block.location)
        # Travel from block to destination
        travel_to_dest = self.shipyard.get_travel_time(block.location, destination)
        empty_distance = travel_to_block
        loaded_distance = travel_to_dest

        # Track SPMT busy time (both empty and loaded travel)
        # Assume travel time in hours (distance units are travel time)
        total_travel_time = empty_distance + loaded_distance
        if "spmt_busy_time" not in self.metrics:
            self.metrics["spmt_busy_time"] = 0.0
        self.metrics["spmt_busy_time"] += total_travel_time

        # Update empty travel metrics
        self.metrics["empty_travel_distance"] += empty_distance

        # Update SPMT state for travel
        spmt.status = SPMTStatus.TRAVELING_EMPTY
        self._log_block_event(block_id, "transport_dispatch", block.current_stage.name, spmt.current_location)

        # Assign load
        spmt.current_load = block_id
        spmt.status = SPMTStatus.TRAVELING_LOADED
        spmt.current_location = destination

        # Drop off block at destination
        block.location = f"queue_{destination}"
        self.facility_queues[destination].append(block_id)
        spmt.current_load = None
        spmt.status = SPMTStatus.IDLE

        self._log_block_event(block_id, "transport_arrival", block.current_stage.name, destination)
        # Reward: +0.2 for successful dispatch, minus empty travel penalty
        dispatch_reward = 0.2
        return dispatch_reward - self.w_empty * empty_distance

    def _dispatch_crane(self, crane_idx: int, request_idx: int) -> float:
        """Dispatch a crane to lift a block onto the dock.

        Validates precedence constraints and models crane travel/lift time.
        """
        if request_idx >= len(self.lift_requests):
            return -0.01  # Penalty for invalid dispatch (no such request)
        request = self.lift_requests.pop(request_idx)
        block_id = request["block_id"]
        crane = self.entities["cranes"][crane_idx]
        block = self._get_block(block_id)
        if crane.status != CraneStatus.IDLE:
            # Put the request back (it wasn't fulfilled)
            self.lift_requests.insert(request_idx, request)
            return -0.01  # Penalty for invalid dispatch (crane busy)

        # Enforce precedence: all predecessors must be placed on dock
        placed_blocks = {b.id: b for b in self.entities["blocks"]
                        if b.status == BlockStatus.ERECTED}
        if not is_predecessor_complete(block, placed_blocks):
            # Re-add request since we can't process it yet
            self.lift_requests.append(request)
            return -0.01  # Penalty for precedence violation

        # Model crane travel time based on rail position
        # Assume block is at pre-erection area at position 0, dock at rail end
        block_rail_pos = 0.0  # pre-erection position
        travel_distance = abs(crane.position_on_rail - block_rail_pos)
        crane_speed = 0.5  # m/s (configurable)
        travel_time = travel_distance / crane_speed if crane_speed > 0 else 0.0

        # Lift duration (configurable, default 2 hours for heavy lift)
        lift_duration = 2.0
        total_crane_time = travel_time / 3600.0 + lift_duration  # convert travel to hours

        # Track crane busy time
        if "crane_busy_time" not in self.metrics:
            self.metrics["crane_busy_time"] = 0.0
        self.metrics["crane_busy_time"] += total_crane_time

        # Update crane state
        crane.status = CraneStatus.POSITIONING
        crane.position_on_rail = block_rail_pos  # crane moves to block
        crane.status = CraneStatus.LIFTING
        crane.current_block = block_id

        # Place block on dock and update stage
        # Treat ERECTION as block-level completion (ship-level stages 8-10 are aggregated separately)
        block.current_stage = ProductionStage.DELIVERY
        block.status = BlockStatus.ERECTED
        block.location = "dock"
        self.metrics["blocks_completed"] += 1

        # Reset crane
        crane.status = CraneStatus.IDLE
        crane.current_block = None

        self._log_block_event(block_id, "lift_to_dock", "DOCK", "dock")
        return self.w_completion

    def _trigger_maintenance(self, equipment_idx: int) -> float:
        """Trigger preventive maintenance on an SPMT or crane."""
        reward = 0.0
        if equipment_idx < self.n_spmts:
            spmt = self.entities["spmts"][equipment_idx]
            if spmt.status == SPMTStatus.IDLE:
                spmt.status = SPMTStatus.IN_MAINTENANCE
                spmt.health_hydraulic = self.degradation_model.perform_maintenance()
                spmt.health_tires = self.degradation_model.perform_maintenance()
                spmt.health_engine = self.degradation_model.perform_maintenance()
                spmt.status = SPMTStatus.IDLE
                reward -= self.w_maintenance
                self.metrics["planned_maintenance"] += 1
                self._log_equipment_event(spmt.id, "spmt", "maintenance_complete")
        else:
            # Crane maintenance
            crane_idx_rel = equipment_idx - self.n_spmts
            if crane_idx_rel < len(self.entities["cranes"]):
                crane = self.entities["cranes"][crane_idx_rel]
                if crane.status == CraneStatus.IDLE:
                    crane.status = CraneStatus.IN_MAINTENANCE
                    crane.health_hoist = self.degradation_model.perform_maintenance()
                    crane.health_trolley = self.degradation_model.perform_maintenance()
                    crane.health_gantry = self.degradation_model.perform_maintenance()
                    crane.status = CraneStatus.IDLE
                    reward -= self.w_maintenance
                    self.metrics["planned_maintenance"] += 1
                    self._log_equipment_event(crane.id, "crane", "maintenance_complete")
        return reward

    # ------------------------------------------------------------------
    # Gym API methods
    # ------------------------------------------------------------------
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep given an action.

        Reward includes all 5 cost components:
        1. Completion reward (+)
        2. Empty travel penalty (-)
        3. Breakdown penalty (-)
        4. Maintenance cost (-)
        5. Tardiness penalty (-)
        """
        reward = 0.0
        # Track metrics before action for delta computation
        empty_travel_before = self.metrics.get("empty_travel_distance", 0.0)
        tardiness_before = self.metrics.get("total_tardiness", 0.0)
        breakdowns_before = self.metrics["breakdowns"]
        blocks_completed_before = self.metrics["blocks_completed"]
        stage_advances_before = self.metrics.get("stage_advances", 0)

        # Decode action
        action_type = int(action.get("action_type", 3))  # default hold
        if action_type == 0:
            spmt_idx = int(action.get("spmt_idx", 0))
            req_idx = int(action.get("request_idx", 0))
            reward += self._dispatch_spmt(spmt_idx, req_idx)
        elif action_type == 1:
            crane_idx = int(action.get("crane_idx", 0))
            req_idx = int(action.get("lift_idx", 0))
            reward += self._dispatch_crane(crane_idx, req_idx)
        elif action_type == 2:
            equip_idx = int(action.get("equipment_idx", 0))
            reward += self._trigger_maintenance(equip_idx)
        elif action_type == 3:
            # Hold (no operation) - only penalize if there were other valid actions
            # Check if SPMT dispatch, crane dispatch, or maintenance was possible
            has_valid_spmt = len(self.transport_requests) > 0
            has_valid_crane = len(self.lift_requests) > 0
            has_valid_maintenance = any(
                e.get_min_health() < 60.0 and e.status not in [SPMTStatus.BROKEN_DOWN, CraneStatus.BROKEN_DOWN]
                for e in self.entities.get("spmts", []) + self.entities.get("cranes", [])
            )
            if has_valid_spmt or has_valid_crane or has_valid_maintenance:
                # There was something to do, but agent chose to hold - penalize
                reward -= 0.3

        # Advance simulation by one hour
        self._advance_simulation(dt=1.0)

        # Penalize only new breakdowns that occurred this step
        new_breakdowns = self.metrics["breakdowns"] - breakdowns_before
        reward -= self.w_breakdown * new_breakdowns

        # Penalize tardiness accumulated this step
        tardiness_delta = self.metrics.get("total_tardiness", 0.0) - tardiness_before
        reward -= self.w_tardy * tardiness_delta * 0.01  # scale down for per-step

        # Reward for block completions (+1.0 per completed block)
        new_completions = self.metrics["blocks_completed"] - blocks_completed_before
        reward += 1.0 * new_completions

        # Reward for stage progression (+0.5 per stage advance)
        # High reward to encourage dispatching blocks through the pipeline
        new_stage_advances = self.metrics.get("stage_advances", 0) - stage_advances_before
        reward += 0.5 * new_stage_advances

        # Reward for successful dispatches (SPMT or crane)
        # Action types 0 and 1 are dispatch actions; give bonus if they succeeded
        if action_type in (0, 1) and new_stage_advances > 0:
            reward += 0.3  # Bonus for dispatch that led to progress

        # Clip rewards to prevent extreme values from destabilizing training
        # Positive capped at +2.0 (allows multiple block completions), negative at -10.0
        reward = float(np.clip(reward, -10.0, 2.0))

        # Periodic DB logging for time-series data (every 50 sim hours)
        if self.db_logging_enabled and int(self.sim_time) % 50 == 0:
            self.log_state_to_db()

        # Determine termination and truncation
        terminated = self.metrics["blocks_completed"] == self.n_blocks
        truncated = self.sim_time >= self.max_time

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Observation, masking and graph data
    # ------------------------------------------------------------------
    def _encode_block(self, block: Block) -> List[float]:
        # Encode block attributes into a feature vector
        stage = block.current_stage.value
        # Simple location encoding: index of facility or staging area, else -1
        loc_enc = 0.0
        if block.location.startswith("queue_"):
            loc_enc = 0.5
        elif block.location == "dock":
            loc_enc = 1.0
        completion = block.completion_pct
        time_to_due = max(0.0, block.due_date - self.sim_time)
        pred_ready = 1.0 if is_predecessor_complete(block, {b.id: b for b in self.entities["blocks"]}) else 0.0
        weight = block.weight / 500.0  # normalized
        in_transit = 1.0 if block.status == BlockStatus.IN_TRANSIT else 0.0
        waiting = 1.0 if block.status == BlockStatus.WAITING else 0.0
        return [stage, loc_enc, completion, time_to_due, pred_ready, weight, in_transit, waiting]

    def _encode_spmt(self, spmt: SPMT) -> List[float]:
        # Location encoding: not used for now (set to 0), status encoding: one‑hot of 4 states
        loc_enc = 0.0
        status_enc = [0.0, 0.0, 0.0, 0.0]
        status_map = {
            SPMTStatus.IDLE: 0,
            SPMTStatus.TRAVELING_EMPTY: 1,
            SPMTStatus.TRAVELING_LOADED: 2,
            SPMTStatus.IN_MAINTENANCE: 3,
        }
        idx = status_map.get(spmt.status, 0)
        status_enc[idx] = 1.0
        load_ratio = 0.0
        if spmt.current_load:
            block = self._get_block(spmt.current_load)
            load_ratio = block.weight / spmt.capacity
        health = spmt.get_health_vector().tolist()
        return [loc_enc] + status_enc + [load_ratio] + health

    def _encode_crane(self, crane: Crane) -> List[float]:
        pos_norm = crane.position_on_rail / 100.0  # assume max rail length 100m
        status_enc = [0.0, 0.0, 0.0]
        status_map = {
            CraneStatus.IDLE: 0,
            CraneStatus.LIFTING: 1,
            CraneStatus.POSITIONING: 2,
        }
        idx = status_map.get(crane.status, 0)
        status_enc[idx] = 1.0
        health = crane.get_health_vector().tolist()
        current_load = 1.0 if crane.current_block else 0.0
        return [pos_norm] + status_enc + health + [current_load]

    def _encode_facility(self, fac_name: str) -> List[float]:
        queue_len = len(self.facility_queues.get(fac_name, []))
        util = float(len(self.facility_processing.get(fac_name, [])))
        avg_wait = float(queue_len)
        return [queue_len, util, avg_wait]

    def _get_observation(self) -> np.ndarray:
        obs: List[float] = []
        # Block features
        for block in self.entities.get("blocks", []):
            obs.extend(self._encode_block(block))
        # SPMT features
        for spmt in self.entities.get("spmts", []):
            obs.extend(self._encode_spmt(spmt))
        # Crane features
        for crane in self.entities.get("cranes", []):
            obs.extend(self._encode_crane(crane))
        # Facility features
        facilities = self.config.get("shipyard", {}).get("facilities", [])
        for f in facilities:
            obs.extend(self._encode_facility(f["name"]))
                # OBS_SANITIZE_STRINGS: ensure obs is numeric (encode strings like stage names)
        str_map = {"paint": 1.0}
        obs_clean = []
        for v in obs:
            if isinstance(v, (int, float)):
                obs_clean.append(float(v))
            elif hasattr(v, "item"):
                # numpy scalar / torch scalar
                try:
                    obs_clean.append(float(v.item()))
                except Exception:
                    obs_clean.append(0.0)
            else:
                s = str(v).lower()
                obs_clean.append(float(str_map.get(s, 0.0)))
        obs = obs_clean
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        return {
            "sim_time": self.sim_time,
            "metrics": self.metrics.copy(),
            "transport_requests": len(self.transport_requests),
            "lift_requests": len(self.lift_requests),
        }

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """Return a mask of valid actions for the current state."""
        mask: Dict[str, Any] = {
            "action_type": np.ones(4, dtype=bool),
            "spmt_dispatch": np.zeros((self.n_spmts, len(self.transport_requests)), dtype=bool),
            "crane_dispatch": np.zeros((self.n_cranes, len(self.lift_requests)), dtype=bool),
            "maintenance": np.zeros(self.n_spmts + self.n_cranes, dtype=bool),
        }
        # SPMT dispatch mask (excludes broken equipment)
        for i, spmt in enumerate(self.entities.get("spmts", [])):
            if spmt.status == SPMTStatus.IDLE and spmt.status != SPMTStatus.BROKEN_DOWN:
                for j, req in enumerate(self.transport_requests):
                    block = self._get_block(req["block_id"])
                    if block.weight <= spmt.capacity and spmt.get_min_health() > 20.0:
                        mask["spmt_dispatch"][i, j] = True
        # Crane dispatch mask (includes precedence check)
        placed_blocks = {b.id: b for b in self.entities.get("blocks", [])
                        if b.status == BlockStatus.ERECTED}
        for i, crane in enumerate(self.entities.get("cranes", [])):
            if crane.status == CraneStatus.IDLE and crane.status != CraneStatus.BROKEN_DOWN:
                for j, req in enumerate(self.lift_requests):
                    block = self._get_block(req["block_id"])
                    # Check weight, crane health, and precedence constraints
                    if (block.weight <= crane.capacity_tons
                        and crane.get_min_health() > 20.0
                        and is_predecessor_complete(block, placed_blocks)):
                        mask["crane_dispatch"][i, j] = True
        # Maintenance mask (excludes broken equipment - they need repair, not PM)
        for i, spmt in enumerate(self.entities.get("spmts", [])):
            if (spmt.status == SPMTStatus.IDLE
                and spmt.status != SPMTStatus.BROKEN_DOWN
                and spmt.get_min_health() < 60.0):
                mask["maintenance"][i] = True
        for i, crane in enumerate(self.entities.get("cranes", [])):
            idx = self.n_spmts + i
            if (crane.status == CraneStatus.IDLE
                and crane.status != CraneStatus.BROKEN_DOWN
                and crane.get_min_health() < 60.0):
                mask["maintenance"][idx] = True
        # Disable action types if no valid options
        if mask["spmt_dispatch"].size == 0 or not mask["spmt_dispatch"].any():
            mask["action_type"][0] = False
        if mask["crane_dispatch"].size == 0 or not mask["crane_dispatch"].any():
            mask["action_type"][1] = False
        if not mask["maintenance"].any():
            mask["action_type"][2] = False
        return mask

    def _log_block_event(self, block_id: str, event_type: str, stage: str, location: str) -> None:
        """Log a block event to the MES database if logging is enabled."""
        if not self.db_logging_enabled:
            return
        try:
            from mes.database import log_block_event
            log_block_event(block_id, self.sim_time, event_type, stage, location)
        except Exception:
            pass

    def _log_equipment_event(self, equipment_id: str, equipment_type: str, event_type: str) -> None:
        """Log an equipment event (maintenance, breakdown) to the MES database."""
        if not self.db_logging_enabled:
            return
        try:
            from mes.database import log_block_event
            # Re-use block_event table with equipment info in stage/location fields
            log_block_event(equipment_id, self.sim_time, event_type, equipment_type, "equipment")
        except Exception:
            pass

    def log_state_to_db(self) -> None:
        """Log full simulation state to the MES database."""
        if not self.db_logging_enabled:
            return
        try:
            from mes.database import log_metrics, log_entities, log_health_snapshot, log_queue_depths
            log_metrics(self.sim_time, self.metrics)
            log_entities(
                self.entities.get("blocks", []),
                self.entities.get("spmts", []),
                self.entities.get("cranes", []),
            )
            log_health_snapshot(
                self.sim_time,
                self.entities.get("spmts", []),
                self.entities.get("cranes", []),
            )
            log_queue_depths(self.sim_time, self.facility_queues, self.facility_processing)
        except Exception:
            pass

    def _get_block(self, block_id: str) -> Block:
        for b in self.entities.get("blocks", []):
            if b.id == block_id:
                return b
        raise KeyError(f"Block {block_id} not found")

    def get_graph_data(self) -> Dict:
        # GRAPH_SANITIZE_FEATURES: ensure node features are numeric floats
        from simulation.entities import HHIProductionStage as ProductionStage
        stage_map = {s.value: float(i+1) for i, s in enumerate(ProductionStage)}
        def _to_float_list(seq):
            out = []
            for v in seq:
                if isinstance(v, (int, float)):
                    out.append(float(v))
                elif hasattr(v, 'item'):
                    try:
                        out.append(float(v.item()))
                    except Exception:
                        out.append(0.0)
                else:
                    s = str(v).lower()
                    out.append(float(stage_map.get(s, 0.0)))
            return out

        """Construct a PyG heterogeneous graph representation of the state.

        This method produces a dictionary compatible with the GNN encoder.
        The graph includes block, spmt, crane, and facility nodes with
        dummy edges indicating potential interactions. Edge attributes
        currently carry travel times or zeros. Each node type has a
        corresponding feature matrix and a batch vector of zeros (single
        graph instance).
        """
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        # Node features
        block_x = []
        spmt_x = []
        crane_x = []
        fac_x = []
        # Node batch indices (all zeros because single graph)
        block_batch = []
        spmt_batch = []
        crane_batch = []
        fac_batch = []
        # Fill features
        for b in self.entities.get("blocks", []):
            block_x.append(torch.tensor(_to_float_list(self._encode_block(b)), dtype=torch.float))
            block_batch.append(0)
        for s in self.entities.get("spmts", []):
            spmt_x.append(torch.tensor(_to_float_list(self._encode_spmt(s)), dtype=torch.float))
            spmt_batch.append(0)
        for c in self.entities.get("cranes", []):
            crane_x.append(torch.tensor(_to_float_list(self._encode_crane(c)), dtype=torch.float))
            crane_batch.append(0)

        # Get facilities from HHI zone structure or flat config
        fac_names = []
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_config = self.config.get(zone, {})
            for f in zone_config.get("facilities", []):
                fac_names.append(f["name"])
                fac_x.append(torch.tensor(self._encode_facility(f["name"]), dtype=torch.float))
                fac_batch.append(0)
        # Fallback to flat facilities list
        if not fac_names:
            for f in self.config.get("shipyard", {}).get("facilities", []):
                fac_names.append(f["name"])
                fac_x.append(torch.tensor(self._encode_facility(f["name"]), dtype=torch.float))
                fac_batch.append(0)
        data["block"].x = torch.stack(block_x) if block_x else torch.empty((0, self.block_features))
        data["block"].batch = torch.tensor(block_batch, dtype=torch.long)
        data["spmt"].x = torch.stack(spmt_x) if spmt_x else torch.empty((0, self.spmt_features))
        data["spmt"].batch = torch.tensor(spmt_batch, dtype=torch.long)
        data["crane"].x = torch.stack(crane_x) if crane_x else torch.empty((0, self.crane_features))
        data["crane"].batch = torch.tensor(crane_batch, dtype=torch.long)
        data["facility"].x = torch.stack(fac_x) if fac_x else torch.empty((0, self.facility_features))
        data["facility"].batch = torch.tensor(fac_batch, dtype=torch.long)
        # Edge indices: we create simple edges to indicate potential interactions
        # Blocks needing transport connect to spmts; spmts connect back; blocks to cranes; cranes back; blocks to facilities
        # The exact structure is less important for demonstration but sets up heterogenous relations.
        block_to_spmt_src = []
        block_to_spmt_dst = []
        for i, b in enumerate(self.entities.get("blocks", [])):
            for j, s in enumerate(self.entities.get("spmts", [])):
                block_to_spmt_src.append(i)
                block_to_spmt_dst.append(j)
        spmt_to_block_src = block_to_spmt_dst.copy()
        spmt_to_block_dst = block_to_spmt_src.copy()
        block_to_crane_src = []
        block_to_crane_dst = []
        for i, b in enumerate(self.entities.get("blocks", [])):
            for j, c in enumerate(self.entities.get("cranes", [])):
                block_to_crane_src.append(i)
                block_to_crane_dst.append(j)
        crane_to_block_src = block_to_crane_dst.copy()
        crane_to_block_dst = block_to_crane_src.copy()
        # Blocks to facilities (current location)
        # Note: fac_names was populated above when creating facility features
        block_to_fac_src = []
        block_to_fac_dst = []
        for i, b in enumerate(self.entities.get("blocks", [])):
            # Map facility name to index
            fac_idx = 0
            for idx, name in enumerate(fac_names):
                if name in b.location:
                    fac_idx = idx
                    break
            # Only add edge if there are facilities
            if fac_names:
                block_to_fac_src.append(i)
                block_to_fac_dst.append(fac_idx)
        # Assign edges
        import torch_geometric as tg  # type: ignore
        # Convert lists to tensors
        if block_to_spmt_src:
            data["block", "needs_transport", "spmt"].edge_index = torch.tensor(
                [block_to_spmt_src, block_to_spmt_dst], dtype=torch.long
            )
        if spmt_to_block_src:
            data["spmt", "can_transport", "block"].edge_index = torch.tensor(
                [spmt_to_block_src, spmt_to_block_dst], dtype=torch.long
            )
        if block_to_crane_src:
            data["block", "needs_lift", "crane"].edge_index = torch.tensor(
                [block_to_crane_src, block_to_crane_dst], dtype=torch.long
            )
        if crane_to_block_src:
            data["crane", "can_lift", "block"].edge_index = torch.tensor(
                [crane_to_block_src, crane_to_block_dst], dtype=torch.long
            )
        if block_to_fac_src:
            data["block", "at", "facility"].edge_index = torch.tensor(
                [block_to_fac_src, block_to_fac_dst], dtype=torch.long
            )
        # Block precedence edges (block -> precedes -> block)
        prec_src, prec_dst = [], []
        block_id_to_idx = {b.id: i for i, b in enumerate(self.entities.get("blocks", []))}
        for i, b in enumerate(self.entities.get("blocks", [])):
            for pred_id in b.predecessors:
                if pred_id in block_id_to_idx:
                    prec_src.append(block_id_to_idx[pred_id])
                    prec_dst.append(i)
        if prec_src:
            data["block", "precedes", "block"].edge_index = torch.tensor(
                [prec_src, prec_dst], dtype=torch.long
            )

        # SPMT -> at -> facility edges (current location)
        # Only create if there are facilities
        if fac_names:
            spmt_fac_src, spmt_fac_dst = [], []
            for j, s in enumerate(self.entities.get("spmts", [])):
                fac_idx = 0
                for idx, name in enumerate(fac_names):
                    if name in s.current_location:
                        fac_idx = idx
                        break
                spmt_fac_src.append(j)
                spmt_fac_dst.append(fac_idx)
            if spmt_fac_src:
                data["spmt", "at", "facility"].edge_index = torch.tensor(
                    [spmt_fac_src, spmt_fac_dst], dtype=torch.long
                )

            # Crane -> at -> facility edges (cranes are at dock, use last facility index)
            crane_fac_src, crane_fac_dst = [], []
            dock_fac_idx = len(fac_names) - 1  # dock is after last facility
            for k, c in enumerate(self.entities.get("cranes", [])):
                crane_fac_src.append(k)
                crane_fac_dst.append(dock_fac_idx)  # all cranes operate at dock area
            if crane_fac_src:
                data["crane", "at", "facility"].edge_index = torch.tensor(
                    [crane_fac_src, crane_fac_dst], dtype=torch.long
                )

        return data
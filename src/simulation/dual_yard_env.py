"""Dual-yard Gymnasium environment for Electric Boat shipyard scheduling.

This environment extends the base ShipyardEnv to model the dual-yard workflow
between EB-Quonset (module fabrication) and EB-Groton (final assembly),
connected by barge transport.

The agent manages:
- Block production across both yards
- SPMT dispatching within each yard
- Barge loading, transit, and unloading
- Crane operations at both yards
- Equipment health and maintenance
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .shipyard import DualShipyardGraph, EB_DUAL_YARD_DEFAULT_CONFIG
from .entities import (
    Block,
    SPMT,
    Crane,
    Barge,
    SuperModule,
    EBProductionStage,
    BlockStatus,
    SPMTStatus,
    CraneStatus,
    BargeStatus,
)
from .degradation import WienerDegradationModel
from .precedence import is_predecessor_complete


class DualShipyardEnv(gym.Env):
    """Gymnasium environment for dual-yard shipyard scheduling.

    Models the Electric Boat workflow:
    - Quonset Point: Steel processing -> Module fabrication -> Super-module assembly
    - Barge transport: Quonset pier -> Groton pier (~36 hours)
    - Groton: Land-level construction -> Final assembly -> Float-off
    """

    metadata = {"render_modes": ["human", "none"]}

    # Map facility names to production stages
    _QUONSET_STAGES = {
        "steel_processing": EBProductionStage.STEEL_PROCESSING,
        "afc_facility": EBProductionStage.CYLINDER_FABRICATION,
        "bldg_9a": EBProductionStage.MODULE_OUTFITTING,
        "bldg_9b": EBProductionStage.MODULE_OUTFITTING,
        "bldg_9c": EBProductionStage.MODULE_OUTFITTING,
        "super_module_assembly": EBProductionStage.SUPER_MODULE_ASSEMBLY,
        "quonset_pier": EBProductionStage.BARGE_LOADING,
    }

    _GROTON_STAGES = {
        "groton_pier": EBProductionStage.BARGE_UNLOADING,
        "land_level_construction": EBProductionStage.FINAL_ASSEMBLY,
        "building_600": EBProductionStage.SYSTEMS_INTEGRATION,
        "graving_dock": EBProductionStage.FLOAT_OFF,
    }

    def __init__(self, config: dict, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.db_logging_enabled = False

        # Dual-yard graph
        dual_yard_config = config.get("dual_yard", EB_DUAL_YARD_DEFAULT_CONFIG)
        self.shipyard = DualShipyardGraph(dual_yard_config)

        # Entity counts
        self.n_blocks = int(config.get("n_blocks", 20))
        self.n_super_modules = int(config.get("n_super_modules", 6))
        self.n_quonset_spmts = int(config.get("n_quonset_spmts", 4))
        self.n_groton_spmts = int(config.get("n_groton_spmts", 3))
        self.n_quonset_cranes = int(config.get("n_quonset_cranes", 2))
        self.n_groton_cranes = int(config.get("n_groton_cranes", 2))
        self.n_barges = int(config.get("n_barges", 1))

        # Count facilities per yard
        self.n_quonset_facilities = len(dual_yard_config.get("quonset", {}).get("facilities", []))
        self.n_groton_facilities = len(dual_yard_config.get("groton", {}).get("facilities", []))

        # Feature dimensions
        self.block_features = 10
        self.spmt_features = 10
        self.crane_features = 8
        self.barge_features = 6
        self.facility_features = 4

        # Observation space (flat vector)
        obs_dim = (
            self.n_blocks * self.block_features
            + (self.n_quonset_spmts + self.n_groton_spmts) * self.spmt_features
            + (self.n_quonset_cranes + self.n_groton_cranes) * self.crane_features
            + self.n_barges * self.barge_features
            + (self.n_quonset_facilities + self.n_groton_facilities) * self.facility_features
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space (hierarchical with barge actions)
        max_requests = self.n_blocks
        max_equipment = (self.n_quonset_spmts + self.n_groton_spmts +
                         self.n_quonset_cranes + self.n_groton_cranes)
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(6),  # 0-3 same as base, 4=barge_load, 5=barge_unload
            "spmt_idx": spaces.Discrete(self.n_quonset_spmts + self.n_groton_spmts),
            "request_idx": spaces.Discrete(max_requests),
            "crane_idx": spaces.Discrete(self.n_quonset_cranes + self.n_groton_cranes),
            "lift_idx": spaces.Discrete(max_requests),
            "equipment_idx": spaces.Discrete(max_equipment),
            "barge_idx": spaces.Discrete(max(1, self.n_barges)),
            "module_idx": spaces.Discrete(max(1, self.n_super_modules)),
        })

        # Reward weights
        self.w_tardy = float(config.get("reward_tardy", 10.0))
        self.w_empty = float(config.get("reward_empty_travel", 0.1))
        self.w_breakdown = float(config.get("reward_breakdown", 100.0))
        self.w_maintenance = float(config.get("reward_maintenance", 5.0))
        self.w_completion = float(config.get("reward_completion", 1.0))
        self.w_barge_delay = float(config.get("reward_barge_delay", 2.0))

        # Simulation state
        self.sim_time: float = 0.0
        self.entities: Dict[str, List[Any]] = {}
        self.transport_requests: Dict[str, List[Dict[str, Any]]] = {"quonset": [], "groton": []}
        self.lift_requests: Dict[str, List[Dict[str, Any]]] = {"quonset": [], "groton": []}
        self.barge_load_requests: List[Dict[str, Any]] = []
        self.facility_queues: Dict[str, Dict[str, List[str]]] = {"quonset": {}, "groton": {}}
        self.facility_processing: Dict[str, Dict[str, List[str]]] = {"quonset": {}, "groton": {}}
        self.facility_remaining_time: Dict[str, Dict[str, Dict[str, float]]] = {"quonset": {}, "groton": {}}
        self.degradation_model = WienerDegradationModel()
        self.max_time = float(config.get("max_time", 20000))

    # ------------------------------------------------------------------
    # Entity creation
    # ------------------------------------------------------------------
    def _create_blocks(self) -> None:
        """Create blocks assigned to super-modules."""
        blocks: List[Block] = []
        rng = np.random.default_rng(seed=42)
        blocks_per_module = self.n_blocks // max(1, self.n_super_modules)

        for i in range(self.n_blocks):
            weight = float(rng.uniform(50.0, 200.0))
            size = (float(rng.uniform(8.0, 15.0)), float(rng.uniform(8.0, 15.0)))
            due_date = float(500 + 20 * i)  # staggered due dates

            # Assign to a super-module
            module_idx = min(i // blocks_per_module, self.n_super_modules - 1)

            block = Block(
                id=f"B{i:03d}",
                weight=weight,
                size=size,
                due_date=due_date,
                current_stage=EBProductionStage.STEEL_PROCESSING,
            )
            block.super_module_id = f"SM{module_idx:02d}"
            block.yard = "quonset"
            blocks.append(block)

        self.entities["blocks"] = blocks

    def _create_super_modules(self) -> None:
        """Create super-modules that group blocks."""
        super_modules: List[SuperModule] = []
        for i in range(self.n_super_modules):
            sm = SuperModule(
                id=f"SM{i:02d}",
                component_modules=[],
                current_stage=EBProductionStage.STEEL_PROCESSING,
                yard="quonset",
            )
            super_modules.append(sm)
        self.entities["super_modules"] = super_modules

    def _create_spmts(self) -> None:
        """Create SPMTs for both yards."""
        spmts: List[SPMT] = []

        # Quonset SPMTs
        for i in range(self.n_quonset_spmts):
            spmt = SPMT(
                id=f"QV{i:02d}",
                capacity=500.0,
                current_location="quonset_wip1",
            )
            spmt.yard = "quonset"
            spmts.append(spmt)

        # Groton SPMTs
        for i in range(self.n_groton_spmts):
            spmt = SPMT(
                id=f"GV{i:02d}",
                capacity=500.0,
                current_location="groton_staging",
            )
            spmt.yard = "groton"
            spmts.append(spmt)

        self.entities["spmts"] = spmts

    def _create_cranes(self) -> None:
        """Create cranes for both yards."""
        cranes: List[Crane] = []

        # Quonset cranes (for loading barges)
        for i in range(self.n_quonset_cranes):
            crane = Crane(id=f"QC{i:02d}")
            crane.yard = "quonset"
            cranes.append(crane)

        # Groton cranes (for dock assembly)
        for i in range(self.n_groton_cranes):
            crane = Crane(id=f"GC{i:02d}")
            crane.yard = "groton"
            cranes.append(crane)

        self.entities["cranes"] = cranes

    def _create_barges(self) -> None:
        """Create barges for inter-yard transport."""
        barges: List[Barge] = []
        for i in range(self.n_barges):
            barge = Barge(
                id=f"HOLLAND{i+1}",
                capacity=2,
                current_location="quonset_pier",
            )
            barges.append(barge)
        self.entities["barges"] = barges

    def _initialize_facilities(self) -> None:
        """Initialize facility queues for both yards."""
        dual_yard_config = self.config.get("dual_yard", EB_DUAL_YARD_DEFAULT_CONFIG)

        for yard in ["quonset", "groton"]:
            facilities = dual_yard_config.get(yard, {}).get("facilities", [])
            for f in facilities:
                name = f["name"]
                self.facility_queues[yard][name] = []
                self.facility_processing[yard][name] = []
                self.facility_remaining_time[yard][name] = {}

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.sim_time = 0.0
        self.entities.clear()
        self.transport_requests = {"quonset": [], "groton": []}
        self.lift_requests = {"quonset": [], "groton": []}
        self.barge_load_requests.clear()
        self.facility_queues = {"quonset": {}, "groton": {}}
        self.facility_processing = {"quonset": {}, "groton": {}}
        self.facility_remaining_time = {"quonset": {}, "groton": {}}

        # Create entities
        self._create_blocks()
        self._create_super_modules()
        self._create_spmts()
        self._create_cranes()
        self._create_barges()
        self._initialize_facilities()

        # Place blocks in initial Quonset queue
        first_fac = "steel_processing"
        for block in self.entities["blocks"]:
            self.facility_queues["quonset"][first_fac].append(block.id)
            block.location = f"queue_{first_fac}"
            block.yard = "quonset"

        # Reset equipment status
        for spmt in self.entities["spmts"]:
            spmt.status = SPMTStatus.IDLE
            spmt.current_load = None
            spmt.health_hydraulic = 100.0
            spmt.health_tires = 100.0
            spmt.health_engine = 100.0

        for crane in self.entities["cranes"]:
            crane.status = CraneStatus.IDLE
            crane.current_block = None
            crane.health_cable = 100.0
            crane.health_motor = 100.0

        for barge in self.entities["barges"]:
            barge.status = BargeStatus.IDLE
            barge.cargo.clear()
            barge.transit_progress = 0.0
            barge.current_location = "quonset_pier"

        # Reset metrics
        self.metrics = {
            "blocks_completed": 0,
            "breakdowns": 0,
            "planned_maintenance": 0,
            "total_tardiness": 0.0,
            "empty_travel_distance": 0.0,
            "barge_trips": 0,
            "quonset_completions": 0,
            "groton_completions": 0,
        }

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Simulation update methods
    # ------------------------------------------------------------------
    def _get_yard_facilities(self, yard: str) -> List[Dict]:
        """Get facility configs for a yard."""
        dual_yard_config = self.config.get("dual_yard", EB_DUAL_YARD_DEFAULT_CONFIG)
        return dual_yard_config.get(yard, {}).get("facilities", [])

    def _assign_blocks_to_facilities(self) -> None:
        """Assign blocks from queues to processing in both yards."""
        for yard in ["quonset", "groton"]:
            facilities_cfg = self._get_yard_facilities(yard)
            fac_info = {f["name"]: f for f in facilities_cfg}

            for fac_name, queue in self.facility_queues[yard].items():
                if fac_name not in fac_info:
                    continue
                capacity = fac_info[fac_name].get("capacity", 1)

                while queue and len(self.facility_processing[yard][fac_name]) < capacity:
                    block_id = queue.pop(0)
                    self.facility_processing[yard][fac_name].append(block_id)

                    # Processing time
                    mean = fac_info[fac_name]["processing_time_mean"]
                    std = fac_info[fac_name]["processing_time_std"]
                    proc_time = float(np.random.lognormal(mean=np.log(mean), sigma=std / mean))
                    self.facility_remaining_time[yard][fac_name][block_id] = proc_time

                    # Update block
                    block = self._get_block(block_id)
                    block.status = BlockStatus.IN_PROCESS
                    block.location = fac_name

                    # Set stage
                    if yard == "quonset" and fac_name in self._QUONSET_STAGES:
                        block.current_stage = self._QUONSET_STAGES[fac_name]
                    elif yard == "groton" and fac_name in self._GROTON_STAGES:
                        block.current_stage = self._GROTON_STAGES[fac_name]

    def _update_processing(self, dt: float) -> None:
        """Advance processing in both yards."""
        completed_blocks: List[Tuple[str, str, str]] = []  # (yard, fac_name, block_id)

        for yard in ["quonset", "groton"]:
            for fac_name, remaining_times in list(self.facility_remaining_time[yard].items()):
                finished_ids: List[str] = []
                for block_id, remaining in list(remaining_times.items()):
                    new_remaining = remaining - dt
                    self.facility_remaining_time[yard][fac_name][block_id] = new_remaining
                    if new_remaining <= 0.0:
                        finished_ids.append(block_id)

                for block_id in finished_ids:
                    completed_blocks.append((yard, fac_name, block_id))
                    self.facility_processing[yard][fac_name].remove(block_id)
                    del self.facility_remaining_time[yard][fac_name][block_id]

        # Handle completions
        for yard, fac_name, block_id in completed_blocks:
            self._handle_block_completion(yard, fac_name, block_id)

    def _handle_block_completion(self, yard: str, fac_name: str, block_id: str) -> None:
        """Handle a block completing processing at a facility."""
        block = self._get_block(block_id)
        block.status = BlockStatus.WAITING
        self._log_block_event(block_id, "stage_complete", block.current_stage.name, fac_name)

        if yard == "quonset":
            self._handle_quonset_completion(fac_name, block)
        else:
            self._handle_groton_completion(fac_name, block)

    def _handle_quonset_completion(self, fac_name: str, block: Block) -> None:
        """Handle block completion at Quonset facilities."""
        facilities = self._get_yard_facilities("quonset")
        fac_order = [f["name"] for f in facilities]

        if fac_name == "quonset_pier":
            # Ready for barge loading
            self.barge_load_requests.append({"block_id": block.id})
            block.status = BlockStatus.WAITING
            block.location = "waiting_barge_load"
            self.metrics["quonset_completions"] += 1
        else:
            # Move to next Quonset facility
            try:
                current_idx = fac_order.index(fac_name)
                if current_idx + 1 < len(fac_order):
                    next_fac = fac_order[current_idx + 1]
                    self.transport_requests["quonset"].append({
                        "block_id": block.id,
                        "destination": next_fac,
                    })
                    block.location = f"waiting_transport_to_{next_fac}"
            except ValueError:
                pass

    def _handle_groton_completion(self, fac_name: str, block: Block) -> None:
        """Handle block completion at Groton facilities."""
        facilities = self._get_yard_facilities("groton")
        fac_order = [f["name"] for f in facilities]

        if fac_name == "graving_dock":
            # Final stage - block is complete
            block.status = BlockStatus.PLACED_ON_DOCK
            block.location = "dock"
            block.current_stage = EBProductionStage.FLOAT_OFF
            self.metrics["blocks_completed"] += 1
            self.metrics["groton_completions"] += 1
            self._log_block_event(block.id, "completed", "FLOAT_OFF", "dock")
        else:
            # Move to next Groton facility
            try:
                current_idx = fac_order.index(fac_name)
                if current_idx + 1 < len(fac_order):
                    next_fac = fac_order[current_idx + 1]
                    self.transport_requests["groton"].append({
                        "block_id": block.id,
                        "destination": next_fac,
                    })
                    block.location = f"waiting_transport_to_{next_fac}"
            except ValueError:
                pass

    def _update_barges(self, dt: float) -> None:
        """Update barge transit progress."""
        for barge in self.entities.get("barges", []):
            if barge.status == BargeStatus.IN_TRANSIT_TO_GROTON:
                barge.transit_progress += dt
                if barge.transit_progress >= self.shipyard.barge_route.transit_time:
                    barge.current_location = "groton_pier"
                    barge.status = BargeStatus.UNLOADING
                    barge.transit_progress = 0.0
                    self._log_barge_event(barge.id, "arrived_groton")

            elif barge.status == BargeStatus.IN_TRANSIT_TO_QUONSET:
                barge.transit_progress += dt
                if barge.transit_progress >= self.shipyard.barge_route.return_time:
                    barge.current_location = "quonset_pier"
                    barge.status = BargeStatus.IDLE
                    barge.transit_progress = 0.0
                    self._log_barge_event(barge.id, "returned_quonset")

    def _degrade_equipment(self, dt: float) -> None:
        """Apply degradation to all equipment."""
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

        for crane in self.entities.get("cranes", []):
            operating = crane.status in {CraneStatus.LIFTING, CraneStatus.POSITIONING}
            crane.health_cable, fail_cable = self.degradation_model.step(
                crane.health_cable, dt, load_ratio=0.2, operating=operating
            )
            crane.health_motor, fail_motor = self.degradation_model.step(
                crane.health_motor, dt, load_ratio=0.2, operating=operating
            )

            if fail_cable or fail_motor:
                crane.status = CraneStatus.BROKEN_DOWN
                self.metrics["breakdowns"] += 1
                self._log_equipment_event(crane.id, "crane", "breakdown")

    def _advance_simulation(self, dt: float = 1.0) -> None:
        """Advance simulation by dt hours."""
        self._assign_blocks_to_facilities()
        self._update_processing(dt)
        self._update_barges(dt)
        self._degrade_equipment(dt)
        self.sim_time += dt

        # Tardiness accumulation
        for block in self.entities.get("blocks", []):
            if block.status != BlockStatus.PLACED_ON_DOCK:
                tardiness = max(0.0, self.sim_time - block.due_date)
                self.metrics["total_tardiness"] += tardiness * dt

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _dispatch_spmt(self, spmt_idx: int, request_idx: int) -> float:
        """Dispatch an SPMT within its yard."""
        spmts = self.entities.get("spmts", [])
        if spmt_idx >= len(spmts):
            return 0.0

        spmt = spmts[spmt_idx]
        yard = getattr(spmt, "yard", "quonset")

        if request_idx >= len(self.transport_requests.get(yard, [])):
            return 0.0

        request = self.transport_requests[yard].pop(request_idx)
        block_id = request["block_id"]
        destination = request["destination"]
        block = self._get_block(block_id)

        if spmt.status != SPMTStatus.IDLE or spmt.current_load is not None:
            return 0.0

        # Get yard graph
        yard_graph = self.shipyard.get_yard(yard)
        if not yard_graph:
            return 0.0

        # Calculate travel
        travel_to_block = yard_graph.get_travel_time(spmt.current_location, block.location)
        travel_to_dest = yard_graph.get_travel_time(block.location, destination)
        empty_distance = travel_to_block

        self.metrics["empty_travel_distance"] += empty_distance

        # Execute transport
        spmt.status = SPMTStatus.TRAVELING_LOADED
        spmt.current_load = block_id
        spmt.current_location = destination

        # Complete transport
        block.location = f"queue_{destination}"
        self.facility_queues[yard][destination].append(block_id)
        spmt.current_load = None
        spmt.status = SPMTStatus.IDLE

        self._log_block_event(block_id, "transport_arrival", block.current_stage.name, destination)
        return -self.w_empty * empty_distance

    def _dispatch_crane(self, crane_idx: int, request_idx: int) -> float:
        """Dispatch a crane for lifting (Groton dock or Quonset barge loading)."""
        cranes = self.entities.get("cranes", [])
        if crane_idx >= len(cranes):
            return 0.0

        crane = cranes[crane_idx]
        yard = getattr(crane, "yard", "groton")

        if request_idx >= len(self.lift_requests.get(yard, [])):
            return 0.0

        if crane.status != CraneStatus.IDLE:
            return 0.0

        request = self.lift_requests[yard].pop(request_idx)
        block_id = request["block_id"]
        block = self._get_block(block_id)

        # Execute lift
        crane.status = CraneStatus.LIFTING
        crane.current_block = block_id

        if yard == "groton":
            # Final dock placement
            block.status = BlockStatus.PLACED_ON_DOCK
            block.location = "dock"
            self.metrics["blocks_completed"] += 1
            self._log_block_event(block_id, "lift_to_dock", "DOCK", "dock")
        else:
            # Quonset - loading onto barge
            self._log_block_event(block_id, "crane_lift", block.current_stage.name, crane.yard)

        crane.status = CraneStatus.IDLE
        crane.current_block = None

        return self.w_completion

    def _load_barge(self, barge_idx: int, module_idx: int) -> float:
        """Load a super-module onto a barge."""
        barges = self.entities.get("barges", [])
        if barge_idx >= len(barges):
            return 0.0

        barge = barges[barge_idx]

        if barge.status not in {BargeStatus.IDLE, BargeStatus.LOADING}:
            return 0.0

        if len(barge.cargo) >= barge.capacity:
            return 0.0

        if not self.barge_load_requests:
            return 0.0

        # Load the first available module
        request = self.barge_load_requests.pop(0)
        block_id = request["block_id"]
        block = self._get_block(block_id)

        barge.cargo.append(block_id)
        barge.status = BargeStatus.LOADING
        block.location = f"barge_{barge.id}"
        block.status = BlockStatus.IN_TRANSIT
        block.current_stage = EBProductionStage.BARGE_TRANSIT

        self._log_block_event(block_id, "barge_loaded", "BARGE_TRANSIT", barge.id)
        return 0.1  # Small reward for loading

    def _start_barge_transit(self, barge_idx: int) -> float:
        """Start barge transit to Groton."""
        barges = self.entities.get("barges", [])
        if barge_idx >= len(barges):
            return 0.0

        barge = barges[barge_idx]

        if barge.status not in {BargeStatus.IDLE, BargeStatus.LOADING}:
            return 0.0

        if not barge.cargo:
            return -0.5  # Penalty for empty transit

        barge.status = BargeStatus.IN_TRANSIT_TO_GROTON
        barge.transit_progress = 0.0
        self.metrics["barge_trips"] += 1
        self._log_barge_event(barge.id, "departed_quonset")

        return 0.2  # Reward for starting trip with cargo

    def _unload_barge(self, barge_idx: int) -> float:
        """Unload cargo from barge at Groton."""
        barges = self.entities.get("barges", [])
        if barge_idx >= len(barges):
            return 0.0

        barge = barges[barge_idx]

        if barge.status != BargeStatus.UNLOADING:
            return 0.0

        if not barge.cargo:
            # Empty, return to Quonset
            barge.status = BargeStatus.IN_TRANSIT_TO_QUONSET
            barge.transit_progress = 0.0
            self._log_barge_event(barge.id, "returning_empty")
            return 0.0

        # Unload one block
        block_id = barge.cargo.pop(0)
        block = self._get_block(block_id)
        block.yard = "groton"
        block.location = "queue_land_level_construction"
        block.status = BlockStatus.WAITING
        block.current_stage = EBProductionStage.FINAL_ASSEMBLY
        self.facility_queues["groton"]["land_level_construction"].append(block_id)

        self._log_block_event(block_id, "barge_unloaded", "FINAL_ASSEMBLY", "groton_pier")

        # If empty, start return trip
        if not barge.cargo:
            barge.status = BargeStatus.IN_TRANSIT_TO_QUONSET
            barge.transit_progress = 0.0
            self._log_barge_event(barge.id, "returning_quonset")

        return 0.3  # Reward for unloading

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
            cranes = self.entities.get("cranes", [])
            if crane_idx < len(cranes):
                crane = cranes[crane_idx]
                if crane.status == CraneStatus.IDLE:
                    crane.status = CraneStatus.IN_MAINTENANCE
                    crane.health_cable = self.degradation_model.perform_maintenance()
                    crane.health_motor = self.degradation_model.perform_maintenance()
                    crane.status = CraneStatus.IDLE
                    self.metrics["planned_maintenance"] += 1
                    self._log_equipment_event(crane.id, "crane", "maintenance_complete")
                    return -self.w_maintenance
        return 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep."""
        reward = 0.0
        breakdowns_before = self.metrics["breakdowns"]
        tardiness_before = self.metrics.get("total_tardiness", 0.0)

        action_type = int(action.get("action_type", 5))  # default hold

        if action_type == 0:  # SPMT dispatch
            spmt_idx = int(action.get("spmt_idx", 0))
            req_idx = int(action.get("request_idx", 0))
            reward += self._dispatch_spmt(spmt_idx, req_idx)

        elif action_type == 1:  # Crane dispatch
            crane_idx = int(action.get("crane_idx", 0))
            req_idx = int(action.get("lift_idx", 0))
            reward += self._dispatch_crane(crane_idx, req_idx)

        elif action_type == 2:  # Maintenance
            equip_idx = int(action.get("equipment_idx", 0))
            reward += self._trigger_maintenance(equip_idx)

        elif action_type == 3:  # Hold
            pass

        elif action_type == 4:  # Barge load or start transit
            barge_idx = int(action.get("barge_idx", 0))
            barges = self.entities.get("barges", [])
            if barge_idx < len(barges):
                barge = barges[barge_idx]
                if self.barge_load_requests:
                    reward += self._load_barge(barge_idx, 0)
                elif barge.cargo:
                    reward += self._start_barge_transit(barge_idx)

        elif action_type == 5:  # Barge unload
            barge_idx = int(action.get("barge_idx", 0))
            reward += self._unload_barge(barge_idx)

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

        terminated = self.metrics["blocks_completed"] == self.n_blocks
        truncated = self.sim_time >= self.max_time

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _encode_block(self, block: Block) -> List[float]:
        stage = block.current_stage.value if hasattr(block.current_stage, 'value') else 0
        yard_enc = 0.0 if getattr(block, 'yard', 'quonset') == 'quonset' else 1.0
        loc_enc = 0.5 if 'queue' in block.location else (1.0 if 'dock' in block.location else 0.0)
        completion = block.completion_pct
        time_to_due = max(0.0, block.due_date - self.sim_time)
        weight = block.weight / 500.0
        in_transit = 1.0 if block.status == BlockStatus.IN_TRANSIT else 0.0
        waiting = 1.0 if block.status == BlockStatus.WAITING else 0.0
        on_barge = 1.0 if 'barge' in block.location.lower() else 0.0
        completed = 1.0 if block.status == BlockStatus.PLACED_ON_DOCK else 0.0
        return [stage, yard_enc, loc_enc, completion, time_to_due, weight, in_transit, waiting, on_barge, completed]

    def _encode_spmt(self, spmt: SPMT) -> List[float]:
        yard_enc = 0.0 if getattr(spmt, 'yard', 'quonset') == 'quonset' else 1.0
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
        return [yard_enc] + status_enc + [load_ratio] + health

    def _encode_crane(self, crane: Crane) -> List[float]:
        yard_enc = 0.0 if getattr(crane, 'yard', 'quonset') == 'quonset' else 1.0
        pos_norm = crane.position_on_rail / 100.0
        status_enc = [0.0] * 3
        status_map = {CraneStatus.IDLE: 0, CraneStatus.LIFTING: 1, CraneStatus.POSITIONING: 2}
        idx = status_map.get(crane.status, 0)
        status_enc[idx] = 1.0
        health = crane.get_health_vector().tolist()
        current_load = 1.0 if crane.current_block else 0.0
        return [yard_enc, pos_norm] + status_enc + health + [current_load]

    def _encode_barge(self, barge: Barge) -> List[float]:
        location_enc = 0.0 if 'quonset' in barge.current_location else 1.0
        transit_progress = barge.transit_progress / 40.0  # Normalize
        cargo_level = len(barge.cargo) / max(1, barge.capacity)
        status_map = {
            BargeStatus.IDLE: 0.0,
            BargeStatus.LOADING: 0.25,
            BargeStatus.IN_TRANSIT_TO_GROTON: 0.5,
            BargeStatus.IN_TRANSIT_TO_QUONSET: 0.75,
            BargeStatus.UNLOADING: 1.0,
        }
        status_enc = status_map.get(barge.status, 0.0)
        capacity = barge.capacity / 4.0
        can_load = 1.0 if (barge.status in {BargeStatus.IDLE, BargeStatus.LOADING}
                          and len(barge.cargo) < barge.capacity) else 0.0
        return [location_enc, transit_progress, cargo_level, status_enc, capacity, can_load]

    def _encode_facility(self, yard: str, fac_name: str) -> List[float]:
        yard_enc = 0.0 if yard == 'quonset' else 1.0
        queue_len = len(self.facility_queues.get(yard, {}).get(fac_name, []))
        util = float(len(self.facility_processing.get(yard, {}).get(fac_name, [])))
        avg_wait = float(queue_len)
        return [yard_enc, queue_len, util, avg_wait]

    def _get_observation(self) -> np.ndarray:
        obs: List[float] = []

        for block in self.entities.get("blocks", []):
            obs.extend(self._encode_block(block))

        for spmt in self.entities.get("spmts", []):
            obs.extend(self._encode_spmt(spmt))

        for crane in self.entities.get("cranes", []):
            obs.extend(self._encode_crane(crane))

        for barge in self.entities.get("barges", []):
            obs.extend(self._encode_barge(barge))

        for yard in ["quonset", "groton"]:
            for f in self._get_yard_facilities(yard):
                obs.extend(self._encode_facility(yard, f["name"]))

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        return {
            "sim_time": self.sim_time,
            "metrics": self.metrics.copy(),
            "quonset_transport_requests": len(self.transport_requests.get("quonset", [])),
            "groton_transport_requests": len(self.transport_requests.get("groton", [])),
            "barge_load_requests": len(self.barge_load_requests),
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _get_block(self, block_id: str) -> Block:
        for b in self.entities.get("blocks", []):
            if b.id == block_id:
                return b
        raise KeyError(f"Block {block_id} not found")

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

    def _log_barge_event(self, barge_id: str, event_type: str) -> None:
        if not self.db_logging_enabled:
            return
        try:
            from mes.database import log_block_event
            log_block_event(barge_id, self.sim_time, event_type, "barge", barge_id)
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

            # Merge queues from both yards
            all_queues = {}
            all_processing = {}
            for yard in ["quonset", "groton"]:
                for fac, q in self.facility_queues.get(yard, {}).items():
                    all_queues[f"{yard}_{fac}"] = q
                for fac, p in self.facility_processing.get(yard, {}).items():
                    all_processing[f"{yard}_{fac}"] = p
            log_queue_depths(self.sim_time, all_queues, all_processing)

            # Log position snapshot for playback
            barge = self.entities.get("barges", [None])[0] if self.entities.get("barges") else None
            log_position_snapshot(
                self.sim_time,
                self.entities.get("blocks", []),
                self.entities.get("spmts", []),
                self.entities.get("cranes", []),
                barge,
            )
        except Exception:
            pass

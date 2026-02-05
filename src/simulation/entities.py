"""Entity classes for the shipyard simulation.

This module defines data structures for blocks, SPMTs (Self‑Propelled
Modular Transporters), cranes, and barges. Entities carry state information
such as their current location, status, health levels and processing history.
Enumerations encode possible statuses and production stages.

Supports both single-yard (legacy) and dual-yard (EB Quonset/Groton) workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class BlockStatus(Enum):
    WAITING = "waiting"
    IN_PROCESS = "in_process"
    IN_TRANSIT = "in_transit"
    AT_STAGING = "at_staging"
    AT_PRE_ERECTION = "at_pre_erection"
    PLACED_ON_DOCK = "placed_on_dock"
    # New statuses for dual-yard workflow
    ON_BARGE = "on_barge"
    AWAITING_BARGE = "awaiting_barge"


class ProductionStage(Enum):
    """Legacy single-yard production stages."""
    CUTTING = 0
    PANEL = 1
    ASSEMBLY = 2
    OUTFITTING = 3
    PAINTING = 4
    PRE_ERECTION = 5
    DOCK = 6


class EBProductionStage(Enum):
    """Electric Boat dual-yard production stages.

    Quonset Point (RI) stages: 0-4
    Transit stage: 5
    Groton (CT) stages: 6-9
    """
    # Quonset Point stages
    STEEL_PROCESSING = 0       # Raw steel cutting, machining, bending
    CYLINDER_FABRICATION = 1   # Vertical hull cylinder construction
    MODULE_OUTFITTING = 2      # Tanks, pipes, wiring, paint
    SUPER_MODULE_ASSEMBLY = 3  # Join modules, rotate horizontal
    BARGE_LOADING = 4          # Load onto Holland barge
    # Transit
    BARGE_TRANSIT = 5          # Ocean transport (~36 hours)
    # Groton stages
    BARGE_UNLOADING = 6        # Unload at Thames River pier
    FINAL_ASSEMBLY = 7         # Join super modules into hull
    SYSTEMS_INTEGRATION = 8    # Final systems in Building 600
    FLOAT_OFF = 9              # Launch via graving dock

    @property
    def yard(self) -> str:
        """Return which yard this stage belongs to."""
        if self.value <= 4:
            return "quonset"
        elif self.value == 5:
            return "transit"
        else:
            return "groton"

    @property
    def is_quonset(self) -> bool:
        return self.value <= 4

    @property
    def is_groton(self) -> bool:
        return self.value >= 6

    @property
    def is_transit(self) -> bool:
        return self.value == 5


class BargeStatus(Enum):
    """Status of the Holland barge for inter-yard transport."""
    IDLE = "idle"                    # At pier, empty
    LOADING = "loading"              # Loading super modules at Quonset
    IN_TRANSIT_TO_GROTON = "in_transit_to_groton"  # Sailing to Groton
    IN_TRANSIT_TO_QUONSET = "in_transit_to_quonset"  # Returning to Quonset
    UNLOADING = "unloading"          # Unloading at Groton
    IN_MAINTENANCE = "in_maintenance"


@dataclass
class Block:
    """Representation of a ship block.

    Parameters
    ----------
    id : str
        Unique identifier for the block.
    weight : float
        Weight in tons; affects SPMT degradation.
    size : Tuple[float, float]
        Length and width in meters.
    due_date : float
        Target completion time (simulation units).
    current_stage : ProductionStage, optional
        The stage the block is currently in. Defaults to `CUTTING`.
    status : BlockStatus, optional
        High‑level status flag.
    location : str, optional
        Name of the node where the block currently resides.
    predecessors : list of str, optional
        Blocks that must be placed on the dock before this block.
    dock_position : Optional[Tuple[int, int]], optional
        Target row/column on the dock. Assigned when scheduling.
    """

    id: str
    weight: float
    size: Tuple[float, float]
    due_date: float
    current_stage: ProductionStage = ProductionStage.CUTTING
    status: BlockStatus = BlockStatus.WAITING
    location: str = "cutting_queue"
    completion_pct: float = 0.0
    predecessors: List[str] = field(default_factory=list)
    dock_position: Optional[Tuple[int, int]] = None
    stage_entry_time: float = 0.0
    history: List[dict] = field(default_factory=list)

    def log_event(self, timestamp: float, description: str) -> None:
        """Append an event to the block's history."""
        self.history.append({"time": timestamp, "desc": description})


class SPMTStatus(Enum):
    IDLE = "idle"
    TRAVELING_EMPTY = "traveling_empty"
    TRAVELING_LOADED = "traveling_loaded"
    LOADING = "loading"
    UNLOADING = "unloading"
    IN_MAINTENANCE = "in_maintenance"
    BROKEN_DOWN = "broken_down"


@dataclass
class SPMT:
    """Self‑propelled modular transporter.

    SPMTs carry blocks between facilities. They degrade over time and may
    require maintenance. Health indicators are expressed on a 0–100 scale.
    """

    id: str
    capacity: float = 500.0
    current_location: str = "yard_depot"
    status: SPMTStatus = SPMTStatus.IDLE
    current_load: Optional[str] = None
    health_hydraulic: float = 100.0
    health_tires: float = 100.0
    health_engine: float = 100.0
    base_degradation_rate: float = 0.01  # per hour
    load_degradation_factor: float = 0.5
    cumulative_operating_hours: float = 0.0
    cumulative_load_ton_hours: float = 0.0
    last_maintenance_time: float = 0.0

    def get_health_vector(self) -> np.ndarray:
        return np.array([
            self.health_hydraulic / 100.0,
            self.health_tires / 100.0,
            self.health_engine / 100.0,
        ], dtype=float)

    def get_min_health(self) -> float:
        return float(min(self.health_hydraulic, self.health_tires, self.health_engine))

    def estimate_rul(self, degradation_model: "WienerDegradationModel") -> float:
        """Estimate remaining useful life using a degradation model.

        This method passes the current health of the worst component to
        the provided `degradation_model` and returns the estimated time
        until failure under a nominal load ratio of 0.3.
        """
        min_health = self.get_min_health()
        return float(degradation_model.estimate_rul(min_health, load_ratio=0.3))


class CraneStatus(Enum):
    IDLE = "idle"
    LIFTING = "lifting"
    POSITIONING = "positioning"
    IN_MAINTENANCE = "in_maintenance"
    BROKEN_DOWN = "broken_down"


@dataclass
class Crane:
    """Gantry crane representation."""

    id: str
    position_on_rail: float = 0.0  # metres along the rail
    status: CraneStatus = CraneStatus.IDLE
    current_block: Optional[str] = None
    health_cable: float = 100.0
    health_motor: float = 100.0
    lift_capacity: float = 800.0
    travel_speed: float = 0.5  # m/s along the rail

    def get_health_vector(self) -> np.ndarray:
        return np.array([
            self.health_cable / 100.0,
            self.health_motor / 100.0,
        ], dtype=float)

    def get_min_health(self) -> float:
        return float(min(self.health_cable, self.health_motor))


@dataclass
class Barge:
    """Holland barge for inter-yard transport between Quonset and Groton.

    The barge transports super modules from Quonset Point (RI) to Groton (CT)
    via Narragansett Bay and the Thames River. Transit typically takes 24-48 hours.

    Parameters
    ----------
    id : str
        Unique identifier for the barge.
    capacity : int
        Maximum number of super modules the barge can carry.
    current_location : str
        Current location: "quonset_pier", "groton_pier", or "in_transit".
    status : BargeStatus
        Current operational status.
    cargo : list of str
        IDs of super modules currently loaded on the barge.
    transit_progress : float
        Progress through current transit (0.0 to 1.0).
    transit_start_time : float
        Simulation time when current transit began.
    """

    id: str
    capacity: int = 2  # Number of super modules per trip
    current_location: str = "quonset_pier"
    status: BargeStatus = BargeStatus.IDLE
    cargo: List[str] = field(default_factory=list)
    transit_progress: float = 0.0
    transit_start_time: float = 0.0
    total_trips: int = 0
    cumulative_cargo_count: int = 0
    health: float = 100.0  # Overall barge health
    history: List[dict] = field(default_factory=list)

    def load_module(self, module_id: str, timestamp: float) -> bool:
        """Attempt to load a super module onto the barge.

        Returns True if successful, False if at capacity.
        """
        if len(self.cargo) >= self.capacity:
            return False
        self.cargo.append(module_id)
        self.history.append({
            "time": timestamp,
            "action": "load",
            "module": module_id,
            "location": self.current_location
        })
        return True

    def unload_module(self, module_id: str, timestamp: float) -> bool:
        """Unload a specific super module from the barge.

        Returns True if successful, False if module not on barge.
        """
        if module_id not in self.cargo:
            return False
        self.cargo.remove(module_id)
        self.history.append({
            "time": timestamp,
            "action": "unload",
            "module": module_id,
            "location": self.current_location
        })
        return True

    def start_transit(self, destination: str, timestamp: float) -> None:
        """Begin transit to the specified destination."""
        if destination == "groton_pier":
            self.status = BargeStatus.IN_TRANSIT_TO_GROTON
        else:
            self.status = BargeStatus.IN_TRANSIT_TO_QUONSET
        self.transit_progress = 0.0
        self.transit_start_time = timestamp
        self.history.append({
            "time": timestamp,
            "action": "depart",
            "from": self.current_location,
            "to": destination
        })

    def complete_transit(self, destination: str, timestamp: float) -> None:
        """Complete transit and arrive at destination."""
        self.current_location = destination
        self.status = BargeStatus.IDLE
        self.transit_progress = 1.0
        self.total_trips += 1
        self.history.append({
            "time": timestamp,
            "action": "arrive",
            "location": destination
        })

    def is_empty(self) -> bool:
        return len(self.cargo) == 0

    def is_full(self) -> bool:
        return len(self.cargo) >= self.capacity

    def cargo_count(self) -> int:
        return len(self.cargo)


@dataclass
class SuperModule:
    """A super module composed of multiple hull cylinders/modules.

    Super modules are assembled at Quonset Point from individual modules
    and transported to Groton for final assembly into a complete submarine.

    Parameters
    ----------
    id : str
        Unique identifier for the super module.
    component_modules : list of str
        IDs of the constituent modules that form this super module.
    weight : float
        Total weight in tons.
    size : Tuple[float, float, float]
        Length, width, height in meters.
    current_stage : EBProductionStage
        Current stage in the dual-yard workflow.
    status : BlockStatus
        High-level status flag.
    location : str
        Current node/location identifier.
    yard : str
        Current yard: "quonset", "groton", or "transit".
    submarine_id : str, optional
        ID of the submarine this module is part of.
    """

    id: str
    component_modules: List[str] = field(default_factory=list)
    weight: float = 0.0
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # length, width, height
    current_stage: EBProductionStage = EBProductionStage.STEEL_PROCESSING
    status: BlockStatus = BlockStatus.WAITING
    location: str = "steel_processing_queue"
    yard: str = "quonset"
    submarine_id: Optional[str] = None
    completion_pct: float = 0.0
    due_date: float = 0.0
    predecessors: List[str] = field(default_factory=list)
    stage_entry_time: float = 0.0
    history: List[dict] = field(default_factory=list)

    def log_event(self, timestamp: float, description: str) -> None:
        """Append an event to the super module's history."""
        self.history.append({"time": timestamp, "desc": description})

    def advance_stage(self, timestamp: float) -> None:
        """Move to the next production stage."""
        current_value = self.current_stage.value
        if current_value < EBProductionStage.FLOAT_OFF.value:
            self.current_stage = EBProductionStage(current_value + 1)
            self.yard = self.current_stage.yard
            self.stage_entry_time = timestamp
            self.log_event(timestamp, f"Advanced to {self.current_stage.name}")

    def is_at_quonset(self) -> bool:
        return self.yard == "quonset"

    def is_at_groton(self) -> bool:
        return self.yard == "groton"

    def is_in_transit(self) -> bool:
        return self.yard == "transit"

    def ready_for_barge(self) -> bool:
        """Check if this module is ready for barge transport."""
        return (self.current_stage == EBProductionStage.BARGE_LOADING and
                self.status != BlockStatus.ON_BARGE)


@dataclass
class Submarine:
    """A submarine under construction, composed of multiple super modules.

    Parameters
    ----------
    id : str
        Unique identifier (e.g., "SSN-801", "SSBN-826").
    hull_number : str
        Official hull number.
    sub_class : str
        Submarine class: "virginia", "columbia", or "generic".
    super_modules : list of str
        IDs of super modules that comprise this submarine.
    """

    id: str
    hull_number: str = ""
    sub_class: str = "generic"  # "virginia", "columbia", "generic"
    super_modules: List[str] = field(default_factory=list)
    target_completion_date: float = 0.0
    current_yard: str = "quonset"
    status: str = "in_production"  # "in_production", "in_transit", "final_assembly", "launched"
    completion_pct: float = 0.0
    history: List[dict] = field(default_factory=list)

    def log_event(self, timestamp: float, description: str) -> None:
        """Append an event to the submarine's history."""
        self.history.append({"time": timestamp, "desc": description})
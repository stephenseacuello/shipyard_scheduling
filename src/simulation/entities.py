"""Entity classes for the shipyard simulation.

This module defines data structures for blocks, SPMTs (Self-Propelled
Modular Transporters), Goliath cranes, and ships. Entities carry state
information such as their current location, status, health levels and
processing history. Enumerations encode possible statuses and production stages.

Models HD Hyundai Heavy Industries (HHI) Ulsan shipyard LNG carrier production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class BlockStatus(Enum):
    """Status of a ship block in the production system."""
    WAITING = "waiting"
    IN_PROCESS = "in_process"
    IN_TRANSIT = "in_transit"
    AT_STAGING = "at_staging"
    AT_PRE_ERECTION = "at_pre_erection"
    IN_ERECTION = "in_erection"
    ERECTED = "erected"
    COMPLETE = "complete"


class HHIProductionStage(Enum):
    """HD Hyundai Heavy Industries commercial shipbuilding stages.

    Follows Korean shipbuilding methodology for LNG carrier production.
    11 stages from steel cutting to delivery.
    """
    STEEL_CUTTING = 0          # NC/plasma cutting of steel plates
    PART_FABRICATION = 1       # Marking, bending, edge preparation
    PANEL_ASSEMBLY = 2         # Flat/curved panel sub-assembly
    BLOCK_ASSEMBLY = 3         # 3D block construction (~300 tons)
    BLOCK_OUTFITTING = 4       # Piping, electrical, HVAC, insulation
    PAINTING = 5               # Block painting (pre-erection)
    PRE_ERECTION = 6           # Grand block staging/joining
    ERECTION = 7               # Goliath crane places block in dock
    QUAY_OUTFITTING = 8        # Post-launch systems installation
    SEA_TRIALS = 9             # Testing at sea
    DELIVERY = 10              # Handover to owner

    @property
    def zone(self) -> str:
        """Return which zone this stage belongs to."""
        if self.value <= 1:
            return "steel_processing"
        elif self.value == 2:
            return "panel_lines"
        elif self.value <= 5:
            return "block_assembly"
        elif self.value == 6:
            return "pre_erection"
        elif self.value == 7:
            return "dry_dock"
        elif self.value == 8:
            return "outfitting_quay"
        elif self.value == 9:
            return "sea_trials"
        else:
            return "delivered"

    @property
    def is_block_stage(self) -> bool:
        """True if this stage operates on individual blocks."""
        return self.value <= 7

    @property
    def is_ship_stage(self) -> bool:
        """True if this stage operates on assembled ships."""
        return self.value >= 8


class BlockType(Enum):
    """Types of blocks for LNG carrier construction."""
    FLAT_BOTTOM = "flat_bottom"          # Double bottom blocks
    FLAT_SIDE = "flat_side"              # Side shell blocks
    DECK = "deck"                        # Weather deck blocks
    CARGO_TANK_SUPPORT = "cargo_tank_support"  # Membrane tank support
    ENGINE_ROOM = "engine_room"          # Machinery blocks
    CURVED_BOW = "curved_bow"            # Bow section (curved)
    CURVED_STERN = "curved_stern"        # Stern section (curved)
    ACCOMMODATION = "accommodation"      # Superstructure blocks

    @property
    def is_curved(self) -> bool:
        """True if this block type requires curved fabrication."""
        return self in (BlockType.CURVED_BOW, BlockType.CURVED_STERN)

    @property
    def processing_multiplier(self) -> float:
        """Processing time multiplier for this block type."""
        multipliers = {
            BlockType.FLAT_BOTTOM: 1.0,
            BlockType.FLAT_SIDE: 1.0,
            BlockType.DECK: 0.9,
            BlockType.CARGO_TANK_SUPPORT: 1.2,
            BlockType.ENGINE_ROOM: 1.3,
            BlockType.CURVED_BOW: 1.4,
            BlockType.CURVED_STERN: 1.4,
            BlockType.ACCOMMODATION: 1.1,
        }
        return multipliers.get(self, 1.0)


class PlateType(Enum):
    """Types of steel plates used in block construction."""
    FLAT = "flat"                     # Standard flat plates
    CURVED = "curved"                 # Curved hull/bow plates
    STIFFENED = "stiffened"           # Flat plate with welded stiffeners
    BRACKET = "bracket"               # Connection brackets between plates
    BULKHEAD = "bulkhead"             # Internal bulkhead plates (thicker)
    SHELL = "shell"                   # Outer hull shell plates


@dataclass
class Plate:
    """A steel plate within a block.

    Dimensions are in millimeters (partner's convention from 3D models).
    Weight is auto-computed from dimensions if not provided.
    """

    id: str
    plate_type: PlateType = PlateType.FLAT
    length_mm: float = 12000.0
    width_mm: float = 3000.0
    thickness_mm: float = 20.0
    weight_kg: float = 0.0
    material_grade: str = "AH36"
    has_stiffeners: bool = False
    n_stiffeners: int = 0
    stiffener_spacing_mm: float = 600.0
    curvature_radius_mm: float = 0.0

    def __post_init__(self):
        if self.weight_kg == 0.0:
            vol_m3 = (self.length_mm / 1000) * (self.width_mm / 1000) * (self.thickness_mm / 1000)
            self.weight_kg = vol_m3 * 7850.0  # Steel density ~7850 kg/m^3
            if self.has_stiffeners and self.n_stiffeners > 0:
                self.weight_kg *= (1.0 + 0.15 * self.n_stiffeners)

    @property
    def area_m2(self) -> float:
        """Surface area in square meters."""
        return (self.length_mm / 1000.0) * (self.width_mm / 1000.0)

    @property
    def is_curved(self) -> bool:
        """True if this is a curved plate."""
        return self.curvature_radius_mm > 0 or self.plate_type == PlateType.CURVED

    @property
    def weld_length_m(self) -> float:
        """Estimated weld length: perimeter + stiffener attachment welds."""
        perimeter = 2 * (self.length_mm + self.width_mm) / 1000.0
        stiffener_welds = self.n_stiffeners * (self.length_mm / 1000.0) * 2
        return perimeter + stiffener_welds


@dataclass
class Block:
    """Representation of a ship block for LNG carrier construction.

    Parameters
    ----------
    id : str
        Unique identifier for the block (e.g., "B001", "B125").
    weight : float
        Weight in tons; affects SPMT degradation and crane requirements.
    size : Tuple[float, float, float]
        Length, width, height in meters.
    due_date : float
        Target completion time (simulation units).
    block_type : BlockType
        Type of block (flat, curved, engine room, etc.).
    current_stage : HHIProductionStage
        The stage the block is currently in.
    status : BlockStatus
        High-level status flag.
    location : str
        Name of the facility where the block currently resides.
    ship_id : str
        ID of the ship this block belongs to.
    erection_sequence : int
        Position in the erection order (1-200 for LNG carrier).
    predecessors : list of str
        Blocks that must be erected before this block.
    """

    id: str
    weight: float
    size: Tuple[float, float, float]
    due_date: float
    block_type: BlockType = BlockType.FLAT_BOTTOM
    current_stage: HHIProductionStage = HHIProductionStage.STEEL_CUTTING
    status: BlockStatus = BlockStatus.WAITING
    location: str = "steel_stockyard"
    ship_id: str = ""
    erection_sequence: int = 0
    completion_pct: float = 0.0
    predecessors: List[str] = field(default_factory=list)
    dock_position: Optional[Tuple[int, int, int]] = None  # x, y, z in dock
    stage_entry_time: float = 0.0
    history: List[dict] = field(default_factory=list)
    # Plate-level decomposition fields (populated when plate data available)
    plates: List["Plate"] = field(default_factory=list)
    n_plates: int = 0
    total_plate_area_m2: float = 0.0
    plate_derived_weight: float = 0.0
    # Sub-stage tracking (optional, for partner's 15-stage model)
    current_substage: Optional["DetailedProductionStage"] = None
    substage_progress: float = 0.0

    def compute_plate_stats(self) -> None:
        """Recompute plate-derived statistics from plate list."""
        self.n_plates = len(self.plates)
        self.total_plate_area_m2 = sum(p.area_m2 for p in self.plates)
        self.plate_derived_weight = sum(p.weight_kg for p in self.plates) / 1000.0
        if self.plate_derived_weight > 0:
            self.weight = self.plate_derived_weight

    def log_event(self, timestamp: float, description: str) -> None:
        """Append an event to the block's history."""
        self.history.append({"time": timestamp, "desc": description})

    def advance_stage(self, timestamp: float) -> None:
        """Move to the next production stage."""
        current_value = self.current_stage.value
        if current_value < HHIProductionStage.ERECTION.value:
            self.current_stage = HHIProductionStage(current_value + 1)
            self.stage_entry_time = timestamp
            self.log_event(timestamp, f"Advanced to {self.current_stage.name}")

    def is_curved(self) -> bool:
        """Check if this block requires curved fabrication."""
        return self.block_type.is_curved

    def get_processing_multiplier(self) -> float:
        """Get processing time multiplier based on block type."""
        return self.block_type.processing_multiplier


class SPMTStatus(Enum):
    """Status of an SPMT (Self-Propelled Modular Transporter)."""
    IDLE = "idle"
    TRAVELING_EMPTY = "traveling_empty"
    TRAVELING_LOADED = "traveling_loaded"
    LOADING = "loading"
    UNLOADING = "unloading"
    IN_MAINTENANCE = "in_maintenance"
    BROKEN_DOWN = "broken_down"


@dataclass
class SPMT:
    """Self-propelled modular transporter.

    SPMTs carry blocks between facilities within the HHI Ulsan shipyard.
    They degrade over time and may require maintenance.
    Health indicators are expressed on a 0-100 scale.
    """

    id: str
    capacity: float = 500.0  # tons
    current_location: str = "spmt_depot"
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
        """Return health values as a numpy array."""
        return np.array([
            self.health_hydraulic / 100.0,
            self.health_tires / 100.0,
            self.health_engine / 100.0,
        ], dtype=float)

    def get_min_health(self) -> float:
        """Return the minimum health across all components."""
        return float(min(self.health_hydraulic, self.health_tires, self.health_engine))

    def estimate_rul(self, degradation_model: "WienerDegradationModel") -> float:
        """Estimate remaining useful life using a degradation model."""
        min_health = self.get_min_health()
        return float(degradation_model.estimate_rul(min_health, load_ratio=0.3))


class GoliathCraneStatus(Enum):
    """Status of a Goliath gantry crane."""
    IDLE = "idle"
    LIFTING = "lifting"
    TRAVELING = "traveling"
    POSITIONING = "positioning"
    LOWERING = "lowering"
    IN_MAINTENANCE = "in_maintenance"
    BROKEN_DOWN = "broken_down"


@dataclass
class GoliathCrane:
    """Goliath gantry crane for block erection.

    HHI Ulsan has 9 Goliath cranes, each 109m tall (36 stories),
    with lifting capacities up to 900 tons. Cranes are rail-mounted
    and assigned to specific dry docks.

    Parameters
    ----------
    id : str
        Unique identifier (e.g., "GC01", "GC09").
    assigned_dock : str
        The dry dock this crane serves.
    capacity_tons : float
        Maximum lifting capacity in tons.
    height_m : float
        Crane height in meters.
    span_m : float
        Rail-to-rail span in meters.
    position_on_rail : float
        Current position along the rail (0-100%).
    """

    id: str
    assigned_dock: str = "dock_1"
    capacity_tons: float = 900.0
    height_m: float = 109.0
    span_m: float = 150.0
    rail_length_m: float = 500.0
    position_on_rail: float = 0.0  # 0-100%
    status: GoliathCraneStatus = GoliathCraneStatus.IDLE
    current_block: Optional[str] = None
    health_hoist: float = 100.0
    health_trolley: float = 100.0
    health_gantry: float = 100.0
    travel_speed: float = 0.3  # m/s along the rail
    lift_speed: float = 0.1   # m/s vertical

    def get_health_vector(self) -> np.ndarray:
        """Return health values as a numpy array."""
        return np.array([
            self.health_hoist / 100.0,
            self.health_trolley / 100.0,
            self.health_gantry / 100.0,
        ], dtype=float)

    def get_min_health(self) -> float:
        """Return the minimum health across all components."""
        return float(min(self.health_hoist, self.health_trolley, self.health_gantry))

    def can_lift(self, weight: float) -> bool:
        """Check if the crane can lift a given weight."""
        return weight <= self.capacity_tons

    def is_available(self) -> bool:
        """Check if the crane is available for a new lift operation."""
        return self.status == GoliathCraneStatus.IDLE


class ShipStatus(Enum):
    """Status of a ship under construction."""
    IN_BLOCK_PRODUCTION = "in_block_production"
    IN_ERECTION = "in_erection"
    AFLOAT = "afloat"
    IN_QUAY_OUTFITTING = "in_quay_outfitting"
    IN_SEA_TRIALS = "in_sea_trials"
    DELIVERED = "delivered"


@dataclass
class LNGCarrier:
    """An LNG carrier under construction at HHI Ulsan.

    Parameters
    ----------
    id : str
        Unique identifier (e.g., "HN2901", "HN2902").
    hull_number : str
        Official hull number.
    capacity_cbm : float
        LNG cargo capacity in cubic meters.
    blocks : list of str
        IDs of blocks that comprise this ship (typically 200).
    assigned_dock : str, optional
        Dry dock where erection is taking place.
    """

    id: str
    hull_number: str = ""
    capacity_cbm: float = 174000.0  # Standard LNG carrier capacity
    length_m: float = 295.0
    beam_m: float = 46.0
    blocks: List[str] = field(default_factory=list)
    total_blocks: int = 200
    blocks_erected: int = 0
    assigned_dock: Optional[str] = None
    assigned_quay: Optional[str] = None
    target_delivery_date: float = 0.0
    status: ShipStatus = ShipStatus.IN_BLOCK_PRODUCTION
    erection_progress: float = 0.0
    completion_pct: float = 0.0
    history: List[dict] = field(default_factory=list)

    def log_event(self, timestamp: float, description: str) -> None:
        """Append an event to the ship's history."""
        self.history.append({"time": timestamp, "desc": description})

    def update_erection_progress(self) -> None:
        """Update erection progress based on erected blocks."""
        if self.total_blocks > 0:
            self.erection_progress = self.blocks_erected / self.total_blocks * 100.0

    def is_erection_complete(self) -> bool:
        """Check if all blocks have been erected."""
        return self.blocks_erected >= self.total_blocks

    def get_next_erection_block(self, blocks: List[Block]) -> Optional[Block]:
        """Get the next block to erect based on sequence."""
        ship_blocks = [b for b in blocks if b.ship_id == self.id]
        ready_blocks = [
            b for b in ship_blocks
            if b.current_stage == HHIProductionStage.PRE_ERECTION
            and b.status == BlockStatus.AT_PRE_ERECTION
        ]
        if not ready_blocks:
            return None
        # Sort by erection sequence
        ready_blocks.sort(key=lambda b: b.erection_sequence)
        return ready_blocks[0]


@dataclass
class DryDock:
    """A dry dock at HHI Ulsan shipyard.

    HHI Ulsan has 10 dry docks of varying sizes.
    """

    id: str
    name: str
    length_m: float = 400.0
    width_m: float = 80.0
    depth_m: float = 12.0
    capacity: int = 1  # Ships at a time
    assigned_cranes: List[str] = field(default_factory=list)
    current_ship: Optional[str] = None
    status: str = "idle"  # "idle", "in_use", "flooding", "draining", "maintenance"
    blocks_in_dock: List[str] = field(default_factory=list)

    def is_available(self) -> bool:
        """Check if the dock is available for a new ship."""
        return self.status == "idle" and self.current_ship is None

    def assign_ship(self, ship_id: str, timestamp: float) -> None:
        """Assign a ship to this dock for erection."""
        self.current_ship = ship_id
        self.status = "in_use"

    def release_ship(self, timestamp: float) -> None:
        """Release the current ship (launch)."""
        self.current_ship = None
        self.status = "draining"
        self.blocks_in_dock.clear()


@dataclass
class OutfittingQuay:
    """An outfitting quay for post-launch work."""

    id: str
    name: str
    length_m: float = 350.0
    capacity: int = 2
    current_ships: List[str] = field(default_factory=list)

    def is_available(self) -> bool:
        """Check if the quay has space for another ship."""
        return len(self.current_ships) < self.capacity

    def assign_ship(self, ship_id: str) -> bool:
        """Assign a ship to this quay."""
        if self.is_available():
            self.current_ships.append(ship_id)
            return True
        return False

    def release_ship(self, ship_id: str) -> bool:
        """Release a ship from this quay."""
        if ship_id in self.current_ships:
            self.current_ships.remove(ship_id)
            return True
        return False


# ---------------------------------------------------------------------------
# Supply Chain Extension: Suppliers, Materials, Labor
# ---------------------------------------------------------------------------


class MaterialType(Enum):
    """Types of materials consumed during block production."""
    STEEL_PLATE = "steel_plate"
    PIPE_SECTION = "pipe_section"
    ELECTRICAL_CABLE = "electrical_cable"
    INSULATION = "insulation"
    PAINT = "paint"
    WELDING_CONSUMABLE = "welding_consumable"


@dataclass
class MaterialInventory:
    """Inventory of a material type at a specific warehouse location."""

    id: str
    material_type: MaterialType = MaterialType.STEEL_PLATE
    quantity: float = 1000.0
    location: str = "steel_stockyard"
    reorder_point: float = 200.0
    max_capacity: float = 5000.0
    holding_cost_per_unit: float = 0.01
    consumption_rate: float = 0.5

    def is_below_reorder_point(self) -> bool:
        return self.quantity <= self.reorder_point

    def is_stockout(self) -> bool:
        return self.quantity <= 0.0

    def consume(self, amount: float) -> float:
        """Consume material. Returns actual amount consumed."""
        consumed = min(self.quantity, amount)
        self.quantity -= consumed
        return consumed

    def receive(self, amount: float) -> None:
        """Receive material from supplier delivery."""
        self.quantity = min(self.quantity + amount, self.max_capacity)

    def get_feature_vector(self) -> List[float]:
        return [
            self.quantity / max(self.max_capacity, 1.0),
            1.0 if self.is_below_reorder_point() else 0.0,
            1.0 if self.is_stockout() else 0.0,
            self.holding_cost_per_unit,
        ]


@dataclass
class Supplier:
    """A supplier of materials or components."""

    id: str
    name: str = ""
    lead_time_mean: float = 72.0
    lead_time_std: float = 12.0
    reliability: float = 0.9
    capacity_per_period: float = 500.0
    current_backlog: int = 0
    price_per_unit: float = 1.0
    rush_multiplier: float = 1.5
    specializations: List[str] = field(default_factory=lambda: ["steel_plate"])
    pending_orders: List[dict] = field(default_factory=list)

    def can_accept_order(self, quantity: float) -> bool:
        total_backlog = self.current_backlog + quantity
        return total_backlog <= self.capacity_per_period

    def place_order(
        self, material_type: str, quantity: float, sim_time: float
    ) -> dict:
        """Place an order; returns order dict with estimated delivery time."""
        delivery_time = sim_time + max(
            1.0, np.random.normal(self.lead_time_mean, self.lead_time_std)
        )
        order = {
            "order_id": f"{self.id}_ORD{len(self.pending_orders)}",
            "material_type": material_type,
            "quantity": quantity,
            "delivery_time": delivery_time,
            "cost": quantity * self.price_per_unit,
        }
        self.pending_orders.append(order)
        self.current_backlog += int(quantity)
        return order

    def check_deliveries(self, sim_time: float) -> List[dict]:
        """Check and return any orders that have been delivered."""
        delivered = []
        remaining = []
        for order in self.pending_orders:
            if sim_time >= order["delivery_time"]:
                if np.random.random() < self.reliability:
                    delivered.append(order)
                    self.current_backlog = max(
                        0, self.current_backlog - int(order["quantity"])
                    )
                else:
                    order["delivery_time"] += self.lead_time_mean * 0.5
                    remaining.append(order)
            else:
                remaining.append(order)
        self.pending_orders = remaining
        return delivered

    def get_feature_vector(self) -> List[float]:
        return [
            self.current_backlog / max(1.0, self.capacity_per_period),
            self.lead_time_mean / 200.0,
            self.reliability,
            self.price_per_unit / 10.0,
            len(self.pending_orders) / 20.0,
        ]


class SkillType(Enum):
    """Types of skilled labor in a shipyard."""
    WELDER = "welder"
    ELECTRICIAN = "electrician"
    FITTER = "fitter"
    PAINTER = "painter"
    CRANE_OPERATOR = "crane_operator"


@dataclass
class LaborPool:
    """A pool of workers with a specific skill type."""

    id: str
    skill_type: SkillType = SkillType.WELDER
    total_workers: int = 20
    available_workers: int = 20
    hourly_rate: float = 50.0
    overtime_rate: float = 75.0
    max_overtime_hours: float = 4.0
    current_overtime_hours: float = 0.0
    shift_hours: float = 8.0
    assigned_tasks: List[str] = field(default_factory=list)

    def can_assign(self, n_workers: int = 1) -> bool:
        return self.available_workers >= n_workers

    def assign(self, block_id: str, n_workers: int = 1) -> bool:
        if not self.can_assign(n_workers):
            return False
        self.available_workers -= n_workers
        self.assigned_tasks.append(block_id)
        return True

    def release(self, block_id: str, n_workers: int = 1) -> None:
        if block_id in self.assigned_tasks:
            self.assigned_tasks.remove(block_id)
            self.available_workers = min(
                self.total_workers, self.available_workers + n_workers
            )

    def reset_shift(self) -> None:
        """Reset at start of new shift."""
        self.current_overtime_hours = 0.0
        self.available_workers = self.total_workers
        self.assigned_tasks.clear()

    def get_feature_vector(self) -> List[float]:
        return [
            self.available_workers / max(1, self.total_workers),
            len(self.assigned_tasks) / max(1, self.total_workers),
            self.current_overtime_hours / max(1.0, self.max_overtime_hours),
            self.hourly_rate / 100.0,
        ]


class DetailedProductionStage(Enum):
    """Partner's 15-stage production model (plate-level granularity).

    More granular at early stages (plate cutting/rolling), less at late stages.
    Maps to our 11-stage HHIProductionStage via DETAILED_TO_HHI_STAGE.
    """
    RAW_MATERIAL_STORAGE = 0
    PLATE_TRANSPORT_TO_FAB = 1
    FAB_CUTTING = 2
    FAB_ROLLING = 3
    CUT_ROLL_STORAGE = 4
    PANEL_STIFFENER_WELDING = 5
    TRANSPORT_TO_BLOCK_STORAGE = 6
    BLOCK_ASSEMBLY_STORAGE = 7
    WELD_PLATE_TO_BLOCK = 8
    PLATES_ON_PARTIAL_BLOCK = 9
    TRANSPORT_TO_SHIP_ERECTION = 10
    WELD_BLOCK_TO_SHIP = 11
    BLOCKS_ON_PARTIAL_SHIP = 12
    SHIP_TRANSPORT_TO_DELIVERY = 13
    SHIP_COMPLETE = 14


DETAILED_TO_HHI_STAGE = {
    DetailedProductionStage.RAW_MATERIAL_STORAGE: HHIProductionStage.STEEL_CUTTING,
    DetailedProductionStage.PLATE_TRANSPORT_TO_FAB: HHIProductionStage.STEEL_CUTTING,
    DetailedProductionStage.FAB_CUTTING: HHIProductionStage.STEEL_CUTTING,
    DetailedProductionStage.FAB_ROLLING: HHIProductionStage.PART_FABRICATION,
    DetailedProductionStage.CUT_ROLL_STORAGE: HHIProductionStage.PART_FABRICATION,
    DetailedProductionStage.PANEL_STIFFENER_WELDING: HHIProductionStage.PANEL_ASSEMBLY,
    DetailedProductionStage.TRANSPORT_TO_BLOCK_STORAGE: HHIProductionStage.PANEL_ASSEMBLY,
    DetailedProductionStage.BLOCK_ASSEMBLY_STORAGE: HHIProductionStage.BLOCK_ASSEMBLY,
    DetailedProductionStage.WELD_PLATE_TO_BLOCK: HHIProductionStage.BLOCK_ASSEMBLY,
    DetailedProductionStage.PLATES_ON_PARTIAL_BLOCK: HHIProductionStage.BLOCK_ASSEMBLY,
    DetailedProductionStage.TRANSPORT_TO_SHIP_ERECTION: HHIProductionStage.PRE_ERECTION,
    DetailedProductionStage.WELD_BLOCK_TO_SHIP: HHIProductionStage.ERECTION,
    DetailedProductionStage.BLOCKS_ON_PARTIAL_SHIP: HHIProductionStage.ERECTION,
    DetailedProductionStage.SHIP_TRANSPORT_TO_DELIVERY: HHIProductionStage.SEA_TRIALS,
    DetailedProductionStage.SHIP_COMPLETE: HHIProductionStage.DELIVERY,
}


# Backward compatibility aliases for old EB code
ProductionStage = HHIProductionStage
Crane = GoliathCrane
CraneStatus = GoliathCraneStatus

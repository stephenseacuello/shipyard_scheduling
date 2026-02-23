"""Simulation module for shipyard scheduling.

Models HD Hyundai Heavy Industries (HHI) Ulsan shipyard for LNG carrier production.
"""

from .shipyard import ShipyardGraph, HHIShipyardGraph, DualShipyardGraph, BargeRoute
from .entities import (
    # Core entities
    Block,
    SPMT,
    GoliathCrane,
    LNGCarrier,
    DryDock,
    OutfittingQuay,
    # Supply chain entities
    Supplier,
    MaterialInventory,
    LaborPool,
    # Plate decomposition entities
    Plate,
    PlateType,
    DetailedProductionStage,
    DETAILED_TO_HHI_STAGE,
    # Enums
    HHIProductionStage,
    BlockType,
    BlockStatus,
    SPMTStatus,
    GoliathCraneStatus,
    ShipStatus,
    MaterialType,
    SkillType,
)
from .degradation import WienerDegradationModel
from .precedence import is_predecessor_complete

# Backward compatibility aliases for old EB code
Crane = GoliathCrane
CraneStatus = GoliathCraneStatus
ProductionStage = HHIProductionStage

__all__ = [
    # Graphs
    "ShipyardGraph",
    "HHIShipyardGraph",
    "DualShipyardGraph",
    "BargeRoute",
    # HHI Entities
    "Block",
    "SPMT",
    "GoliathCrane",
    "LNGCarrier",
    "DryDock",
    "OutfittingQuay",
    # Enums
    "HHIProductionStage",
    "BlockType",
    "BlockStatus",
    "SPMTStatus",
    "GoliathCraneStatus",
    "ShipStatus",
    # Supply chain
    "Supplier",
    "MaterialInventory",
    "LaborPool",
    "MaterialType",
    "SkillType",
    # Plate decomposition
    "Plate",
    "PlateType",
    "DetailedProductionStage",
    "DETAILED_TO_HHI_STAGE",
    # Backward compatibility aliases
    "Crane",
    "CraneStatus",
    "ProductionStage",
    # Other
    "WienerDegradationModel",
    "is_predecessor_complete",
]
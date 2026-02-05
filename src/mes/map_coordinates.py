"""Coordinate mapping for shipyard map visualization.

This module defines the pixel coordinates for facilities, staging areas,
and other elements in the Quonset Point and Groton shipyard visualizations.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class FacilityCoord:
    """Coordinates and metadata for a facility on the map."""
    x: float
    y: float
    width: float = 60
    height: float = 40
    label: str = ""
    color: str = "#3498db"


# ============================================================================
# QUONSET POINT, RI - Module Fabrication & Outfitting
# Canvas: 800 x 600 pixels
# ============================================================================

QUONSET_CANVAS = {"width": 800, "height": 600}

QUONSET_FACILITIES: Dict[str, FacilityCoord] = {
    # Production line (top)
    "steel_processing": FacilityCoord(x=100, y=80, width=100, height=50,
                                       label="STEEL PROCESSING", color="#3498db"),
    "afc_facility": FacilityCoord(x=250, y=80, width=120, height=50,
                                   label="AFC FACILITY", color="#e67e22"),

    # Outfitting buildings (middle)
    "bldg_9a": FacilityCoord(x=150, y=180, width=80, height=45,
                              label="BLDG 9A", color="#27ae60"),
    "bldg_9b": FacilityCoord(x=260, y=180, width=80, height=45,
                              label="BLDG 9B", color="#27ae60"),
    "bldg_9c": FacilityCoord(x=370, y=180, width=80, height=45,
                              label="BLDG 9C", color="#27ae60"),

    # Super module assembly (middle-right)
    "super_module_assembly": FacilityCoord(x=520, y=180, width=140, height=60,
                                            label="SUPER MODULE ASSEMBLY", color="#9b59b6"),

    # Staging areas
    "quonset_wip1": FacilityCoord(x=100, y=280, width=70, height=40,
                                   label="WIP-1", color="#95a5a6"),
    "quonset_wip2": FacilityCoord(x=200, y=280, width=70, height=40,
                                   label="WIP-2", color="#95a5a6"),

    # Pier (bottom)
    "quonset_pier": FacilityCoord(x=350, y=480, width=200, height=60,
                                   label="NARRAGANSETT BAY PIER", color="#1abc9c"),
}

# Flow arrows for Quonset (from -> to)
QUONSET_FLOW_ARROWS: List[Tuple[str, str]] = [
    ("steel_processing", "afc_facility"),
    ("afc_facility", "bldg_9a"),
    ("afc_facility", "bldg_9b"),
    ("afc_facility", "bldg_9c"),
    ("bldg_9a", "super_module_assembly"),
    ("bldg_9b", "super_module_assembly"),
    ("bldg_9c", "super_module_assembly"),
    ("super_module_assembly", "quonset_pier"),
]

# SPMT depot location at Quonset
QUONSET_SPMT_DEPOT = (700, 350)

# Barge position when docked at Quonset
QUONSET_BARGE_POSITION = (450, 530)


# ============================================================================
# GROTON, CT - Final Assembly & Launch
# Canvas: 800 x 600 pixels
# ============================================================================

GROTON_CANVAS = {"width": 800, "height": 600}

GROTON_FACILITIES: Dict[str, FacilityCoord] = {
    # Thames River receiving (top)
    "groton_pier": FacilityCoord(x=100, y=60, width=180, height=50,
                                  label="THAMES RIVER PIER", color="#1abc9c"),

    # Land level construction (middle-top)
    "land_level_construction": FacilityCoord(x=350, y=120, width=180, height=70,
                                              label="LAND LEVEL CONSTRUCTION", color="#3498db"),

    # Building 600 / South Yard Assembly (middle)
    "building_600": FacilityCoord(x=350, y=250, width=200, height=80,
                                   label="BUILDING 600 (SYAB)", color="#e74c3c"),

    # Graving dock (bottom)
    "graving_dock": FacilityCoord(x=350, y=400, width=200, height=80,
                                   label="GRAVING DOCK", color="#9b59b6"),

    # Dry docks (right side)
    "dry_dock_1": FacilityCoord(x=620, y=250, width=100, height=60,
                                 label="DRY DOCK 1", color="#7f8c8d"),
    "dry_dock_2": FacilityCoord(x=620, y=340, width=100, height=60,
                                 label="DRY DOCK 2", color="#7f8c8d"),

    # Staging
    "groton_staging": FacilityCoord(x=100, y=200, width=100, height=50,
                                     label="STAGING", color="#95a5a6"),
}

# Groton dock grid positions (final assembly positions)
GROTON_DOCK_GRID = {
    "origin": (360, 420),
    "cell_width": 60,
    "cell_height": 40,
    "rows": 1,
    "cols": 3,
}

# Flow arrows for Groton
GROTON_FLOW_ARROWS: List[Tuple[str, str]] = [
    ("groton_pier", "land_level_construction"),
    ("groton_pier", "groton_staging"),
    ("groton_staging", "land_level_construction"),
    ("land_level_construction", "building_600"),
    ("building_600", "graving_dock"),
]

# Crane rail at Groton (Building 600)
GROTON_CRANE_RAIL = {
    "y": 290,
    "x_start": 350,
    "x_end": 550,
}

# SPMT depot at Groton
GROTON_SPMT_DEPOT = (700, 450)

# Barge position when docked at Groton
GROTON_BARGE_POSITION = (190, 30)


# ============================================================================
# Helper functions
# ============================================================================

def node_to_coords(node_name: str, yard: str = "quonset") -> Tuple[float, float]:
    """Convert a node name to (x, y) coordinates for the specified yard.

    Parameters
    ----------
    node_name : str
        The node identifier (facility name, dock position, etc.)
    yard : str
        Either "quonset" or "groton"

    Returns
    -------
    Tuple[float, float]
        The (x, y) pixel coordinates for the node.
    """
    facilities = QUONSET_FACILITIES if yard == "quonset" else GROTON_FACILITIES

    # Check if it's a known facility
    if node_name in facilities:
        fac = facilities[node_name]
        return (fac.x + fac.width / 2, fac.y + fac.height / 2)

    # Check for dock position (Groton only)
    if yard == "groton" and node_name.startswith("dock_"):
        parts = node_name.split("_")
        if len(parts) == 3:
            row, col = int(parts[1]), int(parts[2])
            grid = GROTON_DOCK_GRID
            x = grid["origin"][0] + col * grid["cell_width"] + grid["cell_width"] / 2
            y = grid["origin"][1] + row * grid["cell_height"] + grid["cell_height"] / 2
            return (x, y)

    # Default: SPMT depot
    if yard == "quonset":
        return QUONSET_SPMT_DEPOT
    else:
        return GROTON_SPMT_DEPOT


def get_facility_center(facility_name: str, yard: str = "quonset") -> Optional[Tuple[float, float]]:
    """Get the center coordinates of a facility."""
    facilities = QUONSET_FACILITIES if yard == "quonset" else GROTON_FACILITIES
    if facility_name in facilities:
        fac = facilities[facility_name]
        return (fac.x + fac.width / 2, fac.y + fac.height / 2)
    return None


def get_arrow_coords(from_facility: str, to_facility: str, yard: str = "quonset") -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Get the start and end coordinates for a flow arrow between facilities."""
    facilities = QUONSET_FACILITIES if yard == "quonset" else GROTON_FACILITIES

    if from_facility not in facilities or to_facility not in facilities:
        return None

    from_fac = facilities[from_facility]
    to_fac = facilities[to_facility]

    # Calculate edge points (center of right edge -> center of left edge)
    from_x = from_fac.x + from_fac.width
    from_y = from_fac.y + from_fac.height / 2
    to_x = to_fac.x
    to_y = to_fac.y + to_fac.height / 2

    return ((from_x, from_y), (to_x, to_y))


# Color schemes for visualization
COLORS = {
    "primary": "#2c3e50",
    "accent": "#3498db",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "purple": "#9b59b6",
    "teal": "#1abc9c",
    "orange": "#e67e22",
    "gray": "#95a5a6",
    "dark": "#34495e",
}

# Status colors for equipment
STATUS_COLORS = {
    "idle": "#95a5a6",
    "traveling_empty": "#3498db",
    "traveling_loaded": "#27ae60",
    "loading": "#f39c12",
    "unloading": "#f39c12",
    "in_maintenance": "#9b59b6",
    "broken_down": "#e74c3c",
    "lifting": "#27ae60",
    "positioning": "#3498db",
}

# Health-based colors
def health_to_color(health: float) -> str:
    """Convert a health value (0-100) to a color."""
    if health >= 70:
        return COLORS["success"]
    elif health >= 40:
        return COLORS["warning"]
    else:
        return COLORS["danger"]

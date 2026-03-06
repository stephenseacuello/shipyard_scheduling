"""Coordinate mapping for HD Hyundai Heavy Industries Ulsan shipyard visualization.

This module defines the pixel coordinates for facilities, dry docks, staging areas,
and other elements in the HHI Ulsan shipyard visualization.

Canvas: 1200 x 800 pixels (larger for 4km mega-yard)
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
# HHI ULSAN SHIPYARD - LNG Carrier Production
# Canvas: 1200 x 800 pixels
# Layout: Production flows left-to-right, dry docks on right (coastline)
# ============================================================================

HHI_CANVAS = {"width": 1200, "height": 800}

# ============================================================================
# ZONE 1: STEEL PROCESSING (Top-Left)
# ============================================================================

STEEL_PROCESSING: Dict[str, FacilityCoord] = {
    "steel_stockyard": FacilityCoord(
        x=50, y=50, width=120, height=60,
        label="STEEL STOCKYARD", color="#7f8c8d"
    ),
    "cutting_shop": FacilityCoord(
        x=200, y=50, width=100, height=60,
        label="CUTTING SHOP", color="#3498db"
    ),
    "part_fabrication": FacilityCoord(
        x=200, y=130, width=100, height=50,
        label="PART FAB", color="#3498db"
    ),
}

# ============================================================================
# ZONE 2: PANEL ASSEMBLY (Left-Center)
# ============================================================================

PANEL_ASSEMBLY: Dict[str, FacilityCoord] = {
    "flat_panel_line_1": FacilityCoord(
        x=350, y=50, width=120, height=50,
        label="FLAT PANEL LINE 1", color="#27ae60"
    ),
    "flat_panel_line_2": FacilityCoord(
        x=350, y=120, width=120, height=50,
        label="FLAT PANEL LINE 2", color="#27ae60"
    ),
    "curved_block_shop": FacilityCoord(
        x=350, y=190, width=120, height=55,
        label="CURVED BLOCK SHOP", color="#e67e22"
    ),
}

# ============================================================================
# ZONE 3: BLOCK ASSEMBLY (Center)
# ============================================================================

BLOCK_ASSEMBLY: Dict[str, FacilityCoord] = {
    "block_assembly_hall_1": FacilityCoord(
        x=520, y=50, width=130, height=65,
        label="BLOCK ASSEMBLY 1", color="#9b59b6"
    ),
    "block_assembly_hall_2": FacilityCoord(
        x=520, y=135, width=130, height=65,
        label="BLOCK ASSEMBLY 2", color="#9b59b6"
    ),
    "block_assembly_hall_3": FacilityCoord(
        x=520, y=220, width=130, height=65,
        label="BLOCK ASSEMBLY 3", color="#9b59b6"
    ),
    "outfitting_shop": FacilityCoord(
        x=520, y=310, width=130, height=55,
        label="OUTFITTING SHOP", color="#1abc9c"
    ),
    "paint_shop": FacilityCoord(
        x=520, y=385, width=130, height=50,
        label="PAINT SHOP", color="#f39c12"
    ),
}

# ============================================================================
# ZONE 4: PRE-ERECTION STAGING (Center-Right)
# ============================================================================

PRE_ERECTION: Dict[str, FacilityCoord] = {
    "grand_block_staging_north": FacilityCoord(
        x=700, y=100, width=140, height=80,
        label="GRAND BLOCK STAGING (N)", color="#34495e"
    ),
    "grand_block_staging_south": FacilityCoord(
        x=700, y=280, width=140, height=80,
        label="GRAND BLOCK STAGING (S)", color="#34495e"
    ),
}

# ============================================================================
# ZONE 5: DRY DOCKS (Right side - coastline)
# 10 dry docks arranged vertically
# ============================================================================

DRY_DOCKS: Dict[str, FacilityCoord] = {
    "dock_1": FacilityCoord(
        x=900, y=30, width=150, height=50,
        label="DOCK 1 (390m LNG)", color="#e74c3c"
    ),
    "dock_2": FacilityCoord(
        x=900, y=90, width=170, height=52,
        label="DOCK 2 (500m)", color="#e74c3c"
    ),
    "dock_3": FacilityCoord(
        x=900, y=152, width=200, height=60,
        label="DOCK 3 (672m MEGA)", color="#e74c3c"
    ),
    "dock_4": FacilityCoord(
        x=900, y=222, width=150, height=48,
        label="DOCK 4 (390m 150kDWT)", color="#e74c3c"
    ),
    "dock_5": FacilityCoord(
        x=900, y=280, width=120, height=42,
        label="DOCK 5 (300m 70kDWT)", color="#c0392b"
    ),
    "dock_6": FacilityCoord(
        x=900, y=332, width=110, height=40,
        label="DOCK 6 (280m Naval)", color="#c0392b"
    ),
    "dock_7": FacilityCoord(
        x=900, y=382, width=105, height=38,
        label="DOCK 7 (260m Naval)", color="#c0392b"
    ),
    "dock_8": FacilityCoord(
        x=900, y=430, width=140, height=45,
        label="DOCK 8 (350m VLCC)", color="#922b21"
    ),
    "dock_9": FacilityCoord(
        x=900, y=485, width=130, height=42,
        label="DOCK 9 (320m VLCC)", color="#922b21"
    ),
    "h_dock": FacilityCoord(
        x=900, y=537, width=180, height=55,
        label="H-DOCK (490m Offshore)", color="#8e44ad"
    ),
}

# ============================================================================
# ZONE 6: OUTFITTING QUAYS (Bottom-Right)
# ============================================================================

OUTFITTING_QUAYS: Dict[str, FacilityCoord] = {
    "quay_1": FacilityCoord(
        x=900, y=620, width=150, height=45,
        label="OUTFITTING QUAY 1", color="#1abc9c"
    ),
    "quay_2": FacilityCoord(
        x=900, y=680, width=140, height=45,
        label="OUTFITTING QUAY 2", color="#1abc9c"
    ),
    "quay_3": FacilityCoord(
        x=900, y=740, width=130, height=40,
        label="OUTFITTING QUAY 3", color="#1abc9c"
    ),
}

# ============================================================================
# STAGING AREAS
# ============================================================================

STAGING_AREAS: Dict[str, FacilityCoord] = {
    "steel_staging": FacilityCoord(
        x=50, y=130, width=80, height=40,
        label="STEEL STAGING", color="#95a5a6"
    ),
    "panel_staging": FacilityCoord(
        x=350, y=260, width=80, height=35,
        label="PANEL STAGING", color="#95a5a6"
    ),
    "block_staging_west": FacilityCoord(
        x=670, y=200, width=70, height=40,
        label="BLOCK STAGING W", color="#95a5a6"
    ),
    "block_staging_east": FacilityCoord(
        x=700, y=400, width=70, height=40,
        label="BLOCK STAGING E", color="#95a5a6"
    ),
    "painted_block_storage": FacilityCoord(
        x=670, y=385, width=70, height=40,
        label="PAINTED STORAGE", color="#95a5a6"
    ),
    "spmt_depot": FacilityCoord(
        x=50, y=700, width=100, height=50,
        label="SPMT DEPOT", color="#2c3e50"
    ),
}

# ============================================================================
# COMBINED FACILITIES
# ============================================================================

HHI_FACILITIES: Dict[str, FacilityCoord] = {
    **STEEL_PROCESSING,
    **PANEL_ASSEMBLY,
    **BLOCK_ASSEMBLY,
    **PRE_ERECTION,
    **DRY_DOCKS,
    **OUTFITTING_QUAYS,
    **STAGING_AREAS,
}

# ============================================================================
# GOLIATH CRANE POSITIONS (on dock rails)
# ============================================================================

GOLIATH_CRANE_POSITIONS: Dict[str, Tuple[float, float]] = {
    "GC01": (1060, 45),   # Dock 1
    "GC02": (1060, 70),   # Dock 1
    "GC03": (1080, 115),  # Dock 2
    "GC04": (1110, 170),  # Dock 3 (mega dock)
    "GC05": (1110, 200),  # Dock 3 (mega dock)
    "GC06": (1060, 240),  # Dock 4
    "GC07": (1030, 295),  # Dock 5
    "GC08": (1050, 450),  # Dock 8
    "GC09": (1090, 560),  # H-Dock
}

# Crane rail definitions for each dock
CRANE_RAILS: Dict[str, Dict[str, float]] = {
    "dock_1": {"y": 55, "x_start": 900, "x_end": 1050},
    "dock_2": {"y": 115, "x_start": 900, "x_end": 1070},
    "dock_3": {"y": 182, "x_start": 900, "x_end": 1100},
    "dock_4": {"y": 245, "x_start": 900, "x_end": 1050},
    "dock_5": {"y": 298, "x_start": 900, "x_end": 1020},
    "dock_8": {"y": 450, "x_start": 900, "x_end": 1040},
    "h_dock": {"y": 562, "x_start": 900, "x_end": 1080},
}

# ============================================================================
# PRODUCTION FLOW ARROWS
# ============================================================================

HHI_FLOW_ARROWS: List[Tuple[str, str]] = [
    # Steel processing flow
    ("steel_stockyard", "cutting_shop"),
    ("cutting_shop", "part_fabrication"),
    ("cutting_shop", "flat_panel_line_1"),
    ("cutting_shop", "flat_panel_line_2"),
    ("part_fabrication", "curved_block_shop"),
    # Panel to block assembly
    ("flat_panel_line_1", "block_assembly_hall_1"),
    ("flat_panel_line_2", "block_assembly_hall_2"),
    ("curved_block_shop", "block_assembly_hall_3"),
    # Block assembly to outfitting
    ("block_assembly_hall_1", "outfitting_shop"),
    ("block_assembly_hall_2", "outfitting_shop"),
    ("block_assembly_hall_3", "outfitting_shop"),
    # Outfitting to painting
    ("outfitting_shop", "paint_shop"),
    # Painting to pre-erection
    ("paint_shop", "grand_block_staging_north"),
    ("paint_shop", "grand_block_staging_south"),
    # Pre-erection to docks (sample connections)
    ("grand_block_staging_north", "dock_1"),
    ("grand_block_staging_north", "dock_2"),
    ("grand_block_staging_south", "dock_5"),
    ("grand_block_staging_south", "dock_6"),
]

# ============================================================================
# Helper Functions
# ============================================================================

def node_to_coords(node_name: str) -> Tuple[float, float]:
    """Convert a node name to (x, y) pixel coordinates.

    Parameters
    ----------
    node_name : str
        The node identifier (facility name, dock position, etc.)

    Returns
    -------
    Tuple[float, float]
        The (x, y) pixel coordinates for the node center.
    """
    # Check if it's a known facility
    if node_name in HHI_FACILITIES:
        fac = HHI_FACILITIES[node_name]
        return (fac.x + fac.width / 2, fac.y + fac.height / 2)

    # Check for queue_X pattern
    if node_name.startswith("queue_"):
        base_name = node_name.replace("queue_", "")
        if base_name in HHI_FACILITIES:
            fac = HHI_FACILITIES[base_name]
            # Offset slightly to the left of facility
            return (fac.x - 10, fac.y + fac.height / 2)

    # Check for waiting_transport_to_X pattern
    if "waiting_transport_to_" in node_name.lower():
        target = node_name.lower().replace("waiting_transport_to_", "")
        if target in HHI_FACILITIES:
            fac = HHI_FACILITIES[target]
            return (fac.x - 20, fac.y + fac.height / 2)

    # Default: SPMT depot
    depot = STAGING_AREAS["spmt_depot"]
    return (depot.x + depot.width / 2, depot.y + depot.height / 2)


def get_facility_center(facility_name: str) -> Optional[Tuple[float, float]]:
    """Get the center coordinates of a facility."""
    if facility_name in HHI_FACILITIES:
        fac = HHI_FACILITIES[facility_name]
        return (fac.x + fac.width / 2, fac.y + fac.height / 2)
    return None


def get_arrow_coords(
    from_facility: str, to_facility: str
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Get the start and end coordinates for a flow arrow between facilities."""
    if from_facility not in HHI_FACILITIES or to_facility not in HHI_FACILITIES:
        return None

    from_fac = HHI_FACILITIES[from_facility]
    to_fac = HHI_FACILITIES[to_facility]

    # Calculate edge points (center of right edge -> center of left edge)
    from_x = from_fac.x + from_fac.width
    from_y = from_fac.y + from_fac.height / 2
    to_x = to_fac.x
    to_y = to_fac.y + to_fac.height / 2

    return ((from_x, from_y), (to_x, to_y))


def get_goliath_crane_position(crane_id: str, position_pct: float = 0.5) -> Tuple[float, float]:
    """Get the pixel position of a Goliath crane on its rail.

    Parameters
    ----------
    crane_id : str
        Crane identifier (e.g., "GC01")
    position_pct : float
        Position along the rail from 0.0 to 1.0

    Returns
    -------
    Tuple[float, float]
        The (x, y) pixel position
    """
    if crane_id in GOLIATH_CRANE_POSITIONS:
        return GOLIATH_CRANE_POSITIONS[crane_id]

    # Default position
    return (1000, 300)


# ============================================================================
# Color Schemes
# ============================================================================

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

STATUS_COLORS = {
    "idle": "#95a5a6",
    "traveling_empty": "#3498db",
    "traveling_loaded": "#27ae60",
    "loading": "#f39c12",
    "unloading": "#f39c12",
    "in_maintenance": "#9b59b6",
    "broken_down": "#e74c3c",
    "lifting": "#27ae60",
    "traveling": "#3498db",
    "positioning": "#3498db",
    "lowering": "#27ae60",
}

BLOCK_TYPE_COLORS = {
    "flat_bottom": "#3498db",
    "flat_side": "#2980b9",
    "deck": "#1abc9c",
    "cargo_tank_support": "#9b59b6",
    "engine_room": "#e74c3c",
    "curved_bow": "#e67e22",
    "curved_stern": "#d35400",
    "accommodation": "#f39c12",
}


def health_to_color(health: float) -> str:
    """Convert a health value (0-100) to a color."""
    if health >= 70:
        return COLORS["success"]
    elif health >= 40:
        return COLORS["warning"]
    else:
        return COLORS["danger"]


def block_type_to_color(block_type: str) -> str:
    """Get the color for a block type."""
    return BLOCK_TYPE_COLORS.get(block_type, COLORS["accent"])

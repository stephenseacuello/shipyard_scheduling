"""Geographic coordinates for HD Hyundai Heavy Industries Ulsan shipyard.

This module defines the lat/lng coordinates for facilities, dry docks,
staging areas, and other elements in the HHI Ulsan shipyard visualization
using real-world coordinates for dash-leaflet integration.

Location: Mipo Bay, Ulsan, South Korea (~35.50°N, 129.42°E)
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class GeoFacility:
    """Geographic definition for a facility on the map."""
    center: Tuple[float, float]  # (lat, lng)
    polygon: List[Tuple[float, float]] = field(default_factory=list)  # List of (lat, lng) vertices
    label: str = ""
    color: str = "#3498db"
    fill_opacity: float = 0.3
    production_stage: str = ""  # Maps to HHIProductionStage


# ============================================================================
# HD HYUNDAI HEAVY INDUSTRIES - ULSAN SHIPYARD
# Location: Mipo Bay, Ulsan, South Korea
# Real coordinates: ~35.5067° N, 129.4133° E
# Spans approximately 4km along the coastline (1,780 acres)
# ============================================================================

HHI_ULSAN_CENTER = (35.5067, 129.4133)
HHI_ULSAN_ZOOM = 14  # Wider view for 4km shipyard

# Aliases for cleaner imports
HHI_CENTER = HHI_ULSAN_CENTER
HHI_ZOOM = HHI_ULSAN_ZOOM
HHI_BOUNDS = {
    "north": 35.5250,
    "south": 35.4900,
    "east": 129.4350,
    "west": 129.3900,
}

# ============================================================================
# ZONE 1: STEEL PROCESSING (Northwest area)
# ============================================================================

STEEL_PROCESSING_FACILITIES: Dict[str, GeoFacility] = {
    "steel_stockyard": GeoFacility(
        center=(35.5150, 129.4050),
        polygon=[
            (35.5160, 129.4030),
            (35.5160, 129.4070),
            (35.5140, 129.4070),
            (35.5140, 129.4030),
        ],
        label="Steel Stockyard",
        color="#7f8c8d",
        production_stage="STEEL_CUTTING",
    ),
    "cutting_shop": GeoFacility(
        center=(35.5140, 129.4090),
        polygon=[
            (35.5148, 129.4075),
            (35.5148, 129.4105),
            (35.5132, 129.4105),
            (35.5132, 129.4075),
        ],
        label="Cutting Shop (NC Plasma)",
        color="#3498db",
        production_stage="STEEL_CUTTING",
    ),
    "part_fabrication": GeoFacility(
        center=(35.5128, 129.4080),
        polygon=[
            (35.5135, 129.4065),
            (35.5135, 129.4095),
            (35.5121, 129.4095),
            (35.5121, 129.4065),
        ],
        label="Part Fabrication",
        color="#3498db",
        production_stage="PART_FABRICATION",
    ),
}

# ============================================================================
# ZONE 2: PANEL ASSEMBLY (West-Central area)
# ============================================================================

PANEL_ASSEMBLY_FACILITIES: Dict[str, GeoFacility] = {
    "flat_panel_line_1": GeoFacility(
        center=(35.5115, 129.4105),
        polygon=[
            (35.5122, 129.4090),
            (35.5122, 129.4120),
            (35.5108, 129.4120),
            (35.5108, 129.4090),
        ],
        label="Flat Panel Line 1",
        color="#27ae60",
        production_stage="PANEL_ASSEMBLY",
    ),
    "flat_panel_line_2": GeoFacility(
        center=(35.5115, 129.4135),
        polygon=[
            (35.5122, 129.4120),
            (35.5122, 129.4150),
            (35.5108, 129.4150),
            (35.5108, 129.4120),
        ],
        label="Flat Panel Line 2",
        color="#27ae60",
        production_stage="PANEL_ASSEMBLY",
    ),
    "curved_block_shop": GeoFacility(
        center=(35.5100, 129.4090),
        polygon=[
            (35.5108, 129.4075),
            (35.5108, 129.4105),
            (35.5092, 129.4105),
            (35.5092, 129.4075),
        ],
        label="Curved Block Shop",
        color="#e67e22",
        production_stage="PANEL_ASSEMBLY",
    ),
}

# ============================================================================
# ZONE 3: BLOCK ASSEMBLY (Central area)
# ============================================================================

BLOCK_ASSEMBLY_FACILITIES: Dict[str, GeoFacility] = {
    "block_assembly_hall_1": GeoFacility(
        center=(35.5085, 129.4120),
        polygon=[
            (35.5095, 129.4100),
            (35.5095, 129.4140),
            (35.5075, 129.4140),
            (35.5075, 129.4100),
        ],
        label="Block Assembly Hall 1",
        color="#9b59b6",
        production_stage="BLOCK_ASSEMBLY",
    ),
    "block_assembly_hall_2": GeoFacility(
        center=(35.5085, 129.4155),
        polygon=[
            (35.5095, 129.4140),
            (35.5095, 129.4170),
            (35.5075, 129.4170),
            (35.5075, 129.4140),
        ],
        label="Block Assembly Hall 2",
        color="#9b59b6",
        production_stage="BLOCK_ASSEMBLY",
    ),
    "block_assembly_hall_3": GeoFacility(
        center=(35.5070, 129.4120),
        polygon=[
            (35.5080, 129.4105),
            (35.5080, 129.4135),
            (35.5060, 129.4135),
            (35.5060, 129.4105),
        ],
        label="Block Assembly Hall 3",
        color="#9b59b6",
        production_stage="BLOCK_ASSEMBLY",
    ),
    "outfitting_shop": GeoFacility(
        center=(35.5060, 129.4150),
        polygon=[
            (35.5070, 129.4135),
            (35.5070, 129.4165),
            (35.5050, 129.4165),
            (35.5050, 129.4135),
        ],
        label="Outfitting Shop",
        color="#1abc9c",
        production_stage="BLOCK_OUTFITTING",
    ),
    "paint_shop": GeoFacility(
        center=(35.5045, 129.4135),
        polygon=[
            (35.5055, 129.4120),
            (35.5055, 129.4150),
            (35.5035, 129.4150),
            (35.5035, 129.4120),
        ],
        label="Paint Shop",
        color="#f39c12",
        production_stage="PAINTING",
    ),
}

# ============================================================================
# ZONE 4: PRE-ERECTION STAGING (East-Central area)
# ============================================================================

PRE_ERECTION_FACILITIES: Dict[str, GeoFacility] = {
    "grand_block_staging_north": GeoFacility(
        center=(35.5055, 129.4180),
        polygon=[
            (35.5065, 129.4165),
            (35.5065, 129.4195),
            (35.5045, 129.4195),
            (35.5045, 129.4165),
        ],
        label="Grand Block Staging (North)",
        color="#34495e",
        production_stage="PRE_ERECTION",
    ),
    "grand_block_staging_south": GeoFacility(
        center=(35.5025, 129.4180),
        polygon=[
            (35.5040, 129.4165),
            (35.5040, 129.4195),
            (35.5010, 129.4195),
            (35.5010, 129.4165),
        ],
        label="Grand Block Staging (South)",
        color="#34495e",
        production_stage="PRE_ERECTION",
    ),
}

# ============================================================================
# ZONE 5: DRY DOCKS (Eastern coastline along Mipo Bay)
# 10 dry docks, varying sizes
# ============================================================================

DRY_DOCKS: Dict[str, GeoFacility] = {
    "dock_1": GeoFacility(
        center=(35.5080, 129.4230),
        polygon=[
            (35.5095, 129.4210),
            (35.5095, 129.4250),
            (35.5065, 129.4250),
            (35.5065, 129.4210),
        ],
        label="Dock 1 (390m LNG)",
        color="#e74c3c",
        production_stage="ERECTION",
    ),
    "dock_2": GeoFacility(
        center=(35.5055, 129.4232),
        polygon=[
            (35.5065, 129.4210),
            (35.5065, 129.4254),
            (35.5045, 129.4254),
            (35.5045, 129.4210),
        ],
        label="Dock 2 (500m)",
        color="#e74c3c",
        production_stage="ERECTION",
    ),
    "dock_3": GeoFacility(
        center=(35.5030, 129.4235),
        polygon=[
            (35.5045, 129.4205),
            (35.5045, 129.4265),
            (35.5015, 129.4265),
            (35.5015, 129.4205),
        ],
        label="Dock 3 (672m MEGA)",
        color="#e74c3c",
        production_stage="ERECTION",
    ),
    "dock_4": GeoFacility(
        center=(35.5005, 129.4230),
        polygon=[
            (35.5015, 129.4215),
            (35.5015, 129.4245),
            (35.4995, 129.4245),
            (35.4995, 129.4215),
        ],
        label="Dock 4 (390m 150kDWT)",
        color="#e74c3c",
        production_stage="ERECTION",
    ),
    "dock_5": GeoFacility(
        center=(35.4988, 129.4225),
        polygon=[
            (35.4998, 129.4215),
            (35.4998, 129.4235),
            (35.4978, 129.4235),
            (35.4978, 129.4215),
        ],
        label="Dock 5 (300m 70kDWT)",
        color="#c0392b",
        production_stage="ERECTION",
    ),
    "dock_6": GeoFacility(
        center=(35.4970, 129.4222),
        polygon=[
            (35.4980, 129.4213),
            (35.4980, 129.4231),
            (35.4960, 129.4231),
            (35.4960, 129.4213),
        ],
        label="Dock 6 (280m Naval)",
        color="#c0392b",
        production_stage="ERECTION",
    ),
    "dock_7": GeoFacility(
        center=(35.4953, 129.4220),
        polygon=[
            (35.4963, 129.4210),
            (35.4963, 129.4230),
            (35.4943, 129.4230),
            (35.4943, 129.4210),
        ],
        label="Dock 7 (260m Naval)",
        color="#c0392b",
        production_stage="ERECTION",
    ),
    "dock_8": GeoFacility(
        center=(35.4935, 129.4222),
        polygon=[
            (35.4948, 129.4208),
            (35.4948, 129.4236),
            (35.4922, 129.4236),
            (35.4922, 129.4208),
        ],
        label="Dock 8 (350m VLCC)",
        color="#922b21",
        production_stage="ERECTION",
    ),
    "dock_9": GeoFacility(
        center=(35.4915, 129.4220),
        polygon=[
            (35.4925, 129.4208),
            (35.4925, 129.4232),
            (35.4905, 129.4232),
            (35.4905, 129.4208),
        ],
        label="Dock 9 (320m VLCC)",
        color="#922b21",
        production_stage="ERECTION",
    ),
    "h_dock": GeoFacility(
        center=(35.4893, 129.4225),
        polygon=[
            (35.4908, 129.4200),
            (35.4908, 129.4250),
            (35.4878, 129.4250),
            (35.4878, 129.4200),
        ],
        label="H-Dock (490m Offshore)",
        color="#8e44ad",
        production_stage="ERECTION",
    ),
}

# ============================================================================
# ZONE 6: OUTFITTING QUAYS (Southeast coastline)
# ============================================================================

OUTFITTING_QUAYS: Dict[str, GeoFacility] = {
    "quay_1": GeoFacility(
        center=(35.5070, 129.4280),
        polygon=[
            (35.5085, 129.4265),
            (35.5085, 129.4295),
            (35.5055, 129.4295),
            (35.5055, 129.4265),
        ],
        label="Outfitting Quay 1",
        color="#1abc9c",
        production_stage="QUAY_OUTFITTING",
    ),
    "quay_2": GeoFacility(
        center=(35.5025, 129.4275),
        polygon=[
            (35.5040, 129.4260),
            (35.5040, 129.4290),
            (35.5010, 129.4290),
            (35.5010, 129.4260),
        ],
        label="Outfitting Quay 2",
        color="#1abc9c",
        production_stage="QUAY_OUTFITTING",
    ),
    "quay_3": GeoFacility(
        center=(35.4980, 129.4270),
        polygon=[
            (35.4995, 129.4255),
            (35.4995, 129.4285),
            (35.4965, 129.4285),
            (35.4965, 129.4255),
        ],
        label="Outfitting Quay 3",
        color="#1abc9c",
        production_stage="QUAY_OUTFITTING",
    ),
}

# ============================================================================
# STAGING AREAS
# ============================================================================

STAGING_AREAS: Dict[str, GeoFacility] = {
    "steel_staging": GeoFacility(
        center=(35.5145, 129.4035),
        polygon=[
            (35.5152, 129.4025),
            (35.5152, 129.4045),
            (35.5138, 129.4045),
            (35.5138, 129.4025),
        ],
        label="Steel Staging",
        color="#95a5a6",
    ),
    "panel_staging": GeoFacility(
        center=(35.5108, 129.4095),
        polygon=[
            (35.5112, 129.4088),
            (35.5112, 129.4102),
            (35.5104, 129.4102),
            (35.5104, 129.4088),
        ],
        label="Panel Staging",
        color="#95a5a6",
    ),
    "block_staging_west": GeoFacility(
        center=(35.5072, 129.4095),
        polygon=[
            (35.5078, 129.4085),
            (35.5078, 129.4105),
            (35.5066, 129.4105),
            (35.5066, 129.4085),
        ],
        label="Block Staging (West)",
        color="#95a5a6",
    ),
    "block_staging_east": GeoFacility(
        center=(35.5055, 129.4165),
        polygon=[
            (35.5060, 129.4155),
            (35.5060, 129.4175),
            (35.5050, 129.4175),
            (35.5050, 129.4155),
        ],
        label="Block Staging (East)",
        color="#95a5a6",
    ),
    "painted_block_storage": GeoFacility(
        center=(35.5038, 129.4160),
        polygon=[
            (35.5045, 129.4150),
            (35.5045, 129.4170),
            (35.5031, 129.4170),
            (35.5031, 129.4150),
        ],
        label="Painted Block Storage",
        color="#95a5a6",
    ),
    "spmt_depot": GeoFacility(
        center=(35.5065, 129.4075),
        polygon=[
            (35.5072, 129.4065),
            (35.5072, 129.4085),
            (35.5058, 129.4085),
            (35.5058, 129.4065),
        ],
        label="SPMT Depot",
        color="#2c3e50",
    ),
}

# ============================================================================
# GOLIATH CRANE POSITIONS
# 9 cranes, 109m tall (36 stories)
# ============================================================================

GOLIATH_CRANES: Dict[str, Tuple[float, float]] = {
    "GC01": (35.5085, 129.4220),  # Dock 1
    "GC02": (35.5075, 129.4220),  # Dock 1
    "GC03": (35.5055, 129.4220),  # Dock 2
    "GC04": (35.5038, 129.4220),  # Dock 3 (mega)
    "GC05": (35.5022, 129.4220),  # Dock 3 (mega)
    "GC06": (35.5005, 129.4220),  # Dock 4
    "GC07": (35.4988, 129.4218),  # Dock 5
    "GC08": (35.4935, 129.4215),  # Dock 8
    "GC09": (35.4893, 129.4215),  # H-Dock
}

# Crane rail definitions (start and end positions along dock)
CRANE_RAILS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "dock_1": {"start": (35.5095, 129.4215), "end": (35.5065, 129.4215)},
    "dock_2": {"start": (35.5065, 129.4215), "end": (35.5045, 129.4215)},
    "dock_3": {"start": (35.5045, 129.4215), "end": (35.5015, 129.4215)},
    "dock_4": {"start": (35.5015, 129.4215), "end": (35.4995, 129.4215)},
    "dock_5": {"start": (35.4998, 129.4215), "end": (35.4978, 129.4215)},
    "dock_8": {"start": (35.4948, 129.4210), "end": (35.4922, 129.4210)},
    "h_dock": {"start": (35.4908, 129.4210), "end": (35.4878, 129.4210)},
}

# Alias for cleaner imports
GOLIATH_CRANE_RAILS = CRANE_RAILS

# ============================================================================
# COMBINED FACILITIES
# ============================================================================

HHI_FACILITIES_GEO: Dict[str, GeoFacility] = {
    **STEEL_PROCESSING_FACILITIES,
    **PANEL_ASSEMBLY_FACILITIES,
    **BLOCK_ASSEMBLY_FACILITIES,
    **PRE_ERECTION_FACILITIES,
    **DRY_DOCKS,
    **OUTFITTING_QUAYS,
    **STAGING_AREAS,
}

# ============================================================================
# PRODUCTION FLOW (facility connections)
# ============================================================================

HHI_PRODUCTION_FLOW: List[Tuple[str, str]] = [
    # Steel processing flow
    ("steel_stockyard", "cutting_shop"),
    ("cutting_shop", "part_fabrication"),
    ("cutting_shop", "flat_panel_line_1"),
    ("cutting_shop", "flat_panel_line_2"),
    ("cutting_shop", "curved_block_shop"),
    # Panel to block assembly
    ("flat_panel_line_1", "block_assembly_hall_1"),
    ("flat_panel_line_1", "block_assembly_hall_2"),
    ("flat_panel_line_2", "block_assembly_hall_2"),
    ("flat_panel_line_2", "block_assembly_hall_3"),
    ("curved_block_shop", "block_assembly_hall_3"),
    # Block assembly to outfitting
    ("block_assembly_hall_1", "outfitting_shop"),
    ("block_assembly_hall_2", "outfitting_shop"),
    ("block_assembly_hall_3", "outfitting_shop"),
    # Outfitting to painting to pre-erection
    ("outfitting_shop", "paint_shop"),
    ("paint_shop", "grand_block_staging_north"),
    ("paint_shop", "grand_block_staging_south"),
    # Pre-erection to docks
    ("grand_block_staging_north", "dock_1"),
    ("grand_block_staging_north", "dock_2"),
    ("grand_block_staging_north", "dock_3"),
    ("grand_block_staging_south", "dock_5"),
    ("grand_block_staging_south", "dock_6"),
    ("grand_block_staging_south", "dock_7"),
]

# Aliases for cleaner imports
HHI_FLOW = HHI_PRODUCTION_FLOW
HHI_SPMT_DEPOT = STAGING_AREAS.get("spmt_depot")

# ============================================================================
# SEA TRIALS AREA (Ulsan Bay / East Sea)
# ============================================================================

SEA_TRIALS_AREA = {
    "center": (35.48, 129.50),
    "label": "Sea Trials (Ulsan Bay)",
}

# Ship departure path: from quay into Mipo Bay (stays within visible map bounds)
# Ships animate along this path as sea_position increases from 0 to 1
# Kept within bounds to stay visible on the map
SHIP_DEPARTURE_PATH = [
    (35.5050, 129.4285),  # Start: at outfitting quay
    (35.5020, 129.4310),  # Waypoint: just clear of quay
    (35.4980, 129.4340),  # Waypoint: entering bay
    (35.4940, 129.4370),  # Waypoint: sea trials area (in bay)
    (35.4910, 129.4400),  # End: delivered (at edge of Mipo Bay, still visible)
]


def get_ship_position(sea_position: float, quay_center: Tuple[float, float] = None) -> Tuple[float, float]:
    """Get interpolated ship position based on sea_position (0-1).

    Parameters
    ----------
    sea_position : float
        Ship's position along departure path (0 = at quay, 1 = at sea)
    quay_center : Tuple[float, float], optional
        Starting quay coordinates. If None, uses default departure path start.

    Returns
    -------
    Tuple[float, float]
        The (lat, lng) position of the ship
    """
    if sea_position <= 0:
        return quay_center or SHIP_DEPARTURE_PATH[0]

    if sea_position >= 1.0:
        return SHIP_DEPARTURE_PATH[-1]

    # Calculate which segment we're on
    n_segments = len(SHIP_DEPARTURE_PATH) - 1
    segment_size = 1.0 / n_segments
    segment_idx = min(int(sea_position / segment_size), n_segments - 1)
    segment_progress = (sea_position - segment_idx * segment_size) / segment_size

    # Interpolate within segment
    start = SHIP_DEPARTURE_PATH[segment_idx]
    end = SHIP_DEPARTURE_PATH[segment_idx + 1]

    lat = start[0] + (end[0] - start[0]) * segment_progress
    lng = start[1] + (end[1] - start[1]) * segment_progress

    return (lat, lng)


# Ship status colors
SHIP_STATUS_COLORS = {
    "in_block_production": "#3498db",  # Blue - building blocks
    "in_erection": "#9b59b6",          # Purple - in dock
    "afloat": "#1abc9c",               # Teal - just launched
    "in_quay_outfitting": "#27ae60",   # Green - outfitting
    "in_sea_trials": "#f39c12",        # Orange - testing
    "delivered": "#2ecc71",            # Bright green - complete!
}

# ============================================================================
# Helper Functions
# ============================================================================

def facility_to_coords(facility_name: str) -> Tuple[float, float]:
    """Get the center coordinates for a facility.

    Parameters
    ----------
    facility_name : str
        The facility identifier

    Returns
    -------
    Tuple[float, float]
        The (lat, lng) coordinates for the facility center
    """
    # Mapping from standard ShipyardEnv names to HHI facility names
    STANDARD_TO_HHI = {
        "cutting": "cutting_shop",
        "panel": "flat_panel_line_1",
        "assembly": "block_assembly_hall_1",
        "outfitting": "outfitting_shop",
        "paint": "paint_shop",
        "pre_erection": "grand_block_staging_north",
        "dock": "dock_1",
        "wip1": "steel_stockyard",
        "wip2": "panel_staging",
    }

    # Normalize the facility name (handle both standard and HHI names)
    normalized_name = facility_name.lower().strip()

    # Handle direct facility name match
    if facility_name in HHI_FACILITIES_GEO:
        return HHI_FACILITIES_GEO[facility_name].center

    # Handle dock_X format
    if facility_name.startswith("dock_"):
        if facility_name in DRY_DOCKS:
            return DRY_DOCKS[facility_name].center

    # Handle simple "dock" name (standard env)
    if normalized_name == "dock":
        return DRY_DOCKS["dock_1"].center

    # Handle quay_X format
    if facility_name.startswith("quay_"):
        if facility_name in OUTFITTING_QUAYS:
            return OUTFITTING_QUAYS[facility_name].center

    # Handle queue_X pattern
    if facility_name.startswith("queue_"):
        base_name = facility_name.replace("queue_", "")
        # Try to map standard name to HHI name
        if base_name in STANDARD_TO_HHI:
            base_name = STANDARD_TO_HHI[base_name]
        if base_name in HHI_FACILITIES_GEO:
            lat, lng = HHI_FACILITIES_GEO[base_name].center
            return (lat - 0.0003, lng - 0.0003)  # Offset for queue

    # Handle waiting_transport_to_X pattern
    if "waiting_transport_to_" in facility_name.lower():
        target = facility_name.lower().replace("waiting_transport_to_", "")
        # Try to map standard name to HHI name
        if target in STANDARD_TO_HHI:
            target = STANDARD_TO_HHI[target]
        if target in HHI_FACILITIES_GEO:
            lat, lng = HHI_FACILITIES_GEO[target].center
            return (lat - 0.0005, lng - 0.0005)

    # Try direct mapping from standard to HHI
    if normalized_name in STANDARD_TO_HHI:
        hhi_name = STANDARD_TO_HHI[normalized_name]
        if hhi_name in HHI_FACILITIES_GEO:
            return HHI_FACILITIES_GEO[hhi_name].center

    # Default to SPMT depot
    return STAGING_AREAS["spmt_depot"].center


def get_goliath_crane_position(crane_id: str, position_pct: float = 0.5) -> Tuple[float, float]:
    """Get the position of a Goliath crane on its rail.

    Parameters
    ----------
    crane_id : str
        Crane identifier (e.g., "GC01")
    position_pct : float
        Position along the rail from 0.0 to 1.0

    Returns
    -------
    Tuple[float, float]
        The (lat, lng) position of the crane
    """
    if crane_id in GOLIATH_CRANES:
        base_pos = GOLIATH_CRANES[crane_id]
        # Find the dock this crane serves
        for dock_name, cranes in [
            ("dock_1", ["GC01", "GC02"]),
            ("dock_2", ["GC03"]),
            ("dock_3", ["GC04", "GC05"]),
            ("dock_4", ["GC06"]),
            ("dock_5", ["GC07"]),
            ("dock_8", ["GC08"]),
            ("h_dock", ["GC09"]),
        ]:
            if crane_id in cranes and dock_name in CRANE_RAILS:
                rail = CRANE_RAILS[dock_name]
                lat = rail["start"][0] + (rail["end"][0] - rail["start"][0]) * position_pct
                lng = rail["start"][1]
                return (lat, lng)
        return base_pos
    return HHI_ULSAN_CENTER


# ============================================================================
# COLOR SCHEMES
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


# ============================================================================
# ZONE LABELS (for map annotation)
# Positioned at zone edges to avoid blocking facilities
# ============================================================================

ZONE_LABELS: Dict[str, Dict[str, Any]] = {
    "steel": {
        "center": (35.5165, 129.4025),  # Northwest corner, away from facilities
        "label": "Steel Processing",
        "color": "#7f8c8d",
    },
    "panel": {
        "center": (35.5130, 129.4070),  # West edge
        "label": "Panel Assembly",
        "color": "#27ae60",
    },
    "block": {
        "center": (35.5100, 129.4095),  # West edge of block zone
        "label": "Block Assembly",
        "color": "#9b59b6",
    },
    "pre_erection": {
        "center": (35.5070, 129.4200),  # East edge, near docks
        "label": "Pre-Erection",
        "color": "#34495e",
    },
    "docks": {
        "center": (35.5100, 129.4255),  # East side, above docks
        "label": "Dry Docks",
        "color": "#e74c3c",
    },
    "quays": {
        "center": (35.5055, 129.4305),  # Far east, at quays
        "label": "Outfitting Quays",
        "color": "#1abc9c",
    },
}


# ============================================================================
# MIPO BAY / SEA OF JAPAN - Water boundary (east of all facilities)
# ============================================================================

MIPO_BAY_POLYGON = [
    # Water area - entirely east of the docks and quays
    (35.5120, 129.4300),  # North
    (35.5120, 129.4450),  # Northeast
    (35.4880, 129.4450),  # Southeast
    (35.4880, 129.4300),  # South
]

SEA_OF_JAPAN_BOUNDS = {
    "north": 35.5120,
    "south": 35.4880,
    "west": 129.4300,  # Starts east of all facilities
    "east": 129.4450,
}

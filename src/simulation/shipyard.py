"""Shipyard graph model.

This module defines the `ShipyardGraph` class, which represents the physical
layout of a shipyard as a directed graph. Nodes correspond to facilities,
staging areas, dock positions, and maintenance bays. Edges model
transportation routes with associated travel times. The graph is used for
shortest path routing and to provide spatial context to the reinforcement
learning agent.

Includes:
- `ShipyardGraph`: Base graph class for single-yard operations
- `HHIShipyardGraph`: HD Hyundai Heavy Industries Ulsan shipyard model
- `DualShipyardGraph`: Electric Boat dual-yard model (legacy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx


class ShipyardGraph:
    """Directed graph representation of a shipyard.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing facility definitions, staging
        areas, dock grid parameters and a transport network adjacency
        structure. See `config/default.yaml` for an example.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.graph = nx.DiGraph()

        # Add facilities and staging areas as nodes
        facilities = config.get("facilities", [])
        staging = config.get("staging_areas", [])
        dock_grid = config.get("dock_grid", {"rows": 0, "cols": 0})
        transport_network: Dict[str, Dict[str, float]] = config.get(
            "transport_network", {}
        )

        # Register facility and staging nodes
        for f in facilities:
            name = f["name"]
            self.graph.add_node(name, type="facility", data=f)
        for s in staging:
            name = s["name"]
            self.graph.add_node(name, type="staging", data=s)

        # Create dock grid nodes (row,col) -> identifier "dock_r_c"
        self.dock_positions: List[str] = []
        rows, cols = dock_grid.get("rows", 0), dock_grid.get("cols", 0)
        for r in range(rows):
            for c in range(cols):
                node_id = f"dock_{r}_{c}"
                self.dock_positions.append(node_id)
                self.graph.add_node(node_id, type="dock", data={"row": r, "col": c})

        # Build edges from transport_network. Unspecified routes will not be reachable.
        for origin, neighbors in transport_network.items():
            for destination, time in neighbors.items():
                self.graph.add_edge(origin, destination, travel_time=float(time))

        # Connect staging areas to the first facility where appropriate
        # This helps ensure that blocks in wip can enter the production flow.
        for staging_area in staging:
            name = staging_area["name"]
            # Connect to the first facility defined, if not explicitly connected
            if facilities:
                first_fac = facilities[0]["name"]
                if not self.graph.has_edge(name, first_fac):
                    self.graph.add_edge(name, first_fac, travel_time=1.0)

        # Optionally connect dock positions to maintenance bay or exit
        # For simplicity, we leave connections from the last facility to dock grid
        # to be defined by the environment during operations.

    def get_travel_time(self, origin: str, destination: str) -> float:
        """Return travel time between two nodes.

        If a direct edge exists, its weight is returned; otherwise a
        shortest‑path length is computed using Dijkstra's algorithm. If
        no path exists, a large sentinel value is returned.
        """
        if self.graph.has_edge(origin, destination):
            return self.graph[origin][destination]["travel_time"]
        # Compute shortest path travel time
        try:
            path_length = nx.shortest_path_length(
                self.graph, origin, destination, weight="travel_time"
            )
            return float(path_length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 1.0  # default travel time for unknown/unconnected locations

    def get_neighbors(self, node: str) -> List[str]:
        """Return a list of adjacent nodes reachable from the given node."""
        return list(self.graph.successors(node))

    def shortest_path(self, origin: str, destination: str) -> Tuple[List[str], float]:
        """Compute the shortest path and travel time between two nodes.

        Returns a tuple `(path, length)`. If no path exists, the path
        will be empty and length will be infinite.
        """
        try:
            path = nx.shortest_path(
                self.graph, origin, destination, weight="travel_time"
            )
            length = nx.shortest_path_length(
                self.graph, origin, destination, weight="travel_time"
            )
            return path, float(length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], 1.0

    def get_all_nodes(self) -> List[str]:
        """Return a list of all node identifiers in the graph."""
        return list(self.graph.nodes())

    def get_node_type(self, node: str) -> Optional[str]:
        """Return the type of a node (facility, staging, dock, etc.)."""
        if node in self.graph.nodes:
            return self.graph.nodes[node].get("type")
        return None

    def get_node_data(self, node: str) -> Optional[Dict[str, Any]]:
        """Return the data dictionary for a node."""
        if node in self.graph.nodes:
            return self.graph.nodes[node].get("data")
        return None


@dataclass
class BargeRoute:
    """Configuration for barge transport between two shipyards.

    Parameters
    ----------
    origin : str
        Starting pier node name (e.g., "quonset_pier").
    destination : str
        Ending pier node name (e.g., "groton_pier").
    transit_time : float
        Transit time in simulation hours (default 36 hours).
    return_time : float
        Return transit time (empty barge, may be faster).
    """
    origin: str = "quonset_pier"
    destination: str = "groton_pier"
    transit_time: float = 36.0  # hours Quonset -> Groton
    return_time: float = 30.0   # hours Groton -> Quonset (lighter, faster)


class DualShipyardGraph:
    """Graph model for Electric Boat dual-yard operations.

    Models two shipyards (Quonset Point and Groton) connected by barge transport.
    Each yard has its own internal graph for routing, and the barge route
    connects them for inter-yard transport.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - quonset: Quonset Point yard configuration
        - groton: Groton yard configuration
        - transport: Barge transport configuration

    Attributes
    ----------
    quonset : ShipyardGraph
        Graph for Quonset Point facility.
    groton : ShipyardGraph
        Graph for Groton facility.
    barge_route : BargeRoute
        Configuration for barge transport between yards.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # Extract yard-specific configs
        quonset_config = config.get("quonset", {})
        groton_config = config.get("groton", {})
        transport_config = config.get("transport", {})

        # Create individual yard graphs
        self.quonset = self._create_yard_graph(quonset_config, "quonset")
        self.groton = self._create_yard_graph(groton_config, "groton")

        # Configure barge route
        self.barge_route = BargeRoute(
            origin=transport_config.get("origin_pier", "quonset_pier"),
            destination=transport_config.get("destination_pier", "groton_pier"),
            transit_time=transport_config.get("transit_time_hours", 36.0),
            return_time=transport_config.get("return_time_hours", 30.0),
        )

        # Track which nodes are piers for barge operations
        self.quonset_pier = "quonset_pier"
        self.groton_pier = "groton_pier"

    def _create_yard_graph(self, yard_config: dict, yard_id: str) -> ShipyardGraph:
        """Create a ShipyardGraph for a single yard."""
        # Ensure the config has the expected structure
        shipyard_config = {
            "facilities": yard_config.get("facilities", []),
            "staging_areas": yard_config.get("staging_areas", []),
            "dock_grid": yard_config.get("dock_grid", {"rows": 0, "cols": 0}),
            "transport_network": yard_config.get("transport_network", {}),
        }
        graph = ShipyardGraph(shipyard_config)
        graph.yard_id = yard_id
        return graph

    def get_yard(self, yard_id: str) -> Optional[ShipyardGraph]:
        """Get the graph for a specific yard."""
        if yard_id == "quonset":
            return self.quonset
        elif yard_id == "groton":
            return self.groton
        return None

    def get_travel_time(
        self, origin: str, destination: str, origin_yard: str, destination_yard: str
    ) -> float:
        """Get travel time between two locations, possibly across yards.

        If both locations are in the same yard, uses internal routing.
        If locations are in different yards, returns barge transit time
        (assuming travel is via barge).
        """
        if origin_yard == destination_yard:
            yard = self.get_yard(origin_yard)
            if yard:
                return yard.get_travel_time(origin, destination)
            return float("inf")
        else:
            # Inter-yard travel requires barge
            if origin_yard == "quonset" and destination_yard == "groton":
                return self.barge_route.transit_time
            elif origin_yard == "groton" and destination_yard == "quonset":
                return self.barge_route.return_time
            return float("inf")

    def get_barge_transit_time(self, direction: str = "to_groton") -> float:
        """Get barge transit time for the specified direction."""
        if direction == "to_groton":
            return self.barge_route.transit_time
        else:
            return self.barge_route.return_time

    def get_all_facilities(self) -> Dict[str, List[str]]:
        """Return all facility names organized by yard."""
        quonset_facilities = [
            node for node in self.quonset.get_all_nodes()
            if self.quonset.get_node_type(node) == "facility"
        ]
        groton_facilities = [
            node for node in self.groton.get_all_nodes()
            if self.groton.get_node_type(node) == "facility"
        ]
        return {
            "quonset": quonset_facilities,
            "groton": groton_facilities,
        }

    def get_all_dock_positions(self) -> Dict[str, List[str]]:
        """Return all dock positions organized by yard."""
        return {
            "quonset": self.quonset.dock_positions,
            "groton": self.groton.dock_positions,
        }


# Default configurations for EB facilities
QUONSET_DEFAULT_CONFIG = {
    "facilities": [
        {"name": "steel_processing", "processing_time_mean": 16.0, "processing_time_std": 4.0, "capacity": 4},
        {"name": "afc_facility", "processing_time_mean": 24.0, "processing_time_std": 6.0, "capacity": 4},
        {"name": "bldg_9a", "processing_time_mean": 20.0, "processing_time_std": 5.0, "capacity": 2},
        {"name": "bldg_9b", "processing_time_mean": 20.0, "processing_time_std": 5.0, "capacity": 2},
        {"name": "bldg_9c", "processing_time_mean": 20.0, "processing_time_std": 5.0, "capacity": 2},
        {"name": "super_module_assembly", "processing_time_mean": 30.0, "processing_time_std": 8.0, "capacity": 3},
        {"name": "quonset_pier", "processing_time_mean": 4.0, "processing_time_std": 1.0, "capacity": 2},
    ],
    "staging_areas": [
        {"name": "quonset_wip1", "capacity": 10},
        {"name": "quonset_wip2", "capacity": 10},
    ],
    "dock_grid": {"rows": 0, "cols": 0},  # Quonset doesn't have dock assembly
    "transport_network": {
        "steel_processing": {"afc_facility": 0.5},
        "afc_facility": {"bldg_9a": 0.5, "bldg_9b": 0.5, "bldg_9c": 0.5},
        "bldg_9a": {"super_module_assembly": 1.0},
        "bldg_9b": {"super_module_assembly": 1.0},
        "bldg_9c": {"super_module_assembly": 1.0},
        "super_module_assembly": {"quonset_pier": 1.5},
        "quonset_wip1": {"steel_processing": 0.5},
        "quonset_wip2": {"afc_facility": 0.5},
    },
}

GROTON_DEFAULT_CONFIG = {
    "facilities": [
        {"name": "groton_pier", "processing_time_mean": 4.0, "processing_time_std": 1.0, "capacity": 2},
        {"name": "land_level_construction", "processing_time_mean": 40.0, "processing_time_std": 10.0, "capacity": 2},
        {"name": "building_600", "processing_time_mean": 60.0, "processing_time_std": 15.0, "capacity": 1},
        {"name": "graving_dock", "processing_time_mean": 20.0, "processing_time_std": 5.0, "capacity": 1},
    ],
    "staging_areas": [
        {"name": "groton_staging", "capacity": 5},
    ],
    "dock_grid": {"rows": 1, "cols": 3},  # Groton has limited final positions
    "transport_network": {
        "groton_pier": {"land_level_construction": 1.0, "groton_staging": 0.5},
        "land_level_construction": {"building_600": 2.0},
        "building_600": {"graving_dock": 1.5},
        "groton_staging": {"land_level_construction": 0.5},
    },
}

EB_DUAL_YARD_DEFAULT_CONFIG = {
    "quonset": QUONSET_DEFAULT_CONFIG,
    "groton": GROTON_DEFAULT_CONFIG,
    "transport": {
        "origin_pier": "quonset_pier",
        "destination_pier": "groton_pier",
        "transit_time_hours": 36.0,
        "return_time_hours": 30.0,
        "barge_capacity": 2,
    },
}


# =============================================================================
# HD Hyundai Heavy Industries (HHI) Ulsan Shipyard Configuration
# =============================================================================

HHI_ULSAN_DEFAULT_CONFIG = {
    "steel_processing": {
        "facilities": [
            {"name": "steel_stockyard", "processing_time_mean": 4.0, "processing_time_std": 1.0, "capacity": 50},
            {"name": "cutting_shop", "processing_time_mean": 8.0, "processing_time_std": 2.0, "capacity": 20},
            {"name": "part_fabrication", "processing_time_mean": 12.0, "processing_time_std": 3.0, "capacity": 30},
        ],
    },
    "panel_assembly": {
        "facilities": [
            {"name": "flat_panel_line_1", "processing_time_mean": 16.0, "processing_time_std": 4.0, "capacity": 15},
            {"name": "flat_panel_line_2", "processing_time_mean": 16.0, "processing_time_std": 4.0, "capacity": 15},
            {"name": "curved_block_shop", "processing_time_mean": 24.0, "processing_time_std": 6.0, "capacity": 8},
        ],
    },
    "block_assembly": {
        "facilities": [
            {"name": "block_assembly_hall_1", "processing_time_mean": 48.0, "processing_time_std": 12.0, "capacity": 10},
            {"name": "block_assembly_hall_2", "processing_time_mean": 48.0, "processing_time_std": 12.0, "capacity": 10},
            {"name": "block_assembly_hall_3", "processing_time_mean": 48.0, "processing_time_std": 12.0, "capacity": 8},
            {"name": "outfitting_shop", "processing_time_mean": 40.0, "processing_time_std": 10.0, "capacity": 12},
            {"name": "paint_shop", "processing_time_mean": 16.0, "processing_time_std": 4.0, "capacity": 8},
        ],
    },
    "pre_erection": {
        "facilities": [
            {"name": "grand_block_staging_north", "processing_time_mean": 24.0, "processing_time_std": 6.0, "capacity": 20},
            {"name": "grand_block_staging_south", "processing_time_mean": 24.0, "processing_time_std": 6.0, "capacity": 20},
        ],
    },
    # Real HHI Ulsan dry dock specifications (ship-technology.com, Wikipedia)
    "dry_docks": [
        {"name": "dock_1", "length_m": 390, "width_m": 80, "cranes": ["GC01", "GC02"]},   # LNG carriers
        {"name": "dock_2", "length_m": 500, "width_m": 80, "depth_m": 12.7, "cranes": ["GC03"]},  # 2 jib cranes
        {"name": "dock_3", "length_m": 672, "width_m": 92, "cranes": ["GC04", "GC05"]},    # Largest, up to 1M DWT
        {"name": "dock_4", "length_m": 390, "width_m": 80, "cranes": ["GC06"]},             # Up to 150K DWT
        {"name": "dock_5", "length_m": 300, "width_m": 68, "cranes": ["GC07"]},             # Up to 70K DWT
        {"name": "dock_6", "length_m": 280, "width_m": 60, "cranes": []},                   # Naval/special
        {"name": "dock_7", "length_m": 260, "width_m": 55, "cranes": []},                   # Naval/special
        {"name": "dock_8", "length_m": 350, "width_m": 65, "cranes": ["GC08"]},             # VLCCs (1996)
        {"name": "dock_9", "length_m": 320, "width_m": 60, "cranes": []},                   # VLCCs (1996)
        {"name": "h_dock", "length_m": 490, "width_m": 115, "depth_m": 13.5, "cranes": ["GC09"]},  # Offshore vessels
    ],
    "outfitting_quays": [
        {"name": "quay_1", "length_m": 400, "capacity": 2},
        {"name": "quay_2", "length_m": 350, "capacity": 2},
        {"name": "quay_3", "length_m": 300, "capacity": 1},
    ],
    "goliath_cranes": [
        {"id": "GC01", "assigned_dock": "dock_1", "capacity_tons": 900, "height_m": 109},
        {"id": "GC02", "assigned_dock": "dock_1", "capacity_tons": 900, "height_m": 109},
        {"id": "GC03", "assigned_dock": "dock_2", "capacity_tons": 900, "height_m": 109},
        {"id": "GC04", "assigned_dock": "dock_3", "capacity_tons": 900, "height_m": 109},
        {"id": "GC05", "assigned_dock": "dock_3", "capacity_tons": 900, "height_m": 109},
        {"id": "GC06", "assigned_dock": "dock_4", "capacity_tons": 600, "height_m": 90},
        {"id": "GC07", "assigned_dock": "dock_5", "capacity_tons": 600, "height_m": 90},
        {"id": "GC08", "assigned_dock": "dock_8", "capacity_tons": 450, "height_m": 75},
        {"id": "GC09", "assigned_dock": "h_dock", "capacity_tons": 900, "height_m": 109},
    ],
    "staging_areas": [
        {"name": "steel_staging", "capacity": 100},
        {"name": "panel_staging", "capacity": 40},
        {"name": "block_staging_west", "capacity": 30},
        {"name": "block_staging_east", "capacity": 30},
        {"name": "painted_block_storage", "capacity": 25},
        {"name": "spmt_depot", "capacity": 32},
    ],
    "transport_network": {
        "steel_stockyard": {"cutting_shop": 0.25, "steel_staging": 0.1},
        "cutting_shop": {"part_fabrication": 0.5, "flat_panel_line_1": 0.75, "flat_panel_line_2": 0.75, "curved_block_shop": 1.0},
        "part_fabrication": {"flat_panel_line_1": 0.5, "flat_panel_line_2": 0.5, "curved_block_shop": 0.75},
        "flat_panel_line_1": {"block_assembly_hall_1": 1.0, "block_assembly_hall_2": 1.0, "panel_staging": 0.25},
        "flat_panel_line_2": {"block_assembly_hall_1": 1.0, "block_assembly_hall_2": 1.0, "block_assembly_hall_3": 1.25, "panel_staging": 0.25},
        "curved_block_shop": {"block_assembly_hall_1": 1.25, "block_assembly_hall_2": 1.25, "block_assembly_hall_3": 1.0, "panel_staging": 0.5},
        "block_assembly_hall_1": {"outfitting_shop": 0.75, "block_staging_west": 0.25},
        "block_assembly_hall_2": {"outfitting_shop": 0.75, "block_staging_west": 0.25},
        "block_assembly_hall_3": {"outfitting_shop": 0.5, "block_staging_west": 0.5},
        "outfitting_shop": {"paint_shop": 0.5, "block_staging_east": 0.5},
        "paint_shop": {"grand_block_staging_north": 0.75, "grand_block_staging_south": 0.75, "painted_block_storage": 0.25},
        "grand_block_staging_north": {"dock_1": 1.0, "dock_2": 1.25, "dock_3": 1.5, "dock_4": 1.75, "dock_5": 2.0},
        "grand_block_staging_south": {"dock_5": 1.0, "dock_6": 1.25, "dock_7": 1.5, "dock_8": 1.75, "dock_9": 2.0, "h_dock": 2.5},
        "spmt_depot": {"steel_stockyard": 0.5, "block_staging_west": 0.5, "grand_block_staging_north": 0.75},
    },
}


class HHIShipyardGraph(ShipyardGraph):
    """Graph model for HD Hyundai Heavy Industries Ulsan shipyard.

    Models the world's largest shipyard with:
    - 10 dry docks along Mipo Bay
    - 9 Goliath cranes (109m tall)
    - Multiple production zones from steel processing to erection

    Parameters
    ----------
    config : dict
        Configuration dictionary containing facility and transport definitions.
    """

    def __init__(self, config: dict) -> None:
        # Flatten facilities from zones into a single list
        # Support both zone-grouped and flat formats
        facilities = []

        # Try zone-grouped format first (HHI style)
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            zone_facs = config.get(zone, {}).get("facilities", [])
            facilities.extend(zone_facs)

        # Fallback to flat format (default.yaml style)
        if not facilities:
            facilities = config.get("facilities", [])

        # Build base config for parent class
        base_config = {
            "facilities": facilities,
            "staging_areas": config.get("staging_areas", []),
            "dock_grid": {"rows": 0, "cols": 0},  # We handle docks separately
            "transport_network": config.get("transport_network", {}),
        }

        super().__init__(base_config)

        # Store HHI-specific configuration
        self.dry_docks = config.get("dry_docks", [])
        self.goliath_cranes = config.get("goliath_cranes", [])
        self.outfitting_quays = config.get("outfitting_quays", [])

        # Add dry dock nodes
        for dock in self.dry_docks:
            dock_name = dock["name"]
            self.graph.add_node(dock_name, type="dry_dock", data=dock)

        # Add quay nodes
        for quay in self.outfitting_quays:
            quay_name = quay["name"]
            self.graph.add_node(quay_name, type="quay", data=quay)

        # Add dock-to-quay connections (post-launch)
        self._add_dock_quay_connections()

    def _add_dock_quay_connections(self) -> None:
        """Add edges from dry docks to outfitting quays."""
        dock_quay_map = {
            "dock_1": ("quay_1", 0.5),
            "dock_2": ("quay_1", 0.75),
            "dock_3": ("quay_1", 1.0),
            "dock_4": ("quay_2", 0.5),
            "dock_5": ("quay_2", 0.75),
            "dock_6": ("quay_2", 1.0),
            "dock_7": ("quay_3", 0.5),
            "dock_8": ("quay_3", 0.75),
            "dock_9": ("quay_3", 1.0),
            "h_dock": ("quay_3", 1.25),
        }
        for dock, (quay, time) in dock_quay_map.items():
            if self.graph.has_node(dock) and self.graph.has_node(quay):
                self.graph.add_edge(dock, quay, travel_time=time)

    def get_docks_for_ship_type(self, ship_type: str) -> List[str]:
        """Return list of docks suitable for a given ship type."""
        suitable_docks = []
        type_requirements = {
            "lng_carrier": 390,  # Minimum dock length in meters
            "vlcc": 450,
            "container": 350,
            "tanker": 300,
            "bulk": 280,
        }
        min_length = type_requirements.get(ship_type, 300)

        for dock in self.dry_docks:
            if dock.get("length_m", 0) >= min_length:
                suitable_docks.append(dock["name"])

        return suitable_docks

    def get_cranes_for_dock(self, dock_name: str) -> List[str]:
        """Return list of Goliath crane IDs assigned to a dock."""
        for dock in self.dry_docks:
            if dock["name"] == dock_name:
                return dock.get("cranes", [])
        return []

    def get_all_facilities(self) -> List[str]:
        """Return all facility names."""
        return [
            node for node in self.get_all_nodes()
            if self.get_node_type(node) == "facility"
        ]

    def get_all_docks(self) -> List[str]:
        """Return all dry dock names."""
        return [dock["name"] for dock in self.dry_docks]

    def get_all_quays(self) -> List[str]:
        """Return all outfitting quay names."""
        return [quay["name"] for quay in self.outfitting_quays]
"""Shipyard graph model.

This module defines the `ShipyardGraph` class, which represents the physical
layout of a shipyard as a directed graph. Nodes correspond to facilities,
staging areas, dock positions, and maintenance bays. Edges model
transportation routes with associated travel times. The graph is used for
shortest path routing and to provide spatial context to the reinforcement
learning agent.

Also includes `DualShipyardGraph` for modeling the Electric Boat dual-yard
workflow between Quonset Point (RI) and Groton (CT).
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
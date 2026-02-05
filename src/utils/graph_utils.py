"""Graph utility functions.

This module provides helper functions to convert shipyard layouts into
graph data structures and to compute shortest paths or distance
matrices. While the environment handles most graph operations internally,
these functions can be used in analyses or baselines.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx


def build_nx_graph(shipyard_config: Dict) -> nx.DiGraph:
    """Build a NetworkX graph from shipyard configuration."""
    from ..simulation.shipyard import ShipyardGraph
    sg = ShipyardGraph(shipyard_config)
    return sg.graph


def compute_distance_matrix(graph: nx.DiGraph, nodes: List[str]) -> Dict[Tuple[str, str], float]:
    """Compute all‑pairs shortest path lengths between given nodes."""
    dist = {}
    for u in nodes:
        lengths = nx.single_source_dijkstra_path_length(graph, u, weight="travel_time")
        for v in nodes:
            dist[(u, v)] = float(lengths.get(v, float("inf")))
    return dist
"""Block dependency graph visualization.

This module provides functions to build Plotly figures showing block/module
precedence constraints as a directed graph.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import plotly.graph_objects as go

from .map_coordinates import COLORS


def _compute_hierarchical_layout(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    node_status: Dict[str, str],
) -> Dict[str, Tuple[float, float]]:
    """Compute hierarchical left-to-right layout positions for nodes.

    Uses a simple layering approach based on topological depth.
    """
    # Build adjacency list
    children = defaultdict(list)
    parents = defaultdict(list)
    for src, dst in edges:
        children[src].append(dst)
        parents[dst].append(src)

    # Find root nodes (no predecessors)
    roots = [n for n in nodes if not parents[n]]
    if not roots:
        roots = nodes[:1] if nodes else []

    # Compute depth for each node (BFS from roots)
    depth = {}
    queue = [(r, 0) for r in roots]
    visited = set()

    while queue:
        node, d = queue.pop(0)
        if node in visited:
            depth[node] = max(depth.get(node, 0), d)
            continue
        visited.add(node)
        depth[node] = d
        for child in children[node]:
            queue.append((child, d + 1))

    # Assign remaining nodes without depth
    max_depth = max(depth.values()) if depth else 0
    for n in nodes:
        if n not in depth:
            depth[n] = max_depth + 1

    # Group nodes by depth
    layers = defaultdict(list)
    for n, d in depth.items():
        layers[d].append(n)

    # Compute positions
    positions = {}
    x_spacing = 120
    y_spacing = 80

    for d, layer_nodes in layers.items():
        x = d * x_spacing + 50
        n_nodes = len(layer_nodes)
        for i, node in enumerate(layer_nodes):
            y = (i - (n_nodes - 1) / 2) * y_spacing + 300
            positions[node] = (x, y)

    return positions


def build_dependency_graph(
    blocks: List[Dict[str, Any]],
    selected_block: Optional[str] = None,
    show_completed: bool = True,
) -> go.Figure:
    """Build a Plotly figure showing block dependency relationships.

    Parameters
    ----------
    blocks : list of dict
        Block data including id, status, predecessors.
    selected_block : str, optional
        If provided, highlight this block and its dependencies.
    show_completed : bool
        If False, hide completed blocks.

    Returns
    -------
    go.Figure
        The dependency graph figure.
    """
    fig = go.Figure()

    # Filter blocks
    if not show_completed:
        blocks = [b for b in blocks if b.get("status") != "placed_on_dock"]

    if not blocks:
        fig.update_layout(
            annotations=[dict(text="No dependency data available",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=16, color="#95a5a6"))],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=400, margin=dict(t=40, b=20, l=20, r=20),
        )
        return fig

    # Build node and edge lists
    nodes = [b["id"] for b in blocks]
    node_data = {b["id"]: b for b in blocks}
    node_status = {b["id"]: b.get("status", "waiting") for b in blocks}

    edges = []
    for b in blocks:
        for pred in b.get("predecessors", []):
            if pred in node_data:
                edges.append((pred, b["id"]))

    # Compute layout
    positions = _compute_hierarchical_layout(nodes, edges, node_status)

    # Determine which nodes to highlight
    highlight_nodes = set()
    if selected_block and selected_block in node_data:
        # Find all ancestors and descendants
        def find_ancestors(node, visited=None):
            if visited is None:
                visited = set()
            if node in visited:
                return visited
            visited.add(node)
            for pred in node_data.get(node, {}).get("predecessors", []):
                if pred in node_data:
                    find_ancestors(pred, visited)
            return visited

        def find_descendants(node, visited=None):
            if visited is None:
                visited = set()
            if node in visited:
                return visited
            visited.add(node)
            for b in blocks:
                if node in b.get("predecessors", []):
                    find_descendants(b["id"], visited)
            return visited

        highlight_nodes = find_ancestors(selected_block) | find_descendants(selected_block)

    # Draw edges first
    edge_x = []
    edge_y = []
    for src, dst in edges:
        if src in positions and dst in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#bdc3c7", width=1.5),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Add arrow heads
    for src, dst in edges:
        if src in positions and dst in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            # Arrow at 80% of the way
            ax = x0 + 0.85 * (x1 - x0)
            ay = y0 + 0.85 * (y1 - y0)
            fig.add_annotation(
                x=ax, y=ay,
                ax=x0 + 0.75 * (x1 - x0), ay=y0 + 0.75 * (y1 - y0),
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=1.5,
                arrowcolor="#95a5a6",
            )

    # Status colors
    status_colors = {
        "waiting": "#95a5a6",
        "in_process": COLORS["accent"],
        "in_transit": COLORS["warning"],
        "at_staging": "#7f8c8d",
        "at_pre_erection": COLORS["purple"],
        "placed_on_dock": COLORS["success"],
        "on_barge": COLORS["teal"],
        "awaiting_barge": COLORS["orange"],
    }

    # Draw nodes by status
    for status, color in status_colors.items():
        status_nodes = [n for n in nodes if node_status.get(n) == status]
        if not status_nodes:
            continue

        x_coords = [positions[n][0] for n in status_nodes if n in positions]
        y_coords = [positions[n][1] for n in status_nodes if n in positions]
        texts = status_nodes
        hovers = []
        sizes = []

        for n in status_nodes:
            if n not in positions:
                continue
            b = node_data[n]
            preds = b.get("predecessors", [])
            hover = (
                f"<b>{n}</b><br>"
                f"Status: {status}<br>"
                f"Stage: {b.get('current_stage', '?')}<br>"
                f"Completion: {b.get('completion_pct', 0):.0f}%<br>"
                f"Predecessors: {len(preds)}"
            )
            hovers.append(hover)

            # Larger size if highlighted
            if highlight_nodes and n in highlight_nodes:
                sizes.append(35)
            else:
                sizes.append(25)

        # Adjust color for non-highlighted nodes when selection active
        node_colors = []
        for i, n in enumerate(status_nodes):
            if n not in positions:
                continue
            if highlight_nodes and n not in highlight_nodes:
                node_colors.append("#e0e0e0")
            else:
                node_colors.append(color)

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=node_colors,
                line=dict(color="white", width=2),
            ),
            text=texts,
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            name=status.replace("_", " ").title(),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
        ))

    # Layout
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    x_range = [min(all_x) - 50, max(all_x) + 50] if all_x else [0, 100]
    y_range = [min(all_y) - 50, max(all_y) + 50] if all_y else [0, 100]

    fig.update_layout(
        title=dict(
            text="Block Dependency Graph",
            font=dict(size=16, color=COLORS["primary"]),
        ),
        height=500,
        margin=dict(t=60, b=40, l=40, r=40),
        xaxis=dict(
            range=x_range,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        hovermode="closest",
        template="plotly_white",
    )

    # Add flow direction annotation
    fig.add_annotation(
        text="← Predecessors | Successors →",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=11, color="#95a5a6"),
    )

    return fig


def build_critical_path_view(
    blocks: List[Dict[str, Any]],
) -> go.Figure:
    """Build a view highlighting the critical path through the dependency graph.

    The critical path is the longest chain from start to completion.
    """
    # Find the critical path (longest path to any incomplete block)
    node_data = {b["id"]: b for b in blocks}

    # Compute path lengths using dynamic programming
    def longest_path_to(node, memo=None):
        if memo is None:
            memo = {}
        if node in memo:
            return memo[node]

        preds = node_data.get(node, {}).get("predecessors", [])
        if not preds:
            memo[node] = (1, [node])
            return memo[node]

        best_length = 0
        best_path = []
        for pred in preds:
            if pred in node_data:
                length, path = longest_path_to(pred, memo)
                if length > best_length:
                    best_length = length
                    best_path = path

        memo[node] = (best_length + 1, best_path + [node])
        return memo[node]

    # Find the node with longest path (critical end point)
    incomplete = [b["id"] for b in blocks if b.get("status") != "placed_on_dock"]
    if not incomplete:
        incomplete = [b["id"] for b in blocks]

    max_length = 0
    critical_path = []
    for node in incomplete:
        length, path = longest_path_to(node)
        if length > max_length:
            max_length = length
            critical_path = path

    # Build the graph with critical path highlighted
    fig = build_dependency_graph(blocks, selected_block=critical_path[-1] if critical_path else None)

    # Update title
    fig.update_layout(
        title=dict(
            text=f"Critical Path ({max_length} blocks)",
            font=dict(size=16, color=COLORS["primary"]),
        ),
    )

    return fig

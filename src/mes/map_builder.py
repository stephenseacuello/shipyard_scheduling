"""Map figure builders for shipyard visualization.

This module provides functions to build Plotly figures for the Quonset Point
and Groton shipyard maps, including facilities, equipment positions, and
status overlays.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
import plotly.graph_objects as go

from .map_coordinates import (
    QUONSET_FACILITIES, QUONSET_FLOW_ARROWS, QUONSET_CANVAS,
    QUONSET_SPMT_DEPOT, QUONSET_BARGE_POSITION,
    GROTON_FACILITIES, GROTON_FLOW_ARROWS, GROTON_CANVAS,
    GROTON_DOCK_GRID, GROTON_CRANE_RAIL, GROTON_SPMT_DEPOT, GROTON_BARGE_POSITION,
    node_to_coords, COLORS, STATUS_COLORS, health_to_color, FacilityCoord,
)


CHART_TEMPLATE = "plotly_white"


def _empty_map(msg: str = "No data available", height: int = 500) -> go.Figure:
    """Create an empty map figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        template=CHART_TEMPLATE,
        annotations=[dict(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16, color="#95a5a6"))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=height, margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def _create_base_layout(canvas: dict, title: str, height: int = 550) -> dict:
    """Create the base layout configuration for a map figure."""
    return dict(
        template=CHART_TEMPLATE,
        title=dict(text=title, font=dict(size=16, color=COLORS["primary"])),
        height=height,
        margin=dict(t=60, b=40, l=40, r=40),
        xaxis=dict(
            range=[0, canvas["width"]],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[canvas["height"], 0],  # Inverted for top-down view
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
    )


def _add_facilities(fig: go.Figure, facilities: Dict[str, FacilityCoord],
                    queue_data: Optional[Dict[str, int]] = None) -> None:
    """Add facility rectangles and labels to the figure."""
    shapes = []
    annotations = []

    for name, fac in facilities.items():
        # Facility rectangle
        shapes.append(dict(
            type="rect",
            x0=fac.x, y0=fac.y,
            x1=fac.x + fac.width, y1=fac.y + fac.height,
            fillcolor=fac.color + "33",  # Add transparency
            line=dict(color=fac.color, width=2),
        ))

        # Facility label
        annotations.append(dict(
            x=fac.x + fac.width / 2,
            y=fac.y + fac.height / 2,
            text=fac.label,
            showarrow=False,
            font=dict(size=9, color=COLORS["dark"]),
        ))

        # Queue indicator if data available
        if queue_data and name in queue_data:
            queue_count = queue_data[name]
            color = COLORS["warning"] if queue_count > 2 else COLORS["primary"]
            annotations.append(dict(
                x=fac.x + fac.width - 10,
                y=fac.y + 10,
                text=f"Q:{queue_count}",
                showarrow=False,
                font=dict(size=10, color=color),
                bgcolor="white",
                bordercolor=color,
                borderwidth=1,
                borderpad=2,
            ))

    fig.update_layout(shapes=shapes, annotations=annotations)


def _add_flow_arrows(fig: go.Figure, facilities: Dict[str, FacilityCoord],
                     arrows: List[tuple]) -> None:
    """Add flow arrows between facilities."""
    for from_fac, to_fac in arrows:
        if from_fac not in facilities or to_fac not in facilities:
            continue

        f1, f2 = facilities[from_fac], facilities[to_fac]

        # Calculate arrow endpoints
        x0 = f1.x + f1.width
        y0 = f1.y + f1.height / 2
        x1 = f2.x
        y1 = f2.y + f2.height / 2

        # Draw arrow line
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color="#bdc3c7", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))


def _add_spmts(fig: go.Figure, spmts: List[Dict[str, Any]], yard: str,
               show_health: bool = False) -> None:
    """Add SPMT markers to the figure."""
    if not spmts:
        return

    x_coords = []
    y_coords = []
    colors = []
    texts = []
    hovers = []

    for spmt in spmts:
        location = spmt.get("current_location", "")
        x, y = node_to_coords(location, yard)
        # Offset SPMTs slightly to avoid overlap
        idx = int(spmt["id"].replace("spmt_", "").replace("SPMT-", "")) if "spmt" in spmt["id"].lower() else 0
        x += (idx % 3) * 25 - 25
        y += 50 + (idx // 3) * 20

        x_coords.append(x)
        y_coords.append(y)

        status = spmt.get("status", "idle")
        if show_health:
            min_health = min(
                spmt.get("health_hydraulic", 100),
                spmt.get("health_tires", 100),
                spmt.get("health_engine", 100)
            )
            colors.append(health_to_color(min_health))
        else:
            colors.append(STATUS_COLORS.get(status, "#95a5a6"))

        texts.append(spmt["id"])
        load = spmt.get("load", "-")
        hovers.append(
            f"<b>{spmt['id']}</b><br>"
            f"Status: {status}<br>"
            f"Location: {location}<br>"
            f"Load: {load}<br>"
            f"Health: H:{spmt.get('health_hydraulic', '-'):.0f} "
            f"T:{spmt.get('health_tires', '-'):.0f} "
            f"E:{spmt.get('health_engine', '-'):.0f}"
        )

    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode="markers+text",
        marker=dict(size=20, color=colors, symbol="diamond", line=dict(color="white", width=1)),
        text=texts,
        textposition="top center",
        textfont=dict(size=9),
        name="SPMTs",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
    ))


def _add_cranes(fig: go.Figure, cranes: List[Dict[str, Any]], yard: str,
                show_health: bool = False) -> None:
    """Add crane markers to the figure (Groton only has crane rail)."""
    if not cranes or yard != "groton":
        return

    rail = GROTON_CRANE_RAIL

    # Draw crane rail
    fig.add_trace(go.Scatter(
        x=[rail["x_start"], rail["x_end"]],
        y=[rail["y"], rail["y"]],
        mode="lines",
        line=dict(color=COLORS["dark"], width=6),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Add crane markers on rail
    x_coords = []
    y_coords = []
    colors = []
    texts = []
    hovers = []

    rail_length = rail["x_end"] - rail["x_start"]

    for crane in cranes:
        # Position crane along rail based on position_on_rail (0-100)
        pos = crane.get("position_on_rail", 50)
        x = rail["x_start"] + (pos / 100) * rail_length
        y = rail["y"]

        x_coords.append(x)
        y_coords.append(y)

        status = crane.get("status", "idle")
        if show_health:
            min_health = min(
                crane.get("health_cable", 100),
                crane.get("health_motor", 100)
            )
            colors.append(health_to_color(min_health))
        else:
            colors.append(STATUS_COLORS.get(status, "#95a5a6"))

        texts.append(crane["id"])
        hovers.append(
            f"<b>{crane['id']}</b><br>"
            f"Status: {status}<br>"
            f"Position: {pos:.0f}%<br>"
            f"Health: Cable:{crane.get('health_cable', '-'):.0f} "
            f"Motor:{crane.get('health_motor', '-'):.0f}"
        )

    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode="markers+text",
        marker=dict(size=25, color=colors, symbol="triangle-up", line=dict(color="white", width=1)),
        text=texts,
        textposition="top center",
        textfont=dict(size=9),
        name="Cranes",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
    ))


def _add_barge(fig: go.Figure, barge: Optional[Dict[str, Any]], yard: str) -> None:
    """Add barge marker if present at this yard."""
    if not barge:
        return

    location = barge.get("current_location", "")

    # Determine if barge is at this yard
    if yard == "quonset" and location == "quonset_pier":
        x, y = QUONSET_BARGE_POSITION
    elif yard == "groton" and location == "groton_pier":
        x, y = GROTON_BARGE_POSITION
    else:
        return

    cargo_count = len(barge.get("cargo", []))
    status = barge.get("status", "idle")

    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode="markers+text",
        marker=dict(
            size=40,
            color=COLORS["teal"] if cargo_count > 0 else COLORS["gray"],
            symbol="square",
            line=dict(color="white", width=2)
        ),
        text=["HOLLAND"],
        textposition="middle center",
        textfont=dict(size=8, color="white"),
        name="Barge",
        hovertemplate=(
            f"<b>Holland Barge</b><br>"
            f"Status: {status}<br>"
            f"Cargo: {cargo_count} modules<br>"
            f"Location: {location}<extra></extra>"
        ),
    ))


def _add_dock_grid(fig: go.Figure, blocks: List[Dict[str, Any]]) -> None:
    """Add dock grid for Groton (final assembly positions)."""
    grid = GROTON_DOCK_GRID
    shapes = []

    # Create placed blocks lookup
    placed_blocks = {}
    for b in blocks:
        if b.get("dock_row") is not None and b.get("dock_col") is not None:
            key = (b["dock_row"], b["dock_col"])
            placed_blocks[key] = b

    # Draw grid cells
    for row in range(grid["rows"]):
        for col in range(grid["cols"]):
            x = grid["origin"][0] + col * grid["cell_width"]
            y = grid["origin"][1] + row * grid["cell_height"]

            # Check if a block is placed here
            block = placed_blocks.get((row, col))
            if block:
                fill_color = COLORS["success"] + "66"
                line_color = COLORS["success"]
            else:
                fill_color = "#f8f9fa"
                line_color = "#bdc3c7"

            shapes.append(dict(
                type="rect",
                x0=x, y0=y,
                x1=x + grid["cell_width"], y1=y + grid["cell_height"],
                fillcolor=fill_color,
                line=dict(color=line_color, width=1),
            ))

    # Update shapes (append to existing)
    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    fig.update_layout(shapes=existing_shapes + shapes)

    # Add block labels for placed blocks
    for (row, col), block in placed_blocks.items():
        x = grid["origin"][0] + col * grid["cell_width"] + grid["cell_width"] / 2
        y = grid["origin"][1] + row * grid["cell_height"] + grid["cell_height"] / 2

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="text",
            text=[block.get("id", "?")],
            textfont=dict(size=10, color=COLORS["dark"]),
            showlegend=False,
            hovertemplate=f"<b>{block.get('id', '?')}</b><br>Dock position: ({row},{col})<extra></extra>",
        ))


def build_quonset_map(
    map_data: Dict[str, Any],
    show_health: bool = False,
    show_queues: bool = True,
) -> go.Figure:
    """Build the Plotly figure for the Quonset Point shipyard map.

    Parameters
    ----------
    map_data : dict
        Dictionary containing:
        - spmts: List of SPMT status dicts
        - blocks: List of block status dicts (super modules)
        - barge: Barge status dict
        - queue_depths: Dict mapping facility -> queue count
    show_health : bool
        If True, color equipment by health instead of status.
    show_queues : bool
        If True, show queue depth indicators on facilities.

    Returns
    -------
    go.Figure
        The Plotly figure for the Quonset map.
    """
    fig = go.Figure()

    # Set up layout
    fig.update_layout(**_create_base_layout(
        QUONSET_CANVAS,
        "🔧 QUONSET POINT, RI - Module Fabrication & Outfitting",
        height=550
    ))

    # Add flow arrows first (behind facilities)
    _add_flow_arrows(fig, QUONSET_FACILITIES, QUONSET_FLOW_ARROWS)

    # Add facilities
    queue_data = map_data.get("queue_depths", {}) if show_queues else None
    _add_facilities(fig, QUONSET_FACILITIES, queue_data)

    # Add SPMTs
    spmts = [s for s in map_data.get("spmts", [])
             if s.get("current_location", "").startswith("quonset") or
             s.get("current_location", "") in QUONSET_FACILITIES]
    _add_spmts(fig, spmts, "quonset", show_health)

    # Add barge
    _add_barge(fig, map_data.get("barge"), "quonset")

    return fig


def build_groton_map(
    map_data: Dict[str, Any],
    show_health: bool = False,
    show_queues: bool = True,
) -> go.Figure:
    """Build the Plotly figure for the Groton shipyard map.

    Parameters
    ----------
    map_data : dict
        Dictionary containing:
        - spmts: List of SPMT status dicts
        - cranes: List of crane status dicts
        - blocks: List of block status dicts
        - barge: Barge status dict
        - queue_depths: Dict mapping facility -> queue count

    show_health : bool
        If True, color equipment by health instead of status.
    show_queues : bool
        If True, show queue depth indicators on facilities.

    Returns
    -------
    go.Figure
        The Plotly figure for the Groton map.
    """
    fig = go.Figure()

    # Set up layout
    fig.update_layout(**_create_base_layout(
        GROTON_CANVAS,
        "🚢 GROTON, CT - Final Assembly & Launch",
        height=550
    ))

    # Add flow arrows first
    _add_flow_arrows(fig, GROTON_FACILITIES, GROTON_FLOW_ARROWS)

    # Add facilities
    queue_data = map_data.get("queue_depths", {}) if show_queues else None
    _add_facilities(fig, GROTON_FACILITIES, queue_data)

    # Add dock grid
    _add_dock_grid(fig, map_data.get("blocks", []))

    # Add cranes
    _add_cranes(fig, map_data.get("cranes", []), "groton", show_health)

    # Add SPMTs
    spmts = [s for s in map_data.get("spmts", [])
             if s.get("current_location", "").startswith("groton") or
             s.get("current_location", "") in GROTON_FACILITIES]
    _add_spmts(fig, spmts, "groton", show_health)

    # Add barge
    _add_barge(fig, map_data.get("barge"), "groton")

    return fig


def build_transit_map(
    barge_data: Optional[Dict[str, Any]],
    transit_progress: float = 0.0,
) -> go.Figure:
    """Build a simple transit view showing barge between yards.

    Parameters
    ----------
    barge_data : dict or None
        Barge status dict.
    transit_progress : float
        Progress through transit (0.0 to 1.0).

    Returns
    -------
    go.Figure
        The Plotly figure for the transit view.
    """
    fig = go.Figure()

    # Layout
    fig.update_layout(
        template=CHART_TEMPLATE,
        title=dict(text="🌊 BARGE TRANSIT - Narragansett Bay → Thames River",
                   font=dict(size=16, color=COLORS["primary"])),
        height=300,
        margin=dict(t=60, b=40, l=40, r=40),
        xaxis=dict(range=[0, 100], showgrid=False, visible=False),
        yaxis=dict(range=[0, 50], showgrid=False, visible=False),
        showlegend=False,
    )

    # Draw route line
    fig.add_trace(go.Scatter(
        x=[10, 90], y=[25, 25],
        mode="lines",
        line=dict(color=COLORS["accent"], width=4, dash="dash"),
        hoverinfo="skip",
    ))

    # Quonset marker
    fig.add_trace(go.Scatter(
        x=[10], y=[25],
        mode="markers+text",
        marker=dict(size=30, color=COLORS["teal"], symbol="square"),
        text=["QUONSET"],
        textposition="bottom center",
        hoverinfo="skip",
    ))

    # Groton marker
    fig.add_trace(go.Scatter(
        x=[90], y=[25],
        mode="markers+text",
        marker=dict(size=30, color=COLORS["purple"], symbol="square"),
        text=["GROTON"],
        textposition="bottom center",
        hoverinfo="skip",
    ))

    # Barge position
    if barge_data:
        status = barge_data.get("status", "idle")
        progress = barge_data.get("transit_progress", transit_progress)

        if "to_groton" in status:
            barge_x = 10 + progress * 80
        elif "to_quonset" in status:
            barge_x = 90 - progress * 80
        elif barge_data.get("current_location") == "quonset_pier":
            barge_x = 10
        else:
            barge_x = 90

        cargo_count = len(barge_data.get("cargo", []))

        fig.add_trace(go.Scatter(
            x=[barge_x], y=[25],
            mode="markers+text",
            marker=dict(
                size=40,
                color=COLORS["warning"] if "transit" in status else COLORS["gray"],
                symbol="diamond",
                line=dict(color="white", width=2)
            ),
            text=["🚢"],
            textfont=dict(size=20),
            hovertemplate=(
                f"<b>Holland Barge</b><br>"
                f"Status: {status}<br>"
                f"Progress: {progress*100:.0f}%<br>"
                f"Cargo: {cargo_count} modules<extra></extra>"
            ),
        ))

    return fig


def build_split_view(
    quonset_data: Dict[str, Any],
    groton_data: Dict[str, Any],
    show_health: bool = False,
) -> go.Figure:
    """Build a split view showing both Quonset and Groton side by side.

    Note: For better layout, consider using separate figures in the dashboard
    rather than subplots, as subplots don't handle shapes well across domains.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["QUONSET POINT, RI", "GROTON, CT"],
        horizontal_spacing=0.05,
    )

    # For now, return a placeholder - split view is better done at the layout level
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=600,
        title=dict(text="Electric Boat Dual Shipyard Overview",
                   font=dict(size=16, color=COLORS["primary"])),
    )

    fig.add_annotation(
        text="Split view: Use individual tabs for detailed maps",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="#95a5a6"),
    )

    return fig

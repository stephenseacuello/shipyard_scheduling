"""Map figure builders for HHI Ulsan shipyard visualization.

This module provides functions to build Plotly figures for the HD Hyundai Heavy
Industries Ulsan shipyard map, including facilities, equipment positions, dry
docks, Goliath cranes, and status overlays.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
import plotly.graph_objects as go

from .map_coordinates import (
    HHI_FACILITIES, HHI_FLOW_ARROWS, HHI_CANVAS,
    DRY_DOCKS, GOLIATH_CRANE_POSITIONS, CRANE_RAILS,
    STAGING_AREAS, STEEL_PROCESSING, PANEL_ASSEMBLY, BLOCK_ASSEMBLY,
    PRE_ERECTION, OUTFITTING_QUAYS,
    node_to_coords, COLORS, STATUS_COLORS, health_to_color,
    BLOCK_TYPE_COLORS, block_type_to_color, FacilityCoord,
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


def _create_base_layout(canvas: dict, title: str, height: int = 600) -> dict:
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
            font=dict(size=8, color=COLORS["dark"]),
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
                font=dict(size=9, color=color),
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
            line=dict(color="#bdc3c7", width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))


def _add_spmts(fig: go.Figure, spmts: List[Dict[str, Any]],
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
        x, y = node_to_coords(location)
        # Offset SPMTs slightly to avoid overlap
        idx = int(spmt["id"].replace("spmt_", "").replace("SPMT-", "")) if "spmt" in spmt["id"].lower() else 0
        x += (idx % 4) * 20 - 30
        y += 30 + (idx // 4) * 15

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
        marker=dict(size=16, color=colors, symbol="diamond", line=dict(color="white", width=1)),
        text=texts,
        textposition="top center",
        textfont=dict(size=8),
        name="SPMTs",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
    ))


def _add_goliath_cranes(fig: go.Figure, cranes: List[Dict[str, Any]],
                        show_health: bool = False) -> None:
    """Add Goliath crane markers on their dock rails."""
    if not cranes:
        return

    # Draw crane rails for each dock
    rail_shapes = []
    for dock_id, rail in CRANE_RAILS.items():
        rail_shapes.append(dict(
            type="line",
            x0=rail["x_start"], y0=rail["y"],
            x1=rail["x_end"], y1=rail["y"],
            line=dict(color=COLORS["dark"], width=4),
        ))

    # Update shapes (append to existing)
    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    fig.update_layout(shapes=existing_shapes + rail_shapes)

    # Add crane markers
    x_coords = []
    y_coords = []
    colors = []
    texts = []
    hovers = []

    for crane in cranes:
        crane_id = crane.get("id", "")

        # Get position from GOLIATH_CRANE_POSITIONS or calculate from rail
        if crane_id in GOLIATH_CRANE_POSITIONS:
            x, y = GOLIATH_CRANE_POSITIONS[crane_id]
        else:
            # Default position if not found
            x, y = 1000, 300

        # Adjust x based on position_on_rail if available
        assigned_dock = crane.get("assigned_dock", "")
        if assigned_dock in CRANE_RAILS:
            rail = CRANE_RAILS[assigned_dock]
            pos = crane.get("position_on_rail", 50)
            x = rail["x_start"] + (pos / 100) * (rail["x_end"] - rail["x_start"])
            y = rail["y"]

        x_coords.append(x)
        y_coords.append(y)

        status = crane.get("status", "idle")
        if show_health:
            min_health = min(
                crane.get("health_hoist", 100),
                crane.get("health_trolley", 100),
                crane.get("health_gantry", 100)
            )
            colors.append(health_to_color(min_health))
        else:
            colors.append(STATUS_COLORS.get(status, "#95a5a6"))

        texts.append(crane_id)
        hovers.append(
            f"<b>{crane_id}</b><br>"
            f"Status: {status}<br>"
            f"Dock: {assigned_dock}<br>"
            f"Capacity: {crane.get('capacity_tons', 900)} tons<br>"
            f"Health: Hoist:{crane.get('health_hoist', '-'):.0f} "
            f"Trolley:{crane.get('health_trolley', '-'):.0f} "
            f"Gantry:{crane.get('health_gantry', '-'):.0f}"
        )

    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode="markers+text",
        marker=dict(size=22, color=colors, symbol="triangle-up", line=dict(color="white", width=1)),
        text=texts,
        textposition="top center",
        textfont=dict(size=8),
        name="Goliath Cranes",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
    ))


def _add_dry_dock_ships(fig: go.Figure, ships: List[Dict[str, Any]],
                        docks: List[Dict[str, Any]]) -> None:
    """Add ship outlines in dry docks showing construction progress."""
    if not ships and not docks:
        return

    # Create dock occupancy lookup
    dock_ships = {}
    for ship in ships:
        dock_id = ship.get("assigned_dock", "")
        if dock_id:
            dock_ships[dock_id] = ship

    shapes = []
    annotations = []

    for dock_name, dock_fac in DRY_DOCKS.items():
        ship = dock_ships.get(dock_name)

        if ship:
            # Ship in dock - show progress fill
            progress = ship.get("erection_progress", 0) / 100.0
            ship_width = dock_fac.width * 0.8
            ship_height = dock_fac.height * 0.7

            ship_x = dock_fac.x + (dock_fac.width - ship_width) / 2
            ship_y = dock_fac.y + (dock_fac.height - ship_height) / 2

            # Ship outline
            shapes.append(dict(
                type="rect",
                x0=ship_x, y0=ship_y,
                x1=ship_x + ship_width, y1=ship_y + ship_height,
                fillcolor=COLORS["gray"] + "44",
                line=dict(color=COLORS["dark"], width=1),
            ))

            # Progress fill
            shapes.append(dict(
                type="rect",
                x0=ship_x, y0=ship_y,
                x1=ship_x + ship_width * progress, y1=ship_y + ship_height,
                fillcolor=COLORS["success"] + "88",
                line=dict(width=0),
            ))

            # Ship name
            annotations.append(dict(
                x=ship_x + ship_width / 2,
                y=ship_y + ship_height / 2,
                text=f"{ship.get('name', ship.get('id', '?'))}<br>{progress*100:.0f}%",
                showarrow=False,
                font=dict(size=8, color=COLORS["dark"]),
            ))

    # Update shapes and annotations
    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    fig.update_layout(
        shapes=existing_shapes + shapes,
        annotations=existing_annotations + annotations
    )


def _add_blocks(fig: go.Figure, blocks: List[Dict[str, Any]]) -> None:
    """Add block markers at their current locations."""
    if not blocks:
        return

    # Group blocks by location
    block_groups: Dict[str, List[Dict]] = {}
    for block in blocks:
        loc = block.get("current_location", "unknown")
        if loc not in block_groups:
            block_groups[loc] = []
        block_groups[loc].append(block)

    x_coords = []
    y_coords = []
    colors = []
    texts = []
    hovers = []

    for location, loc_blocks in block_groups.items():
        base_x, base_y = node_to_coords(location)

        for i, block in enumerate(loc_blocks[:12]):  # Show max 12 blocks per location
            # Arrange in grid
            row = i // 4
            col = i % 4
            x = base_x + col * 12 - 18
            y = base_y - 40 + row * 12

            x_coords.append(x)
            y_coords.append(y)

            block_type = block.get("block_type", "flat_bottom")
            colors.append(block_type_to_color(block_type))

            texts.append("")  # Too crowded for labels

            stage = block.get("current_stage", "unknown")
            hovers.append(
                f"<b>{block.get('id', '?')}</b><br>"
                f"Type: {block_type}<br>"
                f"Stage: {stage}<br>"
                f"Ship: {block.get('ship_id', '-')}<br>"
                f"Weight: {block.get('weight', '-')} tons<br>"
                f"Progress: {block.get('completion_pct', 0):.0f}%"
            )

        # Show overflow indicator if more blocks
        if len(loc_blocks) > 12:
            overflow = len(loc_blocks) - 12
            x_coords.append(base_x + 30)
            y_coords.append(base_y - 50)
            colors.append(COLORS["warning"])
            texts.append(f"+{overflow}")
            hovers.append(f"+{overflow} more blocks at this location")

    if x_coords:
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode="markers+text",
            marker=dict(size=10, color=colors, symbol="square", line=dict(color="white", width=0.5)),
            text=texts,
            textposition="middle right",
            textfont=dict(size=8),
            name="Blocks",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
        ))


def _add_zone_labels(fig: go.Figure) -> None:
    """Add zone labels to the map."""
    zones = [
        (100, 15, "ZONE 1: STEEL PROCESSING"),
        (400, 15, "ZONE 2: PANEL ASSEMBLY"),
        (550, 15, "ZONE 3: BLOCK ASSEMBLY"),
        (720, 50, "ZONE 4: PRE-ERECTION"),
        (980, 10, "ZONE 5: DRY DOCKS"),
        (950, 595, "ZONE 6: OUTFITTING QUAYS"),
    ]

    existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []

    for x, y, text in zones:
        existing_annotations.append(dict(
            x=x, y=y,
            text=f"<b>{text}</b>",
            showarrow=False,
            font=dict(size=9, color=COLORS["dark"]),
            bgcolor="white",
            borderpad=3,
        ))

    fig.update_layout(annotations=existing_annotations)


def _add_coastline(fig: go.Figure) -> None:
    """Add coastline indicator on the right side (Mipo Bay)."""
    # Coastline wave pattern
    fig.add_trace(go.Scatter(
        x=[1100, 1120, 1100, 1120, 1100, 1120, 1100, 1120, 1100],
        y=[0, 100, 200, 300, 400, 500, 600, 700, 800],
        mode="lines",
        line=dict(color=COLORS["accent"], width=3),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Water label
    existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    existing_annotations.append(dict(
        x=1150, y=400,
        text="MIPO BAY",
        showarrow=False,
        font=dict(size=12, color=COLORS["accent"]),
        textangle=-90,
    ))
    fig.update_layout(annotations=existing_annotations)


def build_hhi_map(
    map_data: Dict[str, Any],
    show_health: bool = False,
    show_queues: bool = True,
    show_blocks: bool = True,
) -> go.Figure:
    """Build the Plotly figure for the HHI Ulsan shipyard map.

    Parameters
    ----------
    map_data : dict
        Dictionary containing:
        - spmts: List of SPMT status dicts
        - goliath_cranes: List of Goliath crane status dicts
        - blocks: List of block status dicts (LNG carrier blocks)
        - ships: List of ship status dicts
        - docks: List of dry dock status dicts
        - queue_depths: Dict mapping facility -> queue count
    show_health : bool
        If True, color equipment by health instead of status.
    show_queues : bool
        If True, show queue depth indicators on facilities.
    show_blocks : bool
        If True, show block markers at their locations.

    Returns
    -------
    go.Figure
        The Plotly figure for the HHI Ulsan map.
    """
    fig = go.Figure()

    # Set up layout
    fig.update_layout(**_create_base_layout(
        HHI_CANVAS,
        "HD HYUNDAI HEAVY INDUSTRIES - Ulsan Shipyard (LNG Carrier Production)",
        height=650
    ))

    # Add coastline first (background)
    _add_coastline(fig)

    # Add flow arrows (behind facilities)
    _add_flow_arrows(fig, HHI_FACILITIES, HHI_FLOW_ARROWS)

    # Add facilities
    queue_data = map_data.get("queue_depths", {}) if show_queues else None
    _add_facilities(fig, HHI_FACILITIES, queue_data)

    # Add zone labels
    _add_zone_labels(fig)

    # Add ships in dry docks
    _add_dry_dock_ships(fig, map_data.get("ships", []), map_data.get("docks", []))

    # Add Goliath cranes
    _add_goliath_cranes(fig, map_data.get("goliath_cranes", []), show_health)

    # Add blocks
    if show_blocks:
        _add_blocks(fig, map_data.get("blocks", []))

    # Add SPMTs
    _add_spmts(fig, map_data.get("spmts", []), show_health)

    return fig


def build_production_overview(map_data: Dict[str, Any]) -> go.Figure:
    """Build a production overview showing stage counts and progress.

    Parameters
    ----------
    map_data : dict
        Dictionary containing blocks and ships data.

    Returns
    -------
    go.Figure
        The Plotly figure for production overview.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Blocks by Production Stage", "Ships Under Construction"],
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # HHI Production Stage names (must match HHIProductionStage enum)
    stages = [
        "STEEL_CUTTING", "PART_FABRICATION", "PANEL_ASSEMBLY", "BLOCK_ASSEMBLY",
        "BLOCK_OUTFITTING", "PAINTING", "PRE_ERECTION", "ERECTION",
        "QUAY_OUTFITTING", "SEA_TRIALS", "DELIVERY"
    ]
    # Shorter labels for chart display
    stage_labels = [
        "Steel Cut", "Part Fab", "Panel", "Block Assy",
        "Outfitting", "Painting", "Pre-Erect", "Erection",
        "Quay Out", "Sea Trial", "Delivery"
    ]

    blocks = map_data.get("blocks", [])
    ships = map_data.get("ships", [])

    # Count blocks by stage
    stage_counts = {i: 0 for i in range(11)}
    for block in blocks:
        stage = block.get("current_stage", 0)
        if isinstance(stage, str):
            # Try to find exact match or partial match
            try:
                stage = stages.index(stage)
            except ValueError:
                # Try case-insensitive match
                stage_upper = stage.upper()
                found = False
                for i, s in enumerate(stages):
                    if s == stage_upper or stage_upper in s or s in stage_upper:
                        stage = i
                        found = True
                        break
                if not found:
                    stage = 0
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    # Blocks by stage bar chart
    fig.add_trace(
        go.Bar(
            x=stage_labels,
            y=[stage_counts.get(i, 0) for i in range(11)],
            marker_color=[
                COLORS["primary"], COLORS["primary"], COLORS["success"],
                COLORS["purple"], COLORS["teal"], COLORS["warning"],
                COLORS["dark"], COLORS["danger"], COLORS["teal"],
                COLORS["accent"], COLORS["success"]
            ],
            name="Blocks",
        ),
        row=1, col=1
    )

    # Ships progress
    if ships:
        ship_names = [s.get("name", s.get("id", "?")) for s in ships[:8]]
        ship_progress = [s.get("erection_progress", 0) for s in ships[:8]]

        fig.add_trace(
            go.Bar(
                x=ship_names,
                y=ship_progress,
                marker_color=COLORS["danger"],
                name="Erection Progress (%)",
            ),
            row=1, col=2
        )

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=dict(text="HHI Ulsan Production Overview",
                   font=dict(size=16, color=COLORS["primary"])),
        height=350,
        showlegend=False,
        margin=dict(t=60, b=40, l=40, r=40),
    )

    fig.update_yaxes(title_text="Block Count", row=1, col=1)
    fig.update_yaxes(title_text="Progress (%)", range=[0, 100], row=1, col=2)

    return fig


def build_equipment_health_summary(map_data: Dict[str, Any]) -> go.Figure:
    """Build a health summary for all equipment.

    Parameters
    ----------
    map_data : dict
        Dictionary containing spmts and goliath_cranes data.

    Returns
    -------
    go.Figure
        The Plotly figure for equipment health summary.
    """
    fig = go.Figure()

    spmts = map_data.get("spmts", [])
    cranes = map_data.get("goliath_cranes", [])

    equipment_ids = []
    health_values = []
    colors = []

    # SPMT health (min of components)
    for spmt in spmts[:12]:  # Limit for display
        equipment_ids.append(spmt.get("id", "?"))
        health = min(
            spmt.get("health_hydraulic", 100),
            spmt.get("health_tires", 100),
            spmt.get("health_engine", 100)
        )
        health_values.append(health)
        colors.append(health_to_color(health))

    # Goliath crane health
    for crane in cranes:
        equipment_ids.append(crane.get("id", "?"))
        health = min(
            crane.get("health_hoist", 100),
            crane.get("health_trolley", 100),
            crane.get("health_gantry", 100)
        )
        health_values.append(health)
        colors.append(health_to_color(health))

    fig.add_trace(go.Bar(
        x=equipment_ids,
        y=health_values,
        marker_color=colors,
        name="Health",
    ))

    # Add threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Warning (70%)", annotation_position="bottom right")
    fig.add_hline(y=40, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Critical (40%)", annotation_position="bottom right")

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=dict(text="Equipment Health Status",
                   font=dict(size=16, color=COLORS["primary"])),
        height=300,
        margin=dict(t=60, b=40, l=40, r=40),
        yaxis=dict(title="Health (%)", range=[0, 100]),
        showlegend=False,
    )

    return fig

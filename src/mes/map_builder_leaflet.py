"""Leaflet map builders for HHI Ulsan shipyard visualization.

This module provides functions to build dash-leaflet map components for
the HD Hyundai Heavy Industries Ulsan shipyard map, including facilities,
equipment positions, Goliath cranes, and dry docks on real OpenStreetMap tiles.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
import dash_leaflet as dl
from dash import html

from .map_coordinates_geo import (
    HHI_FACILITIES_GEO, HHI_CENTER, HHI_ZOOM, HHI_BOUNDS,
    HHI_SPMT_DEPOT, HHI_FLOW, DRY_DOCKS, GOLIATH_CRANE_RAILS,
    OUTFITTING_QUAYS, ZONE_LABELS,
    facility_to_coords, COLORS, STATUS_COLORS, health_to_color, GeoFacility,
    BLOCK_TYPE_COLORS, SHIP_STATUS_COLORS, get_ship_position,
)


# OpenStreetMap tile layer URL
OSM_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

# Alternative tile layers
CARTODB_POSITRON = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTODB_DARK = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"

# Satellite view for shipyard
ESRI_SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def _create_facility_polygons(
    facilities: Dict[str, GeoFacility],
    queue_data: Optional[Dict[str, int]] = None,
) -> List[dl.Polygon]:
    """Create polygon overlays for facilities."""
    polygons = []

    for name, fac in facilities.items():
        if not fac.polygon:
            continue

        # Convert polygon to leaflet format (list of [lat, lng])
        positions = [[pt[0], pt[1]] for pt in fac.polygon]

        # Build tooltip content with Dash components
        tooltip_children = [html.Strong(fac.label)]

        # Queue indicator
        if queue_data and name in queue_data:
            queue_count = queue_data[name]
            tooltip_children.extend([
                html.Br(),
                html.Span([
                    "Queue: ",
                    html.Strong(str(queue_count), style={"color": "#e74c3c" if queue_count > 5 else "#27ae60"}),
                    " blocks"
                ])
            ])

        # Production stage info
        if fac.production_stage:
            tooltip_children.extend([
                html.Br(),
                html.Span(f"Stage: {fac.production_stage}", style={"fontSize": "11px", "color": "#7f8c8d"})
            ])

        tooltip_content = html.Div(tooltip_children)

        polygons.append(
            dl.Polygon(
                positions=positions,
                color=fac.color,
                fillColor=fac.color,
                fillOpacity=fac.fill_opacity,
                weight=2,
                children=[dl.Tooltip(tooltip_content)],
            )
        )

    return polygons


def _create_dry_dock_polygons(
    docks: Dict[str, GeoFacility],
    ships: List[Dict[str, Any]],
) -> List:
    """Create dry dock polygons with ship occupancy indicators."""
    elements = []

    # Create ship lookup by dock
    ships_by_dock = {}
    for ship in ships:
        dock_id = ship.get("assigned_dock", "")
        if dock_id:
            ships_by_dock[dock_id] = ship

    for dock_name, dock in docks.items():
        if not dock.polygon:
            continue

        positions = [[pt[0], pt[1]] for pt in dock.polygon]

        # Check if ship is in dock
        ship = ships_by_dock.get(dock_name)

        if ship:
            # Occupied dock
            progress = ship.get("erection_progress", 0)
            ship_name = ship.get("name", ship.get("id", "Unknown"))

            tooltip_content = html.Div([
                html.Strong(dock.label), html.Br(),
                html.Span(f"Ship: {ship_name}", style={"color": COLORS["success"]}), html.Br(),
                html.Div([
                    f"Erection Progress: {progress:.0f}%",
                    html.Div(style={
                        "width": "100%",
                        "height": "6px",
                        "backgroundColor": "#ecf0f1",
                        "borderRadius": "3px",
                        "marginTop": "4px",
                    }, children=[
                        html.Div(style={
                            "width": f"{progress}%",
                            "height": "100%",
                            "backgroundColor": COLORS["success"],
                            "borderRadius": "3px",
                        })
                    ])
                ]),
                html.Br(),
                html.Span(f"Blocks placed: {ship.get('blocks_erected', 0)}/{ship.get('total_blocks', 200)}",
                         style={"fontSize": "11px"}),
            ])

            fill_color = COLORS["success"]
            fill_opacity = 0.4
        else:
            # Empty dock
            tooltip_content = html.Div([
                html.Strong(dock.label), html.Br(),
                html.Span("Available", style={"color": COLORS["gray"]}),
            ])

            fill_color = dock.color
            fill_opacity = 0.2

        elements.append(
            dl.Polygon(
                positions=positions,
                color=dock.color,
                fillColor=fill_color,
                fillOpacity=fill_opacity,
                weight=2,
                children=[dl.Tooltip(tooltip_content)],
            )
        )

    return elements


def _create_outfitting_quay_polygons(
    quays: Dict[str, GeoFacility],
    ships: List[Dict[str, Any]],
) -> List[dl.Polygon]:
    """Create outfitting quay polygons."""
    elements = []

    # Ships at quays
    ships_at_quay = {}
    for ship in ships:
        location = ship.get("current_location", "")
        if "quay" in location:
            ships_at_quay[location] = ship

    for quay_name, quay in quays.items():
        if not quay.polygon:
            continue

        positions = [[pt[0], pt[1]] for pt in quay.polygon]
        ship = ships_at_quay.get(quay_name)

        if ship:
            tooltip_content = html.Div([
                html.Strong(quay.label), html.Br(),
                html.Span(f"Ship: {ship.get('name', '?')}", style={"color": COLORS["teal"]}), html.Br(),
                f"Outfitting progress: {ship.get('outfitting_progress', 0):.0f}%",
            ])
            fill_opacity = 0.5
        else:
            tooltip_content = html.Div([
                html.Strong(quay.label), html.Br(),
                html.Span("Available", style={"color": COLORS["gray"]}),
            ])
            fill_opacity = 0.2

        elements.append(
            dl.Polygon(
                positions=positions,
                color=quay.color,
                fillColor=quay.color,
                fillOpacity=fill_opacity,
                weight=2,
                children=[dl.Tooltip(tooltip_content)],
            )
        )

    return elements


def _create_flow_arrows(
    facilities: Dict[str, GeoFacility],
    flow: List[tuple],
) -> List[dl.Polyline]:
    """Create flow lines between facilities."""
    lines = []

    for from_name, to_name in flow:
        if from_name not in facilities or to_name not in facilities:
            continue

        from_fac = facilities[from_name]
        to_fac = facilities[to_name]

        positions = [list(from_fac.center), list(to_fac.center)]

        lines.append(
            dl.Polyline(
                positions=positions,
                color="#bdc3c7",
                weight=2,
                dashArray="5, 10",
                opacity=0.5,
            )
        )

    return lines


def _create_spmt_markers(
    spmts: List[Dict[str, Any]],
    show_health: bool = False,
) -> List[dl.CircleMarker]:
    """Create SPMT markers."""
    markers = []

    if not spmts:
        return markers

    for i, spmt in enumerate(spmts):
        location = spmt.get("current_location", "")
        lat, lng = facility_to_coords(location)

        # Offset to avoid overlap
        offset_lat = (i // 4) * 0.0002
        offset_lng = (i % 4) * 0.0003 - 0.00045
        lat += offset_lat
        lng += offset_lng

        status = spmt.get("status", "idle")

        if show_health:
            min_health = min(
                spmt.get("health_hydraulic", 100),
                spmt.get("health_tires", 100),
                spmt.get("health_engine", 100),
            )
            color = health_to_color(min_health)
        else:
            color = STATUS_COLORS.get(status, "#95a5a6")

        load_info = spmt.get("load") or "None"
        tooltip_content = html.Div([
            html.Strong(spmt['id']), html.Br(),
            f"Status: {status}", html.Br(),
            f"Location: {location or 'Depot'}", html.Br(),
            f"Load: {load_info}", html.Br(),
            html.Span([
                "Health: ",
                html.Span(f"H:{spmt.get('health_hydraulic', 100):.0f}", style={"color": "#3498db"}),
                " ",
                html.Span(f"T:{spmt.get('health_tires', 100):.0f}", style={"color": "#e67e22"}),
                " ",
                html.Span(f"E:{spmt.get('health_engine', 100):.0f}", style={"color": "#27ae60"}),
            ]),
        ])

        markers.append(
            dl.CircleMarker(
                center=[lat, lng],
                radius=8,
                color="white",
                weight=2,
                fillColor=color,
                fillOpacity=0.9,
                children=[dl.Tooltip(tooltip_content)],
            )
        )

    return markers


def _create_goliath_crane_markers(
    cranes: List[Dict[str, Any]],
    show_health: bool = False,
) -> List:
    """Create Goliath crane markers along their dock rails."""
    elements = []

    if not cranes:
        return elements

    # Draw crane rails for each dock
    for dock_id, rail in GOLIATH_CRANE_RAILS.items():
        elements.append(
            dl.Polyline(
                positions=[list(rail["start"]), list(rail["end"])],
                color=COLORS["dark"],
                weight=5,
                opacity=0.7,
            )
        )

    # Position cranes on their assigned rails
    for crane in cranes:
        assigned_dock = crane.get("assigned_dock", "")
        crane_id = crane.get("id", "")

        if assigned_dock not in GOLIATH_CRANE_RAILS:
            continue

        rail = GOLIATH_CRANE_RAILS[assigned_dock]
        pos_pct = crane.get("position_on_rail", 50) / 100.0

        # Interpolate position along rail
        lat = rail["start"][0] + (rail["end"][0] - rail["start"][0]) * pos_pct
        lng = rail["start"][1] + (rail["end"][1] - rail["start"][1]) * pos_pct

        status = crane.get("status", "idle")

        if show_health:
            min_health = min(
                crane.get("health_hoist", 100),
                crane.get("health_trolley", 100),
                crane.get("health_gantry", 100),
            )
            color = health_to_color(min_health)
        else:
            color = STATUS_COLORS.get(status, "#95a5a6")

        capacity = crane.get("capacity_tons", 900)
        tooltip_content = html.Div([
            html.Strong(f"{crane_id}"), html.Br(),
            html.Span(f"109m Goliath Crane", style={"fontSize": "10px", "color": "#7f8c8d"}), html.Br(),
            f"Status: {status}", html.Br(),
            f"Dock: {assigned_dock}", html.Br(),
            f"Capacity: {capacity} tons", html.Br(),
            html.Span([
                "Health: ",
                html.Span(f"Hoist:{crane.get('health_hoist', 100):.0f}", style={"color": "#9b59b6"}),
                " ",
                html.Span(f"Trolley:{crane.get('health_trolley', 100):.0f}", style={"color": "#e67e22"}),
                " ",
                html.Span(f"Gantry:{crane.get('health_gantry', 100):.0f}", style={"color": "#3498db"}),
            ]),
        ])

        # Triangle marker for Goliath crane
        elements.append(
            dl.CircleMarker(
                center=[lat, lng],
                radius=12,
                color="white",
                weight=2,
                fillColor=color,
                fillOpacity=0.9,
                children=[dl.Tooltip(tooltip_content)],
            )
        )

    return elements


def _create_block_markers(
    blocks: List[Dict[str, Any]],
) -> List[dl.CircleMarker]:
    """Create block markers showing block locations."""
    markers = []

    if not blocks:
        return markers

    # Group blocks by location to avoid overlap
    # Support both 'location' (hhi_blocks table) and 'current_location' (blocks table)
    blocks_by_location: Dict[str, List] = {}
    for block in blocks:
        loc = block.get("location") or block.get("current_location", "unknown")
        if loc not in blocks_by_location:
            blocks_by_location[loc] = []
        blocks_by_location[loc].append(block)

    for loc, loc_blocks in blocks_by_location.items():
        # Get base position
        lat, lng = facility_to_coords(loc)

        for i, block in enumerate(loc_blocks[:15]):  # Limit to 15 blocks per location
            # Offset each block slightly
            offset_lat = (i // 5) * 0.00008
            offset_lng = (i % 5) * 0.00012 - 0.00024
            blat = lat + offset_lat
            blng = lng + offset_lng

            stage = block.get("current_stage", "unknown")
            completion = block.get("completion_pct", 0)
            block_type = block.get("block_type", "flat_bottom")

            # Color based on block type
            color = BLOCK_TYPE_COLORS.get(block_type, COLORS["accent"])

            tooltip_content = html.Div([
                html.Strong(f"{block.get('id', '?')}"), html.Br(),
                f"Type: {block_type}", html.Br(),
                f"Ship: {block.get('ship_id', '-')}", html.Br(),
                f"Stage: {stage}", html.Br(),
                html.Div([
                    f"Completion: {completion:.0f}%",
                    html.Div(style={
                        "width": "100%",
                        "height": "4px",
                        "backgroundColor": "#ecf0f1",
                        "borderRadius": "2px",
                        "marginTop": "2px",
                    }, children=[
                        html.Div(style={
                            "width": f"{completion}%",
                            "height": "100%",
                            "backgroundColor": color,
                            "borderRadius": "2px",
                        })
                    ])
                ]),
            ])

            markers.append(
                dl.CircleMarker(
                    center=[blat, blng],
                    radius=5,
                    color="white",
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.8,
                    children=[dl.Tooltip(tooltip_content)],
                )
            )

        # If more than 15 blocks, add a summary marker
        if len(loc_blocks) > 15:
            summary_content = html.Div([
                html.Strong(f"+{len(loc_blocks) - 15} more blocks"),
            ])
            markers.append(
                dl.CircleMarker(
                    center=[lat + 0.0003, lng + 0.0004],
                    radius=8,
                    color=COLORS["danger"],
                    fillColor=COLORS["danger"],
                    fillOpacity=0.9,
                    children=[dl.Tooltip(summary_content)],
                )
            )

    return markers


def _create_ship_markers(
    ships: List[Dict[str, Any]],
) -> List:
    """Create LNG carrier ship markers with swim-away animation.

    Ships are shown at different positions based on their status:
    - IN_ERECTION: at assigned dock
    - IN_QUAY_OUTFITTING: at assigned quay
    - IN_SEA_TRIALS: animated position between quay and sea
    - DELIVERED: at sea (final position)
    """
    markers = []

    if not ships:
        return markers

    for ship in ships:
        ship_id = ship.get("id", "?")
        status = ship.get("status", "in_block_production")
        sea_position = ship.get("sea_position", 0.0)

        # Determine ship position based on status
        if status == "in_erection":
            dock_id = ship.get("assigned_dock", "dock_1")
            if dock_id in DRY_DOCKS:
                lat, lng = DRY_DOCKS[dock_id].center
            else:
                lat, lng = (35.5060, 129.4220)
        elif status in ("in_quay_outfitting", "afloat"):
            quay_id = ship.get("assigned_quay", "quay_1")
            if quay_id in OUTFITTING_QUAYS:
                lat, lng = OUTFITTING_QUAYS[quay_id].center
            else:
                lat, lng = (35.5070, 129.4280)
        elif status == "in_sea_trials":
            # Animate ship moving out to sea
            lat, lng = get_ship_position(sea_position)
        elif status == "delivered":
            # Ship has departed - show at sea
            lat, lng = get_ship_position(1.0)
        else:
            # In block production - don't show on map yet
            continue

        # Get color based on status
        color = SHIP_STATUS_COLORS.get(status, COLORS["primary"])

        # Ship statistics
        blocks_erected = ship.get("blocks_erected", 0)
        total_blocks = ship.get("total_blocks", 200)
        erection_progress = ship.get("erection_progress", 0)
        completion = ship.get("completion_pct", 0)
        hull_number = ship.get("hull_number", "")

        # Status display text
        status_display = status.replace("_", " ").title()

        # Build tooltip
        tooltip_children = [
            html.Strong(f"🚢 {ship_id}"),
            html.Br(),
        ]

        if hull_number:
            tooltip_children.extend([
                html.Span(f"Hull: {hull_number}", style={"fontSize": "11px"}),
                html.Br(),
            ])

        tooltip_children.extend([
            html.Span(f"Status: ", style={"fontWeight": "600"}),
            html.Span(status_display, style={"color": color}),
            html.Br(),
            html.Span(f"Blocks: {blocks_erected}/{total_blocks}"),
            html.Br(),
        ])

        # Progress bar for erection or completion
        if status == "in_erection":
            progress = erection_progress
            label = "Erection"
        else:
            progress = completion
            label = "Completion"

        tooltip_children.append(
            html.Div([
                f"{label}: {progress:.0f}%",
                html.Div(style={
                    "width": "100%",
                    "height": "6px",
                    "backgroundColor": "#ecf0f1",
                    "borderRadius": "3px",
                    "marginTop": "4px",
                }, children=[
                    html.Div(style={
                        "width": f"{progress}%",
                        "height": "100%",
                        "backgroundColor": color,
                        "borderRadius": "3px",
                    })
                ])
            ])
        )

        # Sea position indicator for ships in transit
        if status in ("in_sea_trials", "delivered"):
            tooltip_children.extend([
                html.Br(),
                html.Span(f"Sea Position: {sea_position * 100:.0f}%",
                         style={"fontSize": "11px", "color": "#7f8c8d"}),
            ])

        tooltip_content = html.Div(tooltip_children)

        # Ship icon using DivIcon with ship emoji
        ship_icon_html = f"""
            <div style="
                font-size: 24px;
                text-shadow: 1px 1px 2px white, -1px -1px 2px white, 1px -1px 2px white, -1px 1px 2px white;
                transform: rotate(-45deg);
            ">🚢</div>
        """

        markers.append(
            dl.Marker(
                position=[lat, lng],
                icon={
                    "iconUrl": "data:image/svg+xml;charset=utf-8," +
                        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>" +
                        "<text y='80' font-size='80'>🚢</text></svg>".replace("#", "%23"),
                    "iconSize": [32, 32],
                    "iconAnchor": [16, 16],
                },
                children=[dl.Tooltip(tooltip_content)],
            )
        )

        # Add a "wake" trail for ships in sea trials (visual effect)
        if status == "in_sea_trials" and sea_position > 0.1:
            # Small trailing marker to suggest movement
            trail_pos = get_ship_position(max(0, sea_position - 0.15))
            markers.append(
                dl.CircleMarker(
                    center=[trail_pos[0], trail_pos[1]],
                    radius=6,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.3,
                    weight=1,
                )
            )

    return markers


def _create_zone_labels(show_labels: bool = True) -> List[dl.Marker]:
    """Create zone label markers - small, positioned at zone edges."""
    if not show_labels:
        return []

    labels = []

    for zone_name, zone_info in ZONE_LABELS.items():
        # Create a small, semi-transparent label using DivIcon style
        label_html = html.Div(
            zone_info["label"],
            style={
                "fontSize": "10px",
                "fontWeight": "600",
                "color": zone_info["color"],
                "backgroundColor": "rgba(255,255,255,0.75)",
                "padding": "2px 6px",
                "borderRadius": "3px",
                "whiteSpace": "nowrap",
                "border": f"1px solid {zone_info['color']}",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            }
        )

        labels.append(
            dl.Marker(
                position=list(zone_info["center"]),
                children=[
                    dl.Tooltip(
                        label_html,
                        permanent=True,
                        direction="top",
                        offset=[0, -5],
                        className="zone-label-tooltip",
                    ),
                ],
                icon={
                    "iconUrl": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
                    "iconSize": [1, 1],
                },
            )
        )

    return labels


def _create_map_legend() -> html.Div:
    """Create a legend overlay for the map."""
    legend_items = [
        # Entity types
        ("🟠 SPMT", "#e67e22", "Self-Propelled Modular Transporter"),
        ("🟣 Goliath Crane", "#9b59b6", "900-ton capacity crane"),
        ("🔵 Block", "#3498db", "Ship construction block"),
        ("🚢 Ship", "#27ae60", "LNG Carrier under construction"),
        # Facility colors
        ("█ Steel Processing", "#7f8c8d", "Cutting, fabrication"),
        ("█ Panel Assembly", "#3498db", "Flat panel lines"),
        ("█ Block Assembly", "#27ae60", "3D block construction"),
        ("█ Outfitting/Paint", "#e74c3c", "Block completion"),
        ("█ Pre-Erection", "#1abc9c", "Grand block staging"),
        ("█ Dry Dock", "#34495e", "Ship erection"),
        # Status colors
        ("● Idle", "#27ae60", None),
        ("● Busy", "#e67e22", None),
        ("● Maintenance", "#9b59b6", None),
    ]

    items = []
    for label, color, desc in legend_items:
        item_style = {"display": "flex", "alignItems": "center", "marginBottom": "4px"}
        items.append(
            html.Div(style=item_style, children=[
                html.Span(label, style={"color": color, "marginRight": "8px", "fontSize": "11px"}),
                html.Span(desc, style={"fontSize": "10px", "color": "#7f8c8d"}) if desc else None,
            ])
        )

    return html.Div(
        style={
            "position": "absolute",
            "bottom": "20px",
            "left": "10px",
            "backgroundColor": "rgba(255,255,255,0.92)",
            "padding": "10px 14px",
            "borderRadius": "6px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
            "zIndex": "1000",
            "maxWidth": "220px",
            "fontSize": "11px",
            "border": "1px solid #ddd",
        },
        children=[
            html.Div("Map Legend", style={"fontWeight": "700", "marginBottom": "8px", "borderBottom": "1px solid #eee", "paddingBottom": "4px"}),
            *items,
        ]
    )


def _create_mipo_bay_overlay() -> List:
    """Create Mipo Bay water indication - positioned east of all shipyard facilities."""
    # Water polygon for Mipo Bay / Sea of Japan - entirely east of docks and quays
    bay_polygon = [
        [35.5120, 129.4300],  # North - east of quays
        [35.5120, 129.4450],  # Northeast corner
        [35.4880, 129.4450],  # Southeast corner
        [35.4880, 129.4300],  # South - east of docks
    ]

    return [
        dl.Polygon(
            positions=bay_polygon,
            color="#3498db",  # Ocean blue
            fillColor="#3498db",
            fillOpacity=0.15,
            weight=2,
            dashArray="8, 4",
            children=[dl.Tooltip("Mipo Bay (Sea of Japan)")],
        )
    ]


def build_hhi_map(
    map_data: Dict[str, Any],
    show_health: bool = False,
    show_queues: bool = True,
    show_blocks: bool = True,
    use_satellite: bool = False,
    map_id: str = "hhi-map",
) -> dl.Map:
    """Build the dash-leaflet Map component for HHI Ulsan shipyard.

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
        If True, show queue depth in tooltips.
    show_blocks : bool
        If True, show block markers.
    use_satellite : bool
        If True, use satellite imagery instead of map tiles.
    map_id : str
        The component ID for the map.

    Returns
    -------
    dl.Map
        The dash-leaflet Map component.
    """
    queue_data = map_data.get("queue_depths", {}) if show_queues else None

    # Choose tile layer
    tile_url = ESRI_SATELLITE if use_satellite else CARTODB_POSITRON

    # Build layers
    children = [
        # Base tile layer
        dl.TileLayer(url=tile_url, attribution=OSM_ATTRIBUTION),

        # Mipo Bay overlay
        dl.LayerGroup(
            _create_mipo_bay_overlay(),
            id=f"{map_id}-bay",
        ),

        # Flow arrows (behind facilities)
        dl.LayerGroup(
            _create_flow_arrows(HHI_FACILITIES_GEO, HHI_FLOW),
            id=f"{map_id}-flow",
        ),

        # Facility polygons (production areas)
        dl.LayerGroup(
            _create_facility_polygons(HHI_FACILITIES_GEO, queue_data),
            id=f"{map_id}-facilities",
        ),

        # Dry docks
        dl.LayerGroup(
            _create_dry_dock_polygons(DRY_DOCKS, map_data.get("ships", [])),
            id=f"{map_id}-docks",
        ),

        # Outfitting quays
        dl.LayerGroup(
            _create_outfitting_quay_polygons(OUTFITTING_QUAYS, map_data.get("ships", [])),
            id=f"{map_id}-quays",
        ),

        # Goliath cranes
        dl.LayerGroup(
            _create_goliath_crane_markers(map_data.get("goliath_cranes", []), show_health),
            id=f"{map_id}-cranes",
        ),

        # SPMTs
        dl.LayerGroup(
            _create_spmt_markers(map_data.get("spmts", []), show_health),
            id=f"{map_id}-spmts",
        ),

        # Zone labels
        dl.LayerGroup(
            _create_zone_labels(),
            id=f"{map_id}-labels",
        ),

        # Ships (LNG carriers) - with swim-away animation
        dl.LayerGroup(
            _create_ship_markers(map_data.get("ships", [])),
            id=f"{map_id}-ships",
        ),
    ]

    # Add blocks if enabled
    if show_blocks:
        children.append(
            dl.LayerGroup(
                _create_block_markers(map_data.get("blocks", [])),
                id=f"{map_id}-blocks",
            )
        )

    # Create map with legend overlay
    map_component = dl.Map(
        children=children,
        center=list(HHI_CENTER),
        zoom=HHI_ZOOM,
        style={"width": "100%", "height": "600px"},
        id=map_id,
        maxBounds=[[HHI_BOUNDS["south"], HHI_BOUNDS["west"]],
                   [HHI_BOUNDS["north"], HHI_BOUNDS["east"]]],
    )

    # Wrap map with legend overlay
    return html.Div(
        style={"position": "relative", "width": "100%", "height": "600px"},
        children=[
            map_component,
            _create_map_legend(),
        ]
    )


def build_dock_detail_map(
    dock_name: str,
    map_data: Dict[str, Any],
    map_id: str = "dock-detail-map",
) -> dl.Map:
    """Build a detailed view of a specific dry dock.

    Parameters
    ----------
    dock_name : str
        The dock identifier (e.g., "dock_1").
    map_data : dict
        Dictionary containing ships, goliath_cranes, and blocks data.
    map_id : str
        The component ID for the map.

    Returns
    -------
    dl.Map
        The dash-leaflet Map component focused on the dock.
    """
    if dock_name not in DRY_DOCKS:
        # Return empty map
        return dl.Map(
            children=[dl.TileLayer(url=CARTODB_POSITRON)],
            center=list(HHI_CENTER),
            zoom=HHI_ZOOM,
            style={"width": "100%", "height": "400px"},
            id=map_id,
        )

    dock = DRY_DOCKS[dock_name]

    # Get ship in this dock
    ships = map_data.get("ships", [])
    ship_in_dock = next((s for s in ships if s.get("assigned_dock") == dock_name), None)

    # Get cranes for this dock
    cranes = [c for c in map_data.get("goliath_cranes", [])
              if c.get("assigned_dock") == dock_name]

    # Get blocks in erection at this dock
    blocks = [b for b in map_data.get("blocks", [])
              if b.get("current_location") == dock_name]

    children = [
        dl.TileLayer(url=CARTODB_POSITRON, attribution=OSM_ATTRIBUTION),

        # Dock polygon
        dl.LayerGroup(
            _create_dry_dock_polygons({dock_name: dock}, ships),
        ),

        # Cranes for this dock
        dl.LayerGroup(
            _create_goliath_crane_markers(cranes, show_health=True),
        ),

        # Blocks being erected
        dl.LayerGroup(
            _create_block_markers(blocks),
        ),
    ]

    return dl.Map(
        children=children,
        center=list(dock.center),
        zoom=17,
        style={"width": "100%", "height": "400px"},
        id=map_id,
    )


def build_production_zone_map(
    zone: str,
    map_data: Dict[str, Any],
    map_id: str = "zone-map",
) -> dl.Map:
    """Build a focused view of a production zone.

    Parameters
    ----------
    zone : str
        Zone identifier: "steel", "panel", "block", "pre_erection", "docks", "quays"
    map_data : dict
        Dictionary with full shipyard data.
    map_id : str
        The component ID for the map.

    Returns
    -------
    dl.Map
        The dash-leaflet Map component focused on the zone.
    """
    # Zone facility mappings
    zone_facilities = {
        "steel": ["steel_stockyard", "cutting_shop", "part_fabrication"],
        "panel": ["flat_panel_line_1", "flat_panel_line_2", "curved_block_shop"],
        "block": ["block_assembly_hall_1", "block_assembly_hall_2", "block_assembly_hall_3",
                  "outfitting_shop", "paint_shop"],
        "pre_erection": ["grand_block_staging_north", "grand_block_staging_south"],
    }

    if zone in zone_facilities:
        # Filter to zone facilities
        zone_facs = {k: v for k, v in HHI_FACILITIES_GEO.items()
                     if k in zone_facilities[zone]}

        if zone_facs:
            # Calculate zone center
            all_lats = [f.center[0] for f in zone_facs.values()]
            all_lngs = [f.center[1] for f in zone_facs.values()]
            center = [sum(all_lats)/len(all_lats), sum(all_lngs)/len(all_lngs)]
        else:
            center = list(HHI_CENTER)

        children = [
            dl.TileLayer(url=CARTODB_POSITRON, attribution=OSM_ATTRIBUTION),
            dl.LayerGroup(_create_facility_polygons(zone_facs, map_data.get("queue_depths"))),
            dl.LayerGroup(_create_spmt_markers(map_data.get("spmts", []))),
            dl.LayerGroup(_create_block_markers(map_data.get("blocks", []))),
        ]

        return dl.Map(
            children=children,
            center=center,
            zoom=16,
            style={"width": "100%", "height": "400px"},
            id=map_id,
        )

    elif zone == "docks":
        children = [
            dl.TileLayer(url=CARTODB_POSITRON, attribution=OSM_ATTRIBUTION),
            dl.LayerGroup(_create_dry_dock_polygons(DRY_DOCKS, map_data.get("ships", []))),
            dl.LayerGroup(_create_goliath_crane_markers(map_data.get("goliath_cranes", []))),
        ]

        # Center on docks
        dock_lats = [d.center[0] for d in DRY_DOCKS.values()]
        dock_lngs = [d.center[1] for d in DRY_DOCKS.values()]
        center = [sum(dock_lats)/len(dock_lats), sum(dock_lngs)/len(dock_lngs)]

        return dl.Map(
            children=children,
            center=center,
            zoom=15,
            style={"width": "100%", "height": "400px"},
            id=map_id,
        )

    elif zone == "quays":
        children = [
            dl.TileLayer(url=CARTODB_POSITRON, attribution=OSM_ATTRIBUTION),
            dl.LayerGroup(_create_outfitting_quay_polygons(
                OUTFITTING_QUAYS, map_data.get("ships", []))),
        ]

        quay_lats = [q.center[0] for q in OUTFITTING_QUAYS.values()]
        quay_lngs = [q.center[1] for q in OUTFITTING_QUAYS.values()]
        center = [sum(quay_lats)/len(quay_lats), sum(quay_lngs)/len(quay_lngs)]

        return dl.Map(
            children=children,
            center=center,
            zoom=16,
            style={"width": "100%", "height": "400px"},
            id=map_id,
        )

    # Default: full map
    return build_hhi_map(map_data, map_id=map_id)

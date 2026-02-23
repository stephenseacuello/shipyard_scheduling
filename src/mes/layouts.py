"""Layout definitions for the MES dashboard.

Layouts for HD Hyundai Heavy Industries (HHI) Ulsan shipyard visualization,
including LNG carrier production, Goliath cranes, and 10 dry docks.
"""

from __future__ import annotations

from dash import html, dcc, dash_table

try:
    import dash_cytoscape as cyto
    CYTOSCAPE_AVAILABLE = True
except ImportError:
    CYTOSCAPE_AVAILABLE = False

try:
    import dash_leaflet as dl
    LEAFLET_AVAILABLE = True
except ImportError:
    LEAFLET_AVAILABLE = False


def _kpi_card(label: str, value_id: str) -> html.Div:
    return html.Div(className="kpi-card", children=[
        html.P(label, className="kpi-card-label"),
        html.P("—", id=value_id, className="kpi-card-value"),
    ])


def _empty_state(msg: str = "No simulation data yet.") -> html.Div:
    return html.Div(className="empty-state", children=[
        html.P(msg),
        html.P("Run training or evaluation to populate the dashboard."),
    ])


def overview_layout() -> html.Div:
    return html.Div([
        # Header with export buttons
        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"}, children=[
            html.H3("Production Overview", style={"margin": "0"}),
            html.Div([
                html.Button("📊 Export Metrics CSV", id="export-metrics-btn", n_clicks=0,
                            style={"marginRight": "8px", "padding": "6px 12px", "border": "1px solid #3b82f6",
                                   "borderRadius": "4px", "backgroundColor": "#eff6ff", "color": "#3b82f6",
                                   "cursor": "pointer", "fontSize": "12px", "fontWeight": "600"}),
                html.Button("📦 Export Blocks CSV", id="export-blocks-btn", n_clicks=0,
                            style={"padding": "6px 12px", "border": "1px solid #10b981",
                                   "borderRadius": "4px", "backgroundColor": "#ecfdf5", "color": "#10b981",
                                   "cursor": "pointer", "fontSize": "12px", "fontWeight": "600"}),
                dcc.Download(id="download-metrics"),
                dcc.Download(id="download-blocks"),
            ]),
        ]),
        # Primary KPIs
        html.Div(className="kpi-grid", children=[
            _kpi_card("Blocks Completed", "kpi-blocks"),
            _kpi_card("Ships Launched", "kpi-ships"),
            _kpi_card("Breakdowns", "kpi-breakdowns"),
            _kpi_card("Planned Maint.", "kpi-planned"),
            _kpi_card("Avg Tardiness", "kpi-tardiness"),
            _kpi_card("On-Time %", "kpi-ontime"),
            _kpi_card("SPMT Utilization", "kpi-spmt-util"),
            _kpi_card("Crane Utilization", "kpi-crane-util"),
        ]),
        # OEE Metrics Section
        html.Div(style={"marginTop": "16px", "marginBottom": "16px"}, children=[
            html.H4("Overall Equipment Effectiveness (OEE)", style={"marginBottom": "12px", "color": "#2c3e50"}),
            html.Div(className="kpi-grid", children=[
                _kpi_card("OEE Score", "kpi-oee"),
                _kpi_card("Availability", "kpi-availability"),
                _kpi_card("Performance", "kpi-performance"),
                _kpi_card("Quality", "kpi-quality"),
            ]),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            html.Div(className="chart-card", children=[
                html.H4("Production Throughput"),
                dcc.Loading(dcc.Graph(id="throughput-chart")),
            ]),
            html.Div(className="chart-card", children=[
                html.H4("Block Stage Distribution"),
                dcc.Loading(dcc.Graph(id="stage-distribution-chart")),
            ]),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "16px"}, children=[
            html.Div(className="chart-card", children=[
                html.H4("Facility Bottlenecks"),
                dcc.Loading(dcc.Graph(id="bottleneck-chart")),
            ]),
            html.Div(className="chart-card", children=[
                html.H4("Equipment Health Summary"),
                dcc.Loading(dcc.Graph(id="health-summary-chart")),
            ]),
        ]),
        html.Div(className="chart-card", children=[
            html.H3("Metric Trends", className="section-header"),
            dcc.Loading(dcc.Graph(id="overview-kpi-trends")),
        ]),
    ])


def blocks_layout() -> html.Div:
    return html.Div([
        html.H3("LNG Carrier Block Status", className="section-header"),
        html.Div(style={"marginBottom": "16px"}, children=[
            html.Label("Filter by Ship:", style={"marginRight": "8px", "fontWeight": "600"}),
            dcc.Dropdown(
                id="blocks-ship-filter",
                placeholder="All ships",
                style={"width": "200px", "display": "inline-block", "verticalAlign": "middle"},
            ),
            html.Label("Stage:", style={"marginLeft": "16px", "marginRight": "8px", "fontWeight": "600"}),
            dcc.Dropdown(
                id="blocks-stage-filter",
                options=[
                    {"label": "All Stages", "value": "all"},
                    {"label": "Steel Cutting", "value": "STEEL_CUTTING"},
                    {"label": "Part Fabrication", "value": "PART_FABRICATION"},
                    {"label": "Panel Assembly", "value": "PANEL_ASSEMBLY"},
                    {"label": "Block Assembly", "value": "BLOCK_ASSEMBLY"},
                    {"label": "Block Outfitting", "value": "BLOCK_OUTFITTING"},
                    {"label": "Painting", "value": "PAINTING"},
                    {"label": "Pre-Erection", "value": "PRE_ERECTION"},
                    {"label": "Erection", "value": "ERECTION"},
                ],
                value="all",
                style={"width": "180px", "display": "inline-block", "verticalAlign": "middle"},
            ),
        ]),
        dcc.Loading(dash_table.DataTable(
            id="blocks-table",
            columns=[
                {"name": "ID", "id": "id"},
                {"name": "Ship", "id": "ship_id"},
                {"name": "Type", "id": "block_type"},
                {"name": "Stage", "id": "current_stage"},
                {"name": "Location", "id": "current_location"},
                {"name": "Weight (t)", "id": "weight"},
                {"name": "Due Date", "id": "due_date"},
                {"name": "Completion %", "id": "completion_pct"},
            ],
            data=[],
            page_size=20,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_data_conditional=[
                {"if": {"filter_query": "{current_stage} = ERECTION"},
                 "backgroundColor": "#d4edda", "color": "#155724"},
                {"if": {"filter_query": "{current_stage} contains TRANSIT"},
                 "backgroundColor": "#fff3cd", "color": "#856404"},
                {"if": {"filter_query": "{block_type} contains curved"},
                 "backgroundColor": "#e2e3f0", "color": "#383d6e"},
            ],
        )),
    ])


def ships_layout() -> html.Div:
    """Layout for ship construction progress."""
    return html.Div([
        html.H3("LNG Carrier Construction Progress", className="section-header"),
        dcc.Loading(dash_table.DataTable(
            id="ships-table",
            columns=[
                {"name": "Ship Name", "id": "name"},
                {"name": "Hull No.", "id": "hull_number"},
                {"name": "Assigned Dock", "id": "assigned_dock"},
                {"name": "Status", "id": "status"},
                {"name": "Blocks Erected", "id": "blocks_erected"},
                {"name": "Total Blocks", "id": "total_blocks"},
                {"name": "Erection %", "id": "erection_progress", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Launch Date", "id": "target_launch_date"},
            ],
            data=[],
            style_data_conditional=[
                {"if": {"filter_query": "{status} = LAUNCHED"},
                 "backgroundColor": "#d4edda", "color": "#155724"},
                {"if": {"filter_query": "{status} = ERECTION"},
                 "backgroundColor": "#cce5ff", "color": "#004085"},
                {"if": {"filter_query": "{erection_progress} >= 80"},
                 "fontWeight": "bold"},
            ],
        )),
        html.Div(className="chart-card", style={"marginTop": "16px"}, children=[
            html.H4("Ship Erection Progress"),
            dcc.Loading(dcc.Graph(id="ship-progress-chart")),
        ]),
    ])


def fleet_layout() -> html.Div:
    return html.Div([
        html.H3("SPMT Fleet", className="section-header"),
        dcc.Loading(dash_table.DataTable(
            id="fleet-spmt-table",
            columns=[
                {"name": "ID", "id": "id"},
                {"name": "Status", "id": "status"},
                {"name": "Location", "id": "current_location"},
                {"name": "Load", "id": "load"},
                {"name": "Hydraulic", "id": "health_hydraulic", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Tires", "id": "health_tires", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Engine", "id": "health_engine", "type": "numeric", "format": {"specifier": ".1f"}},
            ],
            data=[],
            style_data_conditional=[
                {"if": {"filter_query": "{health_hydraulic} < 40", "column_id": "health_hydraulic"},
                 "backgroundColor": "#f8d7da", "color": "#721c24"},
                {"if": {"filter_query": "{health_tires} < 40", "column_id": "health_tires"},
                 "backgroundColor": "#f8d7da", "color": "#721c24"},
                {"if": {"filter_query": "{health_engine} < 40", "column_id": "health_engine"},
                 "backgroundColor": "#f8d7da", "color": "#721c24"},
            ],
        )),
        html.H3("Goliath Cranes", className="section-header", style={"marginTop": "24px"}),
        dcc.Loading(dash_table.DataTable(
            id="fleet-crane-table",
            columns=[
                {"name": "ID", "id": "id"},
                {"name": "Assigned Dock", "id": "assigned_dock"},
                {"name": "Status", "id": "status"},
                {"name": "Capacity (t)", "id": "capacity_tons"},
                {"name": "Hoist", "id": "health_hoist", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Trolley", "id": "health_trolley", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Gantry", "id": "health_gantry", "type": "numeric", "format": {"specifier": ".1f"}},
            ],
            data=[],
            style_data_conditional=[
                {"if": {"filter_query": "{health_hoist} < 40", "column_id": "health_hoist"},
                 "backgroundColor": "#f8d7da", "color": "#721c24"},
                {"if": {"filter_query": "{health_trolley} < 40", "column_id": "health_trolley"},
                 "backgroundColor": "#f8d7da", "color": "#721c24"},
                {"if": {"filter_query": "{health_gantry} < 40", "column_id": "health_gantry"},
                 "backgroundColor": "#f8d7da", "color": "#721c24"},
                {"if": {"filter_query": "{status} = lifting"},
                 "backgroundColor": "#d4edda", "color": "#155724"},
            ],
        )),
        html.Div(className="chart-card", style={"marginTop": "16px"}, children=[
            html.H3("Equipment Utilization", className="section-header"),
            dcc.Loading(dcc.Graph(id="utilization-heatmap")),
        ]),
    ])


def health_layout() -> html.Div:
    return html.Div([
        html.H3("Equipment Health", className="section-header"),
        html.Div(style={"marginBottom": "16px"}, children=[
            html.Label("Filter by equipment:", style={"marginRight": "8px", "fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="health-equipment-dropdown",
                placeholder="All equipment",
                style={"width": "280px", "display": "inline-block", "verticalAlign": "middle"},
            ),
            html.Label("Type:", style={"marginLeft": "16px", "marginRight": "8px", "fontWeight": "600"}),
            dcc.Dropdown(
                id="health-type-dropdown",
                options=[
                    {"label": "All Types", "value": "all"},
                    {"label": "SPMTs", "value": "spmt"},
                    {"label": "Goliath Cranes", "value": "goliath"},
                ],
                value="all",
                style={"width": "160px", "display": "inline-block", "verticalAlign": "middle"},
            ),
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.H4("Degradation Trends"),
                dcc.Loading(dcc.Graph(id="health-trends-chart")),
            ]),
            html.Div([
                html.H4("RUL Predictions", style={"marginBottom": "12px"}),
                dcc.Loading(dash_table.DataTable(
                    id="rul-table",
                    columns=[
                        {"name": "Equipment", "id": "equipment_id"},
                        {"name": "Type", "id": "equipment_type"},
                        {"name": "Component", "id": "component"},
                        {"name": "Health", "id": "health_value", "type": "numeric"},
                        {"name": "Est. RUL (hrs)", "id": "rul", "type": "numeric"},
                    ],
                    data=[],
                    page_size=12,
                    sort_action="native",
                    style_data_conditional=[
                        {"if": {"filter_query": "{health_value} < 30"},
                         "backgroundColor": "#f8d7da", "color": "#721c24"},
                        {"if": {"filter_query": "{health_value} >= 30 && {health_value} < 50"},
                         "backgroundColor": "#fff3cd", "color": "#856404"},
                        {"if": {"filter_query": "{health_value} >= 50"},
                         "backgroundColor": "#d4edda", "color": "#155724"},
                    ],
                )),
            ]),
        ]),
    ])


def operations_layout() -> html.Div:
    return html.Div([
        html.H3("Operations", className="section-header"),
        html.Div(className="chart-card", children=[
            html.H4("Block Flow Timeline"),
            dcc.Loading(dcc.Graph(id="gantt-chart")),
        ]),
        html.Div(className="chart-card", children=[
            html.H4("Ship Production Timeline"),
            dcc.Loading(dcc.Graph(id="ship-gantt-chart")),
        ]),
        html.Div(className="chart-card", children=[
            html.H4("Facility Queue Depths"),
            dcc.Loading(dcc.Graph(id="queue-depth-chart")),
        ]),
    ])


def kpis_layout() -> html.Div:
    return html.Div([
        html.H3("KPI Analytics", className="section-header"),
        html.Div(className="chart-card", children=[
            dcc.Loading(dcc.Graph(id="kpi-trend-graph")),
        ]),
    ])


# ============================================================================
# HHI ULSAN SHIPYARD MAP LAYOUTS
# ============================================================================

def _map_controls() -> html.Div:
    """Control bar for the shipyard map view."""
    return html.Div(className="map-controls", children=[
        html.Div([
            html.Label("Health Overlay:", style={"marginRight": "8px", "fontWeight": "600"}),
            dcc.Checklist(
                id="map-health-toggle",
                options=[{"label": " Show", "value": "on"}],
                value=[],
                style={"display": "inline-block"},
            ),
        ], style={"marginRight": "24px"}),
        html.Div([
            html.Label("Show Layers:", style={"marginRight": "8px", "fontWeight": "600"}),
            dcc.Checklist(
                id="map-layers",
                options=[
                    {"label": " SPMTs", "value": "spmts"},
                    {"label": " Cranes", "value": "cranes"},
                    {"label": " Blocks", "value": "blocks"},
                    {"label": " Queues", "value": "queues"},
                ],
                value=["spmts", "cranes", "queues"],
                inline=True,
                style={"display": "inline-block"},
            ),
        ], style={"marginRight": "24px"}),
        html.Div([
            html.Label("View:", style={"marginRight": "8px", "fontWeight": "600"}),
            dcc.Dropdown(
                id="map-view-select",
                options=[
                    {"label": "Full Shipyard", "value": "full"},
                    {"label": "Dry Docks", "value": "docks"},
                    {"label": "Production Zone", "value": "production"},
                    {"label": "Pre-Erection", "value": "pre_erection"},
                ],
                value="full",
                clearable=False,
                style={"width": "160px", "display": "inline-block", "verticalAlign": "middle"},
            ),
        ]),
    ])


def hhi_map_layout() -> html.Div:
    """Layout for the HHI Ulsan shipyard map with playback controls."""
    if LEAFLET_AVAILABLE:
        map_content = html.Div(
            id="hhi-map-container",
            className="leaflet-map-container",
            style={"height": "100%", "width": "100%", "minHeight": "750px"},
        )
    else:
        # Fallback to Plotly graph if dash-leaflet not available
        map_content = dcc.Graph(
            id="hhi-map",
            config={
                "displayModeBar": True,
                "scrollZoom": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
            style={"height": "100%", "minHeight": "750px"},
        )

    return html.Div([
        html.H3("HD Hyundai Heavy Industries - Ulsan Shipyard", className="section-header"),
        html.P("Mipo Bay, South Korea | LNG Carrier Production",
               style={"color": "#7f8c8d", "marginBottom": "12px"}),

        # Playback mode toggle and run selector
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "20px", "marginBottom": "12px"}, children=[
            dcc.Checklist(
                id="playback-mode-toggle",
                options=[{"label": " Enable Playback Mode", "value": "on"}],
                value=[],
                style={"fontSize": "13px"},
                inputStyle={"marginRight": "6px"},
            ),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"}, children=[
                html.Label("Historical Run:", style={"fontWeight": "600", "fontSize": "13px", "color": "#475569"}),
                dcc.Dropdown(
                    id="playback-run-selector",
                    placeholder="Current session",
                    style={"width": "280px", "fontSize": "12px"},
                    clearable=True,
                ),
            ]),
        ]),

        # Playback controls
        html.Div(className="playback-controls", style={
            "display": "flex",
            "alignItems": "center",
            "gap": "12px",
            "padding": "10px 16px",
            "backgroundColor": "#f1f5f9",
            "borderRadius": "6px",
            "marginBottom": "12px",
            "border": "1px solid #e2e8f0",
        }, children=[
            html.Div([
                html.Button("⏮", id="playback-rewind", n_clicks=0,
                            style={"padding": "6px 10px", "border": "1px solid #cbd5e1",
                                   "borderRadius": "4px", "backgroundColor": "white",
                                   "cursor": "pointer", "fontSize": "14px"}),
                html.Button("▶", id="playback-play", n_clicks=0,
                            style={"padding": "6px 14px", "border": "1px solid #cbd5e1",
                                   "borderRadius": "4px", "backgroundColor": "white",
                                   "cursor": "pointer", "fontSize": "14px", "marginLeft": "4px"}),
                html.Button("⏭", id="playback-forward", n_clicks=0,
                            style={"padding": "6px 10px", "border": "1px solid #cbd5e1",
                                   "borderRadius": "4px", "backgroundColor": "white",
                                   "cursor": "pointer", "fontSize": "14px", "marginLeft": "4px"}),
            ], style={"display": "flex"}),
            # Playback speed control
            html.Div([
                html.Label("Speed:", style={"fontSize": "12px", "color": "#64748b", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="playback-speed-selector",
                    options=[
                        {"label": "0.5x", "value": 0.5},
                        {"label": "1x", "value": 1},
                        {"label": "2x", "value": 2},
                        {"label": "5x", "value": 5},
                        {"label": "10x", "value": 10},
                    ],
                    value=1,
                    clearable=False,
                    style={"width": "70px", "fontSize": "12px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(style={"flex": "1", "marginLeft": "16px", "marginRight": "16px"}, children=[
                dcc.Slider(
                    id="playback-slider",
                    min=0, max=100, value=100, step=1,
                    marks={0: {"label": "Start", "style": {"fontSize": "11px"}},
                           100: {"label": "Live", "style": {"fontSize": "11px"}}},
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="drag",
                ),
            ]),
            html.Span(id="playback-time", children="Live",
                      style={"fontWeight": "600", "minWidth": "90px", "fontSize": "13px",
                             "color": "#10b981", "textAlign": "center"}),
            html.Button("⟳ Live", id="playback-live", n_clicks=0,
                        style={"padding": "6px 12px", "border": "1px solid #10b981",
                               "borderRadius": "4px", "backgroundColor": "#ecfdf5",
                               "color": "#10b981", "cursor": "pointer", "fontSize": "12px",
                               "fontWeight": "600"}),
        ]),

        # Playback state store
        dcc.Store(id="playback-state", data={"playing": False, "timestamp": None}),
        dcc.Store(id="playback-timeline", data=[]),
        dcc.Store(id="playback-current-time", data=None),
        dcc.Store(id="playback-controls-container", data={}),
        dcc.Interval(id="playback-interval", interval=500, disabled=True),

        _map_controls(),

        # Main content: Map (2/3) + Charts (1/3) side by side
        html.Div(className="map-charts-layout", children=[
            # Map column (2/3 width, tall)
            html.Div(className="map-column", children=[
                html.Div(className="chart-card map-container", style={"height": "100%"}, children=[
                    dcc.Loading(map_content),
                ]),
            ]),
            # Charts column (narrow sidebar on right)
            html.Div(className="charts-column", children=[
                html.Div(className="chart-card", children=[
                    html.H4("Production by Stage"),
                    dcc.Loading(dcc.Graph(id="production-stage-chart", style={"height": "320px"})),
                ]),
                html.Div(className="chart-card", children=[
                    html.H4("Equipment Health"),
                    dcc.Loading(dcc.Graph(id="equipment-health-chart", style={"height": "320px"})),
                ]),
            ]),
        ]),

        # Detail panel
        html.Div(id="hhi-detail-panel", className="detail-panel", children=[
            html.P("Click on equipment for details", className="text-muted"),
        ]),
    ])


def docks_layout() -> html.Div:
    """Layout for dry dock monitoring."""
    return html.Div([
        html.H3("Dry Dock Status", className="section-header"),
        html.P("10 Dry Docks with 9 Goliath Cranes (109m tall, 900-ton capacity)",
               style={"color": "#7f8c8d", "marginBottom": "16px"}),

        # Dock grid
        html.Div(className="dock-grid", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(5, 1fr)",
            "gap": "16px",
            "marginBottom": "24px",
        }, children=[
            _dock_card(f"dock_{i}", f"Dock {i}", f"dock-{i}-status")
            for i in range(1, 11)
        ]),

        # Dock details
        html.Div(className="chart-card", children=[
            html.H4("Selected Dock Details"),
            dcc.Dropdown(
                id="dock-select",
                options=[{"label": f"Dock {i}", "value": f"dock_{i}"} for i in range(1, 11)],
                value="dock_1",
                style={"width": "200px", "marginBottom": "16px"},
            ),
            html.Div(id="dock-detail-content", children=[
                html.P("Select a dock to view details", className="text-muted"),
            ]),
        ]),

        # Dock utilization over time
        html.Div(className="chart-card", style={"marginTop": "16px"}, children=[
            html.H4("Dock Utilization Timeline"),
            dcc.Loading(dcc.Graph(id="dock-utilization-chart")),
        ]),
    ])


def _dock_card(dock_id: str, dock_name: str, status_id: str) -> html.Div:
    """Create a dock status card."""
    return html.Div(className="dock-card", id=f"{dock_id}-card", style={
        "backgroundColor": "#f8f9fa",
        "border": "2px solid #dee2e6",
        "borderRadius": "8px",
        "padding": "12px",
        "textAlign": "center",
    }, children=[
        html.H5(dock_name, style={"marginBottom": "8px", "color": "#2c3e50"}),
        html.Div(id=status_id, children=[
            html.Span("Empty", style={"color": "#7f8c8d"}),
        ]),
    ])


def alerts_banner() -> html.Div:
    """Alerts banner component for critical notifications."""
    return html.Div(id="alerts-banner", className="alerts-banner", children=[])


def playback_controls() -> html.Div:
    """Playback controls for simulation timeline scrubbing."""
    return html.Div(className="playback-controls", style={
        "display": "flex",
        "alignItems": "center",
        "gap": "12px",
        "padding": "8px 16px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "4px",
        "marginBottom": "16px",
    }, children=[
        html.Button("<<", id="playback-rewind", className="playback-btn",
                    style={"padding": "4px 12px", "border": "1px solid #ddd", "borderRadius": "4px"}),
        html.Button("Play", id="playback-play", className="playback-btn",
                    style={"padding": "4px 12px", "border": "1px solid #ddd", "borderRadius": "4px"}),
        html.Button(">>", id="playback-forward", className="playback-btn",
                    style={"padding": "4px 12px", "border": "1px solid #ddd", "borderRadius": "4px"}),
        dcc.Slider(
            id="playback-slider",
            min=0, max=100, value=100, step=1,
            marks={0: "Start", 50: "", 100: "Now"},
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode="drag",
            className="playback-slider",
        ),
        html.Span(id="playback-time", children="Live", style={"fontWeight": "600", "minWidth": "80px"}),
        html.Button("Live", id="playback-live", className="playback-btn",
                    style={"padding": "4px 12px", "border": "1px solid #ddd", "borderRadius": "4px"}),
    ])


def dependencies_layout() -> html.Div:
    """Layout for block dependency visualization."""
    return html.Div([
        html.H3("Block Dependencies", className="section-header"),
        html.Div(className="map-controls", children=[
            html.Div([
                html.Label("Filter Ship:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Dropdown(
                    id="dependency-ship-filter",
                    placeholder="All ships",
                    style={"width": "180px", "display": "inline-block", "verticalAlign": "middle"},
                ),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Filter Block:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Dropdown(
                    id="dependency-block-filter",
                    placeholder="All blocks (click to filter)",
                    style={"width": "200px", "display": "inline-block", "verticalAlign": "middle"},
                ),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Show:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Checklist(
                    id="dependency-show-options",
                    options=[
                        {"label": " Completed", "value": "completed"},
                        {"label": " Critical Path", "value": "critical"},
                    ],
                    value=["completed"],
                    inline=True,
                    style={"display": "inline-block"},
                ),
            ]),
        ]),
        html.Div(className="chart-card", children=[
            dcc.Loading(dcc.Graph(
                id="dependency-graph",
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                },
            )),
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.H4("Dependency Statistics"),
                html.Div(id="dependency-stats", children=[
                    html.P("Select a block to see its dependency chain", className="text-muted"),
                ]),
            ]),
            html.Div(className="chart-card", children=[
                html.H4("Critical Path Blocks"),
                dcc.Loading(dash_table.DataTable(
                    id="critical-path-table",
                    columns=[
                        {"name": "Block", "id": "id"},
                        {"name": "Ship", "id": "ship_id"},
                        {"name": "Stage", "id": "current_stage"},
                        {"name": "Completion", "id": "completion_pct"},
                    ],
                    data=[],
                    page_size=10,
                    style_data_conditional=[
                        {"if": {"filter_query": "{current_stage} = ERECTION"},
                         "backgroundColor": "#d4edda", "color": "#155724"},
                        {"if": {"filter_query": "{current_stage} = PRE_ERECTION"},
                         "backgroundColor": "#cce5ff", "color": "#004085"},
                    ],
                )),
            ]),
        ]),
    ])


# ============================================================================
# GNN GRAPH VISUALIZATION LAYOUT
# ============================================================================

# Cytoscape stylesheet for graph rendering
CYTOSCAPE_STYLESHEET = [
    # Node styles by type
    {
        "selector": "node[type = 'block']",
        "style": {
            "background-color": "#3498db",
            "label": "data(label)",
            "width": 30,
            "height": 30,
            "font-size": "10px",
            "text-valign": "bottom",
            "text-margin-y": 5,
        }
    },
    {
        "selector": "node[type = 'spmt']",
        "style": {
            "background-color": "#27ae60",
            "label": "data(label)",
            "width": 35,
            "height": 35,
            "shape": "rectangle",
            "font-size": "10px",
            "text-valign": "bottom",
            "text-margin-y": 5,
        }
    },
    {
        "selector": "node[type = 'goliath_crane']",
        "style": {
            "background-color": "#e74c3c",
            "label": "data(label)",
            "width": 40,
            "height": 40,
            "shape": "triangle",
            "font-size": "10px",
            "text-valign": "bottom",
            "text-margin-y": 5,
        }
    },
    {
        "selector": "node[type = 'facility']",
        "style": {
            "background-color": "#9b59b6",
            "label": "data(label)",
            "width": 45,
            "height": 45,
            "shape": "hexagon",
            "font-size": "11px",
            "text-valign": "bottom",
            "text-margin-y": 5,
        }
    },
    {
        "selector": "node[type = 'dock']",
        "style": {
            "background-color": "#e74c3c",
            "label": "data(label)",
            "width": 50,
            "height": 50,
            "shape": "rectangle",
            "font-size": "11px",
            "text-valign": "bottom",
            "text-margin-y": 5,
        }
    },
    {
        "selector": "node[type = 'ship']",
        "style": {
            "background-color": "#1abc9c",
            "label": "data(label)",
            "width": 45,
            "height": 45,
            "shape": "ellipse",
            "font-size": "11px",
            "text-valign": "bottom",
            "text-margin-y": 5,
        }
    },
    # Health-based node coloring
    {
        "selector": "node[health_status = 'critical']",
        "style": {"background-color": "#e74c3c", "border-width": 3, "border-color": "#c0392b"}
    },
    {
        "selector": "node[health_status = 'warning']",
        "style": {"background-color": "#f39c12", "border-width": 2, "border-color": "#d68910"}
    },
    {
        "selector": "node[health_status = 'healthy']",
        "style": {"border-width": 0}
    },
    # Selected node
    {
        "selector": ":selected",
        "style": {"border-width": 4, "border-color": "#2c3e50", "background-opacity": 0.9}
    },
    # Edge styles by type
    {
        "selector": "edge[type = 'needs_transport']",
        "style": {"line-color": "#3498db", "width": 2, "curve-style": "bezier", "opacity": 0.6}
    },
    {
        "selector": "edge[type = 'can_transport']",
        "style": {"line-color": "#27ae60", "width": 2, "curve-style": "bezier", "opacity": 0.6}
    },
    {
        "selector": "edge[type = 'needs_lift']",
        "style": {"line-color": "#e74c3c", "width": 2, "curve-style": "bezier", "opacity": 0.6}
    },
    {
        "selector": "edge[type = 'can_lift']",
        "style": {"line-color": "#f39c12", "width": 2, "curve-style": "bezier", "opacity": 0.6}
    },
    {
        "selector": "edge[type = 'at']",
        "style": {"line-color": "#9b59b6", "width": 1, "line-style": "dashed", "opacity": 0.4}
    },
    {
        "selector": "edge[type = 'precedes']",
        "style": {
            "line-color": "#e74c3c",
            "width": 2,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#e74c3c",
            "opacity": 0.7
        }
    },
    {
        "selector": "edge[type = 'assigned_to']",
        "style": {"line-color": "#1abc9c", "width": 2, "curve-style": "bezier", "opacity": 0.5}
    },
]


def gnn_graph_layout() -> html.Div:
    """Layout for visualizing the heterogeneous GNN graph structure."""
    if not CYTOSCAPE_AVAILABLE:
        return html.Div([
            html.H3("GNN Graph Visualization", className="section-header"),
            html.Div(className="chart-card", children=[
                html.P("dash-cytoscape is not installed.", style={"color": "#e74c3c"}),
                html.P("Install with: pip install dash-cytoscape"),
            ]),
        ])

    return html.Div([
        html.H3("GNN Graph Visualization", className="section-header"),
        html.P(
            "Visualizes the heterogeneous graph representation used by the GNN encoder for HHI Ulsan.",
            style={"color": "#7f8c8d", "marginBottom": "16px"}
        ),

        # Controls
        html.Div(className="map-controls", children=[
            html.Div([
                html.Label("Filter Ship:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Dropdown(
                    id="gnn-graph-ship-filter",
                    options=[{"label": "All Ships", "value": "all"}],
                    value="all",
                    clearable=False,
                    style={"width": "160px", "display": "inline-block", "verticalAlign": "middle"},
                ),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Show Edges:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Checklist(
                    id="gnn-graph-edge-filter",
                    options=[
                        {"label": " Transport", "value": "transport"},
                        {"label": " Lift", "value": "lift"},
                        {"label": " Location", "value": "location"},
                        {"label": " Precedence", "value": "precedes"},
                        {"label": " Assignment", "value": "assigned"},
                    ],
                    value=["transport", "lift", "precedes"],
                    inline=True,
                    style={"display": "inline-block"},
                ),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Health Colors:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Checklist(
                    id="gnn-graph-health-toggle",
                    options=[{"label": " Show", "value": "on"}],
                    value=["on"],
                    style={"display": "inline-block"},
                ),
            ]),
        ]),

        # Graph visualization
        html.Div(className="chart-card", style={"marginTop": "16px"}, children=[
            dcc.Loading(
                cyto.Cytoscape(
                    id="gnn-cytoscape-graph",
                    layout={"name": "cose", "animate": False, "randomize": False},
                    style={"width": "100%", "height": "550px"},
                    stylesheet=CYTOSCAPE_STYLESHEET,
                    elements=[],
                ) if CYTOSCAPE_AVAILABLE else html.Div()
            ),
        ]),

        # Legend
        html.Div(className="chart-card", style={"marginTop": "16px", "padding": "16px"}, children=[
            html.H4("Legend", style={"marginBottom": "12px"}),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "24px"}, children=[
                # Node types
                html.Div([
                    html.Strong("Node Types:", style={"display": "block", "marginBottom": "8px"}),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#3498db",
                                       "borderRadius": "50%", "marginRight": "8px"}),
                        html.Span("Blocks (200/ship)"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#27ae60",
                                       "marginRight": "8px"}),
                        html.Span("SPMTs"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "0", "height": "0",
                                       "borderLeft": "8px solid transparent",
                                       "borderRight": "8px solid transparent",
                                       "borderBottom": "16px solid #e74c3c",
                                       "marginRight": "8px"}),
                        html.Span("Goliath Cranes (109m)"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#9b59b6",
                                       "marginRight": "8px"}),
                        html.Span("Facilities"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#1abc9c",
                                       "borderRadius": "50%", "marginRight": "8px"}),
                        html.Span("Ships (LNG Carriers)"),
                    ]),
                ]),
                # Edge types
                html.Div([
                    html.Strong("Edge Types:", style={"display": "block", "marginBottom": "8px"}),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "24px", "height": "2px", "backgroundColor": "#3498db",
                                       "marginRight": "8px"}),
                        html.Span("needs_transport"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "24px", "height": "2px", "backgroundColor": "#e74c3c",
                                       "marginRight": "8px"}),
                        html.Span("needs_lift"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "24px", "height": "2px", "backgroundColor": "#e74c3c",
                                       "marginRight": "8px"}),
                        html.Span("precedes (erection order)"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                        html.Div(style={"width": "24px", "height": "1px", "backgroundColor": "#9b59b6",
                                       "borderStyle": "dashed", "marginRight": "8px"}),
                        html.Span("at (location)"),
                    ]),
                ]),
                # Health status
                html.Div([
                    html.Strong("Health Status:", style={"display": "block", "marginBottom": "8px"}),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#27ae60",
                                       "borderRadius": "50%", "marginRight": "8px"}),
                        html.Span("Healthy (>=60%)"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#f39c12",
                                       "borderRadius": "50%", "border": "2px solid #d68910", "marginRight": "8px"}),
                        html.Span("Warning (40-60%)"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#e74c3c",
                                       "borderRadius": "50%", "border": "3px solid #c0392b", "marginRight": "8px"}),
                        html.Span("Critical (<40%)"),
                    ]),
                ]),
            ]),
        ]),

        # Node details panel
        html.Div(className="chart-card", style={"marginTop": "16px"}, children=[
            html.H4("Selected Node Details", style={"marginBottom": "12px"}),
            html.Div(id="gnn-node-details", children=[
                html.P("Click on a node to see its features", className="text-muted"),
            ]),
        ]),

        # Graph statistics
        html.Div(className="two-col", style={"marginTop": "16px"}, children=[
            html.Div(className="chart-card", children=[
                html.H4("Graph Statistics"),
                html.Div(id="gnn-graph-stats", children=[
                    html.P("Loading...", className="text-muted"),
                ]),
            ]),
            html.Div(className="chart-card", children=[
                html.H4("Edge Counts by Type"),
                dcc.Loading(dcc.Graph(id="gnn-edge-counts-chart", style={"height": "250px"})),
            ]),
        ]),
    ])


def plate_analytics_layout() -> html.Div:
    """Layout for plate-level production analytics."""
    return html.Div([
        html.H3("Plate-Level Analytics", className="section-header"),

        # KPIs
        html.Div(className="kpi-grid", children=[
            _kpi_card("Total Plates", "kpi-total-plates"),
            _kpi_card("Avg Plates/Block", "kpi-avg-plates"),
            _kpi_card("Total Plate Area", "kpi-plate-area"),
            _kpi_card("Processing Source", "kpi-plate-source"),
        ]),

        # Charts row 1
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.H4("Processing Time: Plate-Based vs Lognormal"),
                dcc.Loading(dcc.Graph(id="plate-vs-lognormal-chart", style={"height": "350px"})),
            ]),
            html.Div(className="chart-card", children=[
                html.H4("Plate Count Distribution by Block Type"),
                dcc.Loading(dcc.Graph(id="plate-count-distribution", style={"height": "350px"})),
            ]),
        ]),

        # Charts row 2
        html.Div(className="two-col", style={"marginTop": "16px"}, children=[
            html.Div(className="chart-card", children=[
                html.H4("Processing Time vs Plate Count"),
                dcc.Loading(dcc.Graph(id="plate-time-scatter", style={"height": "350px"})),
            ]),
            html.Div(className="chart-card", children=[
                html.H4("Stage Bottleneck Analysis (Plate-Weighted)"),
                dcc.Loading(dcc.Graph(id="plate-bottleneck-chart", style={"height": "350px"})),
            ]),
        ]),

        # Blocks table with plate columns
        html.Div(className="chart-card", style={"marginTop": "16px"}, children=[
            html.H4("Block Detail (with Plate Data)"),
            dcc.Loading(dash_table.DataTable(
                id="plate-blocks-table",
                columns=[
                    {"name": "Block ID", "id": "id"},
                    {"name": "Ship", "id": "ship_id"},
                    {"name": "Type", "id": "block_type"},
                    {"name": "Plates", "id": "n_plates"},
                    {"name": "Area (m2)", "id": "plate_area_m2"},
                    {"name": "Weight (t)", "id": "weight"},
                    {"name": "Stage", "id": "current_stage"},
                    {"name": "Source", "id": "processing_source"},
                ],
                data=[],
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
            )),
        ]),
    ])

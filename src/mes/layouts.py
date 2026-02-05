"""Layout definitions for the MES dashboard.

Includes layouts for single-yard and dual-yard (EB Quonset/Groton) views.
"""

from __future__ import annotations

from dash import html, dcc, dash_table

try:
    import dash_cytoscape as cyto
    CYTOSCAPE_AVAILABLE = True
except ImportError:
    CYTOSCAPE_AVAILABLE = False


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
        html.Div(className="kpi-grid", children=[
            _kpi_card("Blocks Completed", "kpi-blocks"),
            _kpi_card("Breakdowns", "kpi-breakdowns"),
            _kpi_card("Planned Maint.", "kpi-planned"),
            _kpi_card("Avg Tardiness", "kpi-tardiness"),
            _kpi_card("SPMT Utilization", "kpi-spmt-util"),
            _kpi_card("Crane Utilization", "kpi-crane-util"),
            _kpi_card("OEE", "kpi-oee"),
            _kpi_card("Empty Travel", "kpi-empty"),
        ]),
        html.Div(className="chart-card", children=[
            html.H3("Metric Trends", className="section-header"),
            dcc.Loading(dcc.Graph(id="overview-kpi-trends")),
        ]),
    ])


def blocks_layout() -> html.Div:
    return html.Div([
        html.H3("Block Status", className="section-header"),
        dcc.Loading(dash_table.DataTable(
            id="blocks-table",
            columns=[
                {"name": "ID", "id": "id"},
                {"name": "Status", "id": "status"},
                {"name": "Stage", "id": "current_stage"},
                {"name": "Location", "id": "location"},
                {"name": "Due Date", "id": "due_date"},
                {"name": "Completion %", "id": "completion_pct"},
            ],
            data=[],
            page_size=20,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_data_conditional=[
                {"if": {"filter_query": "{status} = placed_on_dock"}, "backgroundColor": "#d4edda", "color": "#155724"},
                {"if": {"filter_query": "{status} = in_transit"}, "backgroundColor": "#fff3cd", "color": "#856404"},
                {"if": {"filter_query": "{status} = waiting"}, "backgroundColor": "#f8f9fa"},
            ],
        )),
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
                {"if": {"filter_query": "{health_hydraulic} < 40", "column_id": "health_hydraulic"}, "backgroundColor": "#f8d7da", "color": "#721c24"},
                {"if": {"filter_query": "{health_tires} < 40", "column_id": "health_tires"}, "backgroundColor": "#f8d7da", "color": "#721c24"},
                {"if": {"filter_query": "{health_engine} < 40", "column_id": "health_engine"}, "backgroundColor": "#f8d7da", "color": "#721c24"},
            ],
        )),
        html.Div(className="chart-card", children=[
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
                        {"if": {"filter_query": "{health_value} < 30"}, "backgroundColor": "#f8d7da", "color": "#721c24"},
                        {"if": {"filter_query": "{health_value} >= 30 && {health_value} < 50"}, "backgroundColor": "#fff3cd", "color": "#856404"},
                        {"if": {"filter_query": "{health_value} >= 50"}, "backgroundColor": "#d4edda", "color": "#155724"},
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
# DUAL SHIPYARD MAP LAYOUTS (EB Quonset/Groton)
# ============================================================================

def _map_controls() -> html.Div:
    """Control bar for the shipyard map views."""
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
                    {"label": " Queues", "value": "queues"},
                ],
                value=["spmts", "cranes", "queues"],
                inline=True,
                style={"display": "inline-block"},
            ),
        ]),
    ])


def quonset_map_layout() -> html.Div:
    """Layout for the Quonset Point shipyard map."""
    return html.Div([
        html.H3("Quonset Point, RI - Module Fabrication", className="section-header"),
        _map_controls(),
        html.Div(className="chart-card map-container", children=[
            dcc.Loading(dcc.Graph(
                id="quonset-map",
                config={
                    "displayModeBar": True,
                    "scrollZoom": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )),
        ]),
        html.Div(id="quonset-detail-panel", className="detail-panel", children=[
            html.P("Click on equipment for details", className="text-muted"),
        ]),
    ])


def groton_map_layout() -> html.Div:
    """Layout for the Groton shipyard map."""
    return html.Div([
        html.H3("Groton, CT - Final Assembly & Launch", className="section-header"),
        _map_controls(),
        html.Div(className="chart-card map-container", children=[
            dcc.Loading(dcc.Graph(
                id="groton-map",
                config={
                    "displayModeBar": True,
                    "scrollZoom": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )),
        ]),
        html.Div(id="groton-detail-panel", className="detail-panel", children=[
            html.P("Click on equipment for details", className="text-muted"),
        ]),
    ])


def dual_map_layout() -> html.Div:
    """Layout for the dual-yard split view."""
    return html.Div([
        html.H3("Electric Boat Dual Shipyard Overview", className="section-header"),

        # Control bar with health toggle and playback
        html.Div(className="map-controls", children=[
            html.Div([
                html.Label("Health Overlay:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Checklist(
                    id="dual-map-health-toggle",
                    options=[{"label": " Show", "value": "on"}],
                    value=[],
                    style={"display": "inline-block"},
                ),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Playback:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Checklist(
                    id="playback-mode-toggle",
                    options=[{"label": " Enable", "value": "on"}],
                    value=[],
                    style={"display": "inline-block"},
                ),
            ]),
        ]),

        # Playback controls (shown when playback mode is enabled)
        html.Div(id="playback-controls-container", children=[
            playback_controls(),
        ]),

        # Playback state stores
        dcc.Store(id="playback-state", data={"playing": False, "time": None, "speed": 1}),
        dcc.Store(id="playback-timeline", data={"min_time": 0, "max_time": 100, "timestamps": []}),
        dcc.Interval(id="playback-interval", interval=500, disabled=True),

        # Split view with two maps side by side
        html.Div(className="dual-map-container", style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "16px",
        }, children=[
            # Quonset map
            html.Div(className="chart-card", children=[
                html.H4("🔧 Quonset Point, RI", style={"marginBottom": "8px", "color": "#2c3e50"}),
                dcc.Loading(dcc.Graph(id="dual-quonset-map", style={"height": "450px"})),
            ]),
            # Groton map
            html.Div(className="chart-card", children=[
                html.H4("🚢 Groton, CT", style={"marginBottom": "8px", "color": "#2c3e50"}),
                dcc.Loading(dcc.Graph(id="dual-groton-map", style={"height": "450px"})),
            ]),
        ]),

        # Transit view below
        html.Div(className="chart-card", style={"marginTop": "16px"}, children=[
            html.H4("🌊 Barge Transit", style={"marginBottom": "8px"}),
            dcc.Loading(dcc.Graph(id="transit-map", style={"height": "200px"})),
        ]),

        # Current time display
        html.Div(id="playback-current-time", style={
            "textAlign": "center",
            "padding": "8px",
            "fontSize": "13px",
            "color": "#7f8c8d",
        }),
    ])


def alerts_banner() -> html.Div:
    """Alerts banner component for cross-yard notifications."""
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
        html.Button("⏪", id="playback-rewind", className="playback-btn",
                    style={"padding": "4px 12px", "border": "1px solid #ddd", "borderRadius": "4px"}),
        html.Button("▶️ Play", id="playback-play", className="playback-btn",
                    style={"padding": "4px 12px", "border": "1px solid #ddd", "borderRadius": "4px"}),
        html.Button("⏩", id="playback-forward", className="playback-btn",
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
        html.Button("↻ Live", id="playback-live", className="playback-btn",
                    style={"padding": "4px 12px", "border": "1px solid #ddd", "borderRadius": "4px"}),
    ])


def dependencies_layout() -> html.Div:
    """Layout for block dependency visualization."""
    return html.Div([
        html.H3("Block Dependencies", className="section-header"),
        html.Div(className="map-controls", children=[
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
                        {"name": "Status", "id": "status"},
                        {"name": "Stage", "id": "current_stage"},
                        {"name": "Completion", "id": "completion_pct"},
                    ],
                    data=[],
                    page_size=10,
                    style_data_conditional=[
                        {"if": {"filter_query": "{status} = placed_on_dock"},
                         "backgroundColor": "#d4edda", "color": "#155724"},
                        {"if": {"filter_query": "{status} = in_process"},
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
        "selector": "node[type = 'crane']",
        "style": {
            "background-color": "#e67e22",
            "label": "data(label)",
            "width": 35,
            "height": 35,
            "shape": "diamond",
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
        "style": {"line-color": "#e67e22", "width": 2, "curve-style": "bezier", "opacity": 0.6}
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
            "Visualizes the heterogeneous graph representation used by the GNN encoder.",
            style={"color": "#7f8c8d", "marginBottom": "16px"}
        ),

        # Controls
        html.Div(className="map-controls", children=[
            html.Div([
                html.Label("Yard Filter:", style={"marginRight": "8px", "fontWeight": "600"}),
                dcc.Dropdown(
                    id="gnn-graph-yard-filter",
                    options=[
                        {"label": "All", "value": "all"},
                        {"label": "Quonset Point", "value": "quonset"},
                        {"label": "Groton", "value": "groton"},
                    ],
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
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#3498db", "borderRadius": "50%", "marginRight": "8px"}),
                        html.Span("Blocks"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#27ae60", "marginRight": "8px"}),
                        html.Span("SPMTs"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#e67e22", "transform": "rotate(45deg)", "marginRight": "8px"}),
                        html.Span("Cranes"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#9b59b6", "marginRight": "8px"}),
                        html.Span("Facilities"),
                    ]),
                ]),
                # Edge types
                html.Div([
                    html.Strong("Edge Types:", style={"display": "block", "marginBottom": "8px"}),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "24px", "height": "2px", "backgroundColor": "#3498db", "marginRight": "8px"}),
                        html.Span("needs_transport"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "24px", "height": "2px", "backgroundColor": "#e67e22", "marginRight": "8px"}),
                        html.Span("needs_lift"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "24px", "height": "2px", "backgroundColor": "#e74c3c", "marginRight": "8px"}),
                        html.Span("precedes (→)"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                        html.Div(style={"width": "24px", "height": "1px", "backgroundColor": "#9b59b6", "borderStyle": "dashed", "marginRight": "8px"}),
                        html.Span("at (location)"),
                    ]),
                ]),
                # Health status
                html.Div([
                    html.Strong("Health Status:", style={"display": "block", "marginBottom": "8px"}),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#27ae60", "borderRadius": "50%", "marginRight": "8px"}),
                        html.Span("Healthy (≥60%)"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#f39c12", "borderRadius": "50%", "border": "2px solid #d68910", "marginRight": "8px"}),
                        html.Span("Warning (40-60%)"),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                        html.Div(style={"width": "16px", "height": "16px", "backgroundColor": "#e74c3c", "borderRadius": "50%", "border": "3px solid #c0392b", "marginRight": "8px"}),
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

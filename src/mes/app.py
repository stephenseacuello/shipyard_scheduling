"""Dash application for the HD Hyundai Heavy Industries MES dashboard.

HHI Ulsan Shipyard - Mipo Bay, South Korea
LNG Carrier Production Monitoring System
"""

from __future__ import annotations

import os

import dash
from dash import dcc, html


def create_app() -> dash.Dash:
    """Create and return the Dash application."""
    assets_path = os.path.join(os.path.dirname(__file__), "assets")
    app = dash.Dash(__name__, assets_folder=assets_path, suppress_callback_exceptions=True)
    app.title = "HHI Ulsan MES - Shipyard Monitoring"

    app.layout = html.Div(className="app-container", children=[
        # Header with HHI Branding
        html.Div(className="app-header", children=[
            html.H1("HD Hyundai Heavy Industries"),
            html.P("Ulsan Shipyard MES Dashboard", className="subtitle"),
            html.Div(className="location-badge", children=[
                "Mipo Bay, South Korea • 35.51°N, 129.42°E"
            ]),
        ]),

        # Alerts banner (critical notifications)
        html.Div(id="alerts-banner", className="alerts-banner"),

        # Navigation Tabs
        dcc.Tabs(
            id="tabs",
            value="hhi-map",  # Default to HHI map view
            className="custom-tabs",
            children=[
                dcc.Tab(label="Shipyard Map", value="hhi-map", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Dry Docks", value="docks", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Overview", value="overview", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Blocks", value="blocks", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Ships", value="ships", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Fleet", value="fleet", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Health", value="health", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Operations", value="operations", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Dependencies", value="dependencies", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="GNN Graph", value="gnn-graph", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Plates", value="plates", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="KPIs", value="kpis", className="tab", selected_className="tab--selected"),
            ],
        ),

        # Control bar
        html.Div(className="control-bar", children=[
            html.Label("Time Range:"),
            dcc.Dropdown(
                id="time-range",
                options=[
                    {"label": "Last 500 steps", "value": 500},
                    {"label": "Last 1000 steps", "value": 1000},
                    {"label": "Last 2500 steps", "value": 2500},
                    {"label": "All Data", "value": 0},
                ],
                value=0,
                clearable=False,
                style={"width": "160px"},
            ),
            html.Label("Auto-refresh:"),
            dcc.Checklist(
                id="auto-refresh-toggle",
                options=[{"label": " Enabled", "value": "on"}],
                value=["on"],
                style={"display": "inline-block"},
            ),
        ]),

        # Persistent store for time range
        dcc.Store(id="time-range-store", data=0),

        # Tab content
        html.Div(id="tab-content", className="tab-content"),

        # Refresh interval (controlled by toggle)
        dcc.Interval(id="interval", interval=5000, n_intervals=0),

        # Footer
        html.Div(className="app-footer", children=[
            html.P([
                "HD Hyundai Heavy Industries • Ulsan Shipyard MES • ",
                html.A("Shipyard Scheduling RL Framework", href="https://github.com/"),
                " v1.0.0"
            ]),
        ]),
    ])

    from .callbacks import register_callbacks
    register_callbacks(app)

    return app


def main() -> None:
    """Run the dashboard server."""
    app = create_app()
    print("\n" + "="*60)
    print("  HD HYUNDAI HEAVY INDUSTRIES - ULSAN SHIPYARD MES")
    print("  Mipo Bay, South Korea | LNG Carrier Production")
    print("="*60)
    print("  Dashboard: http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=True)


if __name__ == "__main__":
    main()

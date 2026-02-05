"""Dash application for the shipyard MES dashboard."""

from __future__ import annotations

import os

import dash
from dash import dcc, html


def create_app() -> dash.Dash:
    """Create and return the Dash application."""
    assets_path = os.path.join(os.path.dirname(__file__), "assets")
    app = dash.Dash(__name__, assets_folder=assets_path, suppress_callback_exceptions=True)
    app.title = "Shipyard MES"

    app.layout = html.Div(className="app-container", children=[
        # Header
        html.Div(className="app-header", children=[
            html.H1("Shipyard MES Dashboard"),
            html.P("Electric Boat Dual-Yard Monitoring • Quonset Point ↔ Groton", className="subtitle"),
        ]),

        # Alerts banner (cross-yard notifications)
        html.Div(id="alerts-banner", className="alerts-banner"),

        # Tabs
        dcc.Tabs(
            id="tabs",
            value="dual-map",  # Default to dual map view
            className="custom-tabs",
            children=[
                dcc.Tab(label="🗺️ Dual View", value="dual-map", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="🔧 Quonset", value="quonset-map", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="🚢 Groton", value="groton-map", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="🔗 Dependencies", value="dependencies", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="🧠 Graph", value="gnn-graph", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Overview", value="overview", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Blocks", value="blocks", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Fleet", value="fleet", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Health", value="health", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Operations", value="operations", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="KPIs", value="kpis", className="tab", selected_className="tab--selected"),
            ],
        ),

        # Control bar
        html.Div(className="control-bar", children=[
            html.Label("Time Range:"),
            dcc.Dropdown(
                id="time-range",
                options=[
                    {"label": "Last 500", "value": 500},
                    {"label": "Last 1000", "value": 1000},
                    {"label": "Last 2500", "value": 2500},
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
            html.P("Shipyard Scheduling RL Framework v0.1.0"),
        ]),
    ])

    from .callbacks import register_callbacks
    register_callbacks(app)

    return app


def main() -> None:
    app = create_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()

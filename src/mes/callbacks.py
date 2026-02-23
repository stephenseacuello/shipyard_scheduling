"""Callback definitions for the MES dashboard."""

from __future__ import annotations

import dash
from dash import Output, Input, State, html, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

from . import layouts
from .database import (
    fetch_query, fetch_health_history, fetch_queue_history, fetch_block_events,
    fetch_ship_events, fetch_position_at_time, fetch_playback_timeline, fetch_available_timestamps,
    list_simulation_runs, fetch_playback_timeline_for_run,
    fetch_plate_stats, fetch_plate_processing_times,
)
from .map_builder import build_quonset_map, build_groton_map, build_transit_map
from .dependency_graph import build_dependency_graph, build_critical_path_view

# Try to import leaflet map builder
try:
    from .map_builder_leaflet import build_hhi_map as build_leaflet_map
    LEAFLET_AVAILABLE = True
except ImportError:
    LEAFLET_AVAILABLE = False
    build_leaflet_map = None

CHART_TEMPLATE = "plotly_white"
COLORS = {
    "primary": "#2c3e50",
    "accent": "#3498db",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "purple": "#9b59b6",
    "teal": "#1abc9c",
    "orange": "#e67e22",
}
STAGE_COLORS = {
    # Full HHI production stages (uppercase)
    "STEEL_CUTTING": "#3498db",
    "PART_FABRICATION": "#9b59b6",
    "PANEL_ASSEMBLY": "#e67e22",
    "BLOCK_ASSEMBLY": "#f39c12",
    "BLOCK_OUTFITTING": "#1abc9c",
    "PAINTING": "#e74c3c",
    "PRE_ERECTION": "#2c3e50",
    "ERECTION": "#27ae60",
    "QUAY_OUTFITTING": "#8e44ad",
    "SEA_TRIALS": "#16a085",
    "DELIVERY": "#2ecc71",
    "COMPLETED": "#2ecc71",
    # Legacy lowercase names
    "cutting": "#3498db",
    "panel": "#e67e22",
    "assembly": "#27ae60",
    "outfitting": "#e74c3c",
    "paint": "#9b59b6",
    "pre_erection": "#1abc9c",
    "DOCK": "#2c3e50",
    "dock": "#2c3e50",
    # Transit states
    "TRANSIT": "#95a5a6",
    "IN_TRANSIT": "#95a5a6",
    "IDLE": "#bdc3c7",
}
FACILITY_COLORS = ["#3498db", "#e67e22", "#27ae60", "#e74c3c", "#9b59b6", "#1abc9c"]


def _empty_fig(msg: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=CHART_TEMPLATE,
        annotations=[dict(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16, color="#95a5a6"))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=250, margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


def register_callbacks(app: dash.Dash) -> None:
    """Register all callbacks."""

    # ── Auto-refresh toggle ──
    @app.callback(Output("interval", "disabled"), Input("auto-refresh-toggle", "value"))
    def toggle_refresh(value):
        return "on" not in (value or [])

    # ── Tab routing ──
    @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(tab):
        tab_map = {
            "dual-map": layouts.hhi_map_layout,
            "quonset-map": layouts.hhi_map_layout,
            "groton-map": layouts.hhi_map_layout,
            "hhi-map": layouts.hhi_map_layout,
            "dependencies": layouts.dependencies_layout,
            "gnn-graph": layouts.gnn_graph_layout,
            "plates": layouts.plate_analytics_layout,
            "overview": layouts.overview_layout,
            "blocks": layouts.blocks_layout,
            "fleet": layouts.fleet_layout,
            "health": layouts.health_layout,
            "operations": layouts.operations_layout,
            "kpis": layouts.kpis_layout,
            "docks": layouts.docks_layout,
            "ships": layouts.ships_layout,
        }
        fn = tab_map.get(tab)
        return fn() if fn else html.Div()

    # ── Alerts banner ──
    @app.callback(Output("alerts-banner", "children"), Input("interval", "n_intervals"))
    def update_alerts(n):
        alerts = []

        # Check for equipment health issues
        spmts = fetch_query("SELECT * FROM spmts WHERE health_hydraulic < 30 OR health_tires < 30 OR health_engine < 30")
        for s in (spmts or []):
            alerts.append(html.Div(className="alert alert-warning", children=[
                html.Strong("⚠️ HEALTH "),
                f"{s['id']} requires maintenance (low component health)"
            ]))

        # Check for broken equipment
        broken = fetch_query("SELECT * FROM spmts WHERE status = 'broken_down'")
        for b in (broken or []):
            alerts.append(html.Div(className="alert alert-danger", children=[
                html.Strong("🔴 BREAKDOWN "),
                f"{b['id']} is non-operational"
            ]))

        # Check for crane issues
        crane_issues = fetch_query("SELECT * FROM cranes WHERE health_cable < 30 OR health_motor < 30")
        for c in (crane_issues or []):
            alerts.append(html.Div(className="alert alert-warning", children=[
                html.Strong("⚠️ CRANE "),
                f"{c['id']} requires maintenance"
            ]))

        if not alerts:
            return []  # No alerts to show

        return alerts

    # ── HHI Shipyard Map (Leaflet) with Playback Support ──
    if LEAFLET_AVAILABLE:
        @app.callback(
            Output("hhi-map-container", "children"),
            [Input("interval", "n_intervals"),
             Input("map-health-toggle", "value"),
             Input("map-layers", "value"),
             Input("map-view-select", "value"),
             Input("playback-state", "data"),
             Input("playback-mode-toggle", "value"),
             Input("playback-run-selector", "value")],
        )
        def update_hhi_map(n, health_toggle, layers, view_select, playback_state, playback_enabled, selected_run_id):
            """Build and update the HHI shipyard map with playback support."""
            playback_on = "on" in (playback_enabled or [])

            # Determine data source: playback or live
            if playback_on and playback_state and playback_state.get("time") is not None:
                # Use historical data from selected run
                run_id = selected_run_id if selected_run_id else None
                historical_data = fetch_position_at_time(playback_state["time"], run_id=run_id)

                # Map historical data to expected format
                blocks = historical_data.get("blocks", [])
                spmts = historical_data.get("spmts", [])
                cranes = historical_data.get("cranes", [])
                ships = historical_data.get("ships", [])
            else:
                # Fetch current state from database (live mode)
                blocks = fetch_query("SELECT * FROM hhi_blocks") or []
                spmts = fetch_query("SELECT * FROM spmts") or []
                cranes = fetch_query("SELECT * FROM goliath_cranes") or []
                ships = fetch_query("SELECT * FROM ships") or []

            # Build map data dict (use goliath_cranes key for leaflet map)
            map_data = {
                "blocks": blocks,
                "spmts": spmts,
                "goliath_cranes": cranes,
                "ships": ships,
            }

            # Parse toggle options
            show_health = "on" in (health_toggle or [])
            show_blocks = "blocks" in (layers or [])
            show_spmts = "spmts" in (layers or [])
            show_cranes = "cranes" in (layers or [])

            # Build the leaflet map
            try:
                map_element = build_leaflet_map(
                    map_data,
                    map_id="hhi-leaflet-map",
                    show_health=show_health,
                    show_blocks=show_blocks,
                )
                # Add playback indicator if in historical mode
                if playback_on and playback_state and playback_state.get("time") is not None:
                    return html.Div([
                        html.Div(
                            f"📼 Historical: t={int(playback_state['time'])}",
                            style={
                                "position": "absolute", "top": "10px", "left": "60px",
                                "zIndex": "1000", "backgroundColor": "rgba(231, 76, 60, 0.9)",
                                "color": "white", "padding": "4px 8px", "borderRadius": "4px",
                                "fontSize": "12px", "fontWeight": "bold",
                            }
                        ),
                        map_element,
                    ], style={"position": "relative"})
                return map_element
            except Exception as e:
                return html.Div([
                    html.P(f"Error loading map: {e}", style={"color": "red"}),
                    html.P("Run a simulation first to populate the database."),
                ])

    # ── Overview KPI cards ──
    @app.callback(
        [Output("kpi-blocks", "children"), Output("kpi-ships", "children"),
         Output("kpi-breakdowns", "children"), Output("kpi-planned", "children"),
         Output("kpi-tardiness", "children"), Output("kpi-ontime", "children"),
         Output("kpi-spmt-util", "children"), Output("kpi-crane-util", "children"),
         # OEE Metrics
         Output("kpi-oee", "children"), Output("kpi-availability", "children"),
         Output("kpi-performance", "children"), Output("kpi-quality", "children")],
        Input("interval", "n_intervals"),
    )
    def update_kpis(n):
        metrics = fetch_query("SELECT * FROM metrics ORDER BY time DESC LIMIT 1")
        spmts = fetch_query("SELECT status FROM spmts")
        # Try goliath_cranes first (HHI), fall back to cranes
        cranes = fetch_query("SELECT status FROM goliath_cranes")
        if not cranes:
            cranes = fetch_query("SELECT status FROM cranes")
        ships = fetch_query("SELECT status FROM ships")
        ships_delivered_count = len([s for s in (ships or []) if s.get("status") == "delivered"])
        ships_in_progress = len([s for s in (ships or []) if s.get("status") not in ("delivered", None)])
        docks = fetch_query("SELECT current_ship FROM dry_docks WHERE current_ship IS NOT NULL AND current_ship != ''")
        blocks_data = fetch_query("SELECT COUNT(*) as cnt FROM hhi_blocks")
        if not blocks_data:
            blocks_data = fetch_query("SELECT COUNT(*) as cnt FROM blocks")

        # Get on-time data (blocks completed before due date)
        ontime_data = fetch_query("""
            SELECT COUNT(*) as ontime FROM hhi_blocks
            WHERE current_stage = 'COMPLETED' OR current_stage = 'ERECTION'
        """)
        if not ontime_data:
            ontime_data = fetch_query("""
                SELECT COUNT(*) as ontime FROM blocks
                WHERE current_stage = 'COMPLETED' OR current_stage = 'ERECTION'
            """)

        if not metrics:
            # Still show data even without metrics
            total_blocks = blocks_data[0]["cnt"] if blocks_data else 0
            return (
                str(total_blocks), str(ships_delivered_count), "0", "0",
                "0.0", "—", "0%", "0%",
                "—", "—", "—", "—",
            )

        m = metrics[0]
        blocks = m.get("blocks_completed", 0) or 0
        breakdowns = m.get("breakdowns", 0) or 0
        planned = m.get("planned_maintenance", 0) or 0
        tardiness = m.get("total_tardiness", 0.0) or 0.0
        sim_time = m.get("time", 1) or 1

        # Calculate utilization
        spmt_busy = sum(1 for s in (spmts or []) if s.get("status", "idle").lower() not in ("idle", "")) if spmts else 0
        spmt_total = max(len(spmts) if spmts else 1, 1)
        crane_busy = sum(1 for c in (cranes or []) if c.get("status", "idle").lower() not in ("idle", "")) if cranes else 0
        crane_total = max(len(cranes) if cranes else 1, 1)
        spmt_util = spmt_busy / spmt_total
        crane_util = crane_busy / crane_total

        # Calculate On-Time % (blocks completed before due date)
        total_blocks_count = blocks_data[0]["cnt"] if blocks_data else 0
        completed_blocks = blocks
        ontime_pct = (completed_blocks / max(total_blocks_count, 1)) * 100 if total_blocks_count > 0 else 0.0

        # OEE Calculation:
        # Availability = (Total Time - Downtime) / Total Time
        # For shipyard: downtime = breakdown time (estimated as breakdowns * avg_repair_time)
        avg_repair_time = 10.0  # hours per breakdown
        downtime = breakdowns * avg_repair_time
        availability = max(0, (sim_time - downtime) / max(sim_time, 1))

        # Performance = Actual Throughput / Theoretical Max Throughput
        # Theoretical: ~0.1 blocks/hour with full utilization
        theoretical_rate = 0.08  # blocks per hour (conservative estimate)
        actual_rate = blocks / max(sim_time, 1)
        performance = min(1.0, actual_rate / theoretical_rate) if theoretical_rate > 0 else 0.0

        # Quality = (Blocks without rework) / Total Blocks
        # Assume quality is high (99%) since simulation doesn't model rework explicitly
        quality = 0.99 if blocks > 0 else 1.0

        # OEE = Availability × Performance × Quality
        oee = availability * performance * quality

        # Show ships delivered / total
        ships_display = f"{ships_delivered_count}" if ships_in_progress == 0 else f"{ships_delivered_count}/{ships_in_progress + ships_delivered_count}"

        return (
            str(blocks), ships_display, str(breakdowns), str(planned),
            f"{tardiness:.1f}", f"{ontime_pct:.1f}%", f"{spmt_util:.0%}", f"{crane_util:.0%}",
            f"{oee:.1%}", f"{availability:.1%}", f"{performance:.1%}", f"{quality:.1%}",
        )

    # ── Overview trends (dual axis) ──
    @app.callback(Output("overview-kpi-trends", "figure"),
                  [Input("interval", "n_intervals"), Input("time-range", "value")])
    def update_overview_trends(n, time_range):
        rows = fetch_query("SELECT time, blocks_completed, breakdowns, total_tardiness FROM metrics ORDER BY time")
        if not rows:
            return _empty_fig("No metrics recorded yet")
        if time_range and time_range > 0:
            max_t = rows[-1]["time"]
            rows = [r for r in rows if r["time"] >= max_t - time_range]
        times = [r["time"] for r in rows]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=times, y=[r["blocks_completed"] for r in rows],
                                 name="Completed", mode="lines+markers", marker=dict(size=5),
                                 line=dict(color=COLORS["success"], width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=times, y=[r["breakdowns"] for r in rows],
                                 name="Breakdowns", mode="lines+markers", marker=dict(size=5),
                                 line=dict(color=COLORS["danger"], width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=times, y=[r["total_tardiness"] for r in rows],
                                 name="Tardiness", mode="lines",
                                 line=dict(color=COLORS["warning"], width=2, dash="dot")), secondary_y=True)
        fig.update_layout(
            template=CHART_TEMPLATE, height=320,
            margin=dict(t=30, b=40, l=50, r=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_xaxes(title_text="Simulation Time")
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Tardiness", secondary_y=True)
        return fig

    # ── Production Throughput Chart ──
    @app.callback(Output("throughput-chart", "figure"),
                  [Input("interval", "n_intervals"), Input("time-range", "value")])
    def update_throughput_chart(n, time_range):
        """Show production throughput over time (blocks completed per time unit)."""
        rows = fetch_query("SELECT time, blocks_completed FROM metrics ORDER BY time")
        if not rows or len(rows) < 2:
            return _empty_fig("Insufficient metrics for throughput calculation")
        if time_range and time_range > 0:
            max_t = rows[-1]["time"]
            rows = [r for r in rows if r["time"] >= max_t - time_range]

        times = [r["time"] for r in rows]
        blocks = [r["blocks_completed"] or 0 for r in rows]

        # Calculate rolling throughput (blocks completed in last N time steps)
        window = max(5, len(rows) // 20)  # 5% window or minimum 5 points
        throughput = []
        for i in range(len(rows)):
            if i < window:
                rate = blocks[i] / max(times[i], 1) if times[i] > 0 else 0
            else:
                delta_blocks = blocks[i] - blocks[i - window]
                delta_time = times[i] - times[i - window]
                rate = delta_blocks / delta_time if delta_time > 0 else 0
            throughput.append(rate * 100)  # Per 100 time units

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=throughput,
            name="Throughput",
            mode="lines",
            fill="tozeroy",
            line=dict(color=COLORS["accent"], width=2),
            fillcolor="rgba(52, 152, 219, 0.2)",
        ))
        # Add cumulative blocks as secondary trace
        fig.add_trace(go.Scatter(
            x=times, y=blocks,
            name="Cumulative",
            mode="lines",
            line=dict(color=COLORS["success"], width=2, dash="dot"),
            yaxis="y2",
        ))
        fig.update_layout(
            template=CHART_TEMPLATE, height=250,
            margin=dict(t=20, b=40, l=50, r=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            xaxis_title="Simulation Time",
            yaxis_title="Blocks/100 time",
            yaxis2=dict(title="Total Blocks", overlaying="y", side="right"),
        )
        return fig

    # ── Block Stage Distribution Chart ──
    @app.callback(Output("stage-distribution-chart", "figure"),
                  Input("interval", "n_intervals"))
    def update_stage_distribution(n):
        """Show current distribution of blocks across production stages."""
        # Try HHI blocks first
        rows = fetch_query("""
            SELECT current_stage, COUNT(*) as count
            FROM hhi_blocks
            GROUP BY current_stage
        """)
        if not rows:
            rows = fetch_query("""
                SELECT current_stage, COUNT(*) as count
                FROM blocks
                GROUP BY current_stage
            """)
        if not rows:
            return _empty_fig("No blocks in database")

        stages = [r["current_stage"] or "UNKNOWN" for r in rows]
        counts = [r["count"] for r in rows]

        # Define stage order and colors
        stage_order = [
            "STEEL_CUTTING", "PART_FABRICATION", "PANEL_ASSEMBLY",
            "BLOCK_ASSEMBLY", "BLOCK_OUTFITTING", "PAINTING",
            "PRE_ERECTION", "ERECTION", "COMPLETED"
        ]
        stage_colors = {
            "STEEL_CUTTING": "#3498db",
            "PART_FABRICATION": "#9b59b6",
            "PANEL_ASSEMBLY": "#e67e22",
            "BLOCK_ASSEMBLY": "#f39c12",
            "BLOCK_OUTFITTING": "#1abc9c",
            "PAINTING": "#e74c3c",
            "PRE_ERECTION": "#2c3e50",
            "ERECTION": "#27ae60",
            "COMPLETED": "#16a085",
        }

        # Sort by stage order
        stage_data = {s: 0 for s in stage_order}
        for stage, count in zip(stages, counts):
            if stage in stage_data:
                stage_data[stage] = count
            else:
                stage_data[stage] = count  # Handle unknown stages

        labels = list(stage_data.keys())
        values = list(stage_data.values())
        colors = [stage_colors.get(s, "#95a5a6") for s in labels]

        fig = go.Figure(data=[
            go.Pie(
                labels=[s.replace("_", " ").title() for s in labels],
                values=values,
                hole=0.4,
                marker_colors=colors,
                textinfo="percent+value",
                textposition="inside",
            )
        ])
        fig.update_layout(
            template=CHART_TEMPLATE, height=250,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(size=9)),
        )
        return fig

    # ── Facility Bottleneck Chart ──
    @app.callback(Output("bottleneck-chart", "figure"),
                  Input("interval", "n_intervals"))
    def update_bottleneck_chart(n):
        """Show current facility queue depths as horizontal bar chart."""
        rows = fetch_query("""
            SELECT facility_name, queue_depth, processing_count
            FROM queue_history
            WHERE timestamp = (SELECT MAX(timestamp) FROM queue_history)
        """)
        if not rows:
            return _empty_fig("No queue data available")

        facilities = [r["facility_name"] for r in rows]
        queue_depths = [r["queue_depth"] for r in rows]
        processing = [r["processing_count"] for r in rows]
        totals = [q + p for q, p in zip(queue_depths, processing)]

        # Sort by total depth (bottleneck at top)
        sorted_data = sorted(zip(facilities, queue_depths, processing, totals),
                             key=lambda x: x[3], reverse=True)
        facilities = [d[0] for d in sorted_data]
        queue_depths = [d[1] for d in sorted_data]
        processing = [d[2] for d in sorted_data]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=facilities, x=processing, name="Processing",
            orientation="h", marker_color=COLORS["success"],
        ))
        fig.add_trace(go.Bar(
            y=facilities, x=queue_depths, name="Queued",
            orientation="h", marker_color=COLORS["warning"],
        ))
        fig.update_layout(
            template=CHART_TEMPLATE, barmode="stack", height=250,
            margin=dict(t=20, b=40, l=100, r=20),
            xaxis_title="Block Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        )
        return fig

    # ── Equipment Health Summary Chart ──
    @app.callback(Output("health-summary-chart", "figure"),
                  Input("interval", "n_intervals"))
    def update_health_summary(n):
        """Show equipment health summary - grouped by health status."""
        # Get SPMT health
        spmt_rows = fetch_query("""
            SELECT id, health_hydraulic, health_tires, health_engine FROM spmts
        """) or []

        # Get Goliath crane health
        crane_rows = fetch_query("""
            SELECT id, health_hoist, health_trolley, health_gantry FROM goliath_cranes
        """) or []

        if not spmt_rows and not crane_rows:
            return _empty_fig("No equipment data available")

        # Calculate min health for each equipment
        equipment_health = []
        for s in spmt_rows:
            min_h = min(s.get("health_hydraulic", 100),
                       s.get("health_tires", 100),
                       s.get("health_engine", 100))
            equipment_health.append(("SPMT", s["id"], min_h))
        for c in crane_rows:
            min_h = min(c.get("health_hoist", 100),
                       c.get("health_trolley", 100),
                       c.get("health_gantry", 100))
            equipment_health.append(("Goliath", c["id"], min_h))

        # Categorize by health status
        critical = sum(1 for _, _, h in equipment_health if h < 30)
        warning = sum(1 for _, _, h in equipment_health if 30 <= h < 50)
        good = sum(1 for _, _, h in equipment_health if h >= 50)

        fig = go.Figure(data=[
            go.Bar(
                x=["Critical (<30%)", "Warning (30-50%)", "Healthy (>50%)"],
                y=[critical, warning, good],
                marker_color=[COLORS["danger"], COLORS["warning"], COLORS["success"]],
                text=[critical, warning, good],
                textposition="outside",
            )
        ])
        fig.update_layout(
            template=CHART_TEMPLATE, height=250,
            margin=dict(t=30, b=40, l=40, r=20),
            yaxis_title="Equipment Count",
            showlegend=False,
        )
        return fig

    # ── Blocks table ──
    @app.callback(Output("blocks-table", "data"), Input("interval", "n_intervals"))
    def update_blocks(n):
        # Try hhi_blocks first (has ship_id, block_type), fall back to blocks
        # Column names must match DataTable column definitions in layouts.py
        result = fetch_query("""
            SELECT id, ship_id, block_type, current_stage,
                   location as current_location, weight,
                   due_date, ROUND(completion_pct,1) as completion_pct
            FROM hhi_blocks ORDER BY id
        """)
        if result:
            return result
        return fetch_query("""
            SELECT id, '' as ship_id, 'flat' as block_type, current_stage,
                   location as current_location, 0 as weight,
                   due_date, ROUND(completion_pct,1) as completion_pct
            FROM blocks ORDER BY id
        """) or []

    # ── Ships table ──
    @app.callback(Output("ships-table", "data"), Input("interval", "n_intervals"))
    def update_ships_table(n):
        result = fetch_query("""
            SELECT id as name, hull_number, assigned_dock, status,
                   blocks_erected, total_blocks,
                   ROUND(erection_progress, 1) as erection_progress,
                   target_delivery_date as target_launch_date
            FROM ships ORDER BY id
        """)
        return result or []

    # ── Ship progress chart ──
    @app.callback(Output("ship-progress-chart", "figure"), Input("interval", "n_intervals"))
    def update_ship_progress_chart(n):
        ships = fetch_query("""
            SELECT id, hull_number, status, blocks_erected, total_blocks, erection_progress
            FROM ships ORDER BY id
        """)
        if not ships:
            return _empty_fig("No ships in database. Run simulation with HHI config.")

        ship_ids = [s.get("id", f"Ship-{i}") for i, s in enumerate(ships)]
        progress = [s.get("erection_progress", 0) or 0 for s in ships]
        blocks_done = [s.get("blocks_erected", 0) or 0 for s in ships]
        total_blocks = [s.get("total_blocks", 200) or 200 for s in ships]

        # Color by status
        status_colors = {
            "in_block_production": "#95a5a6",
            "in_erection": "#3498db",
            "afloat": "#27ae60",
            "in_quay_outfitting": "#9b59b6",
            "in_sea_trials": "#e67e22",
            "delivered": "#2ecc71",
        }
        colors = [status_colors.get(s.get("status", ""), "#7f8c8d") for s in ships]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ship_ids,
            y=progress,
            marker_color=colors,
            text=[f"{p:.0f}%" for p in progress],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Progress: %{y:.1f}%<br>Blocks: %{customdata[0]}/%{customdata[1]}<extra></extra>",
            customdata=list(zip(blocks_done, total_blocks)),
        ))

        fig.update_layout(
            template=CHART_TEMPLATE,
            height=280,
            margin=dict(t=30, b=60, l=50, r=20),
            xaxis_title="Ship",
            yaxis_title="Erection Progress %",
            yaxis_range=[0, 105],
        )
        return fig

    # ── Fleet table + utilization ──
    @app.callback(Output("fleet-spmt-table", "data"), Input("interval", "n_intervals"))
    def update_fleet(n):
        return fetch_query("SELECT id, status, current_location, load, ROUND(health_hydraulic,1) as health_hydraulic, ROUND(health_tires,1) as health_tires, ROUND(health_engine,1) as health_engine FROM spmts ORDER BY id") or []

    @app.callback(Output("utilization-heatmap", "figure"),
                  [Input("interval", "n_intervals"), Input("time-range", "value")])
    def update_utilization(n, time_range):
        history = fetch_health_history(time_window=time_range if time_range and time_range > 0 else None)
        if not history:
            return _empty_fig("No utilization data yet")
        # Build heatmap from health snapshots — group by equipment × time
        equipment_ids = sorted(set(r["equipment_id"] for r in history if r["equipment_type"] == "spmt"))
        timestamps = sorted(set(r["timestamp"] for r in history))
        if not equipment_ids or len(timestamps) < 2:
            # Fallback: show current status as bar
            spmts = fetch_query("SELECT id, status FROM spmts ORDER BY id")
            if not spmts:
                return _empty_fig()
            fig = go.Figure(go.Bar(
                x=[s["id"] for s in spmts],
                y=[1.0 if s["status"] != "idle" else 0.0 for s in spmts],
                marker_color=[COLORS["success"] if s["status"] != "idle" else "#bdc3c7" for s in spmts],
            ))
            fig.update_layout(template=CHART_TEMPLATE, height=250, yaxis_range=[0, 1.1],
                              yaxis_title="Active", margin=dict(t=20, b=40, l=50, r=20))
            return fig
        # Build matrix: health decline rate as proxy for utilization
        # Lower health = more active
        time_bins = timestamps[::max(1, len(timestamps) // 20)]  # ~20 bins
        z = []
        for eid in equipment_ids:
            row = []
            for t in time_bins:
                vals = [r["health_value"] for r in history
                        if r["equipment_id"] == eid and r["timestamp"] == t and r["component"] == "engine"]
                row.append(round(100 - vals[0], 1) if vals else 0)
            z.append(row)
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"t={int(t)}" for t in time_bins], y=equipment_ids,
            colorscale="YlOrRd", colorbar=dict(title="Wear (100-health)"),
        ))
        fig.update_layout(template=CHART_TEMPLATE, height=max(200, len(equipment_ids) * 35 + 80),
                          margin=dict(t=20, b=40, l=60, r=20), xaxis_title="Time")
        return fig

    # ── Health tab ──
    @app.callback(Output("health-equipment-dropdown", "options"), Input("interval", "n_intervals"))
    def update_equipment_options(n):
        rows = fetch_query("SELECT DISTINCT equipment_id, equipment_type FROM health_history ORDER BY equipment_id")
        return [{"label": f"{r['equipment_id']} ({r['equipment_type']})", "value": r["equipment_id"]} for r in rows] if rows else []

    @app.callback(Output("health-trends-chart", "figure"),
                  [Input("interval", "n_intervals"), Input("health-equipment-dropdown", "value"), Input("time-range", "value")])
    def update_health_trends(n, selected_equip, time_range):
        tw = time_range if time_range and time_range > 0 else None
        history = fetch_health_history(equipment_id=selected_equip, time_window=tw)
        if not history:
            return _empty_fig("No health data recorded")
        series = defaultdict(lambda: {"t": [], "v": []})
        for r in history:
            key = f"{r['equipment_id']} · {r['component']}"
            series[key]["t"].append(r["timestamp"])
            series[key]["v"].append(r["health_value"])
        fig = go.Figure()
        color_list = list(COLORS.values())
        for i, (name, data) in enumerate(sorted(series.items())):
            fig.add_trace(go.Scatter(
                x=data["t"], y=data["v"], name=name, mode="lines",
                line=dict(width=2.5, color=color_list[i % len(color_list)]),
            ))
        fig.add_hline(y=20, line_dash="dash", line_color=COLORS["danger"], line_width=1,
                      annotation_text="Failure", annotation_position="top left")
        fig.add_hline(y=40, line_dash="dash", line_color=COLORS["warning"], line_width=1,
                      annotation_text="PM Threshold", annotation_position="top left")
        fig.update_layout(
            template=CHART_TEMPLATE, height=350,
            xaxis_title="Simulation Time", yaxis_title="Health",
            yaxis_range=[0, 105],
            margin=dict(t=20, b=40, l=50, r=20),
            legend=dict(font=dict(size=11)),
        )
        return fig

    @app.callback(Output("rul-table", "data"), Input("interval", "n_intervals"))
    def update_rul(n):
        rows = fetch_query("""
            SELECT h.equipment_id, h.equipment_type, h.component, h.health_value
            FROM health_history h
            INNER JOIN (
                SELECT equipment_id, component, MAX(timestamp) as max_ts
                FROM health_history GROUP BY equipment_id, component
            ) latest ON h.equipment_id = latest.equipment_id
                AND h.component = latest.component AND h.timestamp = latest.max_ts
            ORDER BY h.health_value ASC
        """)
        if not rows:
            return []
        # Estimate RUL using typical drift rate (base_drift + load_factor * 0.3)
        # With HHI config: 0.15 + 0.2 * 0.3 = 0.21 per hour when operating
        avg_drift = 0.21
        return [{
            "equipment_id": r["equipment_id"],
            "equipment_type": r["equipment_type"],
            "component": r["component"],
            "health_value": round(r["health_value"], 1),
            "rul": round(max(0, (r["health_value"] - 20.0) / avg_drift), 0),
        } for r in rows]

    # ── Operations tab ──
    @app.callback(Output("gantt-chart", "figure"),
                  [Input("interval", "n_intervals"), Input("time-range", "value")])
    def update_gantt(n, time_range):
        events = fetch_block_events()
        if not events:
            return _empty_fig("No block events recorded")
        if time_range and time_range > 0:
            max_t = max(e["timestamp"] for e in events)
            events = [e for e in events if e["timestamp"] >= max_t - time_range]
        blocks = defaultdict(list)
        for e in events:
            blocks[e["block_id"]].append(e)
        block_ids = sorted(blocks.keys())
        # Paginate: show up to 40 blocks
        block_ids = block_ids[:40]
        fig = go.Figure()
        legend_added = set()
        for bid in block_ids:
            block_events = sorted(blocks[bid], key=lambda x: x["timestamp"])
            for j, evt in enumerate(block_events):
                t_start = evt["timestamp"]
                t_end = block_events[j + 1]["timestamp"] if j + 1 < len(block_events) else t_start + 2
                stage = evt.get("stage", "unknown")
                color = STAGE_COLORS.get(stage, "#95a5a6")
                show_legend = stage not in legend_added
                if show_legend:
                    legend_added.add(stage)
                fig.add_trace(go.Bar(
                    base=[t_start], x=[max(0.5, t_end - t_start)], y=[bid],
                    orientation="h", marker_color=color, name=stage,
                    showlegend=show_legend,
                    hovertemplate=f"<b>{bid}</b><br>{evt['event_type']}<br>Stage: {stage}<br>Time: {t_start:.0f}<extra></extra>",
                ))
        fig.update_layout(
            template=CHART_TEMPLATE, barmode="stack",
            xaxis_title="Simulation Time", yaxis_title="Block",
            height=max(300, min(700, len(block_ids) * 16 + 80)),
            margin=dict(t=20, b=40, l=60, r=20),
            yaxis=dict(autorange="reversed", dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        )
        return fig

    # Ship production status colors
    SHIP_STATUS_COLORS = {
        "IN_ERECTION": "#3498db",      # Blue - building
        "AFLOAT": "#9b59b6",           # Purple - launched
        "IN_QUAY_OUTFITTING": "#f39c12",  # Orange - outfitting
        "IN_SEA_TRIALS": "#1abc9c",    # Teal - testing
        "DELIVERED": "#27ae60",        # Green - complete
    }

    @app.callback(Output("ship-gantt-chart", "figure"),
                  [Input("interval", "n_intervals"), Input("time-range", "value")])
    def update_ship_gantt(n, time_range):
        """Update the ship production Gantt chart."""
        events = fetch_ship_events()
        if not events:
            return _empty_fig("No ship events recorded")
        if time_range and time_range > 0:
            max_t = max(e["timestamp"] for e in events)
            events = [e for e in events if e["timestamp"] >= max_t - time_range]

        # Group events by ship
        ships = defaultdict(list)
        for e in events:
            ships[e["ship_id"]].append(e)

        ship_ids = sorted(ships.keys())
        fig = go.Figure()
        legend_added = set()

        for ship_id in ship_ids:
            ship_events = sorted(ships[ship_id], key=lambda x: x["timestamp"])
            for j, evt in enumerate(ship_events):
                t_start = evt["timestamp"]
                # End time is next event or current time + buffer
                t_end = ship_events[j + 1]["timestamp"] if j + 1 < len(ship_events) else t_start + 50
                status = evt.get("status", "unknown")
                color = SHIP_STATUS_COLORS.get(status, "#95a5a6")
                show_legend = status not in legend_added
                if show_legend:
                    legend_added.add(status)

                # Add a hover-friendly bar
                dock_info = f" @ {evt.get('dock_id')}" if evt.get("dock_id") else ""
                fig.add_trace(go.Bar(
                    base=[t_start], x=[max(1, t_end - t_start)], y=[ship_id],
                    orientation="h", marker_color=color, name=status.replace("_", " ").title(),
                    showlegend=show_legend,
                    hovertemplate=f"<b>{ship_id}</b><br>{evt['event_type']}<br>Status: {status}{dock_info}<br>Time: {t_start:.0f}<extra></extra>",
                ))

        fig.update_layout(
            template=CHART_TEMPLATE, barmode="stack",
            xaxis_title="Simulation Time", yaxis_title="Ship",
            height=max(250, min(500, len(ship_ids) * 40 + 80)),
            margin=dict(t=20, b=40, l=100, r=20),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        )
        return fig

    @app.callback(Output("queue-depth-chart", "figure"),
                  [Input("interval", "n_intervals"), Input("time-range", "value")])
    def update_queue_depth(n, time_range):
        tw = time_range if time_range and time_range > 0 else None
        history = fetch_queue_history(time_window=tw)
        if not history:
            return _empty_fig("No queue data recorded")
        facilities = defaultdict(lambda: {"t": [], "total": []})
        for r in history:
            fac = r["facility_name"]
            facilities[fac]["t"].append(r["timestamp"])
            facilities[fac]["total"].append(r["queue_depth"] + r["processing_count"])
        fig = go.Figure()
        for i, (fac, data) in enumerate(sorted(facilities.items())):
            color = FACILITY_COLORS[i % len(FACILITY_COLORS)]
            # Convert hex to rgba for fillcolor (Plotly doesn't support 8-char hex)
            if color.startswith("#"):
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fillcolor = f"rgba({r},{g},{b},0.15)"
            elif color.startswith("rgb"):
                fillcolor = color.replace(")", ",0.15)").replace("rgb", "rgba")
            else:
                fillcolor = color
            fig.add_trace(go.Scatter(
                x=data["t"], y=data["total"], name=fac, mode="lines",
                fill="tozeroy", line=dict(width=1.5, color=color),
                fillcolor=fillcolor,
            ))
        # Find bottleneck
        avg_depths = {}
        for fac, data in facilities.items():
            avg_depths[fac] = sum(data["total"]) / max(len(data["total"]), 1)
        if avg_depths:
            bottleneck = max(avg_depths, key=avg_depths.get)
            fig.add_annotation(
                text=f"Bottleneck: {bottleneck} (avg {avg_depths[bottleneck]:.1f})",
                xref="paper", yref="paper", x=1, y=1, showarrow=False,
                font=dict(size=12, color=COLORS["danger"]),
                xanchor="right", yanchor="top",
            )
        fig.update_layout(
            template=CHART_TEMPLATE, height=300,
            xaxis_title="Simulation Time", yaxis_title="Queue + Processing",
            margin=dict(t=30, b=40, l=50, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        )
        return fig

    # ── KPIs tab (dual axis) ──
    @app.callback(Output("kpi-trend-graph", "figure"),
                  [Input("interval", "n_intervals"), Input("time-range", "value")])
    def update_kpi_trends(n, time_range):
        rows = fetch_query("SELECT time, blocks_completed, breakdowns, planned_maintenance, total_tardiness, empty_travel_distance FROM metrics ORDER BY time")
        if not rows:
            return _empty_fig("No KPI data recorded")
        if time_range and time_range > 0:
            max_t = rows[-1]["time"]
            rows = [r for r in rows if r["time"] >= max_t - time_range]
        times = [r["time"] for r in rows]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Counts", "Tardiness & Empty Travel"))
        fig.add_trace(go.Scatter(x=times, y=[r["blocks_completed"] for r in rows],
                                 name="Completed", line=dict(color=COLORS["success"], width=2), mode="lines+markers", marker=dict(size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=[r["breakdowns"] for r in rows],
                                 name="Breakdowns", line=dict(color=COLORS["danger"], width=2), mode="lines+markers", marker=dict(size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=[r["planned_maintenance"] for r in rows],
                                 name="Planned Maint.", line=dict(color=COLORS["accent"], width=2), mode="lines+markers", marker=dict(size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=[r["total_tardiness"] for r in rows],
                                 name="Tardiness", line=dict(color=COLORS["warning"], width=2), mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=times, y=[r.get("empty_travel_distance", 0) for r in rows],
                                 name="Empty Travel", line=dict(color=COLORS["purple"], width=2, dash="dot"), mode="lines"), row=2, col=1)
        fig.update_layout(
            template=CHART_TEMPLATE, height=500,
            margin=dict(t=40, b=40, l=50, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, font=dict(size=11)),
        )
        fig.update_xaxes(title_text="Simulation Time", row=2, col=1)
        return fig

    # ============================================================================
    # DUAL SHIPYARD MAP CALLBACKS
    # ============================================================================

    def _fetch_map_state():
        """Fetch all data needed for shipyard map visualization."""
        spmts = fetch_query("""
            SELECT id, status, current_location, load,
                   health_hydraulic, health_tires, health_engine
            FROM spmts ORDER BY id
        """) or []

        cranes = fetch_query("""
            SELECT id, status, position_on_rail, health_cable, health_motor
            FROM cranes ORDER BY id
        """) or []

        blocks = fetch_query("""
            SELECT id, status, current_stage, location, dock_row, dock_col, completion_pct
            FROM blocks ORDER BY id
        """) or []

        # Queue depths from latest snapshot
        queue_rows = fetch_query("""
            SELECT facility_name, queue_depth, processing_count
            FROM queue_history
            WHERE timestamp = (SELECT MAX(timestamp) FROM queue_history)
        """) or []
        queue_depths = {r["facility_name"]: r["queue_depth"] + r["processing_count"]
                        for r in queue_rows}

        # Barge data (if exists)
        barge = None
        barge_rows = fetch_query("SELECT * FROM barges ORDER BY id LIMIT 1")
        if barge_rows:
            barge = barge_rows[0]

        return {
            "spmts": spmts,
            "cranes": cranes,
            "blocks": blocks,
            "queue_depths": queue_depths,
            "barge": barge,
        }

    # ── Quonset Map (standalone tab) ──
    @app.callback(
        Output("quonset-map", "figure"),
        [Input("interval", "n_intervals"),
         Input("map-health-toggle", "value"),
         Input("map-layers", "value")],
    )
    def update_quonset_map(n, health_toggle, layers):
        map_data = _fetch_map_state()
        show_health = "on" in (health_toggle or [])
        show_queues = "queues" in (layers or [])
        return build_quonset_map(map_data, show_health=show_health, show_queues=show_queues)

    # ── Groton Map (standalone tab) ──
    @app.callback(
        Output("groton-map", "figure"),
        [Input("interval", "n_intervals"),
         Input("map-health-toggle", "value"),
         Input("map-layers", "value")],
    )
    def update_groton_map(n, health_toggle, layers):
        map_data = _fetch_map_state()
        show_health = "on" in (health_toggle or [])
        show_queues = "queues" in (layers or [])
        return build_groton_map(map_data, show_health=show_health, show_queues=show_queues)


    # ============================================================================
    # DEPENDENCY GRAPH CALLBACKS
    # ============================================================================

    @app.callback(
        Output("dependency-block-filter", "options"),
        Input("interval", "n_intervals"),
    )
    def update_dependency_filter_options(n):
        blocks = fetch_query("SELECT id FROM blocks ORDER BY id") or []
        return [{"label": b["id"], "value": b["id"]} for b in blocks]

    @app.callback(
        Output("dependency-graph", "figure"),
        [Input("interval", "n_intervals"),
         Input("dependency-block-filter", "value"),
         Input("dependency-show-options", "value")],
    )
    def update_dependency_graph(n, selected_block, show_options):
        # Try hhi_blocks first, fall back to blocks
        blocks = fetch_query("""
            SELECT id, status, current_stage, completion_pct, predecessors
            FROM hhi_blocks ORDER BY id
        """)
        if not blocks:
            blocks = fetch_query("""
                SELECT id, status, current_stage, completion_pct, '' as predecessors
                FROM blocks ORDER BY id
            """) or []

        # Parse predecessors if stored as string
        for b in blocks:
            preds = b.get("predecessors", "")
            if isinstance(preds, str):
                if preds:
                    b["predecessors"] = [p.strip() for p in preds.split(",") if p.strip()]
                else:
                    b["predecessors"] = []

        show_completed = "completed" in (show_options or [])
        show_critical = "critical" in (show_options or [])

        if show_critical:
            return build_critical_path_view(blocks)
        else:
            return build_dependency_graph(blocks, selected_block=selected_block, show_completed=show_completed)

    @app.callback(
        Output("dependency-stats", "children"),
        [Input("dependency-block-filter", "value"),
         Input("interval", "n_intervals")],
    )
    def update_dependency_stats(selected_block, n):
        if not selected_block:
            return html.P("Select a block to see its dependency chain", className="text-muted")

        # Try hhi_blocks first, fall back to blocks
        blocks = fetch_query("SELECT id, status, predecessors FROM hhi_blocks")
        if not blocks:
            blocks = fetch_query("SELECT id, status, '' as predecessors FROM blocks") or []
        node_data = {b["id"]: b for b in blocks}

        if selected_block not in node_data:
            return html.P("Block not found", className="text-muted")

        # Parse predecessors
        for b in blocks:
            preds = b.get("predecessors", "")
            if isinstance(preds, str):
                b["predecessors"] = [p.strip() for p in preds.split(",") if p.strip()] if preds else []

        # Count ancestors and descendants
        def count_ancestors(node, visited=None):
            if visited is None:
                visited = set()
            if node in visited or node not in node_data:
                return visited
            visited.add(node)
            for pred in node_data[node].get("predecessors", []):
                count_ancestors(pred, visited)
            return visited

        def count_descendants(node, visited=None):
            if visited is None:
                visited = set()
            if node in visited:
                return visited
            visited.add(node)
            for b in blocks:
                if node in b.get("predecessors", []):
                    count_descendants(b["id"], visited)
            return visited

        ancestors = count_ancestors(selected_block) - {selected_block}
        descendants = count_descendants(selected_block) - {selected_block}

        block = node_data[selected_block]
        direct_preds = block.get("predecessors", [])

        return html.Div([
            html.P([html.Strong("Block: "), selected_block]),
            html.P([html.Strong("Status: "), block.get("status", "unknown")]),
            html.P([html.Strong("Direct predecessors: "), str(len(direct_preds))]),
            html.P([html.Strong("Total ancestors: "), str(len(ancestors))]),
            html.P([html.Strong("Total successors: "), str(len(descendants))]),
            html.Hr(),
            html.P([html.Strong("Predecessors: ")]),
            html.Ul([html.Li(p) for p in direct_preds]) if direct_preds else html.P("None", className="text-muted"),
        ])

    @app.callback(
        Output("critical-path-table", "data"),
        Input("interval", "n_intervals"),
    )
    def update_critical_path_table(n):
        # Try hhi_blocks first, fall back to blocks
        blocks = fetch_query("""
            SELECT id, status, current_stage, ROUND(completion_pct, 1) as completion_pct, predecessors
            FROM hhi_blocks ORDER BY id
        """)
        if not blocks:
            blocks = fetch_query("""
                SELECT id, status, current_stage, ROUND(completion_pct, 1) as completion_pct, '' as predecessors
                FROM blocks ORDER BY id
            """) or []

        # Parse predecessors
        for b in blocks:
            preds = b.get("predecessors", "")
            if isinstance(preds, str):
                b["predecessors"] = [p.strip() for p in preds.split(",") if p.strip()] if preds else []

        # Find critical path
        node_data = {b["id"]: b for b in blocks}

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

        # Find the longest critical path
        max_length = 0
        critical_path = []
        for b in blocks:
            if b.get("status") != "placed_on_dock":
                length, path = longest_path_to(b["id"])
                if length > max_length:
                    max_length = length
                    critical_path = path

        # Return blocks in critical path (convert predecessors list back to string for DataTable)
        result = []
        for bid in critical_path:
            if bid in node_data:
                block = node_data[bid].copy()
                # DataTable requires primitive types - convert list to comma-separated string
                preds = block.get("predecessors", [])
                if isinstance(preds, list):
                    block["predecessors"] = ", ".join(preds) if preds else ""
                result.append(block)
        return result

    # ============================================================================
    # SIMULATION PLAYBACK CALLBACKS
    # ============================================================================

    @app.callback(
        Output("playback-run-selector", "options"),
        Input("playback-mode-toggle", "value"),
    )
    def populate_run_selector(playback_enabled):
        """Populate the historical run selector dropdown."""
        runs = list_simulation_runs()
        if not runs:
            return [{"label": "No historical runs available", "value": "", "disabled": True}]

        options = []
        for run in runs:
            blocks = run.get("blocks_completed", 0) or 0
            ships = run.get("ships_delivered", 0) or 0
            steps = run.get("total_steps", 0) or 0
            policy = run.get("policy_type", "unknown") or "unknown"
            started = run.get("started_at", "")[:16] if run.get("started_at") else ""

            label = f"{run['name']} | {policy} | {blocks} blocks | {started}"
            options.append({"label": label, "value": run["id"]})

        return options

    @app.callback(
        Output("playback-controls-container", "style"),
        Input("playback-mode-toggle", "value"),
    )
    def toggle_playback_controls(playback_enabled):
        """Show/hide playback controls based on toggle."""
        if "on" in (playback_enabled or []):
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        [Output("playback-timeline", "data"),
         Output("playback-slider", "min"),
         Output("playback-slider", "max"),
         Output("playback-slider", "marks")],
        [Input("playback-mode-toggle", "value"),
         Input("playback-run-selector", "value")],
    )
    def initialize_playback_timeline(playback_enabled, selected_run_id):
        """Load timeline data when playback mode is enabled or run is selected."""
        if "on" not in (playback_enabled or []):
            return no_update, no_update, no_update, no_update

        # Use selected run or default to all data
        run_id = selected_run_id if selected_run_id else None
        timeline = fetch_playback_timeline_for_run(run_id)
        timestamps = fetch_available_timestamps(run_id)

        if not timestamps:
            return (
                {"min_time": 0, "max_time": 100, "timestamps": []},
                0, 100,
                {0: "No data", 100: ""},
            )

        min_t = timeline["min_time"]
        max_t = timeline["max_time"]

        # Create marks at regular intervals
        marks = {}
        if max_t > min_t:
            step = (max_t - min_t) / 4
            for i in range(5):
                t = min_t + i * step
                marks[int(t)] = f"t={int(t)}"
        else:
            marks[int(min_t)] = f"t={int(min_t)}"

        return (
            {"min_time": min_t, "max_time": max_t, "timestamps": timestamps},
            int(min_t),
            int(max_t),
            marks,
        )

    @app.callback(
        [Output("playback-state", "data"),
         Output("playback-interval", "disabled"),
         Output("playback-interval", "interval"),
         Output("playback-play", "children")],
        [Input("playback-play", "n_clicks"),
         Input("playback-rewind", "n_clicks"),
         Input("playback-forward", "n_clicks"),
         Input("playback-live", "n_clicks"),
         Input("playback-slider", "value"),
         Input("playback-speed-selector", "value")],
        [State("playback-state", "data"),
         State("playback-timeline", "data")],
        prevent_initial_call=True,
    )
    def control_playback(play_clicks, rewind_clicks, forward_clicks, live_clicks,
                         slider_value, playback_speed, playback_state, timeline_data):
        """Handle playback control button clicks, slider changes, and speed selection."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        speed = playback_speed or 1
        state = playback_state or {"playing": False, "time": None, "speed": speed}
        timeline = timeline_data or {"min_time": 0, "max_time": 100, "timestamps": []}
        # Base interval is 500ms, adjust by speed
        interval_ms = max(50, int(500 / speed))

        min_t = timeline.get("min_time", 0)
        max_t = timeline.get("max_time", 100)
        timestamps = timeline.get("timestamps", [])

        current_time = state.get("time") or max_t

        if trigger_id == "playback-play":
            # Toggle play/pause
            playing = not state.get("playing", False)
            if playing and current_time >= max_t:
                current_time = min_t  # Restart from beginning
            return (
                {"playing": playing, "time": current_time, "speed": speed},
                not playing,  # Disable interval when not playing
                interval_ms,
                "⏸️ Pause" if playing else "▶️ Play",
            )

        elif trigger_id == "playback-rewind":
            # Jump back 10% of timeline
            step = max(1, (max_t - min_t) * 0.1)
            new_time = max(min_t, current_time - step)
            return (
                {"playing": False, "time": new_time, "speed": speed},
                True,
                interval_ms,
                "▶️ Play",
            )

        elif trigger_id == "playback-forward":
            # Jump forward 10% of timeline
            step = max(1, (max_t - min_t) * 0.1)
            new_time = min(max_t, current_time + step)
            return (
                {"playing": False, "time": new_time, "speed": speed},
                True,
                interval_ms,
                "▶️ Play",
            )

        elif trigger_id == "playback-live":
            # Jump to live (most recent)
            return (
                {"playing": False, "time": None, "speed": speed},
                True,
                interval_ms,
                "▶️ Play",
            )

        elif trigger_id == "playback-slider":
            # Slider was dragged
            return (
                {"playing": False, "time": slider_value, "speed": speed},
                True,
                interval_ms,
                "▶️ Play",
            )

        elif trigger_id == "playback-speed-selector":
            # Speed changed, update interval
            return (
                {"playing": state.get("playing", False), "time": state.get("time"), "speed": speed},
                state.get("playing", False) is False,  # Keep current disabled state
                interval_ms,
                "⏸️ Pause" if state.get("playing", False) else "▶️ Play",
            )

        return no_update, no_update, no_update, no_update

    @app.callback(
        [Output("playback-slider", "value"),
         Output("playback-time", "children")],
        [Input("playback-interval", "n_intervals")],
        [State("playback-state", "data"),
         State("playback-timeline", "data")],
    )
    def update_playback_position(n_intervals, playback_state, timeline_data):
        """Update slider position during playback."""
        state = playback_state or {"playing": False, "time": None, "speed": 1}
        timeline = timeline_data or {"min_time": 0, "max_time": 100, "timestamps": []}

        current_time = state.get("time")
        max_t = timeline.get("max_time", 100)
        min_t = timeline.get("min_time", 0)
        timestamps = timeline.get("timestamps", [])

        if current_time is None:
            return int(max_t), "Live"

        # If playing, advance to next timestamp
        if state.get("playing") and timestamps:
            # Find next timestamp after current
            next_times = [t for t in timestamps if t > current_time]
            if next_times:
                current_time = next_times[0]
            else:
                current_time = max_t  # Reached end

        time_display = f"t = {int(current_time)}"
        if current_time >= max_t:
            time_display = "Live"

        return int(current_time), time_display

    @app.callback(
        Output("playback-current-time", "children"),
        [Input("playback-state", "data"),
         Input("playback-mode-toggle", "value")],
    )
    def display_playback_status(playback_state, playback_enabled):
        """Show current playback status below the maps."""
        if "on" not in (playback_enabled or []):
            return ""

        state = playback_state or {}
        current_time = state.get("time")

        if current_time is None:
            return "Viewing: Live data (auto-refreshing)"
        else:
            return f"Viewing: Historical snapshot at simulation time {int(current_time)}"

    # ── Modified Dual Maps Callback with Playback Support ──
    @app.callback(
        [Output("dual-quonset-map", "figure"),
         Output("dual-groton-map", "figure"),
         Output("transit-map", "figure")],
        [Input("interval", "n_intervals"),
         Input("dual-map-health-toggle", "value"),
         Input("playback-state", "data"),
         Input("playback-mode-toggle", "value"),
         Input("playback-run-selector", "value")],
    )
    def update_dual_maps_with_playback(n, health_toggle, playback_state, playback_enabled, selected_run_id):
        """Update dual maps with support for playback mode and historical run selection."""
        show_health = "on" in (health_toggle or [])
        playback_on = "on" in (playback_enabled or [])

        # Determine which data source to use
        if playback_on and playback_state and playback_state.get("time") is not None:
            # Use historical data from selected run
            run_id = selected_run_id if selected_run_id else None
            map_data = fetch_position_at_time(playback_state["time"], run_id=run_id)
        else:
            # Use live data
            map_data = _fetch_map_state()

        quonset_fig = build_quonset_map(map_data, show_health=show_health, show_queues=True)
        groton_fig = build_groton_map(map_data, show_health=show_health, show_queues=True)
        transit_fig = build_transit_map(map_data.get("barge"))

        # Adjust heights for split view
        quonset_fig.update_layout(height=400, margin=dict(t=40, b=30, l=30, r=30))
        groton_fig.update_layout(height=400, margin=dict(t=40, b=30, l=30, r=30))
        transit_fig.update_layout(height=180, margin=dict(t=40, b=20, l=30, r=30))

        # Add playback indicator if in historical mode
        if playback_on and playback_state and playback_state.get("time") is not None:
            historical_time = playback_state["time"]
            for fig in [quonset_fig, groton_fig]:
                fig.add_annotation(
                    text=f"📼 t={int(historical_time)}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=11, color="#e74c3c"),
                    bgcolor="rgba(255,255,255,0.8)",
                    borderpad=4,
                )

        return quonset_fig, groton_fig, transit_fig

    # ============================================================================
    # GNN GRAPH VISUALIZATION CALLBACKS
    # ============================================================================

    @app.callback(
        [Output("gnn-cytoscape-graph", "elements"),
         Output("gnn-graph-stats", "children")],
        [Input("interval", "n_intervals"),
         Input("gnn-graph-ship-filter", "value"),
         Input("gnn-graph-edge-filter", "value"),
         Input("gnn-graph-health-toggle", "value")],
    )
    def update_gnn_graph(n, ship_filter, edge_filters, health_toggle):
        """Build and update the GNN graph visualization."""
        elements = []
        stats_children = []

        # Limit nodes to prevent browser crash - Cytoscape struggles with >1000 elements
        MAX_BLOCKS = 50  # Show sample of blocks to keep graph manageable

        # Fetch data from database - try hhi_blocks first
        blocks = fetch_query(f"SELECT * FROM hhi_blocks LIMIT {MAX_BLOCKS}")
        if not blocks:
            blocks = fetch_query(f"SELECT * FROM blocks LIMIT {MAX_BLOCKS}") or []
        spmts = fetch_query("SELECT * FROM spmts") or []
        # Try goliath_cranes first
        cranes = fetch_query("SELECT * FROM goliath_cranes")
        if not cranes:
            cranes = fetch_query("SELECT * FROM cranes") or []

        # Facility data - derive from blocks and SPMTs locations
        # Try hhi_blocks first (uses 'location' column), then blocks (uses 'current_location')
        block_locations = fetch_query("SELECT DISTINCT location as id FROM hhi_blocks WHERE location IS NOT NULL") or []
        if not block_locations:
            block_locations = fetch_query("SELECT DISTINCT current_location as id FROM blocks WHERE current_location IS NOT NULL") or []
        spmt_locations = fetch_query("SELECT DISTINCT current_location as id FROM spmts WHERE current_location IS NOT NULL") or []

        # Combine all unique locations
        all_locations = set()
        for loc in block_locations + spmt_locations:
            if loc.get("id"):
                all_locations.add(loc["id"])

        facilities = [{"id": loc} for loc in all_locations]

        # Add some default facilities if empty
        if not facilities:
            facilities = [
                {"id": "STEEL_PROCESSING"},
                {"id": "CYLINDER_FAB"},
                {"id": "MODULE_OUTFITTING"},
                {"id": "SUPER_MODULE_ASSEMBLY"},
            ]

        show_health = "on" in (health_toggle or [])
        edge_filters = edge_filters or []

        # Filter by ship if applicable
        if ship_filter and ship_filter != "all":
            blocks = [b for b in blocks if b.get("ship_id") == ship_filter]

        # Helper function to determine health status
        def get_health_status(health_value):
            if health_value is None:
                return "healthy"
            if health_value < 40:
                return "critical"
            elif health_value < 60:
                return "warning"
            return "healthy"

        # Create block nodes
        for block in blocks:
            health_status = "healthy"
            node = {
                "data": {
                    "id": f"block_{block['id']}",
                    "label": block["id"],
                    "type": "block",
                    "health_status": health_status,
                    "status": block.get("status", "unknown"),
                    "stage": block.get("current_stage", "unknown"),
                    "completion": block.get("completion_pct", 0),
                    "location": block.get("location") or block.get("current_location", "unknown"),
                }
            }
            elements.append(node)

        # Create SPMT nodes
        for spmt in spmts:
            min_health = min(
                spmt.get("health_hydraulic", 100),
                spmt.get("health_tires", 100),
                spmt.get("health_engine", 100),
            )
            health_status = get_health_status(min_health) if show_health else "healthy"
            node = {
                "data": {
                    "id": f"spmt_{spmt['id']}",
                    "label": spmt["id"],
                    "type": "spmt",
                    "health_status": health_status,
                    "status": spmt.get("status", "unknown"),
                    "health_hydraulic": spmt.get("health_hydraulic", 100),
                    "health_tires": spmt.get("health_tires", 100),
                    "health_engine": spmt.get("health_engine", 100),
                    "location": spmt.get("current_location", "unknown"),
                }
            }
            elements.append(node)

        # Create Crane nodes (support both goliath_cranes and legacy cranes tables)
        for crane in cranes:
            # Goliath cranes use hoist/trolley/gantry, legacy uses cable/motor
            health_hoist = crane.get("health_hoist", crane.get("health_cable", 100))
            health_trolley = crane.get("health_trolley", crane.get("health_motor", 100))
            health_gantry = crane.get("health_gantry", 100)
            min_health = min(health_hoist, health_trolley, health_gantry)
            health_status = get_health_status(min_health) if show_health else "healthy"
            node = {
                "data": {
                    "id": f"crane_{crane['id']}",
                    "label": crane["id"],
                    "type": "crane",
                    "health_status": health_status,
                    "status": crane.get("status", "unknown"),
                    "health_hoist": health_hoist,
                    "health_trolley": health_trolley,
                    "health_gantry": health_gantry,
                    "position": crane.get("position_on_rail", crane.get("position", 0)),
                }
            }
            elements.append(node)

        # Create Facility nodes
        for facility in facilities:
            node = {
                "data": {
                    "id": f"facility_{facility['id']}",
                    "label": facility["id"].replace("_", " ").title()[:15],
                    "type": "facility",
                    "health_status": "healthy",
                }
            }
            elements.append(node)

        # Create edges
        edge_counts = defaultdict(int)

        # Transport edges (block <-> SPMT) - distribute across SPMTs with round-robin
        if "transport" in edge_filters and spmts:
            for i, block in enumerate(blocks):
                # Round-robin distribution across all SPMTs
                spmt = spmts[i % len(spmts)]
                edge = {
                    "data": {
                        "source": f"block_{block['id']}",
                        "target": f"spmt_{spmt['id']}",
                        "type": "needs_transport",
                    }
                }
                elements.append(edge)
                edge_counts["needs_transport"] += 1

        # Lift edges (block <-> Crane) - distribute across cranes with round-robin
        if "lift" in edge_filters and cranes:
            for i, block in enumerate(blocks):
                # Round-robin distribution across all cranes
                crane = cranes[i % len(cranes)]
                edge = {
                    "data": {
                        "source": f"block_{block['id']}",
                        "target": f"crane_{crane['id']}",
                        "type": "needs_lift",
                    }
                }
                elements.append(edge)
                edge_counts["needs_lift"] += 1

        # Location edges (entity -> facility)
        # First, collect all facility IDs that exist
        facility_ids = {f"facility_{f['id']}" for f in facilities}

        if "location" in edge_filters:
            for block in blocks:
                # Support both 'location' (hhi_blocks) and 'current_location' (blocks)
                loc = block.get("location") or block.get("current_location")
                target_id = f"facility_{loc}"
                # Only create edge if target facility exists
                if loc and target_id in facility_ids:
                    edge = {
                        "data": {
                            "source": f"block_{block['id']}",
                            "target": target_id,
                            "type": "at",
                        }
                    }
                    elements.append(edge)
                    edge_counts["at"] += 1

            for spmt in spmts:
                loc = spmt.get("current_location")
                target_id = f"facility_{loc}"
                # Only create edge if target facility exists
                if loc and target_id in facility_ids:
                    edge = {
                        "data": {
                            "source": f"spmt_{spmt['id']}",
                            "target": target_id,
                            "type": "at",
                        }
                    }
                    elements.append(edge)
                    edge_counts["at"] += 1

        # Precedence edges (block -> block)
        if "precedes" in edge_filters:
            for block in blocks:
                preds = block.get("predecessors", "")
                if isinstance(preds, str) and preds:
                    pred_list = [p.strip() for p in preds.split(",") if p.strip()]
                    for pred in pred_list:
                        edge = {
                            "data": {
                                "source": f"block_{pred}",
                                "target": f"block_{block['id']}",
                                "type": "precedes",
                            }
                        }
                        elements.append(edge)
                        edge_counts["precedes"] += 1

        # Build statistics
        total_nodes = len(blocks) + len(spmts) + len(cranes) + len(facilities)
        total_edges = sum(edge_counts.values())

        stats_children = html.Div([
            html.P([html.Strong("Displayed Nodes: "), f"{total_nodes}"],
                   style={"marginBottom": "4px"}),
            html.P(f"(Showing sample of max {MAX_BLOCKS} blocks for performance)",
                   style={"fontSize": "11px", "color": "#7f8c8d", "marginBottom": "8px"}),
            html.Ul([
                html.Li(f"Blocks: {len(blocks)}"),
                html.Li(f"SPMTs: {len(spmts)}"),
                html.Li(f"Cranes: {len(cranes)}"),
                html.Li(f"Facilities: {len(facilities)}"),
            ]),
            html.P([html.Strong("Total Edges: "), f"{total_edges}"]),
            html.P([html.Strong("Graph Density: "),
                    f"{total_edges / max(1, total_nodes * (total_nodes - 1)):.4f}" if total_nodes > 1 else "N/A"]),
        ])

        return elements, stats_children

    @app.callback(
        Output("gnn-node-details", "children"),
        Input("gnn-cytoscape-graph", "tapNodeData"),
    )
    def display_node_details(node_data):
        """Display details when a node is clicked."""
        if not node_data:
            return html.P("Click on a node to see its features", className="text-muted")

        node_type = node_data.get("type", "unknown")
        label = node_data.get("label", "Unknown")

        # Build details based on node type
        details = [
            html.P([html.Strong("Node ID: "), node_data.get("id", "")]),
            html.P([html.Strong("Label: "), label]),
            html.P([html.Strong("Type: "), node_type.upper()]),
        ]

        if node_type == "block":
            details.extend([
                html.Hr(),
                html.P([html.Strong("Status: "), node_data.get("status", "unknown")]),
                html.P([html.Strong("Stage: "), node_data.get("stage", "unknown")]),
                html.P([html.Strong("Completion: "), f"{node_data.get('completion', 0):.1f}%"]),
                html.P([html.Strong("Location: "), node_data.get("location", "unknown")]),
            ])
        elif node_type == "spmt":
            details.extend([
                html.Hr(),
                html.P([html.Strong("Status: "), node_data.get("status", "unknown")]),
                html.P([html.Strong("Location: "), node_data.get("location", "unknown")]),
                html.P([html.Strong("Health - Hydraulic: "), f"{node_data.get('health_hydraulic', 100):.1f}%"]),
                html.P([html.Strong("Health - Tires: "), f"{node_data.get('health_tires', 100):.1f}%"]),
                html.P([html.Strong("Health - Engine: "), f"{node_data.get('health_engine', 100):.1f}%"]),
            ])
        elif node_type == "crane":
            details.extend([
                html.Hr(),
                html.P([html.Strong("Status: "), node_data.get("status", "unknown")]),
                html.P([html.Strong("Position: "), f"{node_data.get('position', 0)}"]),
                html.P([html.Strong("Health - Cable: "), f"{node_data.get('health_cable', 100):.1f}%"]),
                html.P([html.Strong("Health - Motor: "), f"{node_data.get('health_motor', 100):.1f}%"]),
            ])
        elif node_type == "facility":
            details.extend([
                html.Hr(),
                html.P("Facility node - represents a production stage location"),
            ])

        return html.Div(details)

    @app.callback(
        Output("gnn-edge-counts-chart", "figure"),
        [Input("gnn-graph-edge-filter", "value"),
         Input("interval", "n_intervals")],
    )
    def update_edge_counts_chart(edge_filters, n):
        """Create a bar chart showing edge counts by type."""
        edge_filters = edge_filters or []

        # Calculate approximate edge counts based on data
        blocks = fetch_query("SELECT COUNT(*) as cnt FROM blocks")
        spmts = fetch_query("SELECT COUNT(*) as cnt FROM spmts")
        cranes = fetch_query("SELECT COUNT(*) as cnt FROM cranes")

        n_blocks = blocks[0]["cnt"] if blocks else 0
        n_spmts = spmts[0]["cnt"] if spmts else 0
        n_cranes = cranes[0]["cnt"] if cranes else 0

        edge_types = []
        edge_counts = []
        edge_colors = []

        if "transport" in edge_filters:
            edge_types.append("needs_transport")
            edge_counts.append(n_blocks * n_spmts)
            edge_colors.append("#3498db")

        if "lift" in edge_filters:
            edge_types.append("needs_lift")
            edge_counts.append(n_blocks * n_cranes)
            edge_colors.append("#e67e22")

        if "location" in edge_filters:
            edge_types.append("at (location)")
            edge_counts.append(n_blocks + n_spmts)  # Approximate
            edge_colors.append("#9b59b6")

        if "precedes" in edge_filters:
            edge_types.append("precedes")
            edge_counts.append(max(0, n_blocks - 1))  # Approximate chain
            edge_colors.append("#e74c3c")

        if not edge_types:
            return _empty_fig("Select edge types to display")

        fig = go.Figure(data=[
            go.Bar(
                x=edge_types,
                y=edge_counts,
                marker_color=edge_colors,
                text=edge_counts,
                textposition="auto",
            )
        ])

        fig.update_layout(
            template=CHART_TEMPLATE,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis_title="Edge Type",
            yaxis_title="Count",
            showlegend=False,
        )

        return fig

    # ============================================================================
    # PRODUCTION AND EQUIPMENT CHARTS
    # ============================================================================

    @app.callback(
        Output("production-stage-chart", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_production_stage_chart(n):
        """Show blocks grouped by their current production stage."""
        # Try HHI blocks first, then standard blocks
        blocks = fetch_query("SELECT current_stage, COUNT(*) as count FROM hhi_blocks GROUP BY current_stage")
        if not blocks:
            blocks = fetch_query("SELECT current_stage, COUNT(*) as count FROM blocks GROUP BY current_stage")

        if not blocks:
            return _empty_fig("No blocks in database. Run a simulation first.")

        # Define stage order and colors
        stage_order = [
            "STEEL_CUTTING", "PART_FABRICATION", "SUB_ASSEMBLY", "UNIT_ASSEMBLY",
            "BLOCK_OUTFITTING", "PAINTING", "PRE_ERECTION", "ERECTION",
            "QUAY_OUTFITTING", "SEA_TRIALS", "DELIVERY"
        ]
        stage_colors = {
            "STEEL_CUTTING": "#3498db",
            "PART_FABRICATION": "#1abc9c",
            "SUB_ASSEMBLY": "#e67e22",
            "UNIT_ASSEMBLY": "#f1c40f",
            "BLOCK_OUTFITTING": "#e74c3c",
            "PAINTING": "#9b59b6",
            "PRE_ERECTION": "#2ecc71",
            "ERECTION": "#34495e",
            "QUAY_OUTFITTING": "#16a085",
            "SEA_TRIALS": "#2980b9",
            "DELIVERY": "#27ae60",
        }

        # Organize data
        stage_counts = {b.get("current_stage", "UNKNOWN"): b.get("count", 0) for b in blocks}

        # Create sorted lists for plotting
        stages = []
        counts = []
        colors = []
        for stage in stage_order:
            if stage in stage_counts:
                stages.append(stage.replace("_", " ").title())
                counts.append(stage_counts[stage])
                colors.append(stage_colors.get(stage, "#7f8c8d"))

        # Add any unknown stages
        for stage, count in stage_counts.items():
            if stage not in stage_order:
                stages.append(str(stage).replace("_", " ").title())
                counts.append(count)
                colors.append("#7f8c8d")

        if not stages:
            return _empty_fig("No stage data available")

        fig = go.Figure(data=[
            go.Bar(
                x=stages,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition="auto",
            )
        ])

        fig.update_layout(
            template=CHART_TEMPLATE,
            height=280,
            margin=dict(t=30, b=80, l=50, r=20),
            xaxis_title="Production Stage",
            yaxis_title="Block Count",
            xaxis_tickangle=-45,
        )

        return fig

    @app.callback(
        Output("equipment-health-chart", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_equipment_health_chart(n):
        """Show equipment health summary as a grouped bar chart."""
        spmts = fetch_query("SELECT id, health_hydraulic, health_tires, health_engine FROM spmts")
        cranes = fetch_query("SELECT id, health_cable, health_motor FROM cranes")
        goliath_cranes = fetch_query("SELECT id, health_hoist, health_trolley, health_gantry FROM goliath_cranes")

        if not spmts and not cranes and not goliath_cranes:
            return _empty_fig("No equipment data. Run a simulation first.")

        equipment_ids = []
        health_values = []
        health_components = []
        colors = []

        component_colors = {
            "hydraulic": "#3498db",
            "tires": "#e67e22",
            "engine": "#27ae60",
            "cable": "#9b59b6",
            "motor": "#e74c3c",
            "hoist": "#1abc9c",
            "trolley": "#f1c40f",
            "gantry": "#34495e",
        }

        # Add SPMT data
        for s in (spmts or []):
            for comp, col_name in [("hydraulic", "health_hydraulic"), ("tires", "health_tires"), ("engine", "health_engine")]:
                equipment_ids.append(s["id"])
                health_values.append(s.get(col_name, 100))
                health_components.append(comp)
                colors.append(component_colors[comp])

        # Add crane data (standard cranes)
        for c in (cranes or []):
            for comp, col_name in [("cable", "health_cable"), ("motor", "health_motor")]:
                equipment_ids.append(c["id"])
                health_values.append(c.get(col_name, 100))
                health_components.append(comp)
                colors.append(component_colors[comp])

        # Add goliath crane data
        for gc in (goliath_cranes or []):
            for comp, col_name in [("hoist", "health_hoist"), ("trolley", "health_trolley"), ("gantry", "health_gantry")]:
                equipment_ids.append(gc["id"])
                health_values.append(gc.get(col_name, 100))
                health_components.append(comp)
                colors.append(component_colors[comp])

        if not equipment_ids:
            return _empty_fig("No health data available")

        # Create grouped bar chart
        unique_components = list(dict.fromkeys(health_components))
        unique_equipment = list(dict.fromkeys(equipment_ids))

        fig = go.Figure()

        for comp in unique_components:
            comp_health = []
            for eq in unique_equipment:
                # Find health value for this equipment/component combo
                found = False
                for i, (eid, val, hc) in enumerate(zip(equipment_ids, health_values, health_components)):
                    if eid == eq and hc == comp:
                        comp_health.append(val)
                        found = True
                        break
                if not found:
                    comp_health.append(None)

            fig.add_trace(go.Bar(
                name=comp.title(),
                x=unique_equipment,
                y=comp_health,
                marker_color=component_colors.get(comp, "#7f8c8d"),
            ))

        fig.update_layout(
            template=CHART_TEMPLATE,
            height=280,
            margin=dict(t=30, b=60, l=50, r=20),
            barmode="group",
            xaxis_title="Equipment ID",
            yaxis_title="Health %",
            yaxis_range=[0, 105],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        return fig

    # ── Dry Docks Tab ──
    # Create callbacks for each dock status card (dock-1-status through dock-10-status)
    for dock_num in range(1, 11):
        @app.callback(
            Output(f"dock-{dock_num}-status", "children"),
            Input("interval", "n_intervals"),
            prevent_initial_call=False,
        )
        def update_dock_status(n, dock_id=f"dock_{dock_num}"):
            """Update dock status card."""
            dock = fetch_query(f"SELECT * FROM dry_docks WHERE id = '{dock_id}'")
            if not dock:
                return html.Span("No data", style={"color": "#95a5a6"})

            dock = dock[0]
            ship_id = dock.get("current_ship")
            status = dock.get("status", "idle")

            if ship_id and ship_id.strip():
                # Dock is occupied - get ship info
                ship = fetch_query(f"SELECT * FROM ships WHERE id = '{ship_id}'")
                if ship:
                    ship = ship[0]
                    progress = ship.get("erection_progress", 0)
                    return html.Div([
                        html.Div(f"🚢 {ship_id}", style={
                            "fontWeight": "bold",
                            "color": "#2980b9",
                            "marginBottom": "4px",
                        }),
                        html.Div(f"Progress: {progress:.0f}%", style={
                            "fontSize": "12px",
                            "color": "#27ae60" if progress > 50 else "#e67e22",
                        }),
                        html.Div(style={
                            "height": "6px",
                            "backgroundColor": "#ecf0f1",
                            "borderRadius": "3px",
                            "marginTop": "4px",
                            "overflow": "hidden",
                        }, children=[
                            html.Div(style={
                                "width": f"{progress}%",
                                "height": "100%",
                                "backgroundColor": "#27ae60" if progress > 50 else "#e67e22",
                            })
                        ])
                    ])
                return html.Span(f"🚢 {ship_id}", style={"color": "#2980b9", "fontWeight": "bold"})
            else:
                return html.Span("Empty", style={"color": "#95a5a6"})

    @app.callback(
        Output("dock-detail-content", "children"),
        [Input("dock-select", "value"), Input("interval", "n_intervals")],
    )
    def update_dock_detail(dock_id, n):
        """Show detailed info for selected dock."""
        if not dock_id:
            return html.P("Select a dock to view details", className="text-muted")

        dock = fetch_query(f"SELECT * FROM dry_docks WHERE id = '{dock_id}'")
        if not dock:
            return html.P(f"No data for {dock_id}", className="text-muted")

        dock = dock[0]
        ship_id = dock.get("current_ship")

        details = [
            html.P([html.Strong("Dock ID: "), dock_id]),
            html.P([html.Strong("Dimensions: "), f"{dock.get('length_m', 'N/A')}m x {dock.get('width_m', 'N/A')}m"]),
            html.P([html.Strong("Status: "), dock.get("status", "unknown")]),
        ]

        if ship_id and ship_id.strip():
            ship = fetch_query(f"SELECT * FROM ships WHERE id = '{ship_id}'")
            if ship:
                ship = ship[0]
                details.extend([
                    html.Hr(),
                    html.H5("Current Ship", style={"marginTop": "12px"}),
                    html.P([html.Strong("Ship ID: "), ship_id]),
                    html.P([html.Strong("Hull Number: "), ship.get("hull_number", "N/A")]),
                    html.P([html.Strong("Status: "), ship.get("status", "N/A")]),
                    html.P([html.Strong("Blocks Erected: "), f"{ship.get('blocks_erected', 0)} / {ship.get('total_blocks', 200)}"]),
                    html.P([html.Strong("Erection Progress: "), f"{ship.get('erection_progress', 0):.1f}%"]),
                ])

        # Get cranes assigned to this dock
        cranes = fetch_query(f"SELECT * FROM goliath_cranes WHERE assigned_dock = '{dock_id}'")
        if cranes:
            details.append(html.Hr())
            details.append(html.H5("Assigned Cranes", style={"marginTop": "12px"}))
            for crane in cranes:
                min_health = min(
                    crane.get("health_hoist", 100),
                    crane.get("health_trolley", 100),
                    crane.get("health_gantry", 100),
                )
                status_color = "#27ae60" if min_health > 70 else "#e67e22" if min_health > 40 else "#e74c3c"
                details.append(html.P([
                    html.Strong(f"{crane['id']}: "),
                    html.Span(f"{crane.get('status', 'N/A')} ", style={"color": "#7f8c8d"}),
                    html.Span(f"({min_health:.0f}% health)", style={"color": status_color}),
                ]))

        return html.Div(details)

    @app.callback(
        Output("dock-utilization-chart", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_dock_utilization(n):
        """Show dock utilization timeline."""
        docks = fetch_query("SELECT id, current_ship, status FROM dry_docks ORDER BY id")
        if not docks:
            return _empty_fig("No dock data. Run a simulation first.")

        dock_ids = [d["id"] for d in docks]
        occupied = [1 if d.get("current_ship") else 0 for d in docks]
        colors = ["#27ae60" if occ else "#ecf0f1" for occ in occupied]

        fig = go.Figure(data=[
            go.Bar(
                x=dock_ids,
                y=[1] * len(dock_ids),
                marker_color=colors,
                text=["Occupied" if occ else "Empty" for occ in occupied],
                textposition="inside",
            )
        ])

        fig.update_layout(
            template=CHART_TEMPLATE,
            height=200,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis_title="Dock",
            yaxis_visible=False,
            showlegend=False,
        )

        return fig

    # ============================================================================
    # DATA EXPORT CALLBACKS
    # ============================================================================

    @app.callback(
        Output("download-metrics", "data"),
        Input("export-metrics-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_metrics_csv(n_clicks):
        """Export metrics history as CSV."""
        if not n_clicks:
            return no_update

        metrics = fetch_query("""
            SELECT time, blocks_completed, breakdowns, planned_maintenance,
                   total_tardiness, empty_travel_distance
            FROM metrics ORDER BY time
        """)

        if not metrics:
            return no_update

        # Build CSV content
        import io
        import csv
        from datetime import datetime

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["time", "blocks_completed", "breakdowns", "planned_maintenance",
                         "total_tardiness", "empty_travel_distance"])

        # Data rows
        for m in metrics:
            writer.writerow([
                m.get("time", ""),
                m.get("blocks_completed", 0),
                m.get("breakdowns", 0),
                m.get("planned_maintenance", 0),
                m.get("total_tardiness", 0),
                m.get("empty_travel_distance", 0),
            ])

        csv_content = output.getvalue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return dict(
            content=csv_content,
            filename=f"shipyard_metrics_{timestamp}.csv",
        )

    @app.callback(
        Output("download-blocks", "data"),
        Input("export-blocks-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_blocks_csv(n_clicks):
        """Export block events as CSV."""
        if not n_clicks:
            return no_update

        blocks = fetch_block_events()

        if not blocks:
            return no_update

        # Build CSV content
        import io
        import csv
        from datetime import datetime

        output = io.StringIO()
        writer = csv.writer(output)

        # Header - get keys from first row
        if blocks:
            keys = list(blocks[0].keys())
            writer.writerow(keys)

            # Data rows
            for block in blocks:
                writer.writerow([block.get(k, "") for k in keys])

        csv_content = output.getvalue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return dict(
            content=csv_content,
            filename=f"shipyard_blocks_{timestamp}.csv",
        )

    # ====================================================================
    # PLATE ANALYTICS CALLBACKS
    # ====================================================================

    @app.callback(
        [Output("kpi-total-plates", "children"),
         Output("kpi-avg-plates", "children"),
         Output("kpi-plate-area", "children"),
         Output("kpi-plate-source", "children")],
        Input("interval", "n_intervals"),
    )
    def update_plate_kpis(n):
        rows = fetch_plate_stats()
        if not rows:
            return "0", "0", "0 m2", "No data"
        total_plates = sum(r.get("n_plates", 0) for r in rows)
        n_blocks = len(rows)
        avg_plates = total_plates / max(1, n_blocks)
        total_area = sum(r.get("plate_area_m2", 0) for r in rows)
        sources = defaultdict(int)
        for r in rows:
            sources[r.get("processing_source", "lognormal")] += 1
        source_str = ", ".join(f"{k}: {v}" for k, v in sources.items())
        return (
            f"{total_plates:,}",
            f"{avg_plates:.1f}",
            f"{total_area:,.0f} m2",
            source_str or "N/A",
        )

    @app.callback(
        Output("plate-vs-lognormal-chart", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_plate_vs_lognormal(n):
        records = fetch_plate_processing_times()
        fig = go.Figure()
        if not records:
            fig.update_layout(
                template=CHART_TEMPLATE,
                annotations=[dict(text="No plate processing data yet", showarrow=False,
                                  xref="paper", yref="paper", x=0.5, y=0.5, font_size=14)],
            )
            return fig

        plate_times = [r["processing_time_hours"] for r in records if r.get("method") == "plate_count"]
        log_times = [r["processing_time_hours"] for r in records if r.get("method") == "lognormal"]

        if plate_times:
            fig.add_trace(go.Histogram(x=plate_times, name="Plate-Count", opacity=0.7,
                                       marker_color="#2ecc71"))
        if log_times:
            fig.add_trace(go.Histogram(x=log_times, name="Lognormal", opacity=0.7,
                                       marker_color="#e74c3c"))
        fig.update_layout(
            template=CHART_TEMPLATE, barmode="overlay",
            xaxis_title="Processing Time (hours)", yaxis_title="Count",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        return fig

    @app.callback(
        Output("plate-count-distribution", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_plate_distribution(n):
        rows = fetch_plate_stats()
        fig = go.Figure()
        if not rows:
            fig.update_layout(
                template=CHART_TEMPLATE,
                annotations=[dict(text="No plate data yet", showarrow=False,
                                  xref="paper", yref="paper", x=0.5, y=0.5, font_size=14)],
            )
            return fig

        by_type = defaultdict(list)
        for r in rows:
            by_type[r.get("block_type", "unknown")].append(r.get("n_plates", 0))

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        for i, (btype, plates) in enumerate(sorted(by_type.items())):
            fig.add_trace(go.Box(y=plates, name=btype, marker_color=colors[i % len(colors)]))

        fig.update_layout(
            template=CHART_TEMPLATE,
            yaxis_title="Plates per Block",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        return fig

    @app.callback(
        Output("plate-time-scatter", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_plate_scatter(n):
        records = fetch_plate_processing_times()
        fig = go.Figure()
        if not records:
            fig.update_layout(
                template=CHART_TEMPLATE,
                annotations=[dict(text="No plate processing data yet", showarrow=False,
                                  xref="paper", yref="paper", x=0.5, y=0.5, font_size=14)],
            )
            return fig

        plate_recs = [r for r in records if r.get("method") == "plate_count"]
        if plate_recs:
            fig.add_trace(go.Scatter(
                x=[r["n_plates"] for r in plate_recs],
                y=[r["processing_time_hours"] for r in plate_recs],
                mode="markers",
                marker=dict(size=6, color="#2ecc71", opacity=0.6),
                name="Plate-Count",
            ))
        log_recs = [r for r in records if r.get("method") == "lognormal"]
        if log_recs:
            fig.add_trace(go.Scatter(
                x=[r["n_plates"] for r in log_recs],
                y=[r["processing_time_hours"] for r in log_recs],
                mode="markers",
                marker=dict(size=6, color="#e74c3c", opacity=0.6),
                name="Lognormal",
            ))

        fig.update_layout(
            template=CHART_TEMPLATE,
            xaxis_title="Number of Plates",
            yaxis_title="Processing Time (hours)",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        return fig

    @app.callback(
        Output("plate-bottleneck-chart", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_plate_bottleneck(n):
        records = fetch_plate_processing_times()
        fig = go.Figure()
        if not records:
            fig.update_layout(
                template=CHART_TEMPLATE,
                annotations=[dict(text="No plate processing data yet", showarrow=False,
                                  xref="paper", yref="paper", x=0.5, y=0.5, font_size=14)],
            )
            return fig

        by_stage = defaultdict(list)
        for r in records:
            weighted_time = r.get("processing_time_hours", 0) * max(1, r.get("n_plates", 1))
            by_stage[r.get("stage", "unknown")].append(weighted_time)

        stages = sorted(by_stage.keys())
        means = [sum(by_stage[s]) / len(by_stage[s]) for s in stages]

        fig.add_trace(go.Bar(
            y=stages, x=means, orientation="h",
            marker_color="#3498db",
        ))
        fig.update_layout(
            template=CHART_TEMPLATE,
            xaxis_title="Avg Plate-Weighted Time",
            margin=dict(l=120, r=20, t=30, b=40),
        )
        return fig

    @app.callback(
        Output("plate-blocks-table", "data"),
        Input("interval", "n_intervals"),
    )
    def update_plate_table(n):
        rows = fetch_plate_stats()
        return [dict(r) for r in rows] if rows else []

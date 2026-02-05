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
    fetch_position_at_time, fetch_playback_timeline, fetch_available_timestamps,
)
from .map_builder import build_quonset_map, build_groton_map, build_transit_map
from .dependency_graph import build_dependency_graph, build_critical_path_view

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
    "cutting": "#3498db",
    "panel": "#e67e22",
    "assembly": "#27ae60",
    "outfitting": "#e74c3c",
    "paint": "#9b59b6",
    "PAINTING": "#9b59b6",
    "pre_erection": "#1abc9c",
    "PRE_ERECTION": "#1abc9c",
    "DOCK": "#2c3e50",
    "dock": "#2c3e50",
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
            "dual-map": layouts.dual_map_layout,
            "quonset-map": layouts.quonset_map_layout,
            "groton-map": layouts.groton_map_layout,
            "dependencies": layouts.dependencies_layout,
            "gnn-graph": layouts.gnn_graph_layout,
            "overview": layouts.overview_layout,
            "blocks": layouts.blocks_layout,
            "fleet": layouts.fleet_layout,
            "health": layouts.health_layout,
            "operations": layouts.operations_layout,
            "kpis": layouts.kpis_layout,
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

    # ── Overview KPI cards ──
    @app.callback(
        [Output("kpi-blocks", "children"), Output("kpi-breakdowns", "children"),
         Output("kpi-planned", "children"), Output("kpi-tardiness", "children"),
         Output("kpi-spmt-util", "children"), Output("kpi-crane-util", "children"),
         Output("kpi-oee", "children"), Output("kpi-empty", "children")],
        Input("interval", "n_intervals"),
    )
    def update_kpis(n):
        metrics = fetch_query("SELECT * FROM metrics ORDER BY time DESC LIMIT 1")
        spmts = fetch_query("SELECT status FROM spmts")
        cranes = fetch_query("SELECT status FROM cranes")
        if not metrics:
            return ("—",) * 8
        m = metrics[0]
        blocks = m.get("blocks_completed", 0)
        breakdowns = m.get("breakdowns", 0)
        planned = m.get("planned_maintenance", 0)
        tardiness = m.get("total_tardiness", 0.0)
        empty = m.get("empty_travel_distance", 0.0)
        spmt_busy = sum(1 for s in spmts if s.get("status", "idle") != "idle") if spmts else 0
        spmt_total = max(len(spmts), 1)
        crane_busy = sum(1 for c in cranes if c.get("status", "idle") != "idle") if cranes else 0
        crane_total = max(len(cranes), 1)
        spmt_util = spmt_busy / spmt_total
        crane_util = crane_busy / crane_total
        avail = max(0, 1 - breakdowns / max(blocks + breakdowns, 1))
        oee = avail * spmt_util
        return (
            str(blocks), str(breakdowns), str(planned), f"{tardiness:.1f}",
            f"{spmt_util:.0%}", f"{crane_util:.0%}", f"{oee:.2f}", f"{empty:.1f}",
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

    # ── Blocks table ──
    @app.callback(Output("blocks-table", "data"), Input("interval", "n_intervals"))
    def update_blocks(n):
        return fetch_query("SELECT id, status, current_stage, location, due_date, ROUND(completion_pct,1) as completion_pct FROM blocks ORDER BY id") or []

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
        return [{
            "equipment_id": r["equipment_id"],
            "equipment_type": r["equipment_type"],
            "component": r["component"],
            "health_value": round(r["health_value"], 1),
            "rul": round(max(0, (r["health_value"] - 20.0) / 0.05), 0),
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
            fig.add_trace(go.Scatter(
                x=data["t"], y=data["total"], name=fac, mode="lines",
                fill="tozeroy", line=dict(width=1.5, color=color),
                fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if color.startswith("rgb") else color + "26",
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
        blocks = fetch_query("""
            SELECT id, status, current_stage, completion_pct, predecessors
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

        blocks = fetch_query("SELECT id, status, predecessors FROM blocks") or []
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
        blocks = fetch_query("""
            SELECT id, status, current_stage, ROUND(completion_pct, 1) as completion_pct, predecessors
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

        # Return blocks in critical path
        return [node_data[bid] for bid in critical_path if bid in node_data]

    # ============================================================================
    # SIMULATION PLAYBACK CALLBACKS
    # ============================================================================

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
        Input("playback-mode-toggle", "value"),
    )
    def initialize_playback_timeline(playback_enabled):
        """Load timeline data when playback mode is enabled."""
        if "on" not in (playback_enabled or []):
            return no_update, no_update, no_update, no_update

        timeline = fetch_playback_timeline()
        timestamps = fetch_available_timestamps()

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
         Output("playback-play", "children")],
        [Input("playback-play", "n_clicks"),
         Input("playback-rewind", "n_clicks"),
         Input("playback-forward", "n_clicks"),
         Input("playback-live", "n_clicks"),
         Input("playback-slider", "value")],
        [State("playback-state", "data"),
         State("playback-timeline", "data")],
        prevent_initial_call=True,
    )
    def control_playback(play_clicks, rewind_clicks, forward_clicks, live_clicks,
                         slider_value, playback_state, timeline_data):
        """Handle playback control button clicks and slider changes."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        state = playback_state or {"playing": False, "time": None, "speed": 1}
        timeline = timeline_data or {"min_time": 0, "max_time": 100, "timestamps": []}

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
                {"playing": playing, "time": current_time, "speed": state.get("speed", 1)},
                not playing,  # Disable interval when not playing
                "⏸️ Pause" if playing else "▶️ Play",
            )

        elif trigger_id == "playback-rewind":
            # Jump back 10% of timeline
            step = max(1, (max_t - min_t) * 0.1)
            new_time = max(min_t, current_time - step)
            return (
                {"playing": False, "time": new_time, "speed": state.get("speed", 1)},
                True,
                "▶️ Play",
            )

        elif trigger_id == "playback-forward":
            # Jump forward 10% of timeline
            step = max(1, (max_t - min_t) * 0.1)
            new_time = min(max_t, current_time + step)
            return (
                {"playing": False, "time": new_time, "speed": state.get("speed", 1)},
                True,
                "▶️ Play",
            )

        elif trigger_id == "playback-live":
            # Jump to live (most recent)
            return (
                {"playing": False, "time": None, "speed": state.get("speed", 1)},
                True,
                "▶️ Play",
            )

        elif trigger_id == "playback-slider":
            # Slider was dragged
            return (
                {"playing": False, "time": slider_value, "speed": state.get("speed", 1)},
                True,
                "▶️ Play",
            )

        return no_update, no_update, no_update

    @app.callback(
        [Output("playback-slider", "value"),
         Output("playback-time", "children")],
        [Input("playback-interval", "n_intervals"),
         Input("playback-state", "data")],
        [State("playback-timeline", "data")],
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
         Input("playback-mode-toggle", "value")],
    )
    def update_dual_maps_with_playback(n, health_toggle, playback_state, playback_enabled):
        """Update dual maps with support for playback mode."""
        show_health = "on" in (health_toggle or [])
        playback_on = "on" in (playback_enabled or [])

        # Determine which data source to use
        if playback_on and playback_state and playback_state.get("time") is not None:
            # Use historical data
            map_data = fetch_position_at_time(playback_state["time"])
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
         Input("gnn-graph-yard-filter", "value"),
         Input("gnn-graph-edge-filter", "value"),
         Input("gnn-graph-health-toggle", "value")],
    )
    def update_gnn_graph(n, yard_filter, edge_filters, health_toggle):
        """Build and update the GNN graph visualization."""
        elements = []
        stats_children = []

        # Fetch data from database
        blocks = fetch_query("SELECT * FROM blocks") or []
        spmts = fetch_query("SELECT * FROM spmts") or []
        cranes = fetch_query("SELECT * FROM cranes") or []

        # Facility data - derive from blocks if not in separate table
        facilities_data = fetch_query("SELECT DISTINCT current_location as id FROM blocks WHERE current_location IS NOT NULL") or []
        facilities = [{"id": f["id"]} for f in facilities_data if f.get("id")]

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

        # Filter by yard if applicable
        if yard_filter and yard_filter != "all":
            blocks = [b for b in blocks if b.get("yard", "quonset").lower() == yard_filter.lower()]
            spmts = [s for s in spmts if s.get("yard", "quonset").lower() == yard_filter.lower()]
            cranes = [c for c in cranes if c.get("yard", "quonset").lower() == yard_filter.lower()]

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
                    "location": block.get("current_location", "unknown"),
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

        # Create Crane nodes
        for crane in cranes:
            min_health = min(
                crane.get("health_cable", 100),
                crane.get("health_motor", 100),
            )
            health_status = get_health_status(min_health) if show_health else "healthy"
            node = {
                "data": {
                    "id": f"crane_{crane['id']}",
                    "label": crane["id"],
                    "type": "crane",
                    "health_status": health_status,
                    "status": crane.get("status", "unknown"),
                    "health_cable": crane.get("health_cable", 100),
                    "health_motor": crane.get("health_motor", 100),
                    "position": crane.get("position", 0),
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

        # Transport edges (block <-> SPMT)
        if "transport" in edge_filters:
            for block in blocks:
                for spmt in spmts:
                    # needs_transport: block -> spmt
                    edge = {
                        "data": {
                            "source": f"block_{block['id']}",
                            "target": f"spmt_{spmt['id']}",
                            "type": "needs_transport",
                        }
                    }
                    elements.append(edge)
                    edge_counts["needs_transport"] += 1

        # Lift edges (block <-> Crane)
        if "lift" in edge_filters:
            for block in blocks:
                for crane in cranes:
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
        if "location" in edge_filters:
            for block in blocks:
                loc = block.get("current_location")
                if loc:
                    edge = {
                        "data": {
                            "source": f"block_{block['id']}",
                            "target": f"facility_{loc}",
                            "type": "at",
                        }
                    }
                    elements.append(edge)
                    edge_counts["at"] += 1

            for spmt in spmts:
                loc = spmt.get("current_location")
                if loc:
                    edge = {
                        "data": {
                            "source": f"spmt_{spmt['id']}",
                            "target": f"facility_{loc}",
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
            html.P([html.Strong("Total Nodes: "), f"{total_nodes}"]),
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

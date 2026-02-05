"""SQLite database helpers for the MES dashboard.

Provides functions to initialize the database, log simulation state
(metrics, entities, health history, queue depths, block events) and
retrieve data for display. The simulation calls logging functions
periodically to populate the dashboard with real data.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, Any, List, Optional


DB_PATH = os.environ.get("SHIPYARD_DB", "shipyard.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create database tables if they do not exist."""
    conn = _get_conn()
    cur = conn.cursor()

    # Snapshot tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            time REAL PRIMARY KEY,
            blocks_completed INTEGER,
            breakdowns INTEGER,
            planned_maintenance INTEGER,
            total_tardiness REAL,
            empty_travel_distance REAL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS blocks (
            id TEXT PRIMARY KEY,
            status TEXT,
            location TEXT,
            due_date REAL,
            current_stage TEXT,
            completion_pct REAL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spmts (
            id TEXT PRIMARY KEY,
            status TEXT,
            current_location TEXT,
            load TEXT,
            health_hydraulic REAL,
            health_tires REAL,
            health_engine REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cranes (
            id TEXT PRIMARY KEY,
            status TEXT,
            position_on_rail REAL,
            health_cable REAL,
            health_motor REAL
        )
    """)

    # Time-series tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS health_history (
            timestamp REAL,
            equipment_id TEXT,
            equipment_type TEXT,
            component TEXT,
            health_value REAL,
            PRIMARY KEY (timestamp, equipment_id, component)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queue_history (
            timestamp REAL,
            facility_name TEXT,
            queue_depth INTEGER,
            processing_count INTEGER,
            PRIMARY KEY (timestamp, facility_name)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS block_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            block_id TEXT,
            timestamp REAL,
            event_type TEXT,
            stage TEXT,
            location TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_health_ts ON health_history(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_queue_ts ON queue_history(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_block_events_block ON block_events(block_id)")

    # Position history for playback (timestamped snapshots of all entity positions)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS position_history (
            timestamp REAL,
            yard TEXT,
            entity_type TEXT,
            entity_id TEXT,
            location TEXT,
            status TEXT,
            extra_data TEXT,
            PRIMARY KEY (timestamp, yard, entity_type, entity_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_position_ts ON position_history(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_position_yard ON position_history(yard)")

    # Barge tracking table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS barges (
            id TEXT PRIMARY KEY,
            status TEXT,
            current_location TEXT,
            cargo TEXT,
            transit_progress REAL,
            capacity INTEGER
        )
    """)

    conn.commit()
    conn.close()


def clear_db() -> None:
    """Clear all data (useful between runs)."""
    conn = _get_conn()
    for table in ["metrics", "blocks", "spmts", "cranes", "health_history", "queue_history", "block_events", "position_history", "barges"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()


def log_metrics(time: float, metrics: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO metrics (time, blocks_completed, breakdowns, planned_maintenance, total_tardiness, empty_travel_distance) VALUES (?, ?, ?, ?, ?, ?)",
        (
            time,
            int(metrics.get("blocks_completed", 0)),
            int(metrics.get("breakdowns", 0)),
            int(metrics.get("planned_maintenance", 0)),
            float(metrics.get("total_tardiness", 0.0)),
            float(metrics.get("empty_travel_distance", 0.0)),
        ),
    )
    conn.commit()
    conn.close()


def log_entities(blocks: list, spmts: list, cranes: list) -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM blocks")
    cur.execute("DELETE FROM spmts")
    cur.execute("DELETE FROM cranes")
    for b in blocks:
        stage = b.current_stage.name if hasattr(b.current_stage, 'name') else str(b.current_stage)
        cur.execute(
            "INSERT INTO blocks (id, status, location, due_date, current_stage, completion_pct) VALUES (?, ?, ?, ?, ?, ?)",
            (b.id, b.status.value, b.location, b.due_date, stage, getattr(b, 'completion_pct', 0.0)),
        )
    for s in spmts:
        cur.execute(
            "INSERT INTO spmts (id, status, current_location, load, health_hydraulic, health_tires, health_engine) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (s.id, s.status.value, s.current_location, s.current_load or "", s.health_hydraulic, s.health_tires, s.health_engine),
        )
    for c in cranes:
        cur.execute(
            "INSERT INTO cranes (id, status, position_on_rail, health_cable, health_motor) VALUES (?, ?, ?, ?, ?)",
            (c.id, c.status.value, c.position_on_rail, c.health_cable, c.health_motor),
        )
    conn.commit()
    conn.close()


def log_health_snapshot(time: float, spmts: list, cranes: list) -> None:
    """Log per-component health values for all equipment."""
    conn = _get_conn()
    cur = conn.cursor()
    for s in spmts:
        for comp, val in [("hydraulic", s.health_hydraulic), ("tires", s.health_tires), ("engine", s.health_engine)]:
            cur.execute(
                "INSERT OR REPLACE INTO health_history (timestamp, equipment_id, equipment_type, component, health_value) VALUES (?, ?, ?, ?, ?)",
                (time, s.id, "spmt", comp, val),
            )
    for c in cranes:
        for comp, val in [("cable", c.health_cable), ("motor", c.health_motor)]:
            cur.execute(
                "INSERT OR REPLACE INTO health_history (timestamp, equipment_id, equipment_type, component, health_value) VALUES (?, ?, ?, ?, ?)",
                (time, c.id, "crane", comp, val),
            )
    conn.commit()
    conn.close()


def log_queue_depths(time: float, facility_queues: Dict[str, list], facility_processing: Dict[str, list]) -> None:
    """Log queue depth and processing count per facility."""
    conn = _get_conn()
    cur = conn.cursor()
    all_facilities = set(list(facility_queues.keys()) + list(facility_processing.keys()))
    for fac in all_facilities:
        q_depth = len(facility_queues.get(fac, []))
        p_count = len(facility_processing.get(fac, []))
        cur.execute(
            "INSERT OR REPLACE INTO queue_history (timestamp, facility_name, queue_depth, processing_count) VALUES (?, ?, ?, ?)",
            (time, fac, q_depth, p_count),
        )
    conn.commit()
    conn.close()


def log_block_event(block_id: str, time: float, event_type: str, stage: str, location: str) -> None:
    """Log a block lifecycle event (stage completion, transport, lift)."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO block_events (block_id, timestamp, event_type, stage, location) VALUES (?, ?, ?, ?, ?)",
        (block_id, time, event_type, stage, location),
    )
    conn.commit()
    conn.close()


# Query helpers for dashboard callbacks

def fetch_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return results as list of dicts."""
    if not os.path.exists(DB_PATH):
        return []
    conn = _get_conn()
    cur = conn.execute(query, params)
    records = [dict(row) for row in cur.fetchall()]
    conn.close()
    return records


def fetch_health_history(equipment_id: str | None = None, time_window: float | None = None) -> List[Dict[str, Any]]:
    """Fetch health time-series, optionally filtered."""
    query = "SELECT * FROM health_history"
    conditions, params = [], []
    if equipment_id:
        conditions.append("equipment_id = ?")
        params.append(equipment_id)
    if time_window:
        conditions.append("timestamp >= (SELECT MAX(timestamp) FROM health_history) - ?")
        params.append(time_window)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp"
    return fetch_query(query, tuple(params))


def fetch_queue_history(time_window: float | None = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM queue_history"
    if time_window:
        query += " WHERE timestamp >= (SELECT MAX(timestamp) FROM queue_history) - ?"
        return fetch_query(query, (time_window,))
    query += " ORDER BY timestamp"
    return fetch_query(query)


def fetch_block_events(block_id: str | None = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM block_events"
    if block_id:
        query += " WHERE block_id = ?"
        return fetch_query(query, (block_id,))
    query += " ORDER BY timestamp"
    return fetch_query(query)


# ============================================================================
# POSITION HISTORY (for simulation playback)
# ============================================================================

def log_position_snapshot(time: float, blocks: list, spmts: list, cranes: list, barge: Any = None) -> None:
    """Log a complete position snapshot for playback.

    Each entity's position and status is recorded with a timestamp,
    enabling historical replay of the simulation state.
    """
    import json

    conn = _get_conn()
    cur = conn.cursor()

    # Determine yard for each entity based on location
    def get_yard(location: str) -> str:
        location = location.lower() if location else ""
        if any(x in location for x in ["quonset", "steel_", "afc", "bldg_9", "super_module", "barge_loading"]):
            return "quonset"
        elif any(x in location for x in ["groton", "land_level", "building_600", "graving_dock", "dry_dock"]):
            return "groton"
        elif "transit" in location or "barge" in location:
            return "transit"
        return "quonset"  # Default

    # Log blocks
    for b in blocks:
        location = getattr(b, "location", None) or ""
        status = b.status.value if hasattr(b, "status") and hasattr(b.status, "value") else str(getattr(b, "status", ""))
        stage = b.current_stage.name if hasattr(b.current_stage, "name") else str(getattr(b, "current_stage", ""))
        yard = get_yard(location)

        extra = json.dumps({
            "stage": stage,
            "completion_pct": getattr(b, "completion_pct", 0),
            "due_date": getattr(b, "due_date", None),
        })

        cur.execute(
            "INSERT OR REPLACE INTO position_history (timestamp, yard, entity_type, entity_id, location, status, extra_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time, yard, "block", b.id, location, status, extra),
        )

    # Log SPMTs
    for s in spmts:
        location = getattr(s, "current_location", "") or ""
        status = s.status.value if hasattr(s, "status") and hasattr(s.status, "value") else str(getattr(s, "status", ""))
        yard = get_yard(location)

        extra = json.dumps({
            "load": getattr(s, "current_load", None),
            "health_hydraulic": getattr(s, "health_hydraulic", 100),
            "health_tires": getattr(s, "health_tires", 100),
            "health_engine": getattr(s, "health_engine", 100),
        })

        cur.execute(
            "INSERT OR REPLACE INTO position_history (timestamp, yard, entity_type, entity_id, location, status, extra_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time, yard, "spmt", s.id, location, status, extra),
        )

    # Log cranes
    for c in cranes:
        location = f"rail_{getattr(c, 'position_on_rail', 0)}"
        status = c.status.value if hasattr(c, "status") and hasattr(c.status, "value") else str(getattr(c, "status", ""))
        # Assume cranes are yard-specific (need to determine from crane ID or config)
        yard = "quonset" if "Q" in c.id.upper() else "groton"

        extra = json.dumps({
            "position_on_rail": getattr(c, "position_on_rail", 0),
            "health_cable": getattr(c, "health_cable", 100),
            "health_motor": getattr(c, "health_motor", 100),
        })

        cur.execute(
            "INSERT OR REPLACE INTO position_history (timestamp, yard, entity_type, entity_id, location, status, extra_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time, yard, "crane", c.id, location, status, extra),
        )

    # Log barge
    if barge:
        barge_location = getattr(barge, "current_location", "quonset_pier")
        barge_status = barge.status.value if hasattr(barge, "status") and hasattr(barge.status, "value") else str(getattr(barge, "status", ""))

        extra = json.dumps({
            "cargo": getattr(barge, "cargo", []),
            "transit_progress": getattr(barge, "transit_progress", 0),
            "capacity": getattr(barge, "capacity", 2),
        })

        cur.execute(
            "INSERT OR REPLACE INTO position_history (timestamp, yard, entity_type, entity_id, location, status, extra_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time, "transit", "barge", barge.id, barge_location, barge_status, extra),
        )

        # Also update the barges table for current state
        cur.execute(
            "INSERT OR REPLACE INTO barges (id, status, current_location, cargo, transit_progress, capacity) VALUES (?, ?, ?, ?, ?, ?)",
            (
                barge.id,
                barge_status,
                barge_location,
                json.dumps(getattr(barge, "cargo", [])),
                getattr(barge, "transit_progress", 0),
                getattr(barge, "capacity", 2),
            ),
        )

    conn.commit()
    conn.close()


def fetch_position_at_time(target_time: float) -> Dict[str, Any]:
    """Fetch entity positions at or before a specific timestamp.

    Returns the most recent snapshot at or before target_time.
    """
    import json

    # Find the closest timestamp <= target_time
    closest_ts = fetch_query(
        "SELECT MAX(timestamp) as ts FROM position_history WHERE timestamp <= ?",
        (target_time,)
    )

    if not closest_ts or closest_ts[0]["ts"] is None:
        return {"spmts": [], "cranes": [], "blocks": [], "queue_depths": {}, "barge": None}

    actual_time = closest_ts[0]["ts"]

    rows = fetch_query(
        "SELECT * FROM position_history WHERE timestamp = ?",
        (actual_time,)
    )

    spmts = []
    cranes = []
    blocks = []
    barge = None

    for r in rows:
        extra = json.loads(r.get("extra_data", "{}")) if r.get("extra_data") else {}

        if r["entity_type"] == "spmt":
            spmts.append({
                "id": r["entity_id"],
                "status": r["status"],
                "current_location": r["location"],
                "load": extra.get("load", ""),
                "health_hydraulic": extra.get("health_hydraulic", 100),
                "health_tires": extra.get("health_tires", 100),
                "health_engine": extra.get("health_engine", 100),
            })
        elif r["entity_type"] == "crane":
            cranes.append({
                "id": r["entity_id"],
                "status": r["status"],
                "position_on_rail": extra.get("position_on_rail", 0),
                "health_cable": extra.get("health_cable", 100),
                "health_motor": extra.get("health_motor", 100),
            })
        elif r["entity_type"] == "block":
            blocks.append({
                "id": r["entity_id"],
                "status": r["status"],
                "location": r["location"],
                "current_stage": extra.get("stage", ""),
                "completion_pct": extra.get("completion_pct", 0),
                "due_date": extra.get("due_date"),
            })
        elif r["entity_type"] == "barge":
            barge = {
                "id": r["entity_id"],
                "status": r["status"],
                "current_location": r["location"],
                "cargo": extra.get("cargo", []),
                "transit_progress": extra.get("transit_progress", 0),
                "capacity": extra.get("capacity", 2),
            }

    # Fetch queue depths at this time
    queue_rows = fetch_query(
        """SELECT facility_name, queue_depth, processing_count
           FROM queue_history
           WHERE timestamp = (SELECT MAX(timestamp) FROM queue_history WHERE timestamp <= ?)""",
        (actual_time,)
    )
    queue_depths = {r["facility_name"]: r["queue_depth"] + r["processing_count"]
                    for r in (queue_rows or [])}

    return {
        "spmts": spmts,
        "cranes": cranes,
        "blocks": blocks,
        "queue_depths": queue_depths,
        "barge": barge,
        "actual_time": actual_time,
    }


def fetch_playback_timeline() -> Dict[str, Any]:
    """Get timeline info for playback controls (min/max timestamps, count)."""
    result = fetch_query(
        """SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts,
                  COUNT(DISTINCT timestamp) as snapshot_count
           FROM position_history"""
    )

    if not result or result[0]["min_ts"] is None:
        return {"min_time": 0, "max_time": 0, "snapshot_count": 0}

    return {
        "min_time": result[0]["min_ts"],
        "max_time": result[0]["max_ts"],
        "snapshot_count": result[0]["snapshot_count"],
    }


def fetch_available_timestamps() -> List[float]:
    """Get all available snapshot timestamps for playback scrubbing."""
    rows = fetch_query(
        "SELECT DISTINCT timestamp FROM position_history ORDER BY timestamp"
    )
    return [r["timestamp"] for r in rows] if rows else []

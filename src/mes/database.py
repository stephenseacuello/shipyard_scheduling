"""SQLite database helpers for the MES dashboard.

Provides functions to initialize the database, log simulation state
(metrics, entities, health history, queue depths, block events) and
retrieve data for display. The simulation calls logging functions
periodically to populate the dashboard with real data.

Supports HD Hyundai Heavy Industries (HHI) Ulsan shipyard model with:
- LNG Carrier ships
- Goliath cranes (9 units, 109m tall)
- 200 blocks per ship
- 10 dry docks
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, Any, List, Optional


def _get_db_path() -> str:
    """Get database path, checking multiple locations."""
    # Check environment variable first
    if os.environ.get("SHIPYARD_DB"):
        return os.environ["SHIPYARD_DB"]

    # Check current working directory
    if os.path.exists("shipyard.db"):
        return "shipyard.db"

    # Check project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root_db = os.path.join(project_root, "shipyard.db")
    if os.path.exists(root_db):
        return root_db

    # Default to current directory
    return "shipyard.db"


DB_PATH = _get_db_path()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)  # Wait up to 30s for lock
    conn.row_factory = sqlite3.Row
    # Use WAL mode for better concurrent access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout
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
            completion_pct REAL DEFAULT 0,
            super_module_id TEXT
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ship_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ship_id TEXT,
            timestamp REAL,
            event_type TEXT,
            status TEXT,
            dock_id TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_health_ts ON health_history(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_queue_ts ON queue_history(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_block_events_block ON block_events(block_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ship_events_ship ON ship_events(ship_id)")

    # Simulation runs table (for historical playback selection)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            config_path TEXT,
            policy_type TEXT,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            ended_at TEXT,
            total_steps INTEGER DEFAULT 0,
            blocks_completed INTEGER DEFAULT 0,
            ships_delivered INTEGER DEFAULT 0,
            total_reward REAL DEFAULT 0
        )
    """)

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
            run_id INTEGER DEFAULT NULL,
            PRIMARY KEY (timestamp, yard, entity_type, entity_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_position_ts ON position_history(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_position_yard ON position_history(yard)")

    # Add run_id column to existing databases (migration)
    try:
        cur.execute("ALTER TABLE position_history ADD COLUMN run_id INTEGER DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    cur.execute("CREATE INDEX IF NOT EXISTS idx_position_run ON position_history(run_id)")


    # ========================================================================
    # HHI ULSAN SHIPYARD TABLES
    # ========================================================================

    # LNG Carrier ships under construction
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ships (
            id TEXT PRIMARY KEY,
            hull_number TEXT,
            ship_type TEXT DEFAULT 'lng_carrier',
            capacity_cbm REAL DEFAULT 174000,
            total_blocks INTEGER DEFAULT 200,
            blocks_erected INTEGER DEFAULT 0,
            assigned_dock TEXT,
            assigned_quay TEXT,
            target_delivery_date REAL,
            status TEXT DEFAULT 'in_block_production',
            erection_progress REAL DEFAULT 0,
            completion_pct REAL DEFAULT 0
        )
    """)

    # Goliath cranes (9 units, 109m tall)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS goliath_cranes (
            id TEXT PRIMARY KEY,
            assigned_dock TEXT,
            capacity_tons REAL DEFAULT 900,
            height_m REAL DEFAULT 109,
            position_on_rail REAL DEFAULT 0,
            status TEXT DEFAULT 'idle',
            current_block TEXT,
            health_hoist REAL DEFAULT 100,
            health_trolley REAL DEFAULT 100,
            health_gantry REAL DEFAULT 100
        )
    """)

    # Dry docks (10 units)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dry_docks (
            id TEXT PRIMARY KEY,
            name TEXT,
            length_m REAL,
            width_m REAL,
            current_ship TEXT,
            status TEXT DEFAULT 'idle',
            blocks_in_dock INTEGER DEFAULT 0
        )
    """)

    # Outfitting quays (3 units)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS outfitting_quays (
            id TEXT PRIMARY KEY,
            name TEXT,
            length_m REAL,
            capacity INTEGER DEFAULT 2,
            current_ships TEXT
        )
    """)

    # Extended blocks table for HHI (block type, ship assignment)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hhi_blocks (
            id TEXT PRIMARY KEY,
            ship_id TEXT,
            block_type TEXT,
            weight REAL,
            erection_sequence INTEGER,
            status TEXT,
            location TEXT,
            current_stage TEXT,
            completion_pct REAL DEFAULT 0,
            due_date REAL,
            predecessors TEXT DEFAULT ''
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_ships_dock ON ships(assigned_dock)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hhi_blocks_ship ON hhi_blocks(ship_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_goliath_dock ON goliath_cranes(assigned_dock)")

    # ========================================================================
    # PLATE DECOMPOSITION TABLES
    # ========================================================================

    # Migration: add plate columns to hhi_blocks
    for col, col_def in [
        ("n_plates", "INTEGER DEFAULT 0"),
        ("plate_area_m2", "REAL DEFAULT 0"),
        ("processing_source", "TEXT DEFAULT 'lognormal'"),
    ]:
        try:
            cur.execute(f"ALTER TABLE hhi_blocks ADD COLUMN {col} {col_def}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Plate processing time records
    cur.execute("""
        CREATE TABLE IF NOT EXISTS plate_processing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            block_id TEXT,
            timestamp REAL,
            stage TEXT,
            n_plates INTEGER,
            processing_time_hours REAL,
            method TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_proc_block ON plate_processing(block_id)")

    conn.commit()
    conn.close()


def clear_db() -> None:
    """Clear all data (useful between runs)."""
    conn = _get_conn()
    tables = [
        "metrics", "blocks", "spmts", "cranes", "health_history",
        "queue_history", "block_events", "ship_events", "position_history",
        # HHI tables
        "ships", "goliath_cranes", "dry_docks", "outfitting_quays", "hhi_blocks"
    ]
    for table in tables:
        try:
            conn.execute(f"DELETE FROM {table}")
        except sqlite3.OperationalError:
            pass  # Table may not exist
    conn.commit()
    conn.close()


def log_ships(ships: list) -> None:
    """Log LNG carrier ships to the database."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM ships")
    for ship in ships:
        status = ship.status.value if hasattr(ship.status, 'value') else str(ship.status)
        cur.execute(
            """INSERT INTO ships (id, hull_number, ship_type, capacity_cbm, total_blocks,
               blocks_erected, assigned_dock, assigned_quay, target_delivery_date,
               status, erection_progress, completion_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ship.id,
                getattr(ship, 'hull_number', ''),
                'lng_carrier',
                getattr(ship, 'capacity_cbm', 174000),
                getattr(ship, 'total_blocks', 200),
                getattr(ship, 'blocks_erected', 0),
                getattr(ship, 'assigned_dock', None),
                getattr(ship, 'assigned_quay', None),
                getattr(ship, 'target_delivery_date', 0),
                status,
                getattr(ship, 'erection_progress', 0),
                getattr(ship, 'completion_pct', 0),
            ),
        )
    conn.commit()
    conn.close()


def log_goliath_cranes(cranes: list) -> None:
    """Log Goliath crane state to the database."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM goliath_cranes")
    for crane in cranes:
        status = crane.status.value if hasattr(crane.status, 'value') else str(crane.status)
        cur.execute(
            """INSERT INTO goliath_cranes (id, assigned_dock, capacity_tons, height_m,
               position_on_rail, status, current_block, health_hoist, health_trolley, health_gantry)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                crane.id,
                getattr(crane, 'assigned_dock', ''),
                getattr(crane, 'capacity_tons', 900),
                getattr(crane, 'height_m', 109),
                getattr(crane, 'position_on_rail', 0),
                status,
                getattr(crane, 'current_block', None),
                getattr(crane, 'health_hoist', 100),
                getattr(crane, 'health_trolley', 100),
                getattr(crane, 'health_gantry', 100),
            ),
        )
    conn.commit()
    conn.close()


def log_dry_docks(docks: list) -> None:
    """Log dry dock state to the database."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM dry_docks")
    for dock in docks:
        cur.execute(
            """INSERT INTO dry_docks (id, name, length_m, width_m, current_ship, status, blocks_in_dock)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                dock.id,
                getattr(dock, 'name', dock.id),
                getattr(dock, 'length_m', 400),
                getattr(dock, 'width_m', 80),
                getattr(dock, 'current_ship', None),
                getattr(dock, 'status', 'idle'),
                len(getattr(dock, 'blocks_in_dock', [])),
            ),
        )
    conn.commit()
    conn.close()


def log_hhi_blocks(blocks: list) -> None:
    """Log HHI-specific block data to the database."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM hhi_blocks")
    for block in blocks:
        stage = block.current_stage.name if hasattr(block.current_stage, 'name') else str(block.current_stage)
        block_type = block.block_type.value if hasattr(block.block_type, 'value') else str(getattr(block, 'block_type', 'flat_bottom'))
        # Handle predecessors - can be list or string
        preds = getattr(block, 'predecessors', [])
        if isinstance(preds, list):
            preds_str = ','.join(str(p) for p in preds)
        else:
            preds_str = str(preds) if preds else ''
        cur.execute(
            """INSERT INTO hhi_blocks (id, ship_id, block_type, weight, erection_sequence,
               status, location, current_stage, completion_pct, due_date, predecessors)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                block.id,
                getattr(block, 'ship_id', ''),
                block_type,
                getattr(block, 'weight', 0),
                getattr(block, 'erection_sequence', 0),
                block.status.value if hasattr(block.status, 'value') else str(block.status),
                getattr(block, 'location', ''),
                stage,
                getattr(block, 'completion_pct', 0),
                getattr(block, 'due_date', 0),
                preds_str,
            ),
        )
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
        super_module_id = getattr(b, 'super_module_id', None)
        cur.execute(
            "INSERT INTO blocks (id, status, location, due_date, current_stage, completion_pct, super_module_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (b.id, b.status.value, b.location, b.due_date, stage, getattr(b, 'completion_pct', 0.0), super_module_id),
        )
    for s in spmts:
        cur.execute(
            "INSERT INTO spmts (id, status, current_location, load, health_hydraulic, health_tires, health_engine) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (s.id, s.status.value, s.current_location, s.current_load or "", s.health_hydraulic, s.health_tires, s.health_engine),
        )
    for c in cranes:
        health_1 = getattr(c, 'health_hoist', getattr(c, 'health_cable', 100))
        health_2 = getattr(c, 'health_trolley', getattr(c, 'health_motor', 100))
        cur.execute(
            "INSERT INTO cranes (id, status, position_on_rail, health_cable, health_motor) VALUES (?, ?, ?, ?, ?)",
            (c.id, c.status.value, c.position_on_rail, health_1, health_2),
        )
    conn.commit()
    conn.close()


def log_spmts(spmts: list) -> None:
    """Log SPMT entities to the spmts table."""
    if not spmts:
        return
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM spmts")
    for s in spmts:
        status = s.status.value if hasattr(s.status, 'value') else str(s.status)
        cur.execute(
            "INSERT INTO spmts (id, status, current_location, load, health_hydraulic, health_tires, health_engine) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (s.id, status, s.current_location, s.current_load or "", s.health_hydraulic, s.health_tires, s.health_engine),
        )
    conn.commit()
    conn.close()


def log_health_snapshot(time: float, spmts: list, cranes: list) -> None:
    """Log per-component health values for all equipment.

    For HHI Ulsan, cranes are Goliath cranes with 3 health components:
    - hoist: Main lifting mechanism
    - trolley: Horizontal movement on bridge
    - gantry: Rail movement along dock
    """
    conn = _get_conn()
    cur = conn.cursor()
    for s in spmts:
        for comp, val in [("hydraulic", s.health_hydraulic), ("tires", s.health_tires), ("engine", s.health_engine)]:
            cur.execute(
                "INSERT OR REPLACE INTO health_history (timestamp, equipment_id, equipment_type, component, health_value) VALUES (?, ?, ?, ?, ?)",
                (time, s.id, "spmt", comp, val),
            )
    for c in cranes:
        # Support both old Crane (cable/motor) and new GoliathCrane (hoist/trolley/gantry)
        health_components = []
        if hasattr(c, 'health_hoist'):
            # New GoliathCrane format
            health_components = [
                ("hoist", c.health_hoist),
                ("trolley", c.health_trolley),
                ("gantry", c.health_gantry),
            ]
        else:
            # Legacy Crane format (backward compatibility)
            health_components = [
                ("cable", getattr(c, 'health_cable', 100)),
                ("motor", getattr(c, 'health_motor', 100)),
            ]
        for comp, val in health_components:
            cur.execute(
                "INSERT OR REPLACE INTO health_history (timestamp, equipment_id, equipment_type, component, health_value) VALUES (?, ?, ?, ?, ?)",
                (time, c.id, "goliath_crane", comp, val),
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


def log_ship_event(ship_id: str, time: float, event_type: str, status: str, dock_id: str = "") -> None:
    """Log a ship lifecycle event (status change, dock assignment)."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO ship_events (ship_id, timestamp, event_type, status, dock_id) VALUES (?, ?, ?, ?, ?)",
        (ship_id, time, event_type, status, dock_id),
    )
    conn.commit()
    conn.close()


# Query helpers for dashboard callbacks

def fetch_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return results as list of dicts."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = _get_conn()
        cur = conn.execute(query, params)
        records = [dict(row) for row in cur.fetchall()]
        conn.close()
        return records
    except Exception as e:
        # Log error but don't crash - return empty list
        print(f"Database query error: {e}")
        return []


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


def fetch_ship_events(ship_id: str | None = None) -> List[Dict[str, Any]]:
    """Fetch ship lifecycle events for Gantt chart."""
    query = "SELECT * FROM ship_events"
    if ship_id:
        query += " WHERE ship_id = ?"
        return fetch_query(query, (ship_id,))
    query += " ORDER BY timestamp"
    return fetch_query(query)


# ============================================================================
# POSITION HISTORY (for simulation playback)
# ============================================================================

def log_position_snapshot(time: float, blocks: list, spmts: list, cranes: list,
                          ships: list = None,
                          run_id: int = None) -> None:
    """Log a complete position snapshot for playback.

    Each entity's position and status is recorded with a timestamp,
    enabling historical replay of the simulation state.

    Parameters
    ----------
    time : float
        Simulation time
    blocks : list
        List of Block entities
    spmts : list
        List of SPMT entities
    cranes : list
        List of GoliathCrane entities
    ships : list, optional
        List of LNGCarrier ships (for swim-away animation)
    run_id : int, optional
        Simulation run ID for historical tracking
    """
    import json

    conn = _get_conn()
    cur = conn.cursor()

    # HHI Ulsan: Single yard - all entities are in the same shipyard
    def get_yard(entity, location: str) -> str:
        # HHI Ulsan is a single shipyard - no dual-yard logic needed
        return "hhi_ulsan"

    # Log ships (LNG carriers) with sea_position for swim-away animation
    if ships:
        for ship in ships:
            status = ship.status.value if hasattr(ship.status, 'value') else str(ship.status)

            # Determine ship location based on status
            if status == 'delivered':
                location = 'sea'  # Ship has sailed away
            elif status == 'in_sea_trials':
                location = 'sea_trials_area'
            elif status in ('in_quay_outfitting', 'afloat'):
                location = getattr(ship, 'assigned_quay', 'outfitting_quay')
            elif status == 'in_erection':
                location = getattr(ship, 'assigned_dock', 'dock')
            else:
                location = 'production'

            extra = json.dumps({
                'hull_number': getattr(ship, 'hull_number', ''),
                'assigned_dock': getattr(ship, 'assigned_dock', None),
                'assigned_quay': getattr(ship, 'assigned_quay', None),
                'blocks_erected': getattr(ship, 'blocks_erected', 0),
                'total_blocks': getattr(ship, 'total_blocks', 200),
                'erection_progress': getattr(ship, 'erection_progress', 0),
                'completion_pct': getattr(ship, 'completion_pct', 0),
                'sea_position': getattr(ship, 'sea_position', 0.0),  # 0=at quay, 1=at sea
                'stage_remaining_time': getattr(ship, 'stage_remaining_time', 0),
            })

            cur.execute(
                "INSERT OR REPLACE INTO position_history (timestamp, yard, entity_type, entity_id, location, status, extra_data, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time, "hhi_ulsan", "ship", ship.id, location, status, extra, run_id),
            )

    # Log blocks
    for b in blocks:
        location = getattr(b, "location", None) or ""
        status = b.status.value if hasattr(b, "status") and hasattr(b.status, "value") else str(getattr(b, "status", ""))
        stage = b.current_stage.name if hasattr(b.current_stage, "name") else str(getattr(b, "current_stage", ""))
        yard = get_yard(b, location)

        extra = json.dumps({
            "stage": stage,
            "completion_pct": getattr(b, "completion_pct", 0),
            "due_date": getattr(b, "due_date", None),
        })

        cur.execute(
            "INSERT OR REPLACE INTO position_history (timestamp, yard, entity_type, entity_id, location, status, extra_data, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time, yard, "block", b.id, location, status, extra, run_id),
        )

    # Log SPMTs
    for s in spmts:
        location = getattr(s, "current_location", "") or ""
        status = s.status.value if hasattr(s, "status") and hasattr(s.status, "value") else str(getattr(s, "status", ""))
        yard = get_yard(s, location)

        extra = json.dumps({
            "load": getattr(s, "current_load", None),
            "health_hydraulic": getattr(s, "health_hydraulic", 100),
            "health_tires": getattr(s, "health_tires", 100),
            "health_engine": getattr(s, "health_engine", 100),
        })

        cur.execute(
            "INSERT OR REPLACE INTO position_history (timestamp, yard, entity_type, entity_id, location, status, extra_data, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time, yard, "spmt", s.id, location, status, extra, run_id),
        )

    # Log cranes (Goliath cranes for HHI Ulsan)
    for c in cranes:
        assigned_dock = getattr(c, 'assigned_dock', '')
        location = f"{assigned_dock}_rail_{getattr(c, 'position_on_rail', 0)}"
        status = c.status.value if hasattr(c, "status") and hasattr(c.status, "value") else str(getattr(c, "status", ""))
        yard = "hhi_ulsan"  # HHI Ulsan single shipyard

        # Support both old Crane and new GoliathCrane health components
        extra = json.dumps({
            "position_on_rail": getattr(c, "position_on_rail", 0),
            "assigned_dock": assigned_dock,
            "health_hoist": getattr(c, "health_hoist", getattr(c, "health_cable", 100)),
            "health_trolley": getattr(c, "health_trolley", getattr(c, "health_motor", 100)),
            "health_gantry": getattr(c, "health_gantry", 100),
            "capacity_tons": getattr(c, "capacity_tons", 900),
            "height_m": getattr(c, "height_m", 109),
        })

        cur.execute(
            "INSERT OR REPLACE INTO position_history (timestamp, yard, entity_type, entity_id, location, status, extra_data, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time, yard, "crane", c.id, location, status, extra, run_id),
        )

    conn.commit()
    conn.close()


def fetch_position_at_time(target_time: float, run_id: int = None) -> Dict[str, Any]:
    """Fetch entity positions at or before a specific timestamp.

    Returns the most recent snapshot at or before target_time.
    Includes ships with sea_position for swim-away animation.

    Parameters
    ----------
    target_time : float
        Target simulation timestamp to fetch data for.
    run_id : int, optional
        Specific simulation run ID to filter by. If None, uses all runs.
    """
    import json

    # Find the closest timestamp <= target_time
    try:
        if run_id is not None:
            closest_ts = fetch_query(
                "SELECT MAX(timestamp) as ts FROM position_history WHERE timestamp <= ? AND run_id = ?",
                (target_time, run_id)
            )
        else:
            closest_ts = fetch_query(
                "SELECT MAX(timestamp) as ts FROM position_history WHERE timestamp <= ?",
                (target_time,)
            )
    except Exception:
        # position_history table may not exist
        return {"spmts": [], "cranes": [], "blocks": [], "ships": [], "queue_depths": {}}

    if not closest_ts or closest_ts[0]["ts"] is None:
        return {"spmts": [], "cranes": [], "blocks": [], "ships": [], "queue_depths": {}}

    actual_time = closest_ts[0]["ts"]

    if run_id is not None:
        rows = fetch_query(
            "SELECT * FROM position_history WHERE timestamp = ? AND run_id = ?",
            (actual_time, run_id)
        )
    else:
        rows = fetch_query(
            "SELECT * FROM position_history WHERE timestamp = ?",
            (actual_time,)
        )

    spmts = []
    cranes = []
    blocks = []
    ships = []

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
        elif r["entity_type"] == "ship":
            ships.append({
                "id": r["entity_id"],
                "status": r["status"],
                "current_location": r["location"],
                "hull_number": extra.get("hull_number", ""),
                "assigned_dock": extra.get("assigned_dock"),
                "assigned_quay": extra.get("assigned_quay"),
                "blocks_erected": extra.get("blocks_erected", 0),
                "total_blocks": extra.get("total_blocks", 200),
                "erection_progress": extra.get("erection_progress", 0),
                "completion_pct": extra.get("completion_pct", 0),
                "sea_position": extra.get("sea_position", 0.0),
                "stage_remaining_time": extra.get("stage_remaining_time", 0),
            })

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
        "ships": ships,
        "queue_depths": queue_depths,
        "actual_time": actual_time,
    }


def fetch_playback_timeline() -> Dict[str, Any]:
    """Get timeline info for playback controls (min/max timestamps, count)."""
    try:
        result = fetch_query(
            """SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts,
                      COUNT(DISTINCT timestamp) as snapshot_count
               FROM position_history"""
        )
    except Exception:
        # position_history table may not exist
        return {"min_time": 0, "max_time": 0, "snapshot_count": 0}

    if not result or result[0]["min_ts"] is None:
        return {"min_time": 0, "max_time": 0, "snapshot_count": 0}

    return {
        "min_time": result[0]["min_ts"],
        "max_time": result[0]["max_ts"],
        "snapshot_count": result[0]["snapshot_count"],
    }


def fetch_available_timestamps(run_id: int = None) -> List[float]:
    """Get all available snapshot timestamps for playback scrubbing."""
    try:
        if run_id is not None:
            rows = fetch_query(
                "SELECT DISTINCT timestamp FROM position_history WHERE run_id = ? ORDER BY timestamp",
                (run_id,)
            )
        else:
            rows = fetch_query(
                "SELECT DISTINCT timestamp FROM position_history ORDER BY timestamp"
            )
        return [r["timestamp"] for r in rows] if rows else []
    except Exception:
        # position_history table may not exist
        return []


# ============================================================================
# SIMULATION RUN MANAGEMENT
# ============================================================================

def create_simulation_run(name: str = None, config_path: str = None,
                          policy_type: str = None) -> int:
    """Create a new simulation run and return its ID."""
    conn = _get_conn()
    cur = conn.cursor()

    if name is None:
        # Generate a default name with timestamp
        import datetime
        name = f"Run {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

    cur.execute("""
        INSERT INTO simulation_runs (name, config_path, policy_type)
        VALUES (?, ?, ?)
    """, (name, config_path, policy_type))

    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def update_simulation_run(run_id: int, total_steps: int = None,
                          blocks_completed: int = None, ships_delivered: int = None,
                          total_reward: float = None) -> None:
    """Update a simulation run with final statistics."""
    conn = _get_conn()
    cur = conn.cursor()

    updates = []
    params = []
    if total_steps is not None:
        updates.append("total_steps = ?")
        params.append(total_steps)
    if blocks_completed is not None:
        updates.append("blocks_completed = ?")
        params.append(blocks_completed)
    if ships_delivered is not None:
        updates.append("ships_delivered = ?")
        params.append(ships_delivered)
    if total_reward is not None:
        updates.append("total_reward = ?")
        params.append(total_reward)

    updates.append("ended_at = CURRENT_TIMESTAMP")
    params.append(run_id)

    if updates:
        cur.execute(f"""
            UPDATE simulation_runs
            SET {', '.join(updates)}
            WHERE id = ?
        """, params)

    conn.commit()
    conn.close()


def list_simulation_runs() -> List[Dict[str, Any]]:
    """List all simulation runs for the run selector dropdown."""
    return fetch_query("""
        SELECT id, name, config_path, policy_type, started_at, ended_at,
               total_steps, blocks_completed, ships_delivered, total_reward
        FROM simulation_runs
        ORDER BY started_at DESC
        LIMIT 50
    """) or []


def fetch_playback_timeline_for_run(run_id: int = None) -> Dict[str, Any]:
    """Get timeline info for a specific run (or all runs if None)."""
    try:
        if run_id is not None:
            result = fetch_query(
                """SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts,
                          COUNT(DISTINCT timestamp) as snapshot_count
                   FROM position_history WHERE run_id = ?""",
                (run_id,)
            )
        else:
            result = fetch_query(
                """SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts,
                          COUNT(DISTINCT timestamp) as snapshot_count
                   FROM position_history"""
            )
    except Exception:
        return {"min_time": 0, "max_time": 0, "snapshot_count": 0}

    if not result or result[0]["min_ts"] is None:
        return {"min_time": 0, "max_time": 0, "snapshot_count": 0}

    return {
        "min_time": result[0]["min_ts"],
        "max_time": result[0]["max_ts"],
        "snapshot_count": result[0]["snapshot_count"],
    }


def fetch_plate_stats() -> List[Dict]:
    """Fetch plate processing statistics for dashboard visualization."""
    return fetch_query("""
        SELECT b.id, b.ship_id, b.block_type, b.weight, b.current_stage,
               b.n_plates, b.plate_area_m2, b.processing_source,
               b.status, b.completion_pct
        FROM hhi_blocks b
        WHERE b.n_plates > 0
        ORDER BY b.n_plates DESC
    """) or []


def fetch_plate_processing_times() -> List[Dict]:
    """Fetch plate processing time records for scatter/comparison charts."""
    return fetch_query("""
        SELECT block_id, stage, n_plates, processing_time_hours, method
        FROM plate_processing
        ORDER BY timestamp DESC
        LIMIT 5000
    """) or []


def log_plate_processing(block_id: str, timestamp: float, stage: str,
                         n_plates: int, processing_time: float, method: str) -> None:
    """Log a plate processing time observation."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO plate_processing (block_id, timestamp, stage, n_plates,
               processing_time_hours, method) VALUES (?, ?, ?, ?, ?, ?)""",
            (block_id, timestamp, stage, n_plates, processing_time, method),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

"""Metrics and KPI computation utilities.

This module defines functions to compute key performance indicators (KPIs)
from simulation logs or environment state. Metrics include tardiness,
unplanned breakdowns, planned maintenance events, throughput, OEE,
utilization, schedule variance, and cost.
"""

from __future__ import annotations

from typing import Dict, Any, List
import numpy as np


def compute_kpis(
    metrics: Dict[str, Any],
    total_time: float,
    n_spmts: int | None = None,
    n_cranes: int | None = None,
) -> Dict[str, float]:
    """Compute high-level KPIs given raw metrics and total simulation time.

    Args:
        metrics: Dictionary of raw metrics from simulation.
        total_time: Total simulation time elapsed.
        n_spmts: Number of SPMTs (for utilization calculation). If None, uses metrics dict.
        n_cranes: Number of cranes (for utilization calculation). If None, uses metrics dict.
    """
    kpis = {}
    blocks_completed = metrics.get("blocks_completed", 0)
    total_tardiness = metrics.get("total_tardiness", 0.0)
    breakdowns = metrics.get("breakdowns", 0)
    planned = metrics.get("planned_maintenance", 0)

    # Throughput rate: blocks per unit time
    kpis["throughput"] = blocks_completed / total_time if total_time > 0 else 0.0
    kpis["average_tardiness"] = total_tardiness / blocks_completed if blocks_completed > 0 else 0.0
    kpis["unplanned_breakdown_rate"] = breakdowns / total_time if total_time > 0 else 0.0
    kpis["planned_maintenance_rate"] = planned / total_time if total_time > 0 else 0.0

    # SPMT utilization: fraction of time SPMTs were busy
    spmt_busy = metrics.get("spmt_busy_time", 0.0)
    _n_spmts = n_spmts if n_spmts is not None else metrics.get("n_spmts", 1)
    kpis["spmt_utilization"] = spmt_busy / (_n_spmts * total_time) if total_time > 0 and _n_spmts > 0 else 0.0

    # Crane utilization
    crane_busy = metrics.get("crane_busy_time", 0.0)
    _n_cranes = n_cranes if n_cranes is not None else metrics.get("n_cranes", 1)
    kpis["crane_utilization"] = crane_busy / (_n_cranes * total_time) if total_time > 0 and _n_cranes > 0 else 0.0

    # OEE = Availability * Performance * Quality
    availability = 1.0 - (breakdowns / total_time if total_time > 0 else 0.0)
    performance = kpis["spmt_utilization"]
    quality = 1.0  # assume no rework
    kpis["oee"] = max(0.0, availability * performance * quality)

    # Schedule variance: std dev of (completion_time - due_date) across completed blocks
    completion_deltas = metrics.get("completion_deltas", [])
    if completion_deltas:
        kpis["schedule_variance"] = float(np.std(completion_deltas))
    else:
        kpis["schedule_variance"] = 0.0

    # Cost: weighted sum of tardiness penalty, breakdown cost, maintenance cost, empty travel
    w_tard = metrics.get("cost_weight_tardiness", 10.0)
    w_break = metrics.get("cost_weight_breakdown", 50.0)
    w_maint = metrics.get("cost_weight_maintenance", 5.0)
    w_empty = metrics.get("cost_weight_empty_travel", 1.0)
    empty_travel = metrics.get("empty_travel_time", 0.0)
    kpis["total_cost"] = (
        w_tard * total_tardiness
        + w_break * breakdowns
        + w_maint * planned
        + w_empty * empty_travel
    )

    return kpis


def compute_episode_summary(episode_kpis: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean KPIs over multiple episodes."""
    if not episode_kpis:
        return {}
    keys = episode_kpis[0].keys()
    return {k: float(np.mean([ep.get(k, 0.0) for ep in episode_kpis])) for k in keys}

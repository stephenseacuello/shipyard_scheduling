"""Visualization utilities for simulation results.

Provides Gantt charts, health trend plots, KPI trend charts,
and utilization heatmaps using matplotlib.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Sequence


def plot_gantt(block_histories: Dict[str, List[Dict[str, Any]]], filename: str | None = None) -> None:
    """Plot a Gantt chart from block history logs."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(block_histories) * 0.4)))
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    y_pos = 0
    for block_id, events in block_histories.items():
        events_sorted = sorted(events, key=lambda e: e["time"])
        for i, event in enumerate(events_sorted):
            duration = event.get("duration", 1.0)
            color = colors[i % len(colors)]
            ax.barh(y_pos, duration, left=event["time"], height=0.6, color=color,
                    edgecolor="gray", linewidth=0.5)
        y_pos += 1
    ax.set_yticks(range(len(block_histories)))
    ax.set_yticklabels(list(block_histories.keys()), fontsize=8)
    ax.set_xlabel("Time")
    ax.set_title("Block Processing Timeline")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


def plot_health_trends(
    health_records: Dict[str, List[float]],
    timesteps: Sequence[float] | None = None,
    failure_threshold: float = 20.0,
    filename: str | None = None,
) -> None:
    """Plot health indicator time-series for multiple equipment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for equip_id, values in health_records.items():
        t = timesteps if timesteps is not None else list(range(len(values)))
        ax.plot(t, values, label=equip_id, linewidth=1.5)
    ax.axhline(y=failure_threshold, color="red", linestyle="--", linewidth=1, label="Failure threshold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Health")
    ax.set_title("Equipment Health Trends")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


def plot_kpi_trends(
    kpi_history: List[Dict[str, float]],
    keys: List[str] | None = None,
    filename: str | None = None,
) -> None:
    """Plot KPI values over training epochs."""
    if not kpi_history:
        return
    if keys is None:
        keys = list(kpi_history[0].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = list(range(1, len(kpi_history) + 1))
    for key in keys:
        values = [ep.get(key, 0.0) for ep in kpi_history]
        ax.plot(epochs, values, marker="o", markersize=3, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("KPI Trends")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


def plot_utilization_heatmap(
    utilization_matrix: np.ndarray,
    equipment_ids: List[str],
    time_labels: List[str] | None = None,
    filename: str | None = None,
) -> None:
    """Plot a utilization heatmap (equipment x time)."""
    fig, ax = plt.subplots(figsize=(12, max(3, len(equipment_ids) * 0.5)))
    im = ax.imshow(utilization_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(equipment_ids)))
    ax.set_yticklabels(equipment_ids, fontsize=8)
    if time_labels:
        step = max(1, len(time_labels) // 10)
        ax.set_xticks(range(0, len(time_labels), step))
        ax.set_xticklabels(time_labels[::step], fontsize=7, rotation=45)
    ax.set_xlabel("Time Period")
    ax.set_title("Equipment Utilization Heatmap")
    plt.colorbar(im, ax=ax, label="Utilization")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()

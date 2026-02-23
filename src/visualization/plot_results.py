"""Generate publication-quality figures from training results.

This module provides functions to create high-quality visualizations of
training metrics, baseline comparisons, and multi-objective Pareto fronts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    if SEABORN_AVAILABLE:
        sns.set_palette("colorblind")


def plot_training_curves(
    metrics_file: Union[str, Path],
    output_dir: Union[str, Path],
    smooth_window: int = 5,
) -> None:
    """Plot training loss, entropy, and KPIs.

    Creates a 2x2 grid showing:
    - Policy/Value Loss over epochs
    - Entropy trajectory
    - Throughput
    - On-time delivery rate

    Args:
        metrics_file: Path to training_metrics.csv
        output_dir: Directory to save figures
        smooth_window: Rolling window for smoothing curves
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot_training_curves")
        return

    setup_style()
    df = pd.read_csv(metrics_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Policy/Value Loss
    ax = axes[0, 0]
    if "policy_loss" in df.columns:
        ax.plot(df["epoch"], df["policy_loss"].rolling(smooth_window).mean(),
                label="Policy Loss", color="tab:blue", linewidth=2)
    if "value_loss" in df.columns:
        ax.plot(df["epoch"], df["value_loss"].rolling(smooth_window).mean(),
                label="Value Loss", color="tab:orange", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    # Entropy
    ax = axes[0, 1]
    if "entropy" in df.columns:
        ax.plot(df["epoch"], df["entropy"].rolling(smooth_window).mean(),
                color="tab:green", linewidth=2)
        ax.fill_between(df["epoch"], 0, df["entropy"].rolling(smooth_window).mean(),
                       alpha=0.2, color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy (Exploration)")

    # Throughput
    ax = axes[1, 0]
    if "throughput" in df.columns:
        ax.plot(df["epoch"], df["throughput"].rolling(smooth_window).mean(),
                color="tab:red", linewidth=2)
    if "val_throughput" in df.columns:
        ax.plot(df["epoch"], df["val_throughput"], "o-",
                label="Validation", color="tab:purple", markersize=4)
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Throughput (blocks/hour)")
    ax.set_title("Throughput")

    # On-time Rate
    ax = axes[1, 1]
    if "on_time_rate" in df.columns:
        ax.plot(df["epoch"], df["on_time_rate"].rolling(smooth_window).mean(),
                color="tab:cyan", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("On-Time Rate (%)")
    ax.set_title("On-Time Delivery Rate")

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png")
    plt.savefig(output_dir / "training_curves.pdf")
    plt.close()
    print(f"Saved training curves to {output_dir}")


def plot_baseline_comparison(
    results: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    metrics: Optional[List[str]] = None,
) -> None:
    """Compare RL agent vs OR baselines.

    Creates a grouped bar chart comparing multiple methods across
    specified metrics.

    Args:
        results: Dict mapping method names to DataFrames with results
        output_dir: Directory to save figures
        metrics: List of metric names to compare (default: throughput, tardiness)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot_baseline_comparison")
        return

    if metrics is None:
        metrics = ["throughput", "average_tardiness"]

    setup_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    n_metrics = len(metrics)
    x = np.arange(len(methods))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        means = []
        stds = []
        for method in methods:
            df = results[method]
            if metric in df.columns:
                means.append(df[metric].mean())
                stds.append(df[metric].std())
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=metric.replace("_", " ").title(),
                     yerr=stds, capsize=3)

    ax.set_xlabel("Method")
    ax.set_ylabel("Value")
    ax.set_title("Baseline Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "baseline_comparison.png")
    plt.savefig(output_dir / "baseline_comparison.pdf")
    plt.close()
    print(f"Saved baseline comparison to {output_dir}")


def plot_pareto_front(
    mo_results_file: Union[str, Path],
    output_dir: Union[str, Path],
    objectives: Optional[List[str]] = None,
) -> None:
    """Plot multi-objective Pareto front.

    Creates a 3D scatter plot showing the Pareto frontier across
    three objectives.

    Args:
        mo_results_file: Path to multi-objective results CSV
        output_dir: Directory to save figures
        objectives: List of 3 objective column names
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot_pareto_front")
        return

    if objectives is None:
        objectives = ["throughput", "average_tardiness", "total_cost"]

    setup_style()
    df = pd.read_csv(mo_results_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check we have all objectives
    missing = [obj for obj in objectives if obj not in df.columns]
    if missing:
        print(f"Missing objectives in data: {missing}")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color by hypervolume or epoch if available
    color_col = "hypervolume" if "hypervolume" in df.columns else "epoch"
    if color_col in df.columns:
        colors = df[color_col]
        cmap = "viridis"
    else:
        colors = "tab:blue"
        cmap = None

    scatter = ax.scatter(
        df[objectives[0]],
        df[objectives[1]],
        df[objectives[2]],
        c=colors,
        cmap=cmap,
        alpha=0.7,
        s=50,
    )

    if cmap:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label(color_col.replace("_", " ").title())

    ax.set_xlabel(objectives[0].replace("_", " ").title())
    ax.set_ylabel(objectives[1].replace("_", " ").title())
    ax.set_zlabel(objectives[2].replace("_", " ").title())
    ax.set_title("Multi-Objective Pareto Front")

    plt.tight_layout()
    plt.savefig(output_dir / "pareto_front.png")
    plt.savefig(output_dir / "pareto_front.pdf")
    plt.close()
    print(f"Saved Pareto front to {output_dir}")


def plot_block_stage_distribution(
    metrics_file: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """Plot heatmap of block stage distribution over training.

    Args:
        metrics_file: Path to metrics file with stage counts
        output_dir: Directory to save figures
    """
    if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
        print("matplotlib/seaborn not available, skipping plot_block_stage_distribution")
        return

    setup_style()
    df = pd.read_csv(metrics_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for stage distribution columns
    stage_cols = [col for col in df.columns if col.startswith("stage_")]
    if not stage_cols:
        print("No stage distribution columns found")
        return

    stage_data = df[stage_cols].values
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(stage_data.T, xticklabels=epochs[::10], yticklabels=stage_cols,
                cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Production Stage")
    ax.set_title("Block Stage Distribution Over Training")

    plt.tight_layout()
    plt.savefig(output_dir / "block_stage_heatmap.png")
    plt.savefig(output_dir / "block_stage_heatmap.pdf")
    plt.close()
    print(f"Saved block stage heatmap to {output_dir}")


def plot_equipment_health(
    metrics_file: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """Plot equipment health degradation over training.

    Args:
        metrics_file: Path to metrics file with health data
        output_dir: Directory to save figures
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot_equipment_health")
        return

    setup_style()
    df = pd.read_csv(metrics_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # SPMT health
    ax = axes[0]
    health_cols = [col for col in df.columns if "spmt" in col.lower() and "health" in col.lower()]
    for col in health_cols:
        ax.plot(df["epoch"], df[col], label=col, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Health (%)")
    ax.set_title("SPMT Health Over Training")
    if health_cols:
        ax.legend()

    # Crane health
    ax = axes[1]
    health_cols = [col for col in df.columns if "crane" in col.lower() and "health" in col.lower()]
    for col in health_cols:
        ax.plot(df["epoch"], df[col], label=col, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Health (%)")
    ax.set_title("Crane Health Over Training")
    if health_cols:
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "equipment_health.png")
    plt.savefig(output_dir / "equipment_health.pdf")
    plt.close()
    print(f"Saved equipment health plots to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training visualizations")
    parser.add_argument("--metrics", type=str, required=True, help="Path to training_metrics.csv")
    parser.add_argument("--output", type=str, default="figures", help="Output directory")
    args = parser.parse_args()

    plot_training_curves(args.metrics, args.output)

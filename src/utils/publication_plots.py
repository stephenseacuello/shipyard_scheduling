"""Publication-quality visualization functions.

This module provides functions for generating figures suitable for
academic publications, including learning curves with confidence bands,
baseline comparisons, ablation heatmaps, and schedule visualizations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (6.5, 4.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette (colorblind-friendly)
COLORS = {
    "gnn_ppo": "#2166AC",      # Blue
    "rule_based": "#B2182B",   # Red
    "myopic": "#762A83",       # Purple
    "siloed": "#1B7837",       # Green
    "baseline": "#636363",     # Gray
}


def plot_learning_curves(
    results: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """Plot learning curves with confidence bands.

    Args:
        results: Dictionary mapping method names to {metric: 2D array [seeds, epochs]}.
        output_path: Path to save the figure.
        metrics: List of metrics to plot. Default: reward, policy_loss, value_loss.
        figsize: Figure size.
    """
    if metrics is None:
        metrics = ["episode_reward", "policy_loss", "value_loss", "entropy"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break
        ax = axes[idx]

        for method_name, method_data in results.items():
            if metric not in method_data:
                continue

            data = method_data[metric]  # Shape: [n_seeds, n_epochs]
            epochs = np.arange(1, data.shape[1] + 1)

            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            color = COLORS.get(method_name, "#333333")
            ax.plot(epochs, mean, label=method_name.replace("_", " ").title(),
                    color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        ax.set_title(f"{metric.replace('_', ' ').title()} over Training")

    # Remove extra axes
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_baseline_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """Bar chart comparing methods with error bars.

    Args:
        results: Dictionary mapping method names to {metric: {mean, std, ...}}.
        output_path: Path to save the figure.
        metrics: List of metrics to plot.
        figsize: Figure size.
    """
    if metrics is None:
        metrics = ["throughput", "average_tardiness", "breakdowns", "oee"]

    methods = list(results.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_methods)
    width = 0.6

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        means = []
        stds = []
        colors = []

        for method in methods:
            if metric in results[method]:
                means.append(results[method][metric]["mean"])
                stds.append(results[method][metric]["std"])
            else:
                means.append(0)
                stds.append(0)
            colors.append(COLORS.get(method, "#333333"))

        bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.5)

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in methods],
                          rotation=45, ha="right")

        # Add significance markers if available
        for i, method in enumerate(methods):
            if metric in results[method] and "marker" in results[method][metric]:
                marker = results[method][metric]["marker"]
                if marker:
                    ax.annotate(marker, xy=(i, means[i] + stds[i]),
                                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_ablation_heatmap(
    ablation_results: Dict[str, Dict[str, float]],
    output_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> None:
    """Heatmap showing impact of ablating each component.

    Args:
        ablation_results: Dictionary mapping config names to {metric: value}.
        output_path: Path to save the figure.
        metrics: List of metrics to show.
        figsize: Figure size.
    """
    if metrics is None:
        metrics = ["throughput", "average_tardiness", "breakdowns", "oee"]

    configs = list(ablation_results.keys())
    n_configs = len(configs)
    n_metrics = len(metrics)

    # Build data matrix
    data = np.zeros((n_configs, n_metrics))
    for i, config in enumerate(configs):
        for j, metric in enumerate(metrics):
            data[i, j] = ablation_results[config].get(metric, 0)

    # Normalize to show % change from full model (assume first row is full)
    if n_configs > 0:
        baseline = data[0, :].copy()
        baseline[baseline == 0] = 1e-10  # Avoid division by zero
        data_pct = (data - baseline) / np.abs(baseline) * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Custom diverging colormap (red = worse, green = better)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    im = ax.imshow(data_pct, cmap=cmap, aspect="auto", vmin=-50, vmax=50)

    # Labels
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_configs))
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_yticklabels(configs)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(n_configs):
        for j in range(n_metrics):
            val = data_pct[i, j]
            color = "white" if abs(val) > 25 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    color=color, fontsize=9)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("% Change from Full Model", rotation=-90, va="bottom")

    ax.set_title("Ablation Study: Impact of Removing Components")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_schedule_gantt(
    block_events: List[Dict[str, Any]],
    output_path: str,
    max_blocks: int = 30,
    figsize: Tuple[float, float] = (12, 8),
) -> None:
    """Gantt chart showing block progression through stages.

    Args:
        block_events: List of event dicts with block_id, stage, start_time, end_time.
        output_path: Path to save the figure.
        max_blocks: Maximum number of blocks to display.
        figsize: Figure size.
    """
    # Stage colors
    stage_colors = {
        "CUTTING": "#E41A1C",
        "PANEL": "#377EB8",
        "ASSEMBLY": "#4DAF4A",
        "OUTFITTING": "#984EA3",
        "PAINTING": "#FF7F00",
        "PRE_ERECTION": "#FFFF33",
        "DOCK": "#A65628",
        "TRANSPORT": "#F781BF",
    }

    # Group events by block
    blocks = {}
    for event in block_events:
        block_id = event.get("block_id")
        if block_id not in blocks:
            blocks[block_id] = []
        blocks[block_id].append(event)

    # Limit number of blocks
    block_ids = list(blocks.keys())[:max_blocks]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, block_id in enumerate(block_ids):
        events = blocks[block_id]
        for event in events:
            stage = event.get("stage", "UNKNOWN")
            start = event.get("start_time", 0)
            end = event.get("end_time", start + 1)
            duration = end - start

            color = stage_colors.get(stage, "#999999")
            ax.barh(idx, duration, left=start, height=0.8,
                    color=color, edgecolor="black", linewidth=0.5)

    ax.set_yticks(range(len(block_ids)))
    ax.set_yticklabels([f"Block {bid}" for bid in block_ids])
    ax.set_xlabel("Time")
    ax.set_ylabel("Block")
    ax.set_title("Block Schedule Gantt Chart")

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=s)
                      for s, c in stage_colors.items()]
    ax.legend(handles=legend_patches, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_health_trajectories(
    health_data: Dict[str, np.ndarray],
    output_path: str,
    failure_threshold: float = 20.0,
    pm_threshold: float = 40.0,
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """Plot equipment health degradation over time.

    Args:
        health_data: Dictionary mapping equipment_id to health array over time.
        output_path: Path to save the figure.
        failure_threshold: Health level below which equipment fails.
        pm_threshold: Health level triggering preventive maintenance.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.tab10
    for idx, (equip_id, health) in enumerate(health_data.items()):
        time = np.arange(len(health))
        color = cmap(idx % 10)
        ax.plot(time, health, label=f"Equipment {equip_id}",
                color=color, linewidth=1.5, alpha=0.8)

    # Threshold lines
    ax.axhline(y=failure_threshold, color="red", linestyle="--",
               linewidth=2, label=f"Failure Threshold ({failure_threshold})")
    ax.axhline(y=pm_threshold, color="orange", linestyle="--",
               linewidth=2, label=f"PM Threshold ({pm_threshold})")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Health Level")
    ax.set_title("Equipment Health Degradation Trajectories")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reward_distribution(
    rewards: Dict[str, np.ndarray],
    output_path: str,
    figsize: Tuple[float, float] = (8, 6),
) -> None:
    """Violin/box plot comparing episode reward distributions.

    Args:
        rewards: Dictionary mapping method names to reward arrays.
        output_path: Path to save the figure.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(rewards.keys())
    data = [rewards[m] for m in methods]
    colors = [COLORS.get(m, "#333333") for m in methods]

    parts = ax.violinplot(data, positions=range(len(methods)), showmeans=True)

    # Color the violins
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[idx])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace("_", " ").title() for m in methods])
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward Distribution by Method")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_utilization_heatmap(
    utilization_data: np.ndarray,
    equipment_names: List[str],
    time_labels: List[str],
    output_path: str,
    figsize: Tuple[float, float] = (12, 6),
) -> None:
    """Heatmap of equipment utilization over time.

    Args:
        utilization_data: 2D array [n_equipment, n_time_bins].
        equipment_names: List of equipment names.
        time_labels: List of time bin labels.
        output_path: Path to save the figure.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    im = ax.imshow(utilization_data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_yticks(np.arange(len(equipment_names)))
    ax.set_xticklabels(time_labels, rotation=45, ha="right")
    ax.set_yticklabels(equipment_names)

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Equipment")
    ax.set_title("Equipment Utilization Heatmap")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Utilization Rate", rotation=-90, va="bottom")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_kpi_trends(
    kpi_data: Dict[str, np.ndarray],
    output_path: str,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """Plot KPI trends over episodes with dual y-axes.

    Args:
        kpi_data: Dictionary mapping KPI names to time series arrays.
        output_path: Path to save the figure.
        figsize: Figure size.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    kpis = list(kpi_data.keys())[:4]

    for idx, kpi in enumerate(kpis):
        ax = axes[idx]
        data = kpi_data[kpi]
        episodes = np.arange(1, len(data) + 1)

        ax.plot(episodes, data, linewidth=2, color=COLORS["gnn_ppo"])
        ax.fill_between(episodes, 0, data, alpha=0.3, color=COLORS["gnn_ppo"])

        ax.set_xlabel("Episode")
        ax.set_ylabel(kpi.replace("_", " ").title())
        ax.set_title(kpi.replace("_", " ").title())

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_convergence_comparison(
    results: Dict[str, np.ndarray],
    output_path: str,
    metric: str = "episode_reward",
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """Compare convergence speed across methods.

    Args:
        results: Dictionary mapping method names to metric arrays [seeds, epochs].
        output_path: Path to save the figure.
        metric: Metric name for title/labels.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method_name, data in results.items():
        epochs = np.arange(1, data.shape[1] + 1)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        color = COLORS.get(method_name, "#333333")
        ax.plot(epochs, mean, label=method_name.replace("_", " ").title(),
                color=color, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Convergence Comparison: {metric.replace('_', ' ').title()}")
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_all_figures(
    results_dir: str,
    figures_dir: str,
) -> None:
    """Generate all publication figures from results directory.

    Args:
        results_dir: Directory containing result JSON files.
        figures_dir: Directory to save generated figures.
    """
    import os
    import json

    os.makedirs(figures_dir, exist_ok=True)

    # Load results if available
    # This is a placeholder - actual implementation would parse result files

    print(f"Generating figures to {figures_dir}/")
    print("  - learning_curves.pdf")
    print("  - baseline_comparison.pdf")
    print("  - ablation_heatmap.pdf")
    print("  - health_trajectories.pdf")
    print("  - reward_distribution.pdf")
    print("  - utilization_heatmap.pdf")
    print("\nNote: Run experiments first to populate results.")

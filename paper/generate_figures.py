"""Generate paper-quality figures for ISE 572 shipyard scheduling paper.

Generates:
1. HHI Ulsan shipyard layout with production flow
2. Curriculum DAgger learning curve
3. Cross-config comparison bar chart
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Publication settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def generate_shipyard_layout():
    """Generate HHI Ulsan shipyard layout figure with real dry dock specs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

    # Background
    ax.set_facecolor("#f0f4f8")
    ax.set_xlim(-20, 1220)
    ax.set_ylim(680, -30)

    # Water area (Mipo Bay) - right side
    water = mpatches.FancyBboxPatch(
        (1100, -30), 120, 710, boxstyle="round,pad=5",
        facecolor="#d4e6f1", edgecolor="#85c1e9", linewidth=1, alpha=0.5
    )
    ax.add_patch(water)
    ax.text(1160, 340, "MIPO\nBAY", ha="center", va="center",
            fontsize=8, color="#2980b9", fontstyle="italic", fontweight="bold")

    # Zone backgrounds
    zones = [
        (30, 20, 190, 200, "#eaf2f8", "ZONE 1\nSteel Processing"),
        (240, 20, 200, 280, "#e8f8f5", "ZONE 2\nPanel Assembly"),
        (460, 20, 220, 440, "#f5eef8", "ZONE 3\nBlock Assembly\n& Outfitting"),
        (700, 60, 170, 340, "#fef9e7", "ZONE 4\nPre-Erection"),
        (880, -10, 220, 650, "#fdedec", "ZONE 5\nDry Docks"),
    ]
    for x, y, w, h, color, label in zones:
        zone_bg = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=3",
            facecolor=color, edgecolor="#bdc3c7", linewidth=0.5, linestyle="--"
        )
        ax.add_patch(zone_bg)
        ax.text(x + 5, y + 8, label, fontsize=6, color="#7f8c8d",
                va="top", fontstyle="italic")

    # Facilities
    facilities = [
        # Zone 1: Steel Processing
        (40, 50, 140, 50, "Steel Stockyard\n(1,780 acres total)", "#7f8c8d"),
        (40, 115, 80, 40, "Cutting Shop\n(NC Plasma)", "#3498db"),
        (135, 115, 80, 40, "Part\nFabrication", "#3498db"),

        # Zone 2: Panel Assembly
        (260, 40, 160, 45, "Flat Panel Line 1", "#27ae60"),
        (260, 100, 160, 45, "Flat Panel Line 2", "#27ae60"),
        (260, 165, 160, 50, "Curved Block Shop", "#e67e22"),

        # Zone 3: Block Assembly
        (480, 40, 180, 60, "Block Assembly Hall 1", "#9b59b6"),
        (480, 115, 180, 60, "Block Assembly Hall 2", "#9b59b6"),
        (480, 190, 180, 60, "Block Assembly Hall 3", "#9b59b6"),
        (480, 270, 180, 55, "Outfitting Shop", "#1abc9c"),
        (480, 340, 180, 50, "Paint Shop\n(eco-friendly)", "#f39c12"),

        # Zone 4: Pre-Erection
        (720, 80, 130, 100, "Grand Block\nStaging (N)", "#34495e"),
        (720, 230, 130, 100, "Grand Block\nStaging (S)", "#34495e"),
    ]

    for x, y, w, h, label, color in facilities:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=2",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=6, color="white", fontweight="bold")

    # Dry Docks with real dimensions
    docks = [
        (895, 10, 130, 48, "Dock 1\n390m × 80m\nLNG", "#e74c3c", "GC01, GC02"),
        (895, 66, 155, 50, "Dock 2\n500m × 80m", "#e74c3c", "GC03"),
        (895, 124, 195, 62, "Dock 3 (MEGA)\n672m × 92m\n1M DWT", "#c0392b", "GC04, GC05"),
        (895, 194, 130, 44, "Dock 4\n390m 150kDWT", "#e74c3c", "GC06"),
        (895, 246, 105, 38, "Dock 5\n300m 70kDWT", "#c0392b", "GC07"),
        (895, 292, 95, 34, "Dock 6: 280m\nNaval", "#95a5a6", ""),
        (895, 334, 90, 32, "Dock 7: 260m\nNaval", "#95a5a6", ""),
        (895, 374, 115, 42, "Dock 8\n350m VLCC", "#922b21", "GC08"),
        (895, 424, 108, 38, "Dock 9\n320m VLCC", "#922b21", ""),
        (895, 470, 155, 52, "H-Dock\n490m × 115m\nOffshore", "#8e44ad", "GC09"),
    ]

    for x, y, w, h, label, color, cranes in docks:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=2",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=5, color="white", fontweight="bold")
        if cranes:
            ax.text(x + w + 3, y + h/2, cranes, fontsize=4.5,
                    color="#e74c3c", va="center", fontstyle="italic")

    # Outfitting Quays
    quays = [
        (895, 540, 120, 35, "Outfitting Quay 1", "#1abc9c"),
        (895, 583, 110, 33, "Outfitting Quay 2", "#1abc9c"),
        (895, 624, 100, 30, "Outfitting Quay 3", "#1abc9c"),
    ]
    for x, y, w, h, label, color in quays:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=2",
            facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold")

    # Production flow arrows
    arrow_style = dict(arrowstyle="-|>", color="#2c3e50", lw=0.8, mutation_scale=8)
    flow_arrows = [
        # Steel flow
        ((110, 90), (110, 115)),   # Stockyard → Cutting
        ((120, 115), (135, 130)),  # Cutting → Part Fab
        ((120, 90), (260, 60)),    # Stockyard → Panel 1
        ((120, 100), (260, 120)),  # Stockyard → Panel 2
        ((215, 135), (260, 190)),  # Part Fab → Curved
        # Panel → Block
        ((420, 60), (480, 65)),
        ((420, 120), (480, 140)),
        ((420, 190), (480, 215)),
        # Block → Outfitting → Paint
        ((570, 250), (570, 270)),
        ((570, 325), (570, 340)),
        # Paint → Pre-Erection
        ((660, 360), (720, 125)),
        ((660, 370), (720, 275)),
        # Pre-Erection → Docks
        ((850, 130), (895, 130)),
        ((850, 280), (895, 350)),
    ]
    for start, end in flow_arrows:
        arrow = FancyArrowPatch(start, end, **arrow_style)
        ax.add_patch(arrow)

    # SPMT Depot
    spmt_rect = mpatches.FancyBboxPatch(
        (40, 400), 120, 45, boxstyle="round,pad=2",
        facecolor="#2c3e50", edgecolor="white", linewidth=1, alpha=0.8
    )
    ax.add_patch(spmt_rect)
    ax.text(100, 422, "SPMT Depot\n24 SPMTs + 8 DCTs", ha="center", va="center",
            fontsize=5.5, color="white", fontweight="bold")

    # Sea Trials
    ax.text(1160, 660, "→ Sea Trials\n(Ulsan Bay)", ha="center",
            fontsize=7, color="#2980b9", fontstyle="italic")

    # Title
    ax.set_title("HD Hyundai Heavy Industries — Ulsan Shipyard Layout\n"
                 "World's Largest Shipyard · 4km Coastline · 1,780 Acres · 10 Dry Docks · 9 Goliath Cranes",
                 fontsize=11, fontweight="bold", pad=10)

    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#3498db", label="Steel Processing"),
        mpatches.Patch(facecolor="#27ae60", label="Panel Assembly"),
        mpatches.Patch(facecolor="#9b59b6", label="Block Assembly"),
        mpatches.Patch(facecolor="#1abc9c", label="Outfitting"),
        mpatches.Patch(facecolor="#f39c12", label="Paint Shop"),
        mpatches.Patch(facecolor="#e74c3c", label="Dry Docks"),
        mpatches.Patch(facecolor="#8e44ad", label="H-Dock (Offshore)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", ncol=4,
              frameon=True, fancybox=True, fontsize=6, framealpha=0.9)

    fig.savefig(os.path.join(FIGURE_DIR, "shipyard_layout.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, "shipyard_layout.png"), dpi=300)
    plt.close(fig)
    print("Generated: shipyard_layout.pdf/png")


def generate_curriculum_learning_curve():
    """Generate curriculum DAgger learning curve from training results."""
    # Data from actual training run (19.8 hours)
    stages = ["Tiny\n(10 blocks)", "Small\n(50 blocks)", "Medium\n(200 blocks)"]
    expert_throughput = [0.0202, 0.1000, 0.1123]

    # Per-iteration data
    tiny_iters = list(range(11))  # BC + 10 DAgger
    tiny_throughput = [0.0206, 0.0212, 0.0207, 0.0211, 0.0207, 0.0203, 0.0202, 0.0200, 0.0205, 0.0200, 0.0204]
    tiny_loss = [0.0425, 0.0073, 0.0101, 0.0048, 0.0069, 0.0113, 0.0058, 0.0100, 0.0093, 0.0045, 0.0064]

    small_iters = list(range(11))
    small_throughput = [0.1000] * 11  # All 100%
    small_loss = [0.0972, 0.0630, 0.0358, 0.0214, 0.0207, 0.0183, 0.0234, 0.0249, 0.0196, 0.0204, 0.0234]

    medium_iters = list(range(11))
    medium_throughput = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0003, 0.0100, 0.0140, 0.0393, 0.0405, 0.0385]
    medium_loss = [1.7887, 2.0864, 2.2062, 2.2832, 2.3411, 2.4232, 2.4959, 2.5677, 2.6440, 2.7062, 2.7722]

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex="col")

    # Top row: throughput
    colors = ["#3498db", "#27ae60", "#e74c3c"]
    data_t = [(tiny_iters, tiny_throughput, 0.0202), (small_iters, small_throughput, 0.1000), (medium_iters, medium_throughput, 0.1123)]
    data_l = [(tiny_iters, tiny_loss), (small_iters, small_loss), (medium_iters, medium_loss)]

    for i, (iters, throughput, expert) in enumerate(data_t):
        ax = axes[0, i]
        ax.plot(iters, throughput, "o-", color=colors[i], markersize=4, linewidth=1.5, label="DAgger")
        ax.axhline(y=expert, color="gray", linestyle="--", linewidth=1, label="Expert")
        ax.set_title(stages[i], fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Throughput")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Bottom row: loss
    for i, (iters, loss) in enumerate(data_l):
        ax = axes[1, i]
        ax.plot(iters, loss, "s-", color=colors[i], markersize=4, linewidth=1.5)
        ax.set_xlabel("Iteration (0=BC)")
        if i == 0:
            ax.set_ylabel("Training Loss")
        ax.grid(True, alpha=0.3)

        # Annotate trend
        if loss[-1] < loss[0]:
            ax.annotate("converging ↓", xy=(10, loss[-1]), fontsize=6,
                        color="green", ha="right")
        else:
            ax.annotate("diverging ↑", xy=(10, loss[-1]), fontsize=6,
                        color="red", ha="right")

    fig.suptitle("Curriculum DAgger: 3-Stage Training (19.8h, Apple M1 Pro CPU)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURE_DIR, "curriculum_dagger.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, "curriculum_dagger.png"), dpi=300)
    plt.close(fig)
    print("Generated: curriculum_dagger.pdf/png")


def generate_cross_config_comparison():
    """Generate cross-config agent comparison bar chart with DAgger."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    agents = ["Expert\n(EDD)", "MPC\n(CP-SAT)", "GA", "DAgger"]
    colors = ["#27ae60", "#3498db", "#e67e22", "#9b59b6"]

    # Small instance (DAgger: 0.0623 mean, exceeds expert)
    small_throughput = [0.0528, 0.0551, 0.0576, 0.0623]
    small_ci = [0.0021, 0.0026, 0.0012, 0.0021]
    small_blocks = [50, 50, 50, 50]

    ax = axes[0]
    bars = ax.bar(agents, small_throughput, color=colors, alpha=0.85,
                  yerr=small_ci, capsize=4, edgecolor="white", linewidth=1)
    ax.set_title("Small Instance (50 blocks)", fontweight="bold")
    ax.set_ylabel("Throughput")
    for bar, b in zip(bars, small_blocks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{b} blocks", ha="center", fontsize=7, color="#555")
    ax.set_ylim(0, 0.08)
    ax.grid(True, axis="y", alpha=0.3)

    # Medium instance (DAgger: 0.0 — normalizer mismatch)
    medium_throughput = [0.1108, 0.0238, 0.0120, 0.0000]
    medium_ci = [0.0016, 0.0016, 0.0065, 0.0000]
    medium_blocks = [110.8, 23.8, 12.0, 0.0]

    ax = axes[1]
    bars = ax.bar(agents, medium_throughput, color=colors, alpha=0.85,
                  yerr=medium_ci, capsize=4, edgecolor="white", linewidth=1)
    ax.set_title("Medium HHI (200 blocks)", fontweight="bold")
    ax.set_ylabel("Throughput")
    for bar, b in zip(bars, medium_blocks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{b:.0f} blocks", ha="center", fontsize=7, color="#555")
    ax.set_ylim(0, 0.14)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cross-Instance Statistical Comparison (5 seeds, Mann-Whitney U)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIGURE_DIR, "cross_config_comparison.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, "cross_config_comparison.png"), dpi=300)
    plt.close(fig)
    print("Generated: cross_config_comparison.pdf/png")


def generate_calibration_scatter():
    """Generate calibration R-squared comparison figure."""
    stages = ["Block\nAssembly", "Steel\nCutting", "Part\nFab", "Panel\nAssembly",
              "Painting", "Block\nOutfit", "Pre-\nErection"]
    r2_before = [-0.03, 0.40, 0.39, 0.81, 0.88, 0.58, 0.42]
    r2_after = [0.985, 0.937, 0.964, 0.945, 0.916, 0.510, 0.369]

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2_before, width, label="Before Fix (p0 bug)",
                   color="#e74c3c", alpha=0.7, edgecolor="white")
    bars2 = ax.bar(x + width/2, r2_after, width, label="After Fix (6-param + Ridge)",
                   color="#27ae60", alpha=0.85, edgecolor="white")

    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(6.5, 0.92, "R²=0.90", fontsize=6, color="gray")
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_ylabel("R² (coefficient of determination)")
    ax.set_title("Calibration Coefficient Fitting: Before vs After p0 Fix", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(-0.15, 1.1)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "calibration_r2.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, "calibration_r2.png"), dpi=300)
    plt.close(fig)
    print("Generated: calibration_r2.pdf/png")


def generate_entropy_collapse():
    """Generate entropy collapse figure across training epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    epochs = list(range(1, 21))

    # PPO entropy collapse
    ppo_entropy = [2.27, 1.50, 0.80, 0.30, 0.00] + [0.00] * 15
    sac_entropy = [0.91, 0.72, 0.58, 0.50, 0.45, 0.40, 0.36, 0.32, 0.29, 0.26,
                   0.24, 0.22, 0.21, 0.20, 0.19, 0.18, 0.18, 0.17, 0.17, 0.17]

    ax = axes[0]
    ax.plot(epochs, ppo_entropy, "o-", color="#e74c3c", markersize=3, linewidth=1.5, label="PPO")
    ax.plot(epochs, sac_entropy, "s-", color="#3498db", markersize=3, linewidth=1.5, label="SAC")
    ax.axhline(y=0.1, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Entropy Over Training", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.annotate("Complete collapse\n(epoch 5)", xy=(5, 0.0), xytext=(10, 0.8),
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                fontsize=7, color="#e74c3c")

    # Throughput comparison
    ppo_throughput = [0.040, 0.035, 0.028, 0.022, 0.019, 0.016, 0.014, 0.012,
                      0.010, 0.009, 0.008, 0.007, 0.006, 0.006, 0.005, 0.005,
                      0.004, 0.004, 0.004, 0.004]
    sac_throughput = [0.015, 0.018, 0.020, 0.022, 0.023, 0.023, 0.022, 0.022,
                      0.021, 0.021, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020,
                      0.020, 0.020, 0.020, 0.020]

    ax = axes[1]
    ax.plot(epochs, ppo_throughput, "o-", color="#e74c3c", markersize=3, linewidth=1.5, label="PPO (0.4%)")
    ax.plot(epochs, sac_throughput, "s-", color="#3498db", markersize=3, linewidth=1.5, label="SAC (28.7%)")
    ax.axhline(y=0.112, color="#27ae60", linestyle="--", linewidth=1.5, label="Expert (100%)")
    ax.axhline(y=0.112, color="#27ae60", linestyle="--", linewidth=1, alpha=0.3)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Throughput")
    ax.set_title("Throughput Over Training", fontweight="bold")
    ax.legend(fontsize=7, loc="center right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Entropy Collapse in Hierarchical Action Spaces", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIGURE_DIR, "entropy_collapse.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, "entropy_collapse.png"), dpi=300)
    plt.close(fig)
    print("Generated: entropy_collapse.pdf/png")


def generate_method_comparison():
    """Generate overall method comparison figure."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    methods = ["PPO", "SAC", "BC", "GAIL", "DAgger\n(direct)", "DAgger\n(curriculum)", "DAgger\n(deployed)"]
    vs_expert = [0.4, 28.7, 85.2, 78.4, 97.0, 100.0, 118.0]
    colors = ["#e74c3c", "#e67e22", "#9b59b6", "#8e44ad", "#3498db", "#27ae60", "#2ecc71"]

    bars = ax.barh(methods, vs_expert, color=colors, alpha=0.85, edgecolor="white", linewidth=1)
    ax.axvline(x=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("% of Expert Throughput")
    ax.set_title("Method Comparison on Small Instance (50 blocks)", fontweight="bold")
    ax.set_xlim(0, 115)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, vs_expert):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")

    # Category labels
    ax.text(2, -0.7, "RL Methods", fontsize=7, color="#e74c3c", fontstyle="italic")
    ax.text(60, 1.3, "Imitation Learning", fontsize=7, color="#3498db", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "method_comparison.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, "method_comparison.png"), dpi=300)
    plt.close(fig)
    print("Generated: method_comparison.pdf/png")


def generate_scaling_analysis():
    """Generate throughput scaling analysis across instance sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    configs = ["Tiny\n(10)", "Small\n(50)", "Medium\n(200)", "HHI Plate\n(1600)"]
    n_blocks = [10, 50, 200, 1600]

    expert_tp = [0.0200, 0.0528, 0.1108, 0.0562]
    mpc_tp = [0.0200, 0.0551, 0.0238, 0.0198]
    ga_tp = [0.0200, 0.0576, 0.0120, None]  # GA missing for HHI Plate

    # Left: throughput vs config
    ax = axes[0]
    x = np.arange(len(configs))
    w = 0.25
    ax.bar(x - w, expert_tp, w, color="#27ae60", alpha=0.85, label="Expert (EDD)", edgecolor="white")
    ax.bar(x, mpc_tp, w, color="#3498db", alpha=0.85, label="MPC (CP-SAT)", edgecolor="white")
    ga_vals = [v if v is not None else 0 for v in ga_tp]
    ga_bars = ax.bar(x + w, ga_vals, w, color="#e67e22", alpha=0.85, label="GA", edgecolor="white")
    # Hatch the missing GA bar
    ga_bars[3].set_alpha(0.15)
    ga_bars[3].set_hatch("//")

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Throughput (blocks/step)")
    ax.set_title("Throughput by Instance Size", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: Expert advantage ratio
    ax = axes[1]
    expert_advantage_mpc = [e/m for e, m in zip(expert_tp, mpc_tp)]
    ax.plot(configs, expert_advantage_mpc, "o-", color="#27ae60", markersize=6,
            linewidth=2, label="Expert / MPC")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Expert Advantage Ratio")
    ax.set_title("Expert Dominance at Scale", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(expert_advantage_mpc):
        ax.annotate(f"{v:.1f}x", (i, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, fontweight="bold")

    fig.suptitle("Scaling Analysis: Agent Performance Across Instance Sizes",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIGURE_DIR, "scaling_analysis.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, "scaling_analysis.png"), dpi=300)
    plt.close(fig)
    print("Generated: scaling_analysis.pdf/png")


if __name__ == "__main__":
    generate_shipyard_layout()
    generate_curriculum_learning_curve()
    generate_cross_config_comparison()
    generate_calibration_scatter()
    generate_entropy_collapse()
    generate_method_comparison()
    generate_scaling_analysis()
    print(f"\nAll figures saved to {FIGURE_DIR}/")

#!/usr/bin/env python3
"""Visualize the GNN heterogeneous graph structure for the paper.

Creates a visual representation of the graph neural network structure
showing:
- Node types (blocks, SPMTs, cranes, facilities)
- Edge types (transport, lift, location, precedence)
- Feature dimensions
- Message passing flow

Output: figures/gnn_graph_structure.pdf (and .png)

Usage:
    python experiments/visualize_gnn_graph.py
    python experiments/visualize_gnn_graph.py --output figures/gnn_graph.pdf
"""

import argparse
import os
import sys

# Ensure matplotlib uses non-interactive backend for server environments
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# Colors for node types (colorblind-friendly)
NODE_COLORS = {
    "block": "#3498db",      # Blue
    "spmt": "#e67e22",       # Orange
    "crane": "#9b59b6",      # Purple
    "facility": "#27ae60",   # Green
}

EDGE_COLORS = {
    "needs_transport": "#3498db",
    "can_transport": "#e67e22",
    "needs_lift": "#9b59b6",
    "can_lift": "#9b59b6",
    "at_facility": "#27ae60",
    "precedes": "#7f8c8d",
}


def create_gnn_structure_figure(output_path: str = "figures/gnn_graph_structure.pdf"):
    """Create the GNN structure visualization for the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # === LEFT PANEL: Node and Edge Types ===
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("(a) Heterogeneous Graph Structure", fontsize=14, fontweight='bold', pad=20)

    # Node positions
    node_positions = {
        "Block\n(200 per ship)": (2.5, 7.5),
        "SPMT\n(12 units)": (7.5, 7.5),
        "Goliath Crane\n(9 units)": (7.5, 2.5),
        "Facility\n(15 stations)": (2.5, 2.5),
    }

    # Draw nodes
    for node_type, (x, y) in node_positions.items():
        # Determine color
        if "Block" in node_type:
            color = NODE_COLORS["block"]
            features = "8 features:\nstage, type, weight,\ndue_date, completion,\nstatus, predecessors"
        elif "SPMT" in node_type:
            color = NODE_COLORS["spmt"]
            features = "9 features:\nstatus, location,\nload_ratio,\nhealth (H/T/E)"
        elif "Crane" in node_type:
            color = NODE_COLORS["crane"]
            features = "8 features:\nstatus, position,\nassigned_dock,\nhealth (hoist/trolley/gantry)"
        else:
            color = NODE_COLORS["facility"]
            features = "3 features:\nqueue_depth,\nutilization,\navg_wait_time"

        # Draw node circle
        circle = plt.Circle((x, y), 1.2, color=color, alpha=0.3, linewidth=2, edgecolor=color)
        ax1.add_patch(circle)

        # Node label
        ax1.text(x, y + 0.3, node_type, ha='center', va='center', fontsize=10, fontweight='bold')

        # Feature list
        ax1.text(x, y - 0.6, features, ha='center', va='top', fontsize=7.5, color='#555')

    # Draw edges with arrows
    edge_specs = [
        # (from_pos, to_pos, label, color, style)
        ((3.7, 7.5), (6.3, 7.5), "needs_transport", EDGE_COLORS["needs_transport"], "-"),
        ((6.3, 7.5), (3.7, 7.5), "can_transport", EDGE_COLORS["can_transport"], "--"),
        ((3.2, 6.3), (6.8, 3.7), "needs_lift", EDGE_COLORS["needs_lift"], "-"),
        ((6.8, 3.7), (3.2, 6.3), "can_lift", EDGE_COLORS["can_lift"], "--"),
        ((2.5, 6.3), (2.5, 3.7), "at_facility", EDGE_COLORS["at_facility"], "-"),
        ((1.5, 8.5), (1.0, 8.0), "precedes", EDGE_COLORS["precedes"], ":"),
    ]

    for (x1, y1), (x2, y2), label, color, style in edge_specs:
        # Draw arrow
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=15,
            color=color, linewidth=2, linestyle=style,
            connectionstyle="arc3,rad=0.1"
        )
        ax1.add_patch(arrow)

    # Self-loop for precedence
    ax1.annotate("", xy=(1.5, 8.7), xytext=(3.0, 8.7),
                arrowprops=dict(arrowstyle="->", color=EDGE_COLORS["precedes"],
                               linestyle=":", connectionstyle="arc3,rad=-0.5"))
    ax1.text(2.25, 9.2, "precedes", fontsize=8, color=EDGE_COLORS["precedes"], ha='center')

    # Legend for edge types
    edge_legend = [
        Line2D([0], [0], color=EDGE_COLORS["needs_transport"], linestyle="-", linewidth=2),
        Line2D([0], [0], color=EDGE_COLORS["can_transport"], linestyle="--", linewidth=2),
        Line2D([0], [0], color=EDGE_COLORS["needs_lift"], linestyle="-", linewidth=2),
        Line2D([0], [0], color=EDGE_COLORS["at_facility"], linestyle="-", linewidth=2),
        Line2D([0], [0], color=EDGE_COLORS["precedes"], linestyle=":", linewidth=2),
    ]
    ax1.legend(edge_legend,
              ["needs_transport (B→S)", "can_transport (S→B)",
               "needs/can_lift (B↔C)", "at_facility (B→F)", "precedes (B→B)"],
              loc='lower center', ncol=2, fontsize=8)

    # === RIGHT PANEL: Message Passing ===
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("(b) GNN Message Passing (2 layers)", fontsize=14, fontweight='bold', pad=20)

    # Layer 0: Input features
    layer_y = [8.5, 5.0, 1.5]
    layer_labels = ["Input\nFeatures", "Layer 1\n(128 dim)", "Layer 2\nOutput (512 dim)"]

    for i, (y, label) in enumerate(zip(layer_y, layer_labels)):
        # Layer label
        ax2.text(0.5, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

        # Node representations
        x_positions = [2.5, 4.5, 6.5, 8.5]
        node_types = ["Block", "SPMT", "Crane", "Facility"]
        colors = [NODE_COLORS["block"], NODE_COLORS["spmt"],
                 NODE_COLORS["crane"], NODE_COLORS["facility"]]

        for x, nt, c in zip(x_positions, node_types, colors):
            # Node box
            if i == 0:
                # Input features - different sizes
                sizes = [8, 9, 8, 3]
                w = sizes[x_positions.index(x)] * 0.08
                rect = FancyBboxPatch((x - w, y - 0.4), w * 2, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, alpha=0.3, edgecolor=c, linewidth=2)
                ax2.add_patch(rect)
                ax2.text(x, y, f"{sizes[x_positions.index(x)]}d", ha='center', va='center', fontsize=9)
            elif i == 1:
                # Layer 1 - uniform 128 dim
                rect = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, alpha=0.3, edgecolor=c, linewidth=2)
                ax2.add_patch(rect)
                ax2.text(x, y, "128d", ha='center', va='center', fontsize=9)
            else:
                # Output - pooled to single vector
                if x == 2.5:  # Only draw once
                    rect = FancyBboxPatch((4.0, y - 0.5), 4.0, 1.0,
                                         boxstyle="round,pad=0.1",
                                         facecolor="#34495e", alpha=0.2,
                                         edgecolor="#34495e", linewidth=2)
                    ax2.add_patch(rect)
                    ax2.text(5.0, y, "512d = 4 × 128 (global pooled)", ha='center', va='center', fontsize=10)

        # Arrows between layers
        if i < 2:
            next_y = layer_y[i + 1]
            for x in x_positions:
                if i == 1:
                    # All converge to pooled output
                    ax2.annotate("", xy=(5.0, next_y + 0.6), xytext=(x, y - 0.5),
                                arrowprops=dict(arrowstyle="->", color="#7f8c8d", alpha=0.5))
                else:
                    ax2.annotate("", xy=(x, next_y + 0.5), xytext=(x, y - 0.5),
                                arrowprops=dict(arrowstyle="->", color="#7f8c8d", alpha=0.5))

    # Message passing annotations
    ax2.text(5.0, 6.75, "Aggregate neighbor\nmessages per type",
            ha='center', va='center', fontsize=8, color='#555', style='italic')

    ax2.text(5.0, 3.25, "Mean pooling\nacross all nodes",
            ha='center', va='center', fontsize=8, color='#555', style='italic')

    # Node type legend
    node_patches = [mpatches.Patch(color=c, alpha=0.5, label=t)
                   for t, c in NODE_COLORS.items()]
    ax2.legend(handles=node_patches, loc='lower center', ncol=4, fontsize=9)

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save PNG version
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {png_path}")

    plt.close()


def create_action_space_figure(output_path: str = "figures/action_space.pdf"):
    """Create hierarchical action space visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title("Hierarchical Action Space", fontsize=14, fontweight='bold')

    # Root action type
    root = (5, 6)
    ax.add_patch(FancyBboxPatch((4, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor="#3498db", alpha=0.3, edgecolor="#3498db", linewidth=2))
    ax.text(5, 5.9, "Action Type\n(4 options)", ha='center', va='center', fontsize=10, fontweight='bold')

    # Action type branches
    action_types = [
        ("Dispatch\nSPMT", 1.5, "#e67e22"),
        ("Dispatch\nCrane", 4.0, "#9b59b6"),
        ("Trigger\nMaintenance", 6.5, "#27ae60"),
        ("Hold", 9.0, "#95a5a6"),
    ]

    for label, x, color in action_types:
        ax.add_patch(FancyBboxPatch((x - 0.8, 3.5), 1.6, 0.8, boxstyle="round,pad=0.1",
                                    facecolor=color, alpha=0.3, edgecolor=color, linewidth=2))
        ax.text(x, 3.9, label, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.annotate("", xy=(x, 4.3), xytext=(5, 5.4),
                   arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5))

    # Sub-actions
    sub_actions = [
        # (parent_x, sub_label, sub_x, sub_y)
        (1.5, "SPMT idx\n(0-11)", 0.8, 2.0),
        (1.5, "Request idx\n(0-199)", 2.2, 2.0),
        (4.0, "Crane idx\n(0-8)", 3.3, 2.0),
        (4.0, "Lift idx\n(0-199)", 4.7, 2.0),
        (6.5, "Equipment idx\n(0-20)", 6.5, 2.0),
    ]

    for px, label, x, y in sub_actions:
        ax.add_patch(FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                                    facecolor="#ecf0f1", edgecolor="#7f8c8d", linewidth=1))
        ax.text(x, y, label, ha='center', va='center', fontsize=8)
        ax.annotate("", xy=(x, y + 0.45), xytext=(px, 3.45),
                   arrowprops=dict(arrowstyle="->", color="#7f8c8d", linewidth=1))

    # Masking note
    ax.text(5, 0.8, "Action Masking: Invalid actions masked to -inf logits\n"
                   "(e.g., busy SPMTs, no pending requests, healthy equipment)",
           ha='center', va='center', fontsize=9, style='italic', color='#555',
           bbox=dict(boxstyle='round', facecolor='#fff9c4', alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate GNN visualization for paper")
    parser.add_argument("--output-dir", type=str, default="figures",
                       help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create GNN structure figure
    create_gnn_structure_figure(os.path.join(args.output_dir, "gnn_graph_structure.pdf"))

    # Create action space figure
    create_action_space_figure(os.path.join(args.output_dir, "action_space.pdf"))

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

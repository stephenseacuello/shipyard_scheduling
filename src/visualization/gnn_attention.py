"""Visualize GNN attention weights and embeddings.

This module provides tools to interpret what the GNN has learned by
visualizing attention patterns and learned node representations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


def visualize_attention(
    encoder: torch.nn.Module,
    graph_data: "torch_geometric.data.HeteroData",
    output_dir: Union[str, Path],
    layer_indices: Optional[List[int]] = None,
) -> None:
    """Plot attention heatmaps for each GNN layer.

    This function extracts attention weights from GNN layers and creates
    heatmap visualizations showing which nodes attend to which.

    Args:
        encoder: The GNN encoder module with attention layers
        graph_data: A HeteroData object representing the current state
        output_dir: Directory to save attention visualizations
        layer_indices: Which layers to visualize (default: all)
    """
    if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
        print("matplotlib/seaborn not available, skipping visualize_attention")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder.eval()

    # Try to get attention weights - depends on encoder architecture
    with torch.no_grad():
        # Run forward pass and capture attention if available
        if hasattr(encoder, "get_attention_weights"):
            attentions = encoder.get_attention_weights(graph_data)
        elif hasattr(encoder, "conv_layers"):
            # For GAT-based encoders, try to extract attention from conv layers
            attentions = []
            x_dict = {}
            for node_type, store in graph_data.node_items():
                if hasattr(store, "x"):
                    x_dict[node_type] = store.x

            for i, conv in enumerate(encoder.conv_layers):
                if hasattr(conv, "return_attention_weights"):
                    # Some GATConv implementations support this
                    _, alpha = conv(x_dict, graph_data.edge_index_dict,
                                   return_attention_weights=True)
                    attentions.append(alpha)
        else:
            print("Encoder does not expose attention weights")
            return

    if not attentions:
        print("No attention weights captured")
        return

    # Determine which layers to visualize
    if layer_indices is None:
        layer_indices = range(len(attentions))

    for i in layer_indices:
        if i >= len(attentions):
            continue

        attn = attentions[i]
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()

        # Handle different attention tensor shapes
        if attn.ndim == 3:
            # Multi-head attention: average across heads
            attn = attn.mean(axis=0)
        elif attn.ndim == 1:
            # Edge-level attention: reshape to square if possible
            n = int(np.sqrt(len(attn)))
            if n * n == len(attn):
                attn = attn.reshape(n, n)
            else:
                print(f"Layer {i}: Cannot reshape attention vector to matrix")
                continue

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attn, cmap="viridis", ax=ax, cbar_kws={"label": "Attention Weight"})
        ax.set_title(f"Layer {i + 1} Attention Weights")
        ax.set_xlabel("Key Node")
        ax.set_ylabel("Query Node")

        plt.tight_layout()
        plt.savefig(output_dir / f"attention_layer_{i + 1}.png", dpi=300)
        plt.savefig(output_dir / f"attention_layer_{i + 1}.pdf")
        plt.close()

    print(f"Saved attention visualizations to {output_dir}")


def visualize_embeddings(
    encoder: torch.nn.Module,
    graph_data: "torch_geometric.data.HeteroData",
    output_dir: Union[str, Path],
    perplexity: int = 30,
    n_iter: int = 1000,
) -> None:
    """t-SNE visualization of learned node embeddings.

    Creates a scatter plot showing how different node types are embedded
    in the learned representation space.

    Args:
        encoder: The GNN encoder module
        graph_data: A HeteroData object representing the current state
        output_dir: Directory to save embedding visualization
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualize_embeddings")
        return

    if not TSNE_AVAILABLE:
        print("sklearn.manifold.TSNE not available, skipping visualize_embeddings")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder.eval()

    with torch.no_grad():
        # Get embeddings for each node type
        if hasattr(encoder, "encode_nodes"):
            embeddings_dict = encoder.encode_nodes(graph_data)
        else:
            # Run full forward pass and try to extract intermediate representations
            _ = encoder(graph_data)
            if hasattr(encoder, "last_embeddings"):
                embeddings_dict = encoder.last_embeddings
            else:
                print("Encoder does not expose node embeddings")
                return

    # Collect all embeddings with their types
    all_embeddings = []
    node_types = []
    type_colors = {
        "block": "tab:blue",
        "spmt": "tab:orange",
        "crane": "tab:green",
        "facility": "tab:red",
        "ship": "tab:purple",
        "dock": "tab:brown",
    }

    for node_type, emb in embeddings_dict.items():
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        all_embeddings.append(emb)
        node_types.extend([node_type] * len(emb))

    if not all_embeddings:
        print("No embeddings to visualize")
        return

    all_embeddings = np.vstack(all_embeddings)

    # Reduce dimensionality with t-SNE
    print(f"Running t-SNE on {len(all_embeddings)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(all_embeddings) - 1),
                n_iter=n_iter, random_state=42)
    reduced = tsne.fit_transform(all_embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_types = list(set(node_types))
    for node_type in unique_types:
        mask = [t == node_type for t in node_types]
        color = type_colors.get(node_type, "tab:gray")
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                  label=node_type.title(), alpha=0.7, c=color, s=50)

    ax.legend()
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("GNN Node Embeddings (t-SNE)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "embeddings_tsne.png", dpi=300)
    plt.savefig(output_dir / "embeddings_tsne.pdf")
    plt.close()

    print(f"Saved embedding visualization to {output_dir}")


def visualize_graph_structure(
    graph_data: "torch_geometric.data.HeteroData",
    output_dir: Union[str, Path],
    max_nodes: int = 50,
) -> None:
    """Visualize the heterogeneous graph structure.

    Creates a network diagram showing nodes (colored by type) and edges
    (styled by relation type).

    Args:
        graph_data: A HeteroData object
        output_dir: Directory to save visualization
        max_nodes: Maximum number of nodes to display (for readability)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualize_graph_structure")
        return

    try:
        import networkx as nx
    except ImportError:
        print("networkx not available, skipping visualize_graph_structure")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build NetworkX graph
    G = nx.DiGraph()

    type_colors = {
        "block": "#1f77b4",
        "spmt": "#ff7f0e",
        "crane": "#2ca02c",
        "facility": "#d62728",
        "ship": "#9467bd",
    }

    node_id = 0
    node_mapping = {}  # (type, local_id) -> global_id

    # Add nodes
    for node_type, store in graph_data.node_items():
        n_nodes = store.x.shape[0] if hasattr(store, "x") else 0
        display_nodes = min(n_nodes, max_nodes // len(list(graph_data.node_types)))

        for i in range(display_nodes):
            G.add_node(node_id, node_type=node_type, label=f"{node_type[:3]}_{i}")
            node_mapping[(node_type, i)] = node_id
            node_id += 1

    # Add edges
    for edge_type, store in graph_data.edge_items():
        if hasattr(store, "edge_index"):
            src_type, rel, dst_type = edge_type
            edge_index = store.edge_index.cpu().numpy()

            for src, dst in zip(edge_index[0], edge_index[1]):
                src_key = (src_type, int(src))
                dst_key = (dst_type, int(dst))

                if src_key in node_mapping and dst_key in node_mapping:
                    G.add_edge(node_mapping[src_key], node_mapping[dst_key],
                              relation=rel)

    if len(G.nodes) == 0:
        print("No nodes to visualize")
        return

    # Layout and colors
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    node_colors = [type_colors.get(G.nodes[n].get("node_type", ""), "#7f7f7f")
                   for n in G.nodes]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5,
                          arrows=True, arrowsize=10, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n].get("label", "")
                                            for n in G.nodes},
                           font_size=8, ax=ax)

    # Legend
    legend_elements = [plt.scatter([], [], c=color, s=100, label=ntype.title())
                      for ntype, color in type_colors.items()
                      if any(G.nodes[n].get("node_type") == ntype for n in G.nodes)]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_title("Heterogeneous Graph Structure")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "graph_structure.png", dpi=300)
    plt.savefig(output_dir / "graph_structure.pdf")
    plt.close()

    print(f"Saved graph structure to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize GNN attention and embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/hhi_ulsan.yaml", help="Environment config")
    parser.add_argument("--output", type=str, default="figures/gnn", help="Output directory")
    args = parser.parse_args()

    print("GNN visualization requires loading a checkpoint and running an environment step.")
    print("Use the provided functions in your training/evaluation scripts.")

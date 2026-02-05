"""Graph neural network encoders for shipyard state.

The primary encoder is a heterogeneous GNN that processes blocks,
SPMTs, cranes and facilities using type‑specific projections and
multi‑head attention message passing layers. A simpler homogeneous
encoder is also provided for baseline experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool, HeteroConv


class HeterogeneousGNNEncoder(nn.Module):
    """Heterogeneous graph encoder with multi‑head attention layers."""

    def __init__(
        self,
        block_dim: int = 8,
        spmt_dim: int = 10,
        crane_dim: int = 7,
        facility_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Input projections
        self.block_proj = nn.Linear(block_dim, hidden_dim)
        self.spmt_proj = nn.Linear(spmt_dim, hidden_dim)
        self.crane_proj = nn.Linear(crane_dim, hidden_dim)
        self.facility_proj = nn.Linear(facility_dim, hidden_dim)
        # Message passing layers (each layer gets its own set of GATConv modules)
        self.conv_layers = nn.ModuleList()
        edge_types = [
            ("block", "needs_transport", "spmt"),
            ("spmt", "can_transport", "block"),
            ("block", "needs_lift", "crane"),
            ("crane", "can_lift", "block"),
            ("block", "at", "facility"),
            ("block", "precedes", "block"),
            ("spmt", "at", "facility"),
            ("crane", "at", "facility"),  # crane location (at dock)
        ]
        for _ in range(num_layers):
            relations = {
                et: GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False)
                for et in edge_types
            }
            self.conv_layers.append(HeteroConv(relations, aggr="mean"))
        # Layer norms
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        x_dict = {
            "block": self.block_proj(data["block"].x),
            "spmt": self.spmt_proj(data["spmt"].x),
            "crane": self.crane_proj(data["crane"].x),
            "facility": self.facility_proj(data["facility"].x),
        }
        # Message passing
        for conv, norm in zip(self.conv_layers, self.norms):
            x_dict_new = conv(x_dict, data.edge_index_dict)
            # residual connection and normalization
            x_dict = {
                k: norm(self.dropout(F.relu(x_dict_new.get(k, x_dict[k]))) + x_dict[k])
                for k in x_dict
            }
        # Global pooling: average per node type then concatenate
        pooled = []
        for node_type in ["block", "spmt", "crane", "facility"]:
            if data[node_type].x.shape[0] > 0:
                pooled.append(global_mean_pool(x_dict[node_type], data[node_type].batch))
            else:
                pooled.append(torch.zeros((1, x_dict[node_type].shape[1]), device=x_dict[node_type].device))
        return torch.cat(pooled, dim=-1)


class SimpleGNNEncoder(nn.Module):
    """Homogeneous GNN encoder for comparison purposes."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv(x, edge_index)) + x)
        return global_mean_pool(x, batch)
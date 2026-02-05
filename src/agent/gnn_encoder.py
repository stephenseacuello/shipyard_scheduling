"""Graph neural network encoders for shipyard state.

The primary encoder is a heterogeneous GNN that processes blocks,
SPMTs, cranes and facilities using type-specific projections and
multi-head attention message passing layers.

Encoder variants:
- HeterogeneousGNNEncoder: GAT-based with multi-head attention (baseline)
- HeterogeneousGraphTransformer: Full transformer attention (HGT-style)
- TemporalGNNEncoder: GNN with temporal positional encoding
- SimpleGNNEncoder: Homogeneous GNN for ablation studies
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool, HeteroConv
from torch_geometric.data import HeteroData


# Edge types used in the shipyard heterogeneous graph
EDGE_TYPES = [
    ("block", "needs_transport", "spmt"),
    ("spmt", "can_transport", "block"),
    ("block", "needs_lift", "crane"),
    ("crane", "can_lift", "block"),
    ("block", "at", "facility"),
    ("block", "precedes", "block"),
    ("spmt", "at", "facility"),
    ("crane", "at", "facility"),
]

# Node types
NODE_TYPES = ["block", "spmt", "crane", "facility"]


class HeterogeneousGNNEncoder(nn.Module):
    """Heterogeneous graph encoder with multi-head attention layers (GAT-based)."""

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
        for _ in range(num_layers):
            relations = {
                et: GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False)
                for et in EDGE_TYPES
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
        for node_type in NODE_TYPES:
            if data[node_type].x.shape[0] > 0:
                pooled.append(global_mean_pool(x_dict[node_type], data[node_type].batch))
            else:
                pooled.append(torch.zeros((1, x_dict[node_type].shape[1]), device=x_dict[node_type].device))
        return torch.cat(pooled, dim=-1)


class RelationTypeEmbedding(nn.Module):
    """Learnable embeddings for relation (edge) types.

    Following HGT, each relation has source type, edge type, and target type.
    We learn embeddings for each component.
    """

    def __init__(
        self,
        n_node_types: int,
        n_edge_types: int,
        embed_dim: int,
    ):
        super().__init__()
        self.node_type_embed = nn.Embedding(n_node_types, embed_dim)
        self.edge_type_embed = nn.Embedding(n_edge_types, embed_dim)

    def forward(
        self, src_type: int, edge_type: int, dst_type: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get embeddings for a relation triplet."""
        src_embed = self.node_type_embed(torch.tensor(src_type, device=self.node_type_embed.weight.device))
        edge_embed = self.edge_type_embed(torch.tensor(edge_type, device=self.edge_type_embed.weight.device))
        dst_embed = self.node_type_embed(torch.tensor(dst_type, device=self.node_type_embed.weight.device))
        return src_embed, edge_embed, dst_embed


class HGTAttention(nn.Module):
    """Heterogeneous Graph Transformer attention mechanism.

    Implements relation-specific attention following HGT (Hu et al., 2020):
    - K, Q, V projections are relation-specific
    - Attention uses relation-aware score computation
    - Multi-head attention with relation message aggregation
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_node_types: int,
        n_edge_types: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types

        # Relation-specific projection matrices
        # K, Q for attention computation
        self.k_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types)
        ])
        self.q_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types)
        ])
        self.v_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types)
        ])

        # Relation attention weights (per edge type, per head)
        self.relation_att = nn.Parameter(
            torch.Tensor(n_edge_types, num_heads, self.head_dim, self.head_dim)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(n_edge_types, num_heads, self.head_dim, self.head_dim)
        )

        # Prior tensor for relation importance
        self.relation_pri = nn.Parameter(torch.ones(n_edge_types, num_heads))

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        src_type: int,
        edge_type: int,
        dst_type: int,
    ) -> torch.Tensor:
        """Compute attention and message passing for a specific relation.

        Args:
            x_src: Source node features [n_src, hidden_dim]
            x_dst: Destination node features [n_dst, hidden_dim]
            edge_index: Edge indices [2, n_edges]
            src_type: Source node type index
            edge_type: Edge type index
            dst_type: Destination node type index

        Returns:
            Updated destination node features [n_dst, hidden_dim]
        """
        n_dst = x_dst.size(0)
        n_edges = edge_index.size(1)

        if n_edges == 0:
            return torch.zeros_like(x_dst)

        # Project source and destination nodes
        k = self.k_linears[src_type](x_src).view(-1, self.num_heads, self.head_dim)
        v = self.v_linears[src_type](x_src).view(-1, self.num_heads, self.head_dim)
        q = self.q_linears[dst_type](x_dst).view(-1, self.num_heads, self.head_dim)

        # Get source and destination nodes for each edge
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Relation-specific key transformation: k @ W_att
        # k: [n_src, heads, head_dim] -> k_edge: [n_edges, heads, head_dim]
        k_edge = k[src_idx]  # [n_edges, heads, head_dim]
        # Apply relation attention matrix
        k_edge = torch.einsum("ehi,hio->eho", k_edge, self.relation_att[edge_type])

        # Query for destination nodes involved in edges
        q_edge = q[dst_idx]  # [n_edges, heads, head_dim]

        # Attention scores
        att = (k_edge * q_edge).sum(dim=-1) / self.scale  # [n_edges, heads]
        att = att + self.relation_pri[edge_type].unsqueeze(0)  # Add relation prior

        # Softmax over incoming edges for each destination
        att = self._softmax_over_edges(att, dst_idx, n_dst)
        att = self.dropout(att)

        # Message computation: v @ W_msg
        v_edge = v[src_idx]  # [n_edges, heads, head_dim]
        msg = torch.einsum("ehi,hio->eho", v_edge, self.relation_msg[edge_type])

        # Weighted aggregation
        out = torch.zeros(n_dst, self.num_heads, self.head_dim, device=x_dst.device)
        att_expanded = att.unsqueeze(-1)  # [n_edges, heads, 1]
        out.scatter_add_(0, dst_idx.unsqueeze(-1).unsqueeze(-1).expand_as(msg), att_expanded * msg)

        return out.view(n_dst, self.hidden_dim)

    def _softmax_over_edges(
        self, att: torch.Tensor, dst_idx: torch.Tensor, n_dst: int
    ) -> torch.Tensor:
        """Compute softmax over incoming edges for each destination node."""
        # Subtract max for numerical stability
        att_max = torch.zeros(n_dst, att.size(1), device=att.device)
        att_max.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand_as(att), att, reduce="amax")
        att = att - att_max[dst_idx]

        # Exp and sum
        att_exp = att.exp()
        att_sum = torch.zeros(n_dst, att.size(1), device=att.device)
        att_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(att_exp), att_exp)
        att_sum = att_sum.clamp(min=1e-8)

        return att_exp / att_sum[dst_idx]


class HGTLayer(nn.Module):
    """Single Heterogeneous Graph Transformer layer.

    Processes all relation types and aggregates updates per node type.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_node_types: int,
        n_edge_types: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_node_types = n_node_types

        # Attention module
        self.attention = HGTAttention(
            hidden_dim, num_heads, n_node_types, n_edge_types, dropout
        )

        # Output projections per node type
        self.out_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types)
        ])

        # Layer norm per node type
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_node_types)
        ])

        # FFN per node type
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_node_types)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_node_types)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        node_type_to_idx: Dict[str, int],
        edge_type_to_idx: Dict[Tuple[str, str, str], int],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the HGT layer.

        Args:
            x_dict: Node features per type.
            edge_index_dict: Edge indices per relation type.
            node_type_to_idx: Mapping from node type name to index.
            edge_type_to_idx: Mapping from edge type triplet to index.

        Returns:
            Updated node features per type.
        """
        # Collect messages for each node type
        out_dict = {k: torch.zeros_like(v) for k, v in x_dict.items()}
        count_dict = {k: 0 for k in x_dict.keys()}

        for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
            if edge_index.size(1) == 0:
                continue

            src_idx = node_type_to_idx[src_type]
            edge_idx = edge_type_to_idx[(src_type, rel, dst_type)]
            dst_idx = node_type_to_idx[dst_type]

            msg = self.attention(
                x_dict[src_type],
                x_dict[dst_type],
                edge_index,
                src_idx,
                edge_idx,
                dst_idx,
            )
            out_dict[dst_type] = out_dict[dst_type] + msg
            count_dict[dst_type] += 1

        # Normalize and apply residual + FFN
        result = {}
        for node_type, x in x_dict.items():
            type_idx = node_type_to_idx[node_type]
            if count_dict[node_type] > 0:
                out = out_dict[node_type] / count_dict[node_type]
                out = self.out_linears[type_idx](out)
                out = self.dropout(out)
            else:
                out = torch.zeros_like(x)

            # Residual + norm
            x = self.norms[type_idx](x + out)
            # FFN + residual
            x = self.ffn_norms[type_idx](x + self.ffn[type_idx](x))
            result[node_type] = x

        return result


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal ordering."""

    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Get positional encodings for given positions.

        Args:
            positions: Integer position indices [n_nodes]

        Returns:
            Positional encodings [n_nodes, hidden_dim]
        """
        return self.pe[positions]


class VirtualNode(nn.Module):
    """Virtual node for global graph context.

    Adds a virtual node connected to all other nodes to capture
    global information and enable long-range communication.
    """

    def __init__(self, hidden_dim: int, n_node_types: int):
        super().__init__()
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_dim))
        self.to_virtual = nn.Linear(hidden_dim, hidden_dim)
        self.from_virtual = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
        node_type_to_idx: Dict[str, int],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Update node features with virtual node communication.

        Args:
            x_dict: Node features per type.
            batch_dict: Batch indices per node type.
            node_type_to_idx: Mapping from node type to index.

        Returns:
            Updated node features and virtual node embedding.
        """
        # Aggregate all nodes to virtual node
        all_features = []
        for node_type, x in x_dict.items():
            all_features.append(self.to_virtual(x).mean(dim=0, keepdim=True))

        virtual = self.virtual_node + torch.stack(all_features).mean(dim=0)
        virtual = self.norm(virtual)

        # Broadcast virtual node back to all nodes
        result = {}
        for node_type, x in x_dict.items():
            type_idx = node_type_to_idx[node_type]
            update = self.from_virtual[type_idx](virtual.expand(x.size(0), -1))
            result[node_type] = x + update

        return result, virtual


class HeterogeneousGraphTransformer(nn.Module):
    """Heterogeneous Graph Transformer encoder based on HGT architecture.

    This is a more powerful alternative to the GAT-based encoder, using
    full transformer attention with relation-specific projections.

    Features:
    - Relation-aware multi-head attention (HGT-style)
    - Learnable relation type embeddings
    - Optional temporal positional encoding
    - Optional virtual node for global context
    - FFN after attention (full transformer block)

    Reference:
    Hu, Z., et al. "Heterogeneous Graph Transformer." WWW 2020.

    Args:
        block_dim: Block node feature dimension.
        spmt_dim: SPMT node feature dimension.
        crane_dim: Crane node feature dimension.
        facility_dim: Facility node feature dimension.
        hidden_dim: Hidden dimension throughout the network.
        num_layers: Number of HGT layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        use_positional: Whether to use temporal positional encoding.
        use_virtual_node: Whether to use virtual node for global context.
    """

    def __init__(
        self,
        block_dim: int = 8,
        spmt_dim: int = 10,
        crane_dim: int = 7,
        facility_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_positional: bool = False,
        use_virtual_node: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_positional = use_positional
        self.use_virtual_node = use_virtual_node

        # Node type and edge type mappings
        self.node_types = NODE_TYPES
        self.edge_types = EDGE_TYPES
        self.node_type_to_idx = {t: i for i, t in enumerate(self.node_types)}
        self.edge_type_to_idx = {t: i for i, t in enumerate(self.edge_types)}

        n_node_types = len(self.node_types)
        n_edge_types = len(self.edge_types)

        # Input projections
        self.input_projs = nn.ModuleDict({
            "block": nn.Linear(block_dim, hidden_dim),
            "spmt": nn.Linear(spmt_dim, hidden_dim),
            "crane": nn.Linear(crane_dim, hidden_dim),
            "facility": nn.Linear(facility_dim, hidden_dim),
        })

        # Learnable node type embeddings
        self.node_type_embed = nn.Embedding(n_node_types, hidden_dim)

        # Positional encoding (optional)
        if use_positional:
            self.pos_encoder = PositionalEncoding(hidden_dim)

        # Virtual node (optional)
        if use_virtual_node:
            self.virtual_node = VirtualNode(hidden_dim, n_node_types)

        # HGT layers
        self.layers = nn.ModuleList([
            HGTLayer(hidden_dim, num_heads, n_node_types, n_edge_types, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        output_dim = hidden_dim * n_node_types
        if use_virtual_node:
            output_dim += hidden_dim
        self.output_proj = nn.Linear(output_dim, hidden_dim * n_node_types)

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Encode heterogeneous graph to fixed-size state embedding.

        Args:
            data: PyG HeteroData object with node features and edge indices.

        Returns:
            State embedding tensor [batch_size, hidden_dim * 4].
        """
        # Project input features
        x_dict = {}
        for node_type in self.node_types:
            x = self.input_projs[node_type](data[node_type].x)
            # Add node type embedding
            type_idx = self.node_type_to_idx[node_type]
            type_embed = self.node_type_embed.weight[type_idx].unsqueeze(0)
            x = x + type_embed.expand(x.size(0), -1)
            x_dict[node_type] = x

        # Add positional encoding if enabled
        if self.use_positional and hasattr(data, "positions"):
            for node_type in self.node_types:
                if hasattr(data[node_type], "position"):
                    pos_enc = self.pos_encoder(data[node_type].position)
                    x_dict[node_type] = x_dict[node_type] + pos_enc

        # Get batch info for pooling
        batch_dict = {}
        for node_type in self.node_types:
            if hasattr(data[node_type], "batch"):
                batch_dict[node_type] = data[node_type].batch
            else:
                batch_dict[node_type] = torch.zeros(
                    x_dict[node_type].size(0), dtype=torch.long, device=x_dict[node_type].device
                )

        # Apply HGT layers
        for layer in self.layers:
            x_dict = layer(
                x_dict, data.edge_index_dict, self.node_type_to_idx, self.edge_type_to_idx
            )

        # Virtual node communication
        virtual_emb = None
        if self.use_virtual_node:
            x_dict, virtual_emb = self.virtual_node(x_dict, batch_dict, self.node_type_to_idx)

        # Global pooling per node type
        pooled = []
        for node_type in self.node_types:
            x = x_dict[node_type]
            batch = batch_dict[node_type]
            if x.size(0) > 0:
                pooled.append(global_mean_pool(x, batch))
            else:
                batch_size = 1 if batch.numel() == 0 else batch.max().item() + 1
                pooled.append(torch.zeros(batch_size, self.hidden_dim, device=x.device))

        # Add virtual node embedding if used
        if virtual_emb is not None:
            pooled.append(virtual_emb.expand(pooled[0].size(0), -1))

        # Concatenate and project
        out = torch.cat(pooled, dim=-1)
        out = self.output_proj(out)

        return out


class TemporalGNNEncoder(nn.Module):
    """GNN encoder with temporal state tracking.

    Maintains per-node hidden states across time steps using GRU cells,
    enabling the model to capture temporal patterns in the scheduling process.

    Args:
        block_dim: Block node feature dimension.
        spmt_dim: SPMT node feature dimension.
        crane_dim: Crane node feature dimension.
        facility_dim: Facility node feature dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of message passing layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

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

        self.hidden_dim = hidden_dim
        self.node_types = NODE_TYPES

        # Input projections
        self.input_projs = nn.ModuleDict({
            "block": nn.Linear(block_dim, hidden_dim),
            "spmt": nn.Linear(spmt_dim, hidden_dim),
            "crane": nn.Linear(crane_dim, hidden_dim),
            "facility": nn.Linear(facility_dim, hidden_dim),
        })

        # GRU cells for temporal state
        self.gru_cells = nn.ModuleDict({
            node_type: nn.GRUCell(hidden_dim, hidden_dim)
            for node_type in self.node_types
        })

        # Message passing layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            relations = {
                et: GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False)
                for et in EDGE_TYPES
            }
            self.conv_layers.append(HeteroConv(relations, aggr="mean"))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Temporal hidden states (stored between forward passes during rollout)
        self._hidden_states: Optional[Dict[str, torch.Tensor]] = None

    def reset_hidden(self) -> None:
        """Reset temporal hidden states (call at episode start)."""
        self._hidden_states = None

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Encode graph with temporal state update.

        Args:
            data: PyG HeteroData object.

        Returns:
            State embedding tensor [batch_size, hidden_dim * 4].
        """
        # Project input features
        x_dict = {
            node_type: self.input_projs[node_type](data[node_type].x)
            for node_type in self.node_types
        }

        # Update with GRU cells if we have previous hidden state
        if self._hidden_states is not None:
            for node_type in self.node_types:
                n_current = x_dict[node_type].size(0)
                n_hidden = self._hidden_states[node_type].size(0)

                if n_current == n_hidden:
                    # Same number of nodes, apply GRU
                    x_dict[node_type] = self.gru_cells[node_type](
                        x_dict[node_type], self._hidden_states[node_type]
                    )
                # If node count changed, just use current features (no GRU update)

        # Message passing
        for conv, norm in zip(self.conv_layers, self.norms):
            x_dict_new = conv(x_dict, data.edge_index_dict)
            x_dict = {
                k: norm(self.dropout(F.relu(x_dict_new.get(k, x_dict[k]))) + x_dict[k])
                for k in x_dict
            }

        # Store hidden states for next step
        self._hidden_states = {k: v.detach() for k, v in x_dict.items()}

        # Global pooling
        pooled = []
        for node_type in self.node_types:
            x = x_dict[node_type]
            if x.size(0) > 0:
                batch = data[node_type].batch if hasattr(data[node_type], "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                pooled.append(global_mean_pool(x, batch))
            else:
                pooled.append(torch.zeros(1, self.hidden_dim, device=x.device))

        return torch.cat(pooled, dim=-1)


class SimpleGNNEncoder(nn.Module):
    """Homogeneous GNN encoder for comparison/ablation purposes."""

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


def create_encoder(
    encoder_type: str = "gat",
    block_dim: int = 8,
    spmt_dim: int = 10,
    crane_dim: int = 7,
    facility_dim: int = 3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Factory function to create the appropriate encoder.

    Args:
        encoder_type: One of 'gat', 'transformer', 'temporal'.
        **kwargs: Additional arguments passed to encoder constructor.

    Returns:
        GNN encoder module.
    """
    if encoder_type == "gat":
        return HeterogeneousGNNEncoder(
            block_dim, spmt_dim, crane_dim, facility_dim,
            hidden_dim, num_layers, num_heads, dropout
        )
    elif encoder_type == "transformer":
        return HeterogeneousGraphTransformer(
            block_dim, spmt_dim, crane_dim, facility_dim,
            hidden_dim, num_layers, num_heads, dropout,
            use_virtual_node=kwargs.get("use_virtual_node", True),
            use_positional=kwargs.get("use_positional", False),
        )
    elif encoder_type == "temporal":
        return TemporalGNNEncoder(
            block_dim, spmt_dim, crane_dim, facility_dim,
            hidden_dim, num_layers, num_heads, dropout
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

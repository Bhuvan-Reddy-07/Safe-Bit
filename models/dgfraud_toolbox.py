"""
DGFraud Toolbox: SOTA Graph Fraud Detection Models (GEM, SemiGNN, GAS).
Integrates architectures from the Safe-Graph DGFraud benchmark:
  https://github.com/safe-graph/DGFraud

Models Implemented:
  1. GEM (Heterogeneous Graph Neural Network for Malicious Account Detection - Alibaba CIKM)
  2. SemiGNN (Hierarchical Dual-Level Node & Relation Attention - ICDM)
  3. GAS (GCN-based Dual-Stream Anti-Fraud Architecture)
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import settings


class GEMConv(nn.Module):
    """
    GEM: Heterogeneous Graph Convolutional Layer.
    Applies relation-specific parameter matrices and learns relation importance weights.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # Relation transformation weights
        self.rel_weights = nn.Parameter(torch.randn(num_relations, in_dim, out_dim))
        # Self transformation
        self.self_weight = nn.Linear(in_dim, out_dim)

        # Relation attention
        self.rel_att = nn.Parameter(torch.ones(num_relations) / float(num_relations))

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: List[torch.Tensor],
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        out = self.self_weight(x)

        norm_att = F.softmax(self.rel_att, dim=0)

        for r_idx, edge_index in enumerate(edge_indices):
            if edge_index.size(1) == 0:
                continue

            src, dst = edge_index[0], edge_index[1]
            W_r = self.rel_weights[r_idx]
            # Transform neighbor features: [E, out_dim]
            h_trans = torch.matmul(x[src], W_r)

            # Aggregate into destination node
            agg = torch.zeros(num_nodes, self.out_dim, device=x.device)
            agg = agg.index_add(0, dst, h_trans)

            out = out + norm_att[r_idx] * agg

        return F.relu(out)


class SemiGNNLayer(nn.Module):
    """
    SemiGNN: Hierarchical Dual-Level Attention Layer.
      Level 1: Node-Level Attention (weights specific neighbors)
      Level 2: Relation-Level Attention (weights edge types)
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.node_att = nn.Parameter(torch.randn(2 * out_dim, 1))
        self.rel_att_vector = nn.Parameter(torch.randn(out_dim, 1))

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: List[torch.Tensor],
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        h = self.W(x)  # [N, out_dim]

        rel_outputs: List[torch.Tensor] = []

        for r_idx, edge_index in enumerate(edge_indices):
            if edge_index.size(1) == 0:
                rel_outputs.append(h)
                continue

            src, dst = edge_index[0], edge_index[1]

            # Level 1: Node-Level Attention
            # concat features of connected pairs
            edge_h = torch.cat([h[src], h[dst]], dim=-1)
            alpha = F.leaky_relu(torch.matmul(edge_h, self.node_att).squeeze(-1))
            # Softmax-style normalization approximation
            alpha_weight = torch.sigmoid(alpha).unsqueeze(-1)
            weighted_msg = h[src] * alpha_weight

            agg = torch.zeros(num_nodes, self.out_dim, device=x.device)
            agg = agg.index_add(0, dst, weighted_msg)
            rel_outputs.append(F.relu(agg))

        # Level 2: Relation-Level Attention
        stacked_rels = torch.stack(rel_outputs, dim=1)  # [N, num_relations, out_dim]
        rel_scores = torch.matmul(torch.tanh(stacked_rels), self.rel_att_vector)  # [N, num_relations, 1]
        beta = F.softmax(rel_scores, dim=1)  # [N, num_relations, 1]

        fused = torch.sum(stacked_rels * beta, dim=1)  # [N, out_dim]
        return F.relu(fused + h)


class GASModel(nn.Module):
    """
    GAS: Dual-Stream Anti-Spam / Anti-Fraud Architecture.
    Separates homophilous transaction flows from heterophilous device/IP hyper-edges.
    """

    def __init__(self, node_in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # Stream 1: Transaction Flow Stream (Financial Edges)
        self.stream_tx = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Stream 2: Telemetry Hyper-Edge Stream (Shared Device/IP)
        self.stream_telem = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        tx_edge_index: torch.Tensor,
        telem_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)

        # Stream 1
        h1 = self.stream_tx(x)
        if tx_edge_index.size(1) > 0:
            agg1 = torch.zeros_like(h1)
            agg1 = agg1.index_add(0, tx_edge_index[1], h1[tx_edge_index[0]])
            h1 = h1 + agg1

        # Stream 2
        h2 = self.stream_telem(x)
        if telem_edge_index.size(1) > 0:
            agg2 = torch.zeros_like(h2)
            agg2 = agg2.index_add(0, telem_edge_index[1], h2[telem_edge_index[0]])
            h2 = h2 + agg2

        fused = torch.cat([h1, h2], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, fused


class DGFraudBackbone(nn.Module):
    """
    Unified DGFraud Model supporting GEM, SemiGNN, and GAS backbones.
    """

    def __init__(
        self,
        arch_type: str = "gem",
        node_in_dim: int = settings.NODE_IN_DIM,
        hidden_dim: int = settings.GNN_HIDDEN_DIM,
        out_embed_dim: int = settings.GNN_OUT_EMBED_DIM,
        num_relations: int = 3,
    ):
        super().__init__()
        self.arch_type = arch_type
        self.num_relations = num_relations

        if arch_type == "semignn":
            self.layer1 = SemiGNNLayer(node_in_dim, hidden_dim, num_relations)
            self.layer2 = SemiGNNLayer(hidden_dim, out_embed_dim, num_relations)
        else:  # 'gem' default
            self.layer1 = GEMConv(node_in_dim, hidden_dim, num_relations)
            self.layer2 = GEMConv(hidden_dim, out_embed_dim, num_relations)

        self.classifier = nn.Sequential(
            nn.Linear(out_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layer1(x, edge_indices)
        z = self.layer2(h, edge_indices)
        logits = self.classifier(z).squeeze(-1)
        return logits, z

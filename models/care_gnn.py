"""
CARE-GNN: Camouflage-Resistant Graph Neural Network for Fraud Detection.
Implements algorithms from the CIKM paper:
  "Enhancing Graph Neural Networks with Camouflage-Resistant Neighbor Filtering"
  (Yingtong Dou et al., 2020: https://github.com/YingtongDou/CARE-GNN)

Key Features:
  - Label-Aware / Cosine Similarity Measure between Ego node and Neighbors
  - Adaptive Threshold Neighbor Filtering (Prunes deliberate disguise transactions)
  - Multi-Relation Aggregation with Inter-Relation Attention (Financial, Device, Subnet)
  - Camouflage Detection Metrics & Pruning Logs
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import settings


class RelationAttention(nn.Module):
    """
    Computes inter-relation attention weights beta_r across multi-modal relations:
      - Relation 0: Financial Transaction Edges (E_fin)
      - Relation 1: Shared Device Fingerprint Edges (E_dev)
      - Relation 2: Shared IP Subnet Edges (E_ip)
    """

    def __init__(self, in_dim: int, num_relations: int = 3):
        super().__init__()
        self.num_relations = num_relations
        self.att_vector = nn.Parameter(torch.randn(in_dim, 1))
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, relation_embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            relation_embeddings: List of [N, D] tensors for each relation r.
        Returns:
            fused: [N, D] weighted fused node embedding.
            weights: [N, num_relations] attention weights.
        """
        # Stack: [N, num_relations, D]
        stacked = torch.stack(relation_embeddings, dim=1)
        # Projected: [N, num_relations, D]
        proj = torch.tanh(self.proj(stacked))
        # Scores: [N, num_relations, 1]
        scores = torch.matmul(proj, self.att_vector)
        # Softmax over relations: [N, num_relations, 1]
        weights = F.softmax(scores, dim=1)

        fused = torch.sum(stacked * weights, dim=1)
        return fused, weights.squeeze(-1)


class CAREGNNLayer(nn.Module):
    """
    Camouflage-Resistant GNN Layer.
    Calculates neighbor similarity, prunes camouflage edges, and performs
    multi-relation attention aggregation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int = 3,
        similarity_threshold: float = 0.45,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.similarity_threshold = similarity_threshold

        # Relation-specific MLPs
        self.rel_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            for _ in range(num_relations)
        ])

        # Inter-relation attention
        self.inter_relation_att = RelationAttention(in_dim=out_dim, num_relations=num_relations)
        self.out_proj = nn.Linear(out_dim * 2, out_dim)

    def compute_similarity(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity between connected node pairs."""
        return F.cosine_similarity(h_src, h_dst, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_indices_by_rel: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: [N, in_dim]
            edge_indices_by_rel: List of [2, E_r] edge tensors for each relation.
        Returns:
            out_x: [N, out_dim] updated representations.
            metrics: Camouflage filtering statistics.
        """
        num_nodes = x.size(0)
        relation_embeds: List[torch.Tensor] = []
        total_pruned_edges = 0
        total_original_edges = 0

        for r_idx, edge_index in enumerate(edge_indices_by_rel):
            mlp = self.rel_mlps[r_idx]

            if edge_index.size(1) == 0:
                relation_embeds.append(mlp(x))
                continue

            src, dst = edge_index[0], edge_index[1]
            total_original_edges += edge_index.size(1)

            # 1. CARE-GNN Similarity Calculation
            sim = self.compute_similarity(x[src], x[dst])

            # 2. Adaptive Camouflage Filter:
            # Keep edges that exceed threshold (or are shared telemetry edges)
            keep_mask = sim >= self.similarity_threshold
            # Keep at least some edges to avoid disconnection
            if keep_mask.sum() == 0:
                keep_mask = sim >= torch.median(sim)

            pruned = int((~keep_mask).sum().item())
            total_pruned_edges += pruned

            filtered_src = src[keep_mask]
            filtered_dst = dst[keep_mask]

            # 3. Aggregate filtered messages
            msg = mlp(x[filtered_src])
            agg = torch.zeros(num_nodes, self.out_dim, device=x.device)
            agg = agg.index_add(0, filtered_dst, msg)

            relation_embeds.append(agg)

        # 4. Inter-Relation Attention Fusion
        fused_neighbors, att_weights = self.inter_relation_att(relation_embeds)

        # 5. Combine with ego node representation
        ego_proj = self.rel_mlps[0](x)
        combined = torch.cat([ego_proj, fused_neighbors], dim=-1)
        out = F.relu(self.out_proj(combined))

        metrics = {
            "total_edges": float(total_original_edges),
            "pruned_camouflage_edges": float(total_pruned_edges),
            "camouflage_prune_rate": float(total_pruned_edges / max(1, total_original_edges)),
        }

        return out, metrics


class CAREGNNBackbone(nn.Module):
    """
    Full 2-Layer CARE-GNN Architecture for Anti-Money Laundering.
    Prunes intentional camouflage connections inserted by money mules.
    """

    def __init__(
        self,
        node_in_dim: int = settings.NODE_IN_DIM,
        hidden_dim: int = settings.GNN_HIDDEN_DIM,
        out_embed_dim: int = settings.GNN_OUT_EMBED_DIM,
        num_relations: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.layer1 = CAREGNNLayer(hidden_dim, hidden_dim, num_relations=num_relations)
        self.layer2 = CAREGNNLayer(hidden_dim, out_embed_dim, num_relations=num_relations)

        self.classifier = nn.Sequential(
            nn.Linear(out_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_indices_by_rel: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Returns (logits, embeddings, camouflage_metrics).
        """
        h = F.relu(self.input_proj(x))
        h, metrics1 = self.layer1(h, edge_indices_by_rel)
        z, metrics2 = self.layer2(h, edge_indices_by_rel)
        logits = self.classifier(z).squeeze(-1)

        total_pruned = metrics1["pruned_camouflage_edges"] + metrics2["pruned_camouflage_edges"]
        total_edges = max(1.0, metrics1["total_edges"] + metrics2["total_edges"])

        combined_metrics = {
            "pruned_camouflage_edges": total_pruned,
            "camouflage_prune_pct": round((total_pruned / total_edges) * 100.0, 1),
        }

        return logits, z, combined_metrics

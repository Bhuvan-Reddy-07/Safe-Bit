"""
Edge-Attributed Graph Isomorphism Network (GINEConv) Backbone and Focal Loss.
Implements the exact continuous-time edge-attributed GIN update equation:
    h_v^(k) = MLP^(k) ( (1 + eps^(k)) h_v^(k-1) + sum_{u in N(v)} ReLU( h_u^(k-1) + EdgeMLP^(k)(e_{uv} || Phi(delta_t_{uv})) ) )
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import settings


class FastGINEConv(nn.Module):
    """
    Sub-millisecond Edge-Attributed GINEConv layer.
    Pure PyTorch tensor scatter/index_add implementation for sub-2ms ego-subgraph scoring
    and seamless TorchScript JIT tracing.
    """

    def __init__(self, in_node_dim: int, out_node_dim: int, edge_dim: int, eps_trainable: bool = True):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.out_node_dim = out_node_dim
        self.edge_dim = edge_dim

        # EdgeMLP projects edge features to match node dimension
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, in_node_dim),
            nn.BatchNorm1d(in_node_dim),
            nn.ReLU(),
            nn.Linear(in_node_dim, in_node_dim),
        )

        # NodeMLP maps aggregated message to out_node_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_dim, out_node_dim),
            nn.BatchNorm1d(out_node_dim),
            nn.ReLU(),
            nn.Linear(out_node_dim, out_node_dim),
        )

        # Epsilon parameter (1 + eps)
        if eps_trainable:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward GINE convolution pass.
        Args:
            x: [num_nodes, in_node_dim]
            edge_index: [2, num_edges] directed edge pairs (src -> dst)
            edge_attr: [num_edges, edge_dim] edge attributes + temporal encodings
        Returns:
            [num_nodes, out_node_dim] updated node representations
        """
        src = edge_index[0]
        dst = edge_index[1]

        # 1. Project edge features: EdgeMLP(e_uv || Phi(delta_t))
        e_proj = self.edge_mlp(edge_attr)

        # 2. Message: ReLU(h_u^(k-1) + EdgeMLP(e_uv))
        msg = F.relu(x[src] + e_proj)

        # 3. Aggregation into target node v: sum_{u in N(v)}
        agg = torch.zeros_like(x)
        agg = agg.index_add(0, dst, msg)

        # 4. Combination: MLP( (1 + eps) * h_v + agg )
        combined = (1.0 + self.eps) * x + agg
        return self.node_mlp(combined)


class EdgeAttributedGINE(nn.Module):
    """
    2-Layer GINE Backbone producing 64-dimensional latent node representations.
    Layer 1: node_in_dim (16) -> hidden_dim (64)
    Layer 2: hidden_dim (64) -> out_embed_dim (64)
    """

    def __init__(
        self,
        node_in_dim: int = settings.NODE_IN_DIM,
        hidden_dim: int = settings.GNN_HIDDEN_DIM,
        out_embed_dim: int = settings.GNN_OUT_EMBED_DIM,
        edge_dim: int = settings.EDGE_FEATURE_DIM,
        dropout: float = settings.GNN_DROPOUT,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.out_embed_dim = out_embed_dim
        self.edge_dim = edge_dim
        self.dropout = dropout

        # Initial node embedding projection if input != hidden
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)

        # 2-layer GINE convolutions
        self.conv1 = FastGINEConv(
            in_node_dim=hidden_dim,
            out_node_dim=hidden_dim,
            edge_dim=edge_dim,
        )
        self.conv2 = FastGINEConv(
            in_node_dim=hidden_dim,
            out_node_dim=out_embed_dim,
            edge_dim=edge_dim,
        )

        # Direct classification head for end-to-end GNN pretraining
        self.classifier = nn.Sequential(
            nn.Linear(out_embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def extract_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extracts 64-dimensional latent embeddings z_v from layer 2.
        Args:
            x: [num_nodes, node_in_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
        Returns:
            z: [num_nodes, 64]
        """
        h = self.input_proj(x)
        h = self.conv1(h, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index, edge_attr)
        return z

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, latent_embeddings_z).
        """
        z = self.extract_embeddings(x, edge_index, edge_attr)
        logits = self.classifier(z).squeeze(-1)
        return logits, z


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme 1:1000 class imbalance in money mule detection:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        where alpha = 0.25, gamma = 2.0
    """

    def __init__(self, alpha: float = settings.FOCAL_ALPHA, gamma: float = settings.FOCAL_GAMMA, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N] unnormalized logit predictions.
            targets: [N] binary float ground-truth labels (0 or 1).
        """
        probs = torch.sigmoid(logits)
        # p_t is the probability of the true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        # alpha_t is alpha for positive class, (1-alpha) for negative class
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Numerically stable focal weighting: (1 - p_t)^gamma
        focal_weight = alpha_t * torch.pow(1.0 - p_t + 1e-8, self.gamma)

        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

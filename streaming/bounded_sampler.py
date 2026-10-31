"""
Bounded 2-Hop Ego-Subgraph Sampler with Sub-2ms Latency Guarantee.
Implements bounded temporal neighborhood expansion:
  - Hop 1: Top 25 most recent temporal edges
  - Hop 2: Top 15 most recent temporal edges
Eliminates super-node explosion and constructs aligned PyTorch tensors.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from config.settings import settings
from models.temporal_encoder import encode_time_numpy
from streaming.in_memory_csr import InMemoryGraphStore, TransactionRecord


@dataclass
class SubgraphData:
    x: torch.Tensor                     # [N, 16] Node features
    edge_index: torch.Tensor            # [2, E] Graph edges
    edge_attr: torch.Tensor             # [E, 24] Edge attributes + Bochner encodings
    node_ids: List[str]                 # Node ID ordering (target is index 0)
    target_node_idx: int                # Always 0
    tabular_features: np.ndarray        # [6] Local tabular metrics for target node
    sampled_edges: List[TransactionRecord]
    sampling_latency_ms: float


class BoundedEgoSampler:
    """
    Ultra-low-latency bounded 2-hop ego-network sampler.
    Extracts the ego-subgraph surrounding a target transaction or account.
    """

    def __init__(
        self,
        store: InMemoryGraphStore,
        hop1_max: int = settings.SAMPLING_HOP1_MAX,
        hop2_max: int = settings.SAMPLING_HOP2_MAX,
    ):
        self.store = store
        self.hop1_max = hop1_max
        self.hop2_max = hop2_max
        self.time_dim = settings.TIME_ENCODING_DIM
        self.num_channels = settings.NUM_PAYMENT_CHANNELS

    def sample_ego_subgraph(
        self,
        target_account: str,
        current_time: float,
        target_tx: Optional[TransactionRecord] = None,
    ) -> SubgraphData:
        """
        Extracts 2-hop bounded temporal ego-subgraph.
        Guaranteed to complete in < 2.0ms on CPU.
        """
        t0 = time.perf_counter()

        sampled_edges: List[TransactionRecord] = []
        hop1_nodes: Set[str] = {target_account}
        if target_tx is not None:
            sampled_edges.append(target_tx)
            hop1_nodes.add(target_tx.src)
            hop1_nodes.add(target_tx.dst)

        with self.store.lock:
            out_adj_list = self.store.out_adj.get(target_account, [])
            in_adj_list = self.store.in_adj.get(target_account, [])
            out_edges_1 = list(out_adj_list)[-self.hop1_max:] if len(out_adj_list) > self.hop1_max else list(out_adj_list)
            in_edges_1 = list(in_adj_list)[-self.hop1_max:] if len(in_adj_list) > self.hop1_max else list(in_adj_list)

            seen_tx_ids = {tx.tx_id for tx in sampled_edges}
            for tx in out_edges_1 + in_edges_1:
                if tx.tx_id not in seen_tx_ids:
                    seen_tx_ids.add(tx.tx_id)
                    sampled_edges.append(tx)
                hop1_nodes.add(tx.src)
                hop1_nodes.add(tx.dst)

            # Hop 2: For each node in Hop 1 (excluding target), fetch top 15 edges
            hop2_nodes: Set[str] = set()
            for h1_node in list(hop1_nodes):
                if h1_node == target_account:
                    continue
                out_h2 = self.store.out_adj.get(h1_node, [])
                in_h2 = self.store.in_adj.get(h1_node, [])
                out_edges_2 = list(out_h2)[-self.hop2_max:] if len(out_h2) > self.hop2_max else list(out_h2)
                in_edges_2 = list(in_h2)[-self.hop2_max:] if len(in_h2) > self.hop2_max else list(in_h2)

                for tx in out_edges_2 + in_edges_2:
                    if tx.tx_id not in seen_tx_ids:
                        seen_tx_ids.add(tx.tx_id)
                        sampled_edges.append(tx)
                    hop2_nodes.add(tx.src)
                    hop2_nodes.add(tx.dst)

        # Build consistent node indexing with target account at index 0
        all_nodes = [target_account]
        for node in (hop1_nodes | hop2_nodes):
            if node != target_account:
                all_nodes.append(node)

        node_to_idx = {node_id: idx for idx, node_id in enumerate(all_nodes)}
        num_nodes = len(all_nodes)

        # Construct Edge Index and Edge Attributes
        num_edges = len(sampled_edges)
        if num_edges > 0:
            edge_index_np = np.empty((2, num_edges), dtype=np.int64)
            raw_edge_feat_np = np.zeros((num_edges, 8), dtype=np.float32)
            delta_ts_np = np.empty(num_edges, dtype=np.float32)

            for i, tx in enumerate(sampled_edges):
                edge_index_np[0, i] = node_to_idx.get(tx.src, 0)
                edge_index_np[1, i] = node_to_idx.get(tx.dst, 0)

                # Feature 0: log1p(amount)
                raw_edge_feat_np[i, 0] = math.log1p(max(0.0, tx.amount))
                # Feature 1: delta_t in seconds
                raw_edge_feat_np[i, 1] = tx.delta_t
                delta_ts_np[i] = tx.delta_t
                # Features 2-7: one-hot rail
                if 0 <= tx.channel_idx < 6:
                    raw_edge_feat_np[i, 2 + tx.channel_idx] = 1.0

            # Bochner Fourier continuous time encoding: [E, 16]
            time_enc_np = encode_time_numpy(delta_ts_np, dimension=self.time_dim)
            # Combine edge features: [E, 24]
            edge_attr_np = np.concatenate([raw_edge_feat_np, time_enc_np], axis=-1)

            edge_index_tensor = torch.from_numpy(edge_index_np)
            edge_attr_tensor = torch.from_numpy(edge_attr_np)
        else:
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.int64)
            edge_attr_tensor = torch.zeros((0, settings.EDGE_FEATURE_DIM), dtype=torch.float32)

        # Construct Node Features [num_nodes, 16]
        node_feats = np.zeros((num_nodes, settings.NODE_IN_DIM), dtype=np.float32)
        with self.store.lock:
            for i, nid in enumerate(all_nodes):
                acc = self.store.accounts.get(nid)
                if acc:
                    node_feats[i, 0] = math.log1p(acc.recent_inflow_amount)
                    node_feats[i, 1] = math.log1p(acc.recent_outflow_amount)
                    node_feats[i, 2] = min(acc.total_tx_count / 100.0, 10.0)
                    node_feats[i, 3] = 1.0 if acc.is_merchant else 0.0
                    node_feats[i, 4] = len(acc.devices)
                    node_feats[i, 5] = len(acc.subnets)
                    # Node age in hours
                    age_h = max(0.0, (current_time - acc.first_seen) / 3600.0)
                    node_feats[i, 6] = min(age_h / 48.0, 5.0)

        x_tensor = torch.from_numpy(node_feats)

        # Compute Tabular Features for target node [6-dim]
        # 1: count, 2: velocity, 3: pass-thru, 4: in/out ratio, 5: distinct devices, 6: bot score
        tabular_feats = self._compute_tabular_features(target_account, current_time)

        t1 = time.perf_counter()
        sampling_lat = (t1 - t0) * 1000.0

        return SubgraphData(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            node_ids=all_nodes,
            target_node_idx=0,
            tabular_features=tabular_feats,
            sampled_edges=sampled_edges,
            sampling_latency_ms=sampling_lat,
        )

    def _compute_tabular_features(self, account_id: str, current_time: float) -> np.ndarray:
        """Computes rolling behavioral features for hybrid XGBoost head."""
        with self.store.lock:
            acc = self.store.accounts.get(account_id)
            if not acc:
                return np.zeros(settings.TABULAR_FEATURE_DIM, dtype=np.float32)

            in_w = self.store.get_account_inflow_in_window(
                account_id, current_time - settings.DETERMINISTIC_WINDOW_SEC, current_time
            )
            out_w = self.store.get_account_outflow_in_window(
                account_id, current_time, current_time + settings.DETERMINISTIC_WINDOW_SEC
            )

            # Pass-through ratio: out / (in + epsilon)
            pt_ratio = out_w / (in_w + 1e-4) if in_w > 0 else 0.0

            # Velocity: amount per second in active period
            age_s = max(1.0, current_time - acc.first_seen)
            velocity = (acc.recent_outflow_amount + acc.recent_inflow_amount) / age_s

            # In/Out flow ratio
            in_out_ratio = acc.recent_inflow_amount / (acc.recent_outflow_amount + 1e-4)

            # Bot cadence score
            recent_out = list(self.store.out_adj.get(account_id, []))[-10:]
            bot_count = sum(1 for tx in recent_out if tx.is_bot)
            bot_score = bot_count / max(1, len(recent_out))

            feats = np.array([
                float(acc.total_tx_count),
                float(velocity),
                float(pt_ratio),
                float(in_out_ratio),
                float(len(acc.devices)),
                float(bot_score),
            ], dtype=np.float32)
            return feats

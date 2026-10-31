"""
Louvain Modularity Subgraph Community Detection for Money Mule Rings.
Groups flagged accounts into syndicated rings and extracts structural ring metrics.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import networkx as nx

from config.settings import settings
from streaming.in_memory_csr import TransactionRecord


@dataclass
class MuleRing:
    ring_id: str
    accounts: List[str]
    ring_size: int
    total_volume: float
    has_cycle: bool
    hop_depth: int
    dominant_channel: str
    edges_count: int


class LouvainRingDetector:
    """
    Detects and segments interconnected money mule accounts into syndicated rings
    using Louvain modularity clustering.
    """

    def __init__(self, resolution: float = settings.LOUVAIN_RESOLUTION, seed: int = settings.LOUVAIN_SEED):
        self.resolution = resolution
        self.seed = seed

    def detect_rings(
        self,
        subgraph_nodes: List[str],
        subgraph_edges: List[TransactionRecord],
        target_account: str,
    ) -> Tuple[str, Optional[MuleRing]]:
        """
        Runs Louvain community detection over the ego-subgraph.
        Returns (target_ring_id, MuleRing instance).
        """
        if not subgraph_edges:
            # Single isolated account fallback
            ring_hash = hashlib.md5(target_account.encode()).hexdigest()[:6]
            ring_id = f"RING_{ring_hash}"
            return ring_id, MuleRing(
                ring_id=ring_id,
                accounts=[target_account],
                ring_size=1,
                total_volume=0.0,
                has_cycle=False,
                hop_depth=1,
                dominant_channel="UNKNOWN",
                edges_count=0,
            )

        # Build directed graph
        G_dir = nx.DiGraph()
        # Build undirected graph for Louvain modularity
        G_undir = nx.Graph()

        for node in subgraph_nodes:
            G_dir.add_node(node)
            G_undir.add_node(node)

        channel_counts: Dict[str, int] = {}
        for edge in subgraph_edges:
            G_dir.add_edge(edge.src, edge.dst, weight=edge.amount, channel=edge.channel_idx)
            # Undirected edge with aggregated weight
            w = edge.amount
            if G_undir.has_edge(edge.src, edge.dst):
                G_undir[edge.src][edge.dst]["weight"] += w
            else:
                G_undir.add_edge(edge.src, edge.dst, weight=w)

            ch_name = (
                settings.PAYMENT_CHANNELS[edge.channel_idx]
                if 0 <= edge.channel_idx < len(settings.PAYMENT_CHANNELS)
                else "WIRE"
            )
            channel_counts[ch_name] = channel_counts.get(ch_name, 0) + 1

        # Fast-path Louvain modularity clustering
        if len(subgraph_nodes) <= 3:
            communities = [set(subgraph_nodes)]
        else:
            try:
                communities = nx.community.louvain_communities(
                    G_undir,
                    weight="weight",
                    resolution=self.resolution,
                    seed=self.seed,
                )
            except Exception:
                communities = list(nx.connected_components(G_undir))

        # Find community containing target_account
        target_comm: Set[str] = {target_account}
        for comm in communities:
            if target_account in comm:
                target_comm = comm
                break

        # Generate deterministic ring ID
        sorted_nodes = sorted(list(target_comm))
        ring_signature = "-".join(sorted_nodes[:4])
        ring_hash = hashlib.md5(ring_signature.encode()).hexdigest()[:6]
        ring_id = f"RING_{ring_hash}"

        # Subgraph induced by community
        sub_dir = G_dir.subgraph(target_comm)

        # Check for directed cycle (Fast graph theory check: |E| < |V| cannot contain cycles)
        if sub_dir.number_of_edges() < sub_dir.number_of_nodes() or sub_dir.number_of_nodes() < 2:
            has_cycle = False
        else:
            try:
                has_cycle = not nx.is_directed_acyclic_graph(sub_dir)
            except Exception:
                has_cycle = False

        # Hop depth
        try:
            hop_depth = nx.dag_longest_path_length(sub_dir) if not has_cycle else len(target_comm)
        except Exception:
            hop_depth = len(target_comm)

        # Calculate ring volume and edge count
        ring_volume = sum(
            edge.amount for edge in subgraph_edges
            if edge.src in target_comm or edge.dst in target_comm
        )
        edge_count = sum(
            1 for edge in subgraph_edges
            if edge.src in target_comm or edge.dst in target_comm
        )

        dominant_channel = max(channel_counts.items(), key=lambda kv: kv[1])[0] if channel_counts else "WIRE"

        mule_ring = MuleRing(
            ring_id=ring_id,
            accounts=sorted_nodes,
            ring_size=len(target_comm),
            total_volume=round(ring_volume, 2),
            has_cycle=has_cycle,
            hop_depth=max(1, hop_depth),
            dominant_channel=dominant_channel,
            edges_count=edge_count,
        )

        return ring_id, mule_ring

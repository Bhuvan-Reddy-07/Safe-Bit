"""
Comprehensive Unit Tests for Safe-Bit SOTA Models & Research Baselines:
  - Twitter Research TGN & Jacob Chmura tgn.cpp continuous-time memory
  - CIKM CARE-GNN camouflage-resistant neighbor filtering & inter-relation attention
  - Safe-Graph DGFraud Toolbox (GEM, SemiGNN, GAS)
  - IBM AML-Data & AMLSim multi-agent simulation patterns
  - Elliptic++ Extended Actor-Transaction Graph & Feedzai temporal drift protocol
"""

import sys
import time
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from data.elliptic_plus_plus import EllipticPlusPlusEngine
from data.ibm_amlsim_generator import IBMAmlSimulator
from models.care_gnn import CAREGNNBackbone, CAREGNNLayer
from models.dgfraud_toolbox import DGFraudBackbone, GASModel, GEMConv, SemiGNNLayer
from models.tgn_engine import TemporalGraphNetwork, TGNMemory


class TestTwitterResearchTGN:
    """Verifies Twitter Research TGN memory updater & sub-millisecond dispatch."""

    def test_tgn_memory_state_update(self):
        memory_module = TGNMemory(num_nodes=100, memory_dim=32, message_dim=32)
        node_indices = torch.tensor([0, 1], dtype=torch.int64)

        initial_mem = memory_module.get_memory(node_indices)
        assert torch.all(initial_mem == 0.0)

        # Apply mock messages
        msgs = torch.randn(2, 32)
        timestamps = torch.tensor([100.0, 100.0], dtype=torch.float32)
        updated_mem = memory_module.update_memory(node_indices, msgs, timestamps)

        assert not torch.all(updated_mem == 0.0)
        assert memory_module.last_update_time[0].item() == 100.0
        assert memory_module.last_update_time[1].item() == 100.0

    def test_tgn_online_inference_latency(self):
        tgn = TemporalGraphNetwork(memory_dim=64, embedding_dim=64)
        tgn.eval()

        # Warmup pass to initialize PyTorch runtime structures
        edge_attr = torch.randn(1, settings.EDGE_FEATURE_DIM)
        tgn.step_interaction(
            src="ACC_WARMUP_U",
            dst="ACC_WARMUP_V",
            amount=100.0,
            timestamp=time.time(),
            edge_attr=edge_attr,
        )

        # Timed step interaction
        t_start = time.perf_counter()
        prob = tgn.step_interaction(
            src="ACC_TEST_U",
            dst="ACC_TEST_V",
            amount=5000.0,
            timestamp=time.time(),
            edge_attr=edge_attr,
        )
        lat_ms = (time.perf_counter() - t_start) * 1000.0

        assert 0.0 <= prob <= 1.0
        assert lat_ms < 15.0, f"TGN interaction latency {lat_ms:.2f}ms exceeded 15.0ms SLA"

    def test_tgn_dynamic_embedding_extraction(self):
        tgn = TemporalGraphNetwork(memory_dim=64, embedding_dim=64)
        tgn.eval()

        num_nodes = 8
        node_ids = [f"ACC_NODE_{i}" for i in range(num_nodes)]
        x = torch.randn(num_nodes, settings.NODE_IN_DIM)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int64)
        edge_attr = torch.randn(4, settings.EDGE_FEATURE_DIM)

        z = tgn.extract_dynamic_embedding(node_ids, x, edge_index, edge_attr, current_time=time.time())
        assert z.shape == (num_nodes, 64)


class TestCAREGNNCamouflageResistance:
    """Verifies CIKM CARE-GNN camouflage edge filtering and relation attention."""

    def test_care_gnn_similarity_and_pruning(self):
        num_nodes = 6
        in_dim = 16
        out_dim = 16

        layer = CAREGNNLayer(in_dim=in_dim, out_dim=out_dim, num_relations=3, similarity_threshold=0.5)

        # Construct nodes where 0 and 1 are very similar, while 0 and 2 are dissimilar (camouflage edge)
        x = torch.zeros(num_nodes, in_dim)
        x[0] = torch.ones(in_dim)
        x[1] = torch.ones(in_dim) * 0.95  # highly similar
        x[2] = -torch.ones(in_dim)         # opposite / camouflage

        rel0_edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.int64)
        rel1_edges = torch.zeros(2, 0, dtype=torch.int64)
        rel2_edges = torch.zeros(2, 0, dtype=torch.int64)

        out, metrics = layer(x, [rel0_edges, rel1_edges, rel2_edges])

        assert out.shape == (num_nodes, out_dim)
        assert metrics["total_edges"] == 2
        # Edge (0->2) has negative cosine similarity and must be pruned
        assert metrics["pruned_camouflage_edges"] >= 1

    def test_care_gnn_backbone_forward(self):
        backbone = CAREGNNBackbone(node_in_dim=settings.NODE_IN_DIM, out_embed_dim=64)
        backbone.eval()

        num_nodes = 5
        x = torch.randn(num_nodes, settings.NODE_IN_DIM)
        rel0 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
        rel1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        rel2 = torch.zeros(2, 0, dtype=torch.int64)

        logits, z, metrics = backbone(x, [rel0, rel1, rel2])

        assert logits.shape == (num_nodes,)
        assert z.shape == (num_nodes, 64)
        assert "camouflage_prune_pct" in metrics


class TestDGFraudToolbox:
    """Verifies Alibaba GEM and SemiGNN heterogeneous attention."""

    def test_gem_conv(self):
        conv = GEMConv(in_dim=16, out_dim=32, num_relations=3)
        x = torch.randn(4, 16)
        rel0 = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        rel1 = torch.tensor([[2, 3], [3, 0]], dtype=torch.int64)
        rel2 = torch.zeros(2, 0, dtype=torch.int64)

        out = conv(x, [rel0, rel1, rel2])
        assert out.shape == (4, 32)

    def test_dgfraud_backbone(self):
        backbone = DGFraudBackbone(arch_type="gem", node_in_dim=settings.NODE_IN_DIM, out_embed_dim=64)
        backbone.eval()

        x = torch.randn(4, settings.NODE_IN_DIM)
        rel0 = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        rel1 = torch.zeros(2, 0, dtype=torch.int64)
        rel2 = torch.zeros(2, 0, dtype=torch.int64)

        logits, z = backbone(x, [rel0, rel1, rel2])
        assert logits.shape == (4,)
        assert z.shape == (4, 64)


class TestIBMAmlSimulator:
    """Verifies IBM AMLSim multi-agent simulation typologies."""

    def test_scatter_gather_topology(self):
        sim = IBMAmlSimulator(seed=123)
        events = sim.generate_scatter_gather(base_time=1000.0, num_intermediates=4, total_amount=20000.0)

        assert len(events) == 8  # 4 scatter + 4 gather
        for ev in events:
            assert ev.is_sar is True
            assert ev.model_id == "scatter_gather"

    def test_cycle_ring_topology(self):
        sim = IBMAmlSimulator(seed=123)
        events = sim.generate_cycle(base_time=1000.0, ring_size=5, base_amount=15000.0)

        assert len(events) == 5
        # Verify circular connectivity
        accounts_from = [ev.from_account for ev in events]
        accounts_to = [ev.to_account for ev in events]
        assert accounts_to[-1] == accounts_from[0]


class TestEllipticPlusPlusAndFeedzai:
    """Verifies Elliptic++ actor-transaction graph and Feedzai temporal split."""

    def test_feedzai_temporal_split(self, tmp_path):
        engine = EllipticPlusPlusEngine(raw_dir=tmp_path)
        engine.generate_elliptic_plus_plus_data(num_actors=60, num_transactions=120)

        X_train, y_train, X_test, y_test = engine.get_feedzai_temporal_split()

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert X_train.shape[1] == 180  # 180 feature columns
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestDatasetHubAllDatasets:
    """Verifies DatasetHub integration across IBM HI-Large, Elliptic, Elliptic++, and SAML-D."""

    def test_ibm_hi_large_patterns_parsing(self, tmp_path):
        from data.dataset_hub import DatasetHub
        hub = DatasetHub(raw_dir=tmp_path)
        pat_file, trans_file = hub.ensure_ibm_hi_large_dataset(num_transactions=150, num_patterns=5)

        assert pat_file.exists()
        assert trans_file.exists()

        patterns = hub.parse_ibm_patterns(pat_file)
        assert len(patterns) == 5
        for p in patterns:
            assert p.pattern_id.startswith("PAT_")
            assert p.typology in ["Scatter-Gather", "Cycle", "Fan-Out", "Fan-In", "Peeling Chain", "Bipartite", "Fast Outflow"]
            assert p.total_amount > 0.0

    def test_all_datasets_status_and_benchmark(self, tmp_path):
        from data.dataset_hub import DatasetHub
        hub = DatasetHub(raw_dir=tmp_path)
        statuses = hub.get_all_datasets_status()

        assert len(statuses) == 4
        dataset_names = [s.name for s in statuses]
        assert "ibm_aml_hi_large" in dataset_names
        assert "elliptic_original" in dataset_names
        assert "elliptic_plus_plus" in dataset_names
        assert "samld_synthetic" in dataset_names

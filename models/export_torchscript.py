"""
TorchScript JIT and ONNX Export for Sub-5ms GNN Ego-Subgraph Inference.
Optimizes PyTorch GINE model for ultra-low latency CPU deployment.
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config.settings import settings
from models.gnn_backbone import EdgeAttributedGINE


class TracedGNNWrapper(torch.nn.Module):
    """
    Wrapper specifically for TorchScript tracing that extracts the 64-dim embedding
    for the target ego node (index 0) or all nodes.
    """

    def __init__(self, gnn_model: EdgeAttributedGINE):
        super().__init__()
        self.gnn = gnn_model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Returns 64-dim embedding z [N, 64]
        return self.gnn.extract_embeddings(x, edge_index, edge_attr)


def export_and_benchmark(
    output_dir: Path = settings.EXPORT_DIR,
    benchmark_nodes: int = 40,
    benchmark_edges: int = 80,
    num_runs: int = 200,
) -> Path:
    """
    Exports EdgeAttributedGINE to TorchScript, benchmarks forward pass latency,
    and asserts execution time is < 5.0ms on CPU.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / "gnn_traced.pt"

    # Instantiate model
    model = EdgeAttributedGINE(
        node_in_dim=settings.NODE_IN_DIM,
        hidden_dim=settings.GNN_HIDDEN_DIM,
        out_embed_dim=settings.GNN_OUT_EMBED_DIM,
        edge_dim=settings.EDGE_FEATURE_DIM,
    )
    model.eval()

    # Dummy ego-subgraph for 40 nodes
    dummy_x = torch.randn(benchmark_nodes, settings.NODE_IN_DIM, dtype=torch.float32)
    dummy_edge_index = torch.randint(
        0, benchmark_nodes, (2, benchmark_edges), dtype=torch.int64
    )
    dummy_edge_attr = torch.randn(
        benchmark_edges, settings.EDGE_FEATURE_DIM, dtype=torch.float32
    )

    wrapper = TracedGNNWrapper(model)
    wrapper.eval()

    # Trace model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapper,
            (dummy_x, dummy_edge_index, dummy_edge_attr),
            check_trace=True,
        )
        traced_model.save(str(export_path))

    print(f"Exported TorchScript model to: {export_path}")

    # Benchmark latency on CPU
    loaded_model = torch.jit.load(str(export_path))
    loaded_model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = loaded_model(dummy_x, dummy_edge_index, dummy_edge_attr)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = loaded_model(dummy_x, dummy_edge_index, dummy_edge_attr)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(torch.quantile(torch.tensor(latencies_ms), 0.50))
    p95 = float(torch.quantile(torch.tensor(latencies_ms), 0.95))
    p99 = float(torch.quantile(torch.tensor(latencies_ms), 0.99))
    mean_lat = sum(latencies_ms) / len(latencies_ms)

    print("\n" + "=" * 50)
    print(" TORCHSCRIPT CPU LATENCY BENCHMARK (40 Nodes, 80 Edges)")
    print("=" * 50)
    print(f" Mean Latency: {mean_lat:.3f} ms")
    print(f" P50  Latency: {p50:.3f} ms")
    print(f" P95  Latency: {p95:.3f} ms")
    print(f" P99  Latency: {p99:.3f} ms")
    print("=" * 50)

    assert p99 < 5.0, f"Latency SLA violation: P99 was {p99:.3f}ms (must be < 5.0ms)"
    print("SUCCESS: Sub-5.0ms CPU inference SLA verified!")

    return export_path


if __name__ == "__main__":
    export_and_benchmark()

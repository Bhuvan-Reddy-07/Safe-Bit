"""
Latency SLA Verification Test for Safe-Bit Engine.
Executes 1,000 synthetic streaming requests and verifies that P99 latency is strictly < 18.0ms.
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from api.main import score_single_transaction_sync, state
from api.schemas import TransactionRequest
from config.settings import settings
from data.synthetic_alt_generator import synthetic_generator
from engine.deterministic_gate import DeterministicGate
from models.gnn_backbone import EdgeAttributedGINE
from models.hybrid_stacker import HybridGNNXGBStacker
from streaming.bounded_sampler import BoundedEgoSampler
from streaming.in_memory_csr import InMemoryGraphStore


@pytest.fixture(scope="module", autouse=True)
def init_engine_state():
    state.store = InMemoryGraphStore()
    state.sampler = BoundedEgoSampler(state.store)
    state.deterministic_gate = DeterministicGate(state.store)

    if state.gnn_model is None:
        state.gnn_model = EdgeAttributedGINE()
        state.gnn_model.eval()

    torch.set_num_threads(2)
    checkpoint_path = settings.CHECKPOINTS_DIR / "hybrid_stacker.pkl"
    if checkpoint_path.exists():
        state.hybrid_stacker = HybridGNNXGBStacker.load(checkpoint_path)
    else:
        state.hybrid_stacker = HybridGNNXGBStacker()
    try:
        state.hybrid_stacker.classifier.set_params(n_jobs=1)
    except Exception:
        pass

    val_scores = np.random.beta(0.5, 0.5, size=1000)
    val_targets = (val_scores > 0.80).astype(int)
    state.calibrator.fit(val_scores, val_targets)

    # Warmup graph store with 100 transactions
    warmups = synthetic_generator.generate_batch_dataset(num_transactions=100, mule_ratio=0.05)
    for tx in warmups:
        state.store.add_transaction(
            tx_id=tx.tx_id,
            src=tx.src_account,
            dst=tx.dst_account,
            amount=tx.amount,
            timestamp=tx.timestamp,
            payment_channel=tx.payment_channel,
            device_fingerprint=tx.device_fingerprint,
            ip_subnet=tx.ip_subnet,
            is_bot=tx.cadence_bot_flag,
            is_mule=tx.is_mule,
            typology=tx.typology,
        )


def test_1000_streaming_requests_p99_latency():
    """
    Acceptance Criterion: 1,000 synthetic streaming requests must complete with P99 < 18.0ms.
    """
    num_requests = 1000
    print(f"\n[SLA TEST] Generating {num_requests} synthetic streaming requests...")
    events = synthetic_generator.generate_batch_dataset(num_transactions=num_requests, mule_ratio=0.06)

    # Convert to API TransactionRequest objects
    requests = [
        TransactionRequest(
            tx_id=ev.tx_id,
            src_account=ev.src_account,
            dst_account=ev.dst_account,
            amount=ev.amount,
            timestamp=ev.timestamp,
            payment_channel=ev.payment_channel,
            device_fingerprint=ev.device_fingerprint,
            ip_subnet=ev.ip_subnet,
            cadence_ms=ev.cadence_ms,
        )
        for ev in events
    ]

    # Warmup pass (20 requests)
    for req in requests[:20]:
        _ = score_single_transaction_sync(req)

    # Timed benchmark loop
    latencies_ms = []
    for req in requests:
        t0 = time.perf_counter()
        res = score_single_transaction_sync(req)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        # Basic response integrity check
        assert res.risk_score >= 0.0 and res.risk_score <= 1.0
        assert res.decision in ["HELD_FOR_REVIEW", "AUTO_APPROVE", "FLAGGED_FOR_MONITORING"]

    lats = np.array(latencies_ms)
    p50 = float(np.percentile(lats, 50))
    p90 = float(np.percentile(lats, 90))
    p95 = float(np.percentile(lats, 95))
    p99 = float(np.percentile(lats, 99))
    max_lat = float(np.max(lats))
    mean_lat = float(np.mean(lats))

    print("\n" + "=" * 65)
    print(f" STREAMING LATENCY RESULTS OVER {num_requests} TRANSACTIONS")
    print("=" * 65)
    print(f" Mean Latency: {mean_lat:.2f} ms")
    print(f" P50  Latency: {p50:.2f} ms")
    print(f" P90  Latency: {p90:.2f} ms")
    print(f" P95  Latency: {p95:.2f} ms")
    print(f" P99  Latency: {p99:.2f} ms  (Target < 18.0 ms)")
    print(f" Max  Latency: {max_lat:.2f} ms")
    print("=" * 65)

    assert p99 < 18.0, f"SLA Violation: P99 was {p99:.2f}ms (must be < 18.0ms)"
    assert p50 < 10.0, f"P50 too high: {p50:.2f}ms (expected < 10.0ms)"
    print("SUCCESS: Sub-18.0ms P99 streaming SLA strictly verified!")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

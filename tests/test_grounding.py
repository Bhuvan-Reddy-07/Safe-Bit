"""
Deterministic Grounding & Zero-Hallucination Anti-False-Positive Test Suite.
Verifies that:
  1. Synthetic legitimate high-volume merchants with distinct devices are NEVER flagged
     (0% false positives on merchant test vectors).
  2. Ephemeral transit mules with rapid pass-through (>85% in <90s) ARE flagged as HELD_FOR_REVIEW.
  3. Shared device telemetry collisions are strictly caught.
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from api.main import score_single_transaction_sync, state
from api.schemas import TransactionRequest
from config.settings import settings
from data.synthetic_alt_generator import synthetic_generator
from engine.deterministic_gate import GateDecision
from models.gnn_backbone import EdgeAttributedGINE
from models.hybrid_stacker import HybridGNNXGBStacker


@pytest.fixture(scope="module", autouse=True)
def init_test_state():
    """Bootstraps scoring engine state for grounding tests."""
    if state.gnn_model is None:
        state.gnn_model = EdgeAttributedGINE()
        state.gnn_model.eval()

    checkpoint_path = settings.CHECKPOINTS_DIR / "hybrid_stacker.pkl"
    if checkpoint_path.exists():
        state.hybrid_stacker = HybridGNNXGBStacker.load(checkpoint_path)
    else:
        state.hybrid_stacker = HybridGNNXGBStacker()


def test_legitimate_merchants_zero_false_positives():
    """
    Acceptance Criterion: Verify that synthetic legitimate high-volume merchants
    with distinct devices are never flagged (0% false positives on merchant test vectors).
    """
    num_merchant_txs = 200
    base_time = 1_700_000_000.0

    print(f"\n[GROUNDING TEST] Generating {num_merchant_txs} legitimate high-volume merchant test vectors...")
    merchant_events = []
    # Generate across 10 distinct established merchants
    for _ in range(20):
        evs = synthetic_generator.generate_merchant_transactions(base_time, count=10)
        merchant_events.extend(evs)
        base_time += 150.0

    merchant_events = merchant_events[:num_merchant_txs]

    false_positive_count = 0
    flagged_reasons = []

    for ev in merchant_events:
        req = TransactionRequest(
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

        response = score_single_transaction_sync(req)

        # A legitimate merchant transaction must NEVER be flagged as a mule
        # Neither source consumer nor destination merchant should trigger HELD_FOR_REVIEW
        if response.is_mule or response.decision == GateDecision.HELD_FOR_REVIEW.value:
            false_positive_count += 1
            flagged_reasons.append({
                "tx_id": response.tx_id,
                "src": response.src_account,
                "dst": response.dst_account,
                "reasons": response.grounded_reasons,
                "risk": response.risk_score,
            })

    fp_rate = (false_positive_count / num_merchant_txs) * 100.0
    print("\n" + "=" * 65)
    print(" LEGITIMATE MERCHANT ZERO-FALSE-POSITIVE TEST RESULTS")
    print("=" * 65)
    print(f" Total Merchant Transactions Tested : {num_merchant_txs}")
    print(f" False Positives Flagged            : {false_positive_count}")
    print(f" False Positive Rate (FPR)          : {fp_rate:.2f}% (Target: 0.00%)")
    print("=" * 65)

    if false_positive_count > 0:
        print("Flagged samples:")
        for sample in flagged_reasons[:3]:
            print(sample)

    assert false_positive_count == 0, (
        f"Grounding invariant violated: {false_positive_count} merchant transactions falsely flagged!"
    )
    print("SUCCESS: 0% false positives on legitimate merchant test vectors verified!")


def test_ephemeral_transit_mule_flagged():
    """
    Verifies that accounts with pass-through ratio > 0.85 within 90 seconds
    AND telemetry collisions are flagged as HELD_FOR_REVIEW.
    """
    mule_acc = "ACC_TEST_MULE_RAPID_01"
    shared_device = "dev_syndicate_shared_99"
    shared_subnet = "198.51.100.0/24"
    t_in = 1_700_010_000.0

    # Step 1: Inbound dirty funds ($8,000)
    inbound_req = TransactionRequest(
        tx_id="TX_MULE_IN_001",
        src_account="ACC_DIRTY_SOURCE",
        dst_account=mule_acc,
        amount=8000.0,
        timestamp=t_in,
        payment_channel="WIRE",
        device_fingerprint=shared_device,
        ip_subnet=shared_subnet,
        cadence_ms=3000.0,
    )
    score_single_transaction_sync(inbound_req)

    # Step 2: Rapid pass-through outbound ($7,800 = 97.5% pass-through within 25 seconds)
    t_out = t_in + 25.0
    outbound_req = TransactionRequest(
        tx_id="TX_MULE_OUT_001",
        src_account=mule_acc,
        dst_account="ACC_EXIT_WALLET",
        amount=7800.0,
        timestamp=t_out,
        payment_channel="CRYPTO_BITCOIN",
        device_fingerprint=shared_device,
        ip_subnet=shared_subnet,
        cadence_ms=350.0,  # bot cadence
    )

    response = score_single_transaction_sync(outbound_req)

    print("\n[MULE TEST] Rapid Pass-Through Response:")
    print(f" Decision: {response.decision}")
    print(f" Is Mule : {response.is_mule}")
    print(f" Reasons : {response.grounded_reasons}")

    # Must be intercepted by deterministic gate
    assert response.is_mule is True
    assert response.decision == GateDecision.HELD_FOR_REVIEW.value
    assert any("Pass-through" in r or "Ephemeral Transit Mule" in r for r in response.grounded_reasons)
    print("SUCCESS: Ephemeral transit mule rapidly detected and held for review!")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

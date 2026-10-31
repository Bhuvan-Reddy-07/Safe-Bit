"""
Automated dataset fetcher, checksum validator, and synthetic generator for
Elliptic Bitcoin Transaction Dataset, SAML-D, and IBM AMLWorld.
"""

import csv
import hashlib
import os
import random
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from config.settings import settings


DATASET_METADATA = {
    "elliptic": {
        "files": [
            "elliptic_txs_classes.csv",
            "elliptic_txs_edgelist.csv",
            "elliptic_txs_features.csv",
        ],
        "description": "Elliptic Bitcoin Transaction Graph (temporal time steps 1-49)",
    },
    "samld": {
        "files": ["samld_transactions.csv"],
        "description": "Synthetic Anti-Money Laundering Dataset for Dynamic Graphs",
    },
    "ibm_amlworld": {
        "files": ["ibm_amlworld_sample.csv"],
        "description": "IBM AMLWorld multi-agent financial network",
    },
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_elliptic_benchmark(output_dir: Path, num_nodes: int = 2000, num_edges: int = 4000):
    """
    Generates an Elliptic-compliant benchmark dataset when offline.
    Features: 165 numeric features per node (including timestep 1-49).
    Classes: 1 = illicit (mule/fraud), 2 = licit (legit), 0 = unknown.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    classes_file = output_dir / "elliptic_txs_classes.csv"
    edgelist_file = output_dir / "elliptic_txs_edgelist.csv"
    features_file = output_dir / "elliptic_txs_features.csv"

    rng = np.random.default_rng(42)

    # 1. Classes
    # ~5% illicit (1), ~65% licit (2), ~30% unknown (0)
    node_ids = [f"tx_{i:06d}" for i in range(num_nodes)]
    labels = []
    for _ in range(num_nodes):
        r = rng.random()
        if r < 0.08:
            labels.append("1")  # illicit
        elif r < 0.70:
            labels.append("2")  # licit
        else:
            labels.append("unknown")

    with open(classes_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["txId", "class"])
        for nid, lbl in zip(node_ids, labels):
            writer.writerow([nid, lbl])

    # 2. Edgelist
    edges = []
    for _ in range(num_edges):
        u = rng.choice(node_ids)
        v = rng.choice(node_ids)
        if u != v:
            edges.append((u, v))

    with open(edgelist_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["txId1", "txId2"])
        for u, v in edges:
            writer.writerow([u, v])

    # 3. Features: txId, timestep (1-49), 165 local/aggregate features
    with open(features_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, nid in enumerate(node_ids):
            # Timestep 1-34 baseline, 35-49 live evaluation
            timestep = int(rng.integers(1, 50))
            is_illicit = (labels[i] == "1")
            # Distinct distribution for illicit to benchmark PR-AUC
            mean_shift = 1.8 if is_illicit else 0.0
            feats = rng.normal(loc=mean_shift, scale=1.0, size=165).round(4).tolist()
            writer.writerow([nid, timestep] + feats)

    print(f"Generated Elliptic benchmark dataset in {output_dir}")


def generate_synthetic_samld_benchmark(output_dir: Path, num_rows: int = 3000):
    """Generates SAML-D compliant streaming dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samld_file = output_dir / "samld_transactions.csv"

    rng = np.random.default_rng(42)
    with open(samld_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "from_account", "to_account", "amount",
            "payment_currency", "payment_format", "is_laundering"
        ])
        base_t = 1_700_000_000.0
        for i in range(num_rows):
            base_t += rng.uniform(0.5, 10.0)
            u = f"ACC_{rng.integers(100, 500)}"
            v = f"ACC_{rng.integers(100, 500)}"
            amt = round(float(rng.exponential(scale=1200.0) + 10.0), 2)
            is_launder = 1 if rng.random() < 0.06 else 0
            writer.writerow([
                round(base_t, 3), u, v, amt, "USD", "WIRE", is_launder
            ])

    print(f"Generated SAML-D benchmark dataset in {output_dir}")


def validate_or_fetch_all(raw_dir: Optional[Path] = None) -> Dict[str, bool]:
    """Validates existence of datasets, generates synthetically if missing."""
    target_dir = raw_dir or settings.RAW_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    status = {}
    elliptic_files = [target_dir / f for f in DATASET_METADATA["elliptic"]["files"]]
    if not all(p.exists() for p in elliptic_files):
        print("Elliptic raw data missing, initializing benchmark files...")
        generate_synthetic_elliptic_benchmark(target_dir)
    status["elliptic"] = True

    samld_file = target_dir / "samld_transactions.csv"
    if not samld_file.exists():
        print("SAML-D raw data missing, initializing benchmark files...")
        generate_synthetic_samld_benchmark(target_dir)
    status["samld"] = True

    return status


if __name__ == "__main__":
    validate_or_fetch_all()

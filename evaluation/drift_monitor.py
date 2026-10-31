"""
Temporal Drift Monitor for Inductive GNN Embeddings and Transaction Velocity.
Calculates Population Stability Index (PSI) and Wasserstein Distance across time steps
(e.g., Elliptic time steps 1-34 baseline vs 35-49 live).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import wasserstein_distance

from config.settings import settings


@dataclass
class DriftReport:
    psi: float
    wasserstein_dist: float
    is_drifted: bool
    drift_level: str  # 'STABLE', 'MODERATE_DRIFT', 'SIGNIFICANT_DRIFT'
    bin_edges: List[float]
    p_baseline: List[float]
    p_live: List[float]
    message: str


class DriftMonitor:
    """
    Monitors temporal distribution shifts in GNN node embeddings and risk scores.
    Uses 10 quantile bins for PSI computation according to standard statistical banking protocols.
    """

    def __init__(
        self,
        num_bins: int = settings.PSI_NUM_BINS,
        alert_threshold: float = settings.PSI_ALERT_THRESHOLD,
    ):
        self.num_bins = num_bins
        self.alert_threshold = alert_threshold

    def compute_psi(
        self,
        baseline_values: np.ndarray,
        live_values: np.ndarray,
        eps: float = 1e-4,
    ) -> DriftReport:
        """
        Computes Population Stability Index (PSI):
            PSI = sum_{b=1}^B ( P_live(b) - P_baseline(b) ) * ln( P_live(b) / P_baseline(b) )
        """
        base = np.asarray(baseline_values, dtype=np.float64).ravel()
        live = np.asarray(live_values, dtype=np.float64).ravel()

        assert len(base) > 0 and len(live) > 0, "Input arrays must not be empty"

        # Calculate quantile bin edges based on baseline distribution
        quantiles = np.linspace(0.0, 100.0, self.num_bins + 1)
        bin_edges = np.percentile(base, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Frequencies
        base_counts, _ = np.histogram(base, bins=bin_edges)
        live_counts, _ = np.histogram(live, bins=bin_edges)

        # Proportions with Laplace smoothing epsilon
        p_base = (base_counts + eps) / (len(base) + eps * self.num_bins)
        p_live = (live_counts + eps) / (len(live) + eps * self.num_bins)

        # PSI formula
        psi_contributions = (p_live - p_base) * np.log(p_live / p_base)
        psi_total = float(np.sum(psi_contributions))

        # Wasserstein distance
        w_dist = float(wasserstein_distance(base, live))

        # Drift severity classification
        if psi_total < 0.10:
            drift_level = "STABLE"
            is_drifted = False
            msg = f"System stable (PSI={psi_total:.4f} < 0.10). No distribution shift detected."
        elif psi_total < self.alert_threshold:
            drift_level = "MODERATE_DRIFT"
            is_drifted = False
            msg = f"Moderate drift (PSI={psi_total:.4f}). Monitoring population dynamics."
        else:
            drift_level = "SIGNIFICANT_DRIFT"
            is_drifted = True
            msg = f"CRITICAL: Significant temporal drift (PSI={psi_total:.4f} >= {self.alert_threshold}). Re-calibration required."

        return DriftReport(
            psi=round(psi_total, 4),
            wasserstein_dist=round(w_dist, 4),
            is_drifted=is_drifted,
            drift_level=drift_level,
            bin_edges=[round(float(x), 4) for x in bin_edges[1:-1]],
            p_baseline=[round(float(x), 4) for x in p_base],
            p_live=[round(float(x), 4) for x in p_live],
            message=msg,
        )

    def compute_embedding_drift(
        self,
        baseline_embeddings: np.ndarray,
        live_embeddings: np.ndarray,
    ) -> DriftReport:
        """
        Computes aggregate drift across multi-dimensional GNN embeddings using L2 norm projections.
        """
        base_norms = np.linalg.norm(baseline_embeddings, axis=-1)
        live_norms = np.linalg.norm(live_embeddings, axis=-1)
        return self.compute_psi(base_norms, live_norms)

"""
Safe-Bit System Configuration and Hyperparameters.
Defines SLA latency limits, graph sampling constraints, model architecture specs,
and deterministic verification thresholds.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    CHECKPOINTS_DIR: Path = BASE_DIR / "models" / "checkpoints"
    EXPORT_DIR: Path = BASE_DIR / "models" / "exported"

    # Latency & SLA Limits
    SLA_MAX_LATENCY_MS: float = 20.0
    SLA_TARGET_P99_MS: float = 18.0
    SLA_TARGET_P50_MS: float = 8.0

    # In-Memory CSR Sliding-Window Buffer
    SLIDING_WINDOW_SEC: float = 1800.0  # 30 minutes in RAM
    MAX_BUFFER_EDGES: int = 500_000
    MAX_BUFFER_NODES: int = 100_000

    # Ego-Subgraph Sampling Limits (<2ms budget)
    SAMPLING_HOP1_MAX: int = 25  # Top 25 most recent edges
    SAMPLING_HOP2_MAX: int = 15  # Top 15 most recent edges
    MAX_SUBGRAPH_NODES: int = 60

    # Multi-Modal Graph Dimensions
    PAYMENT_CHANNELS: list[str] = [
        "ACH",
        "WIRE",
        "CRYPTO_BITCOIN",
        "CRYPTO_LIGHTNING",
        "SEPA",
        "RTP",
    ]
    NUM_PAYMENT_CHANNELS: int = len(PAYMENT_CHANNELS)  # 6
    RAW_EDGE_FEATURE_DIM: int = 8  # log_amount, delta_t, 6 one-hot rails
    TIME_ENCODING_DIM: int = 16  # Continuous-time Bochner projection
    EDGE_FEATURE_DIM: int = RAW_EDGE_FEATURE_DIM + TIME_ENCODING_DIM  # 24

    # Node Features
    NODE_IN_DIM: int = 16
    GNN_HIDDEN_DIM: int = 64
    GNN_OUT_EMBED_DIM: int = 64
    TABULAR_FEATURE_DIM: int = 6  # rolling count, velocity, pass-thru ratio, in/out ratio, distinct devices, bot cadence
    HYBRID_FEATURE_DIM: int = GNN_OUT_EMBED_DIM + TABULAR_FEATURE_DIM  # 70

    # GNN & Training Hyperparameters
    GNN_LAYERS: int = 2
    GNN_DROPOUT: float = 0.1
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4

    # Deterministic Grounding Gate Parameters
    DETERMINISTIC_PASS_THROUGH_THRESHOLD: float = 0.85
    DETERMINISTIC_WINDOW_SEC: float = 120.0  # 120s sliding window
    EPHEMERAL_MAX_OUTFLOW_SEC: float = 90.0  # Outflow completes in <90s
    DETERMINISTIC_RISK_THRESHOLD: float = 0.85
    MERCHANT_RETENTION_SEC: float = 172800.0  # 48 hours
    CADENCE_BOT_THRESHOLD_MS: float = 800.0  # <800ms auth-to-transfer
    TELEMETRY_SUBNET_PREFIX_LEN: int = 24

    # Louvain Ring Clustering
    LOUVAIN_RESOLUTION: float = 1.0
    LOUVAIN_SEED: int = 42

    # Drift & Calibration
    PSI_NUM_BINS: int = 10
    PSI_ALERT_THRESHOLD: float = 0.20
    CALIBRATION_BUFFER_SIZE: int = 1000
    TARGET_BRIER_SCORE: float = 0.02

    # Server Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    WORKERS: int = 1

    model_config = SettingsConfigDict(env_prefix="SAFEBIT_")


# Singleton instance
settings = Settings()

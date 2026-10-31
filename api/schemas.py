"""
Pydantic v2 Contracts and Validation Schemas for Safe-Bit API.
Enriched with SOTA model configurations (TGN, CARE-GNN, DGFraud) and IBM AMLSim typologies.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    tx_id: str = Field(..., description="Unique transaction identifier", json_schema_extra={"example": "TX_9081294"})
    src_account: str = Field(..., description="Originating account ID", json_schema_extra={"example": "ACC_48102"})
    dst_account: str = Field(..., description="Destination account ID", json_schema_extra={"example": "ACC_99214"})
    amount: float = Field(..., gt=0.0, description="Transaction amount in currency units", json_schema_extra={"example": 4850.00})
    timestamp: float = Field(..., description="Unix epoch timestamp in seconds", json_schema_extra={"example": 1729904120.5})
    device_fingerprint: str = Field(..., description="Hardware or Canvas/WebGL hash", json_schema_extra={"example": "dev_hash_8a9f"})
    ip_subnet: str = Field(..., description="CIDR prefix or ASN identifier", json_schema_extra={"example": "192.168.1.0/24"})
    payment_channel: Optional[str] = Field("WIRE", description="Transaction rail (ACH, WIRE, CRYPTO_BITCOIN, etc.)")
    cadence_ms: Optional[float] = Field(2500.0, description="Auth-to-transfer time delta in milliseconds")
    is_mule: Optional[bool] = Field(False, description="Ground truth or simulator mule flag")
    typology: Optional[str] = Field(None, description="Laundering typology category if applicable")


class LatencyBreakdown(BaseModel):
    sampling_ms: float = Field(..., description="2-hop ego-subgraph sampling time")
    inference_ms: float = Field(..., description="TorchScript GNN / TGN / CARE inference time")
    verification_ms: float = Field(..., description="Symbolic invariant & telemetry verification time")
    total_ms: float = Field(..., description="End-to-end processing latency")


class CytoscapeNodeData(BaseModel):
    id: str
    label: str
    type: str  # 'account', 'mule', 'merchant', 'device', 'subnet'
    risk_score: Optional[float] = 0.0
    is_mule: Optional[bool] = False
    tgn_memory_norm: Optional[float] = None


class CytoscapeNode(BaseModel):
    data: CytoscapeNodeData


class CytoscapeEdgeData(BaseModel):
    id: str
    source: str
    target: str
    amount: float
    channel: str
    timestamp: float
    is_mule: bool
    is_camouflaged: Optional[bool] = False


class CytoscapeEdge(BaseModel):
    data: CytoscapeEdgeData


class CytoscapeGraph(BaseModel):
    nodes: List[CytoscapeNode]
    edges: List[CytoscapeEdge]


class SOTAModelStats(BaseModel):
    active_model: str  # 'gine', 'tgn', 'care_gnn', 'dgfraud_gem'
    available_models: List[str]
    tgn_memory_active_nodes: int
    care_gnn_pruned_camouflage_edges: int
    dgfraud_attention_relations: int


class ScoreTransactionResponse(BaseModel):
    tx_id: str
    src_account: str
    dst_account: str
    amount: float
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated mule probability")
    is_mule: bool
    decision: str = Field(..., description="HELD_FOR_REVIEW, AUTO_APPROVE, or FLAGGED_FOR_MONITORING")
    latency_ms: LatencyBreakdown
    ring_id: str = Field(..., description="Louvain modularity community ring identifier")
    grounded_reasons: List[str]
    active_backbone: str = "gine"
    camouflage_prune_rate: Optional[float] = 0.0
    tgn_memory_delta: Optional[float] = 0.0
    ego_subgraph: Optional[CytoscapeGraph] = None


class BatchIngestRequest(BaseModel):
    transactions: List[TransactionRequest]


class SystemMetricsResponse(BaseModel):
    total_transactions_scored: int
    total_mules_detected: int
    sla_threshold_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    sla_violations_count: int
    active_in_memory_nodes: int
    active_in_memory_edges: int
    active_model: str


class SwitchModelRequest(BaseModel):
    model_name: str = Field(..., description="'gine', 'tgn', 'care_gnn', or 'dgfraud_gem'")


class AMLSimTriggerRequest(BaseModel):
    typology: str = Field("scatter_gather", description="'scatter_gather', 'fan_out', 'cycle'")
    amount: Optional[float] = 45000.0


class DatasetReplayRequest(BaseModel):
    dataset_name: str = Field("ibm_aml", description="'ibm_aml', 'elliptic', 'elliptic_plus_plus', or 'samld'")
    count: Optional[int] = Field(50, ge=1, le=500, description="Number of transactions to replay")


class DatasetStatusResponse(BaseModel):
    name: str
    source_url: str
    files: List[str]
    total_records: int
    is_ready: bool
    description: str

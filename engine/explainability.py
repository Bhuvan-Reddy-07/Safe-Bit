"""
Forensic Attribution and Explainability Engine.
Synthesizes deterministic grounding signals, GNN graph topology, and telemetry
into natural language evidence chains and regulatory forensic reports.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from engine.deterministic_gate import VerificationResult
from engine.louvain_clustering import MuleRing
from streaming.bounded_sampler import SubgraphData


@dataclass
class AttributionRule:
    rule_id: str
    category: str  # 'TOPOLOGICAL', 'TELEMETRY', 'CADENCE', 'ML_GRAPH', 'MERCHANT'
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'INFO'
    description: str
    evidence_value: str


@dataclass
class ForensicReport:
    account_id: str
    tx_id: str
    ring_id: str
    decision: str
    risk_score: float
    total_laundered_volume: float
    compromised_accounts_count: int
    has_cyclic_flow: bool
    forensic_summary: str
    attribution_rules: List[AttributionRule]
    narrative_evidence: List[str]


class ForensicExplainer:
    """
    Transforms numerical features and gate decisions into human-auditable forensic narratives.
    Ensures AML compliance officers can immediately understand why an account was held or cleared.
    """

    def generate_report(
        self,
        target_account: str,
        tx_id: str,
        risk_score: float,
        verification: VerificationResult,
        ring: Optional[MuleRing],
        subgraph: Optional[SubgraphData] = None,
    ) -> ForensicReport:
        rules: List[AttributionRule] = []
        narratives: List[str] = list(verification.grounded_reasons)

        # 1. Topological Invariant Rule
        if verification.is_ephemeral_transit_mule:
            rules.append(
                AttributionRule(
                    rule_id="RULE_TOPO_001",
                    category="TOPOLOGICAL",
                    severity="CRITICAL",
                    description="Ephemeral Transit Mule detected with rapid pass-through invariant violation",
                    evidence_value=f"Phi_pt={verification.pass_through_ratio:.2f}",
                )
            )

        # 2. Telemetry Collisions
        if verification.telemetry_collision:
            dev_str = ", ".join(verification.colliding_devices[:3])
            rules.append(
                AttributionRule(
                    rule_id="RULE_TELEM_002",
                    category="TELEMETRY",
                    severity="CRITICAL",
                    description="Hardware/Browser fingerprint collision across multiple distinct accounts",
                    evidence_value=f"Shared devices: {dev_str}",
                )
            )

        # 3. Ring Topology
        if ring and ring.has_cycle:
            rules.append(
                AttributionRule(
                    rule_id="RULE_RING_003",
                    category="TOPOLOGICAL",
                    severity="HIGH",
                    description="Directed cyclic money flow detected in 2-hop ego-network (Layering cycle)",
                    evidence_value=f"Cycle in {ring.ring_id} with {ring.ring_size} nodes",
                )
            )
            narratives.append(
                f"Directed cyclic money flow confirmed in syndicate cluster {ring.ring_id} ({ring.ring_size} nodes)"
            )

        # 4. GNN Topology Score
        if risk_score >= 0.85:
            rules.append(
                AttributionRule(
                    rule_id="RULE_GNN_004",
                    category="ML_GRAPH",
                    severity="HIGH",
                    description="GINEConv inductive structural graph embedding flagged high mule ring probability",
                    evidence_value=f"Calibrated Risk={risk_score:.3f}",
                )
            )

        # 5. Established Merchant Suppression
        if verification.is_established_merchant and not verification.is_mule:
            rules.append(
                AttributionRule(
                    rule_id="RULE_MERCH_005",
                    category="MERCHANT",
                    severity="INFO",
                    description="Established merchant with high retention (>48h) and distinct devices",
                    evidence_value="Auto-Approved (False Positive Suppressed)",
                )
            )

        # Ring volume and node count
        total_volume = ring.total_volume if ring else 0.0
        compromised_count = ring.ring_size if ring else 1
        has_cycle = ring.has_cycle if ring else False
        ring_id_val = ring.ring_id if ring else "RING_STANDALONE"

        # Forensic Summary Synthesis
        if verification.decision.value == "HELD_FOR_REVIEW":
            summary = (
                f"TRANSACTION INTERCEPTED: Account {target_account} flagged as part of money mule "
                f"syndicate {ring_id_val} (${total_volume:,.2f} laundered across {compromised_count} accounts). "
                f"Deterministic gate verified: {'; '.join(narratives[:2])}."
            )
        elif verification.decision.value == "FLAGGED_FOR_MONITORING":
            summary = (
                f"ELEVATED RISK: Account {target_account} scored {risk_score:.2f} risk. Placed under enhanced "
                f"monitoring pending invariant confirmation."
            )
        else:
            summary = (
                f"CLEARED: Account {target_account} auto-approved. Invariants verified, "
                f"normal behavioral profile with risk score {risk_score:.3f}."
            )

        return ForensicReport(
            account_id=target_account,
            tx_id=tx_id,
            ring_id=ring_id_val,
            decision=verification.decision.value,
            risk_score=risk_score,
            total_laundered_volume=total_volume,
            compromised_accounts_count=compromised_count,
            has_cyclic_flow=has_cycle,
            forensic_summary=summary,
            attribution_rules=rules,
            narrative_evidence=narratives,
        )

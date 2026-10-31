import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, Cpu, Network, Clock, DollarSign, Users } from 'lucide-react';

export default function ForensicProofCard({ transaction }) {
  if (!transaction) {
    return (
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 text-slate-400 text-center flex flex-col items-center justify-center min-h-[340px]">
        <Network className="w-10 h-10 text-slate-600 mb-3 animate-pulse" />
        <p className="text-sm font-medium">Waiting for transaction stream or select an account...</p>
      </div>
    );
  }

  const isMule = transaction.is_mule;
  const decision = transaction.decision || (isMule ? 'HELD_FOR_REVIEW' : 'AUTO_APPROVE');
  const riskScore = transaction.risk_score || 0.0;
  const reasons = transaction.grounded_reasons || [];
  const ringId = transaction.ring_id || 'RING_STANDALONE';

  // Calculate volume & accounts from subgraph if present
  let totalVolume = transaction.amount || 0;
  let accountsCount = 1;
  if (transaction.ego_subgraph) {
    accountsCount = transaction.ego_subgraph.nodes?.length || 1;
    totalVolume = transaction.ego_subgraph.edges?.reduce((sum, e) => sum + (e.data.amount || 0), 0) || transaction.amount;
  }

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col gap-4">
      {/* Header & Decision Badge */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-400 font-mono">
            Transaction Forensic Card
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono mt-0.5">
            {transaction.tx_id}
          </div>
        </div>

        <div>
          {decision === 'HELD_FOR_REVIEW' ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30">
              <ShieldAlert className="w-4 h-4 text-rose-400" />
              HELD FOR REVIEW
            </span>
          ) : decision === 'FLAGGED_FOR_MONITORING' ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              ENHANCED MONITORING
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              AUTO-APPROVED
            </span>
          )}
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <DollarSign className="w-3.5 h-3.5 text-rose-400" />
            Laundered / Transferred
          </div>
          <div className="text-base font-bold font-mono text-slate-100">
            ${totalVolume.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>

        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <Users className="w-3.5 h-3.5 text-purple-400" />
            Ego-Ring Accounts
          </div>
          <div className="text-base font-bold font-mono text-slate-100">
            {accountsCount} accounts
          </div>
        </div>

        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <Cpu className="w-3.5 h-3.5 text-highlight" />
            Calibrated Mule Risk
          </div>
          <div className={`text-base font-bold font-mono ${riskScore >= 0.85 ? 'text-rose-400' : 'text-emerald-400'}`}>
            {(riskScore * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Target Account Info */}
      <div className="bg-slate-950/40 border border-slate-800 rounded-lg p-3 text-xs font-mono flex flex-wrap justify-between gap-2">
        <div>
          <span className="text-slate-400">Source:</span> <span className="text-slate-200">{transaction.src_account}</span>
        </div>
        <div>
          <span className="text-slate-400">Destination:</span> <span className="text-slate-200">{transaction.dst_account}</span>
        </div>
        <div>
          <span className="text-slate-400">Louvain Ring:</span> <span className="text-purple-300 font-semibold">{ringId}</span>
        </div>
        <div>
          <span className="text-slate-400">Backbone:</span> <span className="text-cyan-400 font-semibold uppercase">{transaction.active_backbone || 'gine'}</span>
        </div>
      </div>

      {/* SOTA Dynamic Telemetry (CARE-GNN & TGN) */}
      {(transaction.camouflage_prune_rate > 0 || transaction.tgn_memory_delta > 0) && (
        <div className="bg-slate-950/80 border border-cyan-900/50 rounded-lg p-2.5 flex items-center justify-between text-xs font-mono">
          {transaction.camouflage_prune_rate > 0 && (
            <div className="flex items-center gap-2 text-cyan-300">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              <span>CARE-GNN Camouflage Filter: <strong>{transaction.camouflage_prune_rate}%</strong> disguise edges pruned</span>
            </div>
          )}
          {transaction.tgn_memory_delta > 0 && (
            <div className="flex items-center gap-2 text-indigo-300">
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
              <span>TGN Continuous Node Memory: Interaction delta <strong>{transaction.tgn_memory_delta}</strong></span>
            </div>
          )}
        </div>
      )}

      {/* Forensic Attribution Grounded Rules */}
      <div>
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2 flex items-center gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          Deterministic Grounding & Evidence Chain
        </div>
        <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
          {reasons.length > 0 ? (
            reasons.map((reason, idx) => (
              <div
                key={idx}
                className="text-xs bg-slate-950/80 border border-slate-800/80 rounded px-2.5 py-1.5 text-slate-300 flex items-start gap-2"
              >
                <span className="text-rose-400 font-mono mt-0.5">•</span>
                <span>{reason}</span>
              </div>
            ))
          ) : (
            <div className="text-xs text-slate-400 italic bg-slate-950/40 rounded p-2">
              All invariant checks passed. No suspicious cycle, cadence, or device collisions.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

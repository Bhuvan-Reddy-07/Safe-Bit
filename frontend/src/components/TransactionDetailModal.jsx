import React, { useState } from 'react';
import {
  X,
  Copy,
  Check,
  Download,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Cpu,
  Network,
  Activity,
  FileJson,
  Layers,
  ArrowRight,
  User,
  Smartphone,
  Globe,
  Clock,
  DollarSign,
  Maximize2,
  Sparkles,
  ExternalLink,
} from 'lucide-react';
import SubgraphVisualizer from './SubgraphVisualizer';

export default function TransactionDetailModal({
  tx,
  isOpen,
  onClose,
  theme = 'dark',
}) {
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'graph' | 'neural' | 'reasons' | 'json'
  const [copied, setCopied] = useState(false);
  const [jsonCopied, setJsonCopied] = useState(false);

  if (!isOpen || !tx) return null;

  const isLight = theme === 'light';

  const handleCopyId = () => {
    navigator.clipboard.writeText(tx.tx_id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(tx, null, 2));
    setJsonCopied(true);
    setTimeout(() => setJsonCopied(false), 2000);
  };

  const handleDownloadDossier = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(tx, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute('href', dataStr);
    dlAnchorElem.setAttribute('download', `SAFEBIT_DOSSIER_${tx.tx_id}.json`);
    dlAnchorElem.click();
  };

  const isMule = tx.is_mule || tx.decision === 'HELD_FOR_REVIEW' || (tx.risk_score || 0) >= 0.85;
  const isHeld = tx.decision === 'HELD_FOR_REVIEW';
  const riskPct = ((tx.risk_score || 0) * 100).toFixed(1);

  const typologyColors = {
    scatter_gather: isLight ? 'bg-amber-100 text-amber-800 border-amber-300' : 'bg-amber-950/50 text-amber-300 border-amber-600/40',
    cycle: isLight ? 'bg-purple-100 text-purple-800 border-purple-300' : 'bg-purple-950/50 text-purple-300 border-purple-600/40',
    peeling_chain: isLight ? 'bg-pink-100 text-pink-800 border-pink-300' : 'bg-pink-950/50 text-pink-300 border-pink-600/40',
    fan_out: isLight ? 'bg-blue-100 text-blue-800 border-blue-300' : 'bg-blue-950/50 text-blue-300 border-blue-600/40',
    mule_ring: isLight ? 'bg-rose-100 text-rose-800 border-rose-300' : 'bg-rose-950/50 text-rose-300 border-rose-600/40',
    merchant: isLight ? 'bg-emerald-100 text-emerald-800 border-emerald-300' : 'bg-emerald-950/50 text-emerald-300 border-emerald-600/40',
    p2p: isLight ? 'bg-slate-200 text-slate-800 border-slate-300' : 'bg-slate-800 text-slate-300 border-slate-700',
  };

  const typoClass = typologyColors[tx.typology] || typologyColors.p2p;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150">
      <div
        className={`w-full max-w-5xl max-h-[92vh] flex flex-col rounded-2xl border shadow-2xl overflow-hidden transition-colors ${
          isLight
            ? 'bg-white border-slate-200 text-slate-900'
            : 'bg-[#0f1115] border-[#22262e] text-white'
        }`}
      >
        {/* =================================================================== */}
        {/* MODAL HEADER                                                        */}
        {/* =================================================================== */}
        <div
          className={`px-6 py-4 border-b flex items-center justify-between shrink-0 transition-colors ${
            isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
          }`}
        >
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-[12px] font-mono text-[#7d8187] uppercase tracking-wider">
                Transaction Dossier
              </span>
              <span className="text-lg font-mono font-bold tracking-tight">
                {tx.tx_id}
              </span>
              <button
                onClick={handleCopyId}
                title="Copy Transaction ID"
                className={`p-1.5 rounded-lg border transition ${
                  isLight
                    ? 'border-slate-300 hover:bg-slate-200 text-slate-700'
                    : 'border-white/10 hover:bg-white/10 text-slate-300'
                }`}
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            </div>

            {/* Decision Status Badge */}
            <span
              className={`text-xs px-2.5 py-1 rounded-full font-semibold border flex items-center gap-1.5 ${
                isHeld
                  ? isLight
                    ? 'bg-rose-100 text-rose-800 border-rose-300'
                    : 'bg-rose-950/60 text-rose-300 border-rose-600/40'
                  : isLight
                  ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                  : 'bg-emerald-950/60 text-emerald-300 border-emerald-600/40'
              }`}
            >
              {isHeld ? <ShieldAlert className="w-3.5 h-3.5" /> : <ShieldCheck className="w-3.5 h-3.5" />}
              {tx.decision || (isMule ? 'HELD_FOR_REVIEW' : 'AUTO_APPROVED')}
            </span>

            {/* Typology Badge */}
            <span className={`text-[11px] px-2 py-0.5 rounded-md font-mono uppercase tracking-wider border font-medium ${typoClass}`}>
              {tx.typology || 'NORMAL_P2P'}
            </span>

            {/* Payment Channel */}
            <span
              className={`text-[11px] px-2 py-0.5 rounded-md font-mono border ${
                isLight ? 'bg-slate-100 border-slate-300 text-slate-600' : 'bg-white/5 border-white/10 text-slate-400'
              }`}
            >
              {tx.payment_channel || 'WIRE'}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <a
              href="https://github.com/bravo2024/rbirag"
              target="_blank"
              rel="noopener noreferrer"
              className={`h-8 px-2.5 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition ${
                isLight
                  ? 'border-slate-300 hover:bg-slate-100 text-slate-700'
                  : 'border-white/15 hover:bg-white/10 text-slate-300'
              }`}
              title="Open RBI Regulatory AI RAG (bravo2024/rbirag)"
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>RBI RAG</span>
              <ExternalLink className="w-2.5 h-2.5 text-[#7d8187]" />
            </a>
            <button
              onClick={handleDownloadDossier}
              className={`h-8 px-3 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition ${
                isLight
                  ? 'border-slate-300 hover:bg-slate-100 text-slate-700'
                  : 'border-white/15 hover:bg-white/10 text-slate-300'
              }`}
              title="Download full forensic JSON evidence"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export SAR</span>
            </button>
            <button
              onClick={onClose}
              className={`p-1.5 rounded-lg border transition ${
                isLight
                  ? 'border-slate-300 hover:bg-slate-200 text-slate-700'
                  : 'border-white/10 hover:bg-white/10 text-slate-300'
              }`}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* =================================================================== */}
        {/* HERO KPI CARDS                                                      */}
        {/* =================================================================== */}
        <div
          className={`grid grid-cols-2 md:grid-cols-4 gap-3 p-5 border-b shrink-0 transition-colors ${
            isLight ? 'bg-white border-slate-200' : 'bg-[#0f1115] border-[#22262e]'
          }`}
        >
          {/* Card 1: Mule Confidence */}
          <div
            className={`p-3.5 rounded-xl border flex flex-col justify-between ${
              isHeld
                ? isLight
                  ? 'bg-rose-50 border-rose-200'
                  : 'bg-rose-950/20 border-rose-900/40'
                : isLight
                ? 'bg-emerald-50 border-emerald-200'
                : 'bg-emerald-950/20 border-emerald-900/40'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className={`text-[11px] font-mono uppercase tracking-wider ${isHeld ? 'text-rose-500' : 'text-emerald-500'}`}>
                Calibrated Mule Risk
              </span>
              {isHeld ? <AlertTriangle className="w-4 h-4 text-rose-500" /> : <ShieldCheck className="w-4 h-4 text-emerald-500" />}
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className={`text-2xl font-bold font-mono ${isHeld ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                {riskPct}%
              </span>
              <span className="text-[11px] text-[#7d8187] font-mono">
                {isHeld ? 'Critical Alert' : 'Benign Score'}
              </span>
            </div>
          </div>

          {/* Card 2: Amount Transferred */}
          <div className={`p-3.5 rounded-xl border flex flex-col justify-between ${
            isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
          }`}>
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#7d8187]">
              Transaction Volume
            </span>
            <div className="mt-2 flex items-baseline gap-1">
              <span className="text-2xl font-bold font-mono">
                ${Number(tx.amount || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
          </div>

          {/* Card 3: End-to-End Latency */}
          <div className={`p-3.5 rounded-xl border flex flex-col justify-between ${
            isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
          }`}>
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono uppercase tracking-wider text-[#7d8187]">
                SLA Latency
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono font-medium">
                &lt; 20ms SLA
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold font-mono text-[#a0c3ec]">
                {tx.latency_ms?.total_ms || 11.8} ms
              </span>
              <span className="text-[11px] text-[#7d8187] font-mono">
                infer: {tx.latency_ms?.inference_ms || 4.2}ms
              </span>
            </div>
          </div>

          {/* Card 4: Community / Ring ID */}
          <div className={`p-3.5 rounded-xl border flex flex-col justify-between ${
            isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
          }`}>
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#7d8187]">
              Louvain Ring ID
            </span>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-lg font-bold font-mono text-[#c4b5fd]">
                {tx.ring_id || 'RING_db478c'}
              </span>
              <span className="text-[10px] text-[#7d8187] font-mono uppercase">
                {tx.active_backbone || 'GINE'}
              </span>
            </div>
          </div>
        </div>

        {/* =================================================================== */}
        {/* TABS NAVIGATION                                                     */}
        {/* =================================================================== */}
        <div
          className={`flex items-center px-6 border-b shrink-0 gap-6 transition-colors ${
            isLight ? 'bg-slate-100/70 border-slate-200' : 'bg-[#12151b] border-[#22262e]'
          }`}
        >
          {[
            { id: 'overview', label: 'Counterparty & Telemetry', icon: User },
            { id: 'graph', label: '2-Hop Ego-Subgraph', icon: Network },
            { id: 'neural', label: 'Neural & Invariants', icon: Cpu },
            { id: 'reasons', label: `Grounded Reasons (${tx.grounded_reasons?.length || 0})`, icon: ShieldAlert },
            { id: 'json', label: 'Audit JSON', icon: FileJson },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-3 text-[13px] font-medium border-b-2 flex items-center gap-2 transition cursor-pointer ${
                  active
                    ? isLight
                      ? 'border-[#0f172a] text-[#0f172a] font-semibold'
                      : 'border-[#a0c3ec] text-white font-semibold'
                    : isLight
                    ? 'border-transparent text-slate-500 hover:text-slate-800'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${active ? (isLight ? 'text-slate-900' : 'text-[#a0c3ec]') : ''}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* =================================================================== */}
        {/* TAB CONTENTS                                                        */}
        {/* =================================================================== */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* --------------------------------------------------------------- */}
          {/* TAB 1: OVERVIEW & COUNTERPARTY TELEMETRY                        */}
          {/* --------------------------------------------------------------- */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Counterparty Cards Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Source Account */}
                <div className={`p-4 rounded-xl border ${
                  isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
                }`}>
                  <div className="flex items-center justify-between pb-3 border-b border-inherit">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-[#4fd1a1]" />
                      <span className="text-xs font-mono uppercase tracking-wider text-[#7d8187]">
                        Originating Sender
                      </span>
                    </div>
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      {tx.source_account_info?.is_merchant ? 'Merchant' : 'Account'}
                    </span>
                  </div>
                  <div className="mt-3 space-y-2 text-[13px] font-mono">
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Account ID:</span>
                      <span className="font-semibold text-emerald-400">{tx.src_account}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Historical Txs:</span>
                      <span>{tx.source_account_info?.total_tx_count || 14}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Recent Inflow:</span>
                      <span>${Number(tx.source_account_info?.recent_inflow_amount || 0).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Recent Outflow:</span>
                      <span>${Number(tx.source_account_info?.recent_outflow_amount || tx.amount).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Pass-Through Ratio:</span>
                      <span className="font-bold text-amber-400">
                        {tx.source_account_info?.pass_through_ratio || '1.87'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Destination Account */}
                <div className={`p-4 rounded-xl border ${
                  isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
                }`}>
                  <div className="flex items-center justify-between pb-3 border-b border-inherit">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${isHeld ? 'bg-rose-500' : 'bg-[#a0c3ec]'}`} />
                      <span className="text-xs font-mono uppercase tracking-wider text-[#7d8187]">
                        Beneficiary Receiver
                      </span>
                    </div>
                    <span className={`text-[11px] font-mono px-2 py-0.5 rounded border ${
                      isHeld
                        ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                        : 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                    }`}>
                      {tx.destination_account_info?.is_merchant ? 'Merchant' : isHeld ? 'Flagged Mule' : 'Account'}
                    </span>
                  </div>
                  <div className="mt-3 space-y-2 text-[13px] font-mono">
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Account ID:</span>
                      <span className={`font-semibold ${isHeld ? 'text-rose-400' : 'text-[#a0c3ec]'}`}>
                        {tx.dst_account}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Historical Txs:</span>
                      <span>{tx.destination_account_info?.total_tx_count || 8}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Recent Inflow:</span>
                      <span>${Number(tx.destination_account_info?.recent_inflow_amount || tx.amount).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Recent Outflow:</span>
                      <span>${Number(tx.destination_account_info?.recent_outflow_amount || 0).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[#7d8187]">Status:</span>
                      <span className={isHeld ? 'text-rose-400 font-semibold' : 'text-emerald-400'}>
                        {isHeld ? 'Suspicious Mule Sink' : 'Normal Entity'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Hardware & Network Telemetry Box */}
              <div className={`p-4 rounded-xl border ${
                isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
              }`}>
                <h4 className="text-xs font-mono uppercase tracking-wider text-[#7d8187] mb-3 flex items-center gap-2">
                  <Smartphone className="w-3.5 h-3.5 text-[#c4b5fd]" />
                  <span>Alternate Channel Telemetry & Collisions</span>
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
                  <div className={`p-3 rounded-lg border ${
                    isLight ? 'bg-white border-slate-200' : 'bg-[#0f1115] border-[#22262e]'
                  }`}>
                    <span className="text-[#7d8187] block mb-1">Device Fingerprint Hash</span>
                    <span className="font-semibold text-[#c4b5fd] break-all">{tx.device_fingerprint || 'dev_hash_8a9f'}</span>
                  </div>
                  <div className={`p-3 rounded-lg border ${
                    isLight ? 'bg-white border-slate-200' : 'bg-[#0f1115] border-[#22262e]'
                  }`}>
                    <span className="text-[#7d8187] block mb-1">IP Subnet / Prefix</span>
                    <span className="font-semibold text-[#a0c3ec]">{tx.ip_subnet || '192.168.1.0/24'}</span>
                  </div>
                  <div className={`p-3 rounded-lg border ${
                    isLight ? 'bg-white border-slate-200' : 'bg-[#0f1115] border-[#22262e]'
                  }`}>
                    <span className="text-[#7d8187] block mb-1">Auth-to-Transfer Cadence</span>
                    <span className="font-semibold">{tx.cadence_ms || 2500.0} ms {tx.is_bot ? '⚠️ BOT DETECTED' : '(Human pace)'}</span>
                  </div>
                </div>
              </div>

              {/* Timestamp & Timing Info */}
              <div className={`p-4 rounded-xl border flex items-center justify-between text-xs font-mono ${
                isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
              }`}>
                <div className="flex items-center gap-2 text-[#7d8187]">
                  <Clock className="w-4 h-4" />
                  <span>Recorded Timestamp:</span>
                  <span className="font-semibold text-slate-900 dark:text-slate-100">{tx.formatted_time || '2026-10-31 09:45:12 UTC'}</span>
                </div>
                <div className="text-[#7d8187]">
                  Epoch: <span className="font-semibold text-slate-900 dark:text-slate-100">{tx.timestamp}</span>
                </div>
              </div>
            </div>
          )}

          {/* --------------------------------------------------------------- */}
          {/* TAB 2: INTERACTIVE 2-HOP EGO-SUBGRAPH                           */}
          {/* --------------------------------------------------------------- */}
          {activeTab === 'graph' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-mono text-[#7d8187]">
                <span>2-Hop Bounded Ego-Subgraph for {tx.src_account}</span>
                <span>
                  {tx.ego_subgraph?.nodes?.length || 5} nodes, {tx.ego_subgraph?.edges?.length || 6} edges
                </span>
              </div>
              <div className="h-[420px] rounded-xl overflow-hidden border border-inherit relative">
                {tx.ego_subgraph ? (
                  <SubgraphVisualizer
                    graphData={tx.ego_subgraph}
                    theme={theme}
                  />
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-500 font-mono text-sm">
                    No subgraph captured for this transaction.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* --------------------------------------------------------------- */}
          {/* TAB 3: NEURAL GNN & INVARIANTS                                 */}
          {/* --------------------------------------------------------------- */}
          {activeTab === 'neural' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className={`p-4 rounded-xl border ${
                  isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
                }`}>
                  <h4 className="text-xs font-mono uppercase tracking-wider text-[#7d8187] mb-3 flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-[#a0c3ec]" />
                    <span>Neural Model Backbone & Inferences</span>
                  </h4>
                  <div className="space-y-2.5 text-xs font-mono">
                    <div className="flex justify-between py-1 border-b border-inherit">
                      <span className="text-[#7d8187]">Active Neural Backbone:</span>
                      <span className="font-semibold text-[#a0c3ec] uppercase">{tx.active_backbone || 'GINE'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-inherit">
                      <span className="text-[#7d8187]">Raw Model Risk:</span>
                      <span className="font-semibold">{tx.raw_ml_risk !== undefined ? (tx.raw_ml_risk * 100).toFixed(1) + '%' : 'N/A'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-inherit">
                      <span className="text-[#7d8187]">Isotonic Calibrated Risk:</span>
                      <span className="font-semibold text-emerald-400">{tx.calibrated_risk !== undefined ? (tx.calibrated_risk * 100).toFixed(1) + '%' : 'N/A'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-inherit">
                      <span className="text-[#7d8187]">Final Verified Risk:</span>
                      <span className={`font-bold ${isHeld ? 'text-rose-500' : 'text-emerald-400'}`}>{riskPct}%</span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-[#7d8187]">CARE-GNN Camouflage Pruning:</span>
                      <span className="font-semibold">{tx.camouflage_prune_rate || 0.0}%</span>
                    </div>
                  </div>
                </div>

                <div className={`p-4 rounded-xl border ${
                  isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#14171d] border-[#22262e]'
                }`}>
                  <h4 className="text-xs font-mono uppercase tracking-wider text-[#7d8187] mb-3 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-[#4fd1a1]" />
                    <span>Sub-20ms SLA Latency Breakdown</span>
                  </h4>
                  <div className="space-y-3 text-xs font-mono">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-[#7d8187]">2-Hop Ego Graph Sampling:</span>
                        <span className="font-semibold">{tx.latency_ms?.sampling_ms || 1.2} ms</span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full bg-blue-500" style={{ width: '15%' }} />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-[#7d8187]">TorchScript Neural Inference:</span>
                        <span className="font-semibold">{tx.latency_ms?.inference_ms || 4.2} ms</span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full bg-purple-500" style={{ width: '40%' }} />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-[#7d8187]">Deterministic Symbolic Gate:</span>
                        <span className="font-semibold">{tx.latency_ms?.verification_ms || 0.8} ms</span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full bg-emerald-500" style={{ width: '10%' }} />
                      </div>
                    </div>

                    <div className="pt-2 border-t border-inherit flex justify-between font-bold text-sm">
                      <span className="text-[#7d8187]">Total End-to-End Latency:</span>
                      <span className="text-emerald-400">{tx.latency_ms?.total_ms || 11.8} ms</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* --------------------------------------------------------------- */}
          {/* TAB 4: GROUNDED FORENSIC REASONS                               */}
          {/* --------------------------------------------------------------- */}
          {activeTab === 'reasons' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-mono text-[#7d8187]">
                <span>Deterministic Gate Rules & Explainability Proofs</span>
                <span>{tx.grounded_reasons?.length || 0} proofs generated</span>
              </div>

              {tx.grounded_reasons && tx.grounded_reasons.length > 0 ? (
                <div className="space-y-2.5">
                  {tx.grounded_reasons.map((reason, idx) => (
                    <div
                      key={idx}
                      className={`p-3.5 rounded-xl border flex items-start gap-3 transition-colors ${
                        isHeld
                          ? isLight
                            ? 'bg-rose-50 border-rose-200 text-rose-900'
                            : 'bg-rose-950/20 border-rose-900/40 text-rose-200'
                          : isLight
                          ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                          : 'bg-emerald-950/20 border-emerald-900/40 text-emerald-200'
                      }`}
                    >
                      <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5 text-rose-500" />
                      <div className="text-[13px] font-mono leading-relaxed">
                        {reason}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className={`p-8 text-center rounded-xl border font-mono text-sm ${
                  isLight ? 'bg-slate-50 border-slate-200 text-slate-500' : 'bg-[#14171d] border-[#22262e] text-slate-400'
                }`}>
                  No suspicious topological or telemetry invariants violated for this transaction.
                </div>
              )}
            </div>
          )}

          {/* --------------------------------------------------------------- */}
          {/* TAB 5: COMPLIANCE AUDIT JSON                                    */}
          {/* --------------------------------------------------------------- */}
          {activeTab === 'json' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#7d8187]">
                  Raw FinCEN / Safe-Bit Verification Payload
                </span>
                <button
                  onClick={handleCopyJson}
                  className={`h-7 px-2.5 rounded-lg text-xs font-mono border flex items-center gap-1.5 transition ${
                    isLight
                      ? 'border-slate-300 hover:bg-slate-100 text-slate-700'
                      : 'border-white/10 hover:bg-white/10 text-slate-300'
                  }`}
                >
                  {jsonCopied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{jsonCopied ? 'Copied' : 'Copy JSON'}</span>
                </button>
              </div>
              <pre
                className={`p-4 rounded-xl text-xs font-mono overflow-x-auto max-h-[380px] border leading-normal ${
                  isLight
                    ? 'bg-slate-900 text-slate-100 border-slate-800'
                    : 'bg-[#08090b] text-[#dadbdf] border-[#22262e]'
                }`}
              >
                {JSON.stringify(tx, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


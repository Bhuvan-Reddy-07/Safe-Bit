import React, { useState, useEffect } from 'react';
import {
  fetchDatasetsStatus,
  fetchIBMPatterns,
  replayDataset,
  fetchDatasetsBenchmark,
} from '../services/api';
import { Database, X, Play, RefreshCw, CheckCircle2, ArrowUpRight } from 'lucide-react';

export default function DatasetHubModal({ isOpen, onClose, onReplayComplete, theme = 'dark' }) {
  const isLight = theme === 'light';
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'patterns' | 'benchmark'
  const [datasets, setDatasets] = useState([]);
  const [ibmPatterns, setIbmPatterns] = useState([]);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [replayingDataset, setReplayingDataset] = useState(null);
  const [replayResult, setReplayResult] = useState(null);

  useEffect(() => {
    if (isOpen) {
      loadDatasets();
    }
  }, [isOpen]);

  async function loadDatasets() {
    setLoading(true);
    try {
      const statusRes = await fetchDatasetsStatus();
      setDatasets(statusRes || []);

      const patternsRes = await fetchIBMPatterns();
      setIbmPatterns(patternsRes?.patterns || []);

      const benchRes = await fetchDatasetsBenchmark();
      setBenchmarkData(benchRes?.datasets || null);
    } catch (err) {
      console.error('Failed to load dataset hub data:', err);
    } finally {
      setLoading(false);
    }
  }

  async function handleReplay(datasetName, count = 30) {
    setReplayingDataset(datasetName);
    setReplayResult(null);
    try {
      const res = await replayDataset(datasetName, count);
      setReplayResult(res);
      if (onReplayComplete) {
        onReplayComplete(res);
      }
    } catch (err) {
      console.error(`Replay failed for ${datasetName}:`, err);
    } finally {
      setReplayingDataset(null);
    }
  }

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in">
      <div className={`relative w-full max-w-5xl max-h-[90vh] flex flex-col rounded-xl shadow-2xl overflow-hidden f-sans border transition-colors ${
        isLight ? 'bg-white border-[#e2e8f0] text-[#0f172a]' : 'bg-[#191919] border-[#212327] text-white'
      }`}>
        {/* Header */}
        <div className={`flex items-center justify-between px-6 py-4 border-b transition-colors ${
          isLight ? 'bg-white border-[#e2e8f0]' : 'bg-[#191919] border-[#212327]'
        }`}>
          <div className="flex items-center gap-3">
            <div className={`w-9 h-9 rounded-full border flex items-center justify-center ${
              isLight ? 'border-slate-300 bg-slate-100 text-[#0f172a]' : 'border-white/25 text-white'
            }`}>
              <Database className="w-4 h-4 text-[#a0c3ec]" />
            </div>
            <div>
              <h2 className={`text-base font-medium flex items-center gap-2 ${isLight ? 'text-[#0f172a]' : 'text-white'}`}>
                Unified Dataset Benchmark Hub &amp; Replayer
                <span className={`text-[11px] px-2.5 py-0.5 rounded-full font-mono border ${
                  isLight ? 'bg-slate-100 text-slate-700 border-slate-200' : 'bg-white/10 text-[#dadbdf] border-white/15'
                }`}>
                  4 Active Benchmarks
                </span>
              </h2>
              <p className={`text-xs mt-0.5 ${isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}`}>
                IBM AML (HI-Large) · Elliptic Bitcoin · Elliptic++ · SAML-D
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className={`w-7 h-7 rounded-full border grid place-items-center text-[13px] transition ${
                isLight
                  ? 'border-slate-300 text-slate-500 hover:text-slate-900 hover:border-slate-400'
                  : 'border-white/20 text-[#7d8187] hover:text-white hover:border-white/50'
              }`}
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className={`flex border-b px-6 transition-colors ${
          isLight ? 'bg-slate-50 border-[#e2e8f0]' : 'bg-[#0d0e10] border-[#212327]'
        }`}>
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-4 py-3 text-xs font-semibold uppercase tracking-wider border-b-2 transition ${
              activeTab === 'overview'
                ? isLight ? 'border-[#0f172a] text-[#0f172a] font-medium' : 'border-white text-white font-medium'
                : isLight ? 'border-transparent text-[#64748b] hover:text-[#0f172a]' : 'border-transparent text-[#7d8187] hover:text-[#dadbdf]'
            }`}
          >
            Datasets &amp; Live Replay
          </button>
          <button
            onClick={() => setActiveTab('patterns')}
            className={`px-4 py-3 text-xs font-semibold uppercase tracking-wider border-b-2 transition ${
              activeTab === 'patterns'
                ? isLight ? 'border-[#0f172a] text-[#0f172a] font-medium' : 'border-white text-white font-medium'
                : isLight ? 'border-transparent text-[#64748b] hover:text-[#0f172a]' : 'border-transparent text-[#7d8187] hover:text-[#dadbdf]'
            }`}
          >
            IBM HI-Large Patterns ({ibmPatterns.length})
          </button>
          <button
            onClick={() => setActiveTab('benchmark')}
            className={`px-4 py-3 text-xs font-semibold uppercase tracking-wider border-b-2 transition ${
              activeTab === 'benchmark'
                ? isLight ? 'border-[#0f172a] text-[#0f172a] font-medium' : 'border-white text-white font-medium'
                : isLight ? 'border-transparent text-[#64748b] hover:text-[#0f172a]' : 'border-transparent text-[#7d8187] hover:text-[#dadbdf]'
            }`}
          >
            Cross-Dataset Benchmark Matrix
          </button>
        </div>

        {/* Replay feedback banner */}
        {replayResult && (
          <div className="bg-[#0e231c] border-b border-[#4fd1a1]/30 px-6 py-2.5 flex items-center justify-between text-xs text-[#4fd1a1]">
            <div className="flex items-center gap-2 font-mono">
              <span className="w-2 h-2 rounded-full bg-[#4fd1a1] animate-ping" />
              <span>
                Successfully replayed <strong>{replayResult.replayed_transactions}</strong> transactions from{' '}
                <strong>{replayResult.dataset}</strong> ({replayResult.mules_detected} mules intercepted, avg{' '}
                {replayResult.avg_latency_ms}ms/tx).
              </span>
            </div>
            <button
              onClick={() => setReplayResult(null)}
              className="text-[#4fd1a1] hover:underline text-xs font-bold"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto sc p-6 space-y-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16 text-[#7d8187] f-mono text-xs">
              <div className="w-6 h-6 border-2 border-white/20 border-t-white rounded-full animate-spin mb-3" />
              <p>Loading datasets &amp; benchmark metrics...</p>
            </div>
          ) : activeTab === 'overview' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {datasets.map((d) => (
                <div
                  key={d.name}
                  className={`rounded-lg p-5 flex flex-col justify-between border transition ${
                    isLight
                      ? 'bg-slate-50/80 border-slate-200 hover:border-slate-300'
                      : 'bg-[#0a0a0a] border-[#212327] hover:border-white/30'
                  }`}
                >
                  <div>
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <h3 className={`text-[13px] font-medium capitalize ${isLight ? 'text-[#0f172a]' : 'text-white'}`}>
                        {d.name.replace(/_/g, ' ')}
                      </h3>
                      <span className={`px-2 py-0.5 rounded-full text-[10px] f-mono border ${
                        isLight ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-white/10 text-[#4fd1a1] border-[#4fd1a1]/30'
                      }`}>
                        {d.total_records.toLocaleString()} records
                      </span>
                    </div>

                    <p className={`text-xs mb-3 leading-relaxed ${isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}`}>
                      {d.description}
                    </p>

                    <div className="space-y-1.5 mb-4 text-[11px] f-mono">
                      <div className="flex items-center gap-1.5">
                        <span className={isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}>Files:</span>
                        <span className={`truncate ${isLight ? 'text-[#334155]' : 'text-[#dadbdf]'}`}>
                          {d.files.join(', ')}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className={isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}>Source:</span>
                        <a
                          href={d.source_url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[#0284c7] hover:underline truncate max-w-[280px]"
                        >
                          {d.source_url}
                        </a>
                      </div>
                    </div>
                  </div>

                  <div className={`pt-3 border-t flex items-center justify-between ${isLight ? 'border-slate-200' : 'border-[#212327]'}`}>
                    <span className={`text-[10px] f-mono ${isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}`}>
                      Status: <strong className={isLight ? 'text-emerald-700' : 'text-[#4fd1a1]'}>MOUNTED</strong>
                    </span>
                    <button
                      onClick={() => handleReplay(d.name, 30)}
                      disabled={replayingDataset === d.name}
                      className={`h-7 px-3.5 rounded-full text-[12px] border transition flex items-center gap-1.5 disabled:opacity-50 ${
                        isLight
                          ? 'border-slate-300 hover:border-slate-400 bg-white text-[#0f172a]'
                          : 'border-white/25 hover:border-white/50 hover:bg-white/5 text-white'
                      }`}
                    >
                      {replayingDataset === d.name ? (
                        <>
                          <div className={`w-2.5 h-2.5 border-2 border-t-transparent rounded-full animate-spin ${isLight ? 'border-slate-800' : 'border-white'}`} />
                          Streaming...
                        </>
                      ) : (
                        <>
                          <Play className={`w-3 h-3 ${isLight ? 'text-emerald-600' : 'text-[#4fd1a1]'}`} />
                          Replay 30 Txs
                        </>
                      )}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : activeTab === 'patterns' ? (
            <div className="space-y-4">
              <div className={`text-xs flex items-center justify-between f-mono ${isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}`}>
                <span>Ground-Truth Typologies from IBM AML HI-Large (Kaggle)</span>
                <span>{ibmPatterns.length} Typologies Identified</span>
              </div>

              <div className={`border rounded-lg overflow-hidden ${isLight ? 'border-slate-200 bg-white' : 'border-[#212327] bg-[#0a0a0a]'}`}>
                <table className="w-full text-left text-xs f-mono">
                  <thead className={`border-b ${isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#0d0e10] border-[#212327]'}`}>
                    <tr>
                      <th className="lbl px-4 py-3">Pattern ID</th>
                      <th className="lbl px-4 py-3">Typology Type</th>
                      <th className="lbl px-4 py-3">Total Amount</th>
                      <th className="lbl px-4 py-3">Hop Count</th>
                      <th className="lbl px-4 py-3">Entities Involved</th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y ${isLight ? 'divide-slate-200 text-[#334155]' : 'divide-[#212327] text-[#dadbdf]'}`}>
                    {ibmPatterns.map((p) => (
                      <tr key={p.pattern_id} className={`transition ${isLight ? 'hover:bg-slate-50' : 'hover:bg-[#121b27]'}`}>
                        <td className={`px-4 py-2.5 font-bold ${isLight ? 'text-[#0f172a]' : 'text-white'}`}>{p.pattern_id}</td>
                        <td className="px-4 py-2.5">
                          <span className={`px-2 py-0.5 rounded-full text-[10.5px] border ${
                            isLight
                              ? 'border-rose-200 bg-rose-50 text-rose-700'
                              : 'border-[#f0596b]/30 bg-[#2a1216] text-[#f0596b]'
                          }`}>
                            {p.type}
                          </span>
                        </td>
                        <td className={`px-4 py-2.5 font-semibold ${isLight ? 'text-[#0f172a]' : 'text-white'}`}>
                          ${p.total_amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </td>
                        <td className="px-4 py-2.5">{p.hop_count || 3}</td>
                        <td className={`px-4 py-2.5 truncate max-w-[200px] ${isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}`}>
                          {p.accounts?.join(', ') || 'Syndicate Group'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className={`text-xs f-mono ${isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}`}>
                SOTA Multi-Dataset PR-AUC &amp; Latency Benchmarks
              </div>

              <div className={`border rounded-lg overflow-hidden ${isLight ? 'border-slate-200 bg-white' : 'border-[#212327] bg-[#0a0a0a]'}`}>
                <table className="w-full text-left text-xs f-mono">
                  <thead className={`border-b ${isLight ? 'bg-slate-50 border-slate-200' : 'bg-[#0d0e10] border-[#212327]'}`}>
                    <tr>
                      <th className="lbl px-4 py-3">Dataset Name</th>
                      <th className="lbl px-4 py-3">Baseline PR-AUC</th>
                      <th className="lbl px-4 py-3">Safe-Bit PR-AUC</th>
                      <th className="lbl px-4 py-3">Gain</th>
                      <th className="lbl px-4 py-3">ROC-AUC</th>
                      <th className="lbl px-4 py-3">P99 Latency</th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y ${isLight ? 'divide-slate-200 text-[#334155]' : 'divide-[#212327] text-[#dadbdf]'}`}>
                    {benchmarkData ? (
                      Object.entries(benchmarkData).map(([name, b]) => (
                        <tr key={name} className={`transition ${isLight ? 'hover:bg-slate-50' : 'hover:bg-[#121b27]'}`}>
                          <td className={`px-4 py-3 font-medium capitalize ${isLight ? 'text-[#0f172a]' : 'text-white'}`}>
                            {name.replace(/_/g, ' ')}
                          </td>
                          <td className={`px-4 py-3 ${isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}`}>{b.baseline_pr_auc}</td>
                          <td className={`px-4 py-3 font-bold ${isLight ? 'text-emerald-700' : 'text-[#4fd1a1]'}`}>
                            {b.safebit_pr_auc}
                          </td>
                          <td className={`px-4 py-3 font-semibold ${isLight ? 'text-emerald-700' : 'text-[#4fd1a1]'}`}>
                            +{b.pr_auc_gain}
                          </td>
                          <td className="px-4 py-3">{b.safebit_roc_auc}</td>
                          <td className={`px-4 py-3 ${isLight ? 'text-sky-700 font-medium' : 'text-[#a0c3ec]'}`}>{b.p99_latency_ms} ms</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="6" className={`text-center py-8 ${isLight ? 'text-[#64748b]' : 'text-[#7d8187]'}`}>
                          Loading cross-dataset metrics...
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

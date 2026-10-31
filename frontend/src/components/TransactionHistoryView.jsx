import React, { useState, useEffect, useCallback } from 'react';
import {
  Search,
  Filter,
  ArrowUpDown,
  RefreshCw,
  SlidersHorizontal,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  ArrowRight,
  Eye,
  Clock,
  Layers,
  ChevronLeft,
  ChevronRight,
  Radio,
  FileSpreadsheet,
} from 'lucide-react';
import { fetchTransactionHistory } from '../services/api';

export default function TransactionHistoryView({
  theme = 'dark',
  onSelectTransaction,
  onBackToLive,
}) {
  const [transactions, setTransactions] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [decision, setDecision] = useState('ALL');
  const [typology, setTypology] = useState('ALL');
  const [sortBy, setSortBy] = useState('newest');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const isLight = theme === 'light';

  const loadData = useCallback(async () => {
    try {
      const offset = (page - 1) * pageSize;
      const res = await fetchTransactionHistory({
        limit: pageSize,
        offset,
        search,
        decision,
        typology,
        sort_by: sortBy,
      });
      if (res && res.transactions) {
        setTransactions(res.transactions);
        setTotal(res.total || 0);
      }
    } catch (err) {
      console.error('Failed to load transaction history:', err);
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, search, decision, typology, sortBy]);

  useEffect(() => {
    setLoading(true);
    loadData();
  }, [loadData]);

  // Periodic Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      loadData();
    }, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadData]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  // Compute summary stats for current slice
  const totalVolume = transactions.reduce((acc, t) => acc + (t.amount || 0), 0);
  const muleCount = transactions.filter(
    (t) => t.is_mule || t.decision === 'HELD_FOR_REVIEW' || (t.risk_score || 0) >= 0.85
  ).length;
  const slaMetCount = transactions.filter(
    (t) => (t.latency_ms?.total_ms || 12) <= 20.0
  ).length;
  const slaPct = transactions.length > 0
    ? ((slaMetCount / transactions.length) * 100).toFixed(0)
    : '100';

  const typologyColors = {
    scatter_gather: isLight ? 'bg-amber-100 text-amber-800 border-amber-300' : 'bg-amber-950/60 text-amber-300 border-amber-700/40',
    cycle: isLight ? 'bg-purple-100 text-purple-800 border-purple-300' : 'bg-purple-950/60 text-purple-300 border-purple-700/40',
    peeling_chain: isLight ? 'bg-pink-100 text-pink-800 border-pink-300' : 'bg-pink-950/60 text-pink-300 border-pink-700/40',
    fan_out: isLight ? 'bg-blue-100 text-blue-800 border-blue-300' : 'bg-blue-950/60 text-blue-300 border-blue-700/40',
    mule_ring: isLight ? 'bg-rose-100 text-rose-800 border-rose-300' : 'bg-rose-950/60 text-rose-300 border-rose-700/40',
    merchant: isLight ? 'bg-emerald-100 text-emerald-800 border-emerald-300' : 'bg-emerald-950/60 text-emerald-300 border-emerald-700/40',
    p2p: isLight ? 'bg-slate-200 text-slate-700 border-slate-300' : 'bg-slate-800 text-slate-300 border-slate-700',
  };

  return (
    <div className={`flex-1 flex flex-col overflow-hidden transition-colors ${
      isLight ? 'bg-[#f4f5f7] text-[#0f172a]' : 'bg-[#0a0a0a] text-white'
    }`}>
      {/* =================================================================== */}
      {/* 1. TOP SUBHEADER / BREADCRUMB                                       */}
      {/* =================================================================== */}
      <div className={`px-6 py-3 border-b flex items-center justify-between shrink-0 transition-colors ${
        isLight ? 'bg-white border-[#e2e8f0]' : 'bg-[#0e1014] border-[#212327]'
      }`}>
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToLive}
            className={`text-xs px-2.5 py-1 rounded-md border font-mono flex items-center gap-1.5 transition ${
              isLight
                ? 'border-slate-300 hover:bg-slate-100 text-slate-700'
                : 'border-white/10 hover:bg-white/10 text-slate-300'
            }`}
          >
            <ChevronLeft className="w-3.5 h-3.5" />
            <span>Return to Live Stream</span>
          </button>
          <span className="text-[#7d8187] font-mono text-xs">/</span>
          <span className="text-sm font-semibold tracking-tight">
            Transaction History Explorer
          </span>
          <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-[#a0c3ec]/10 text-[#a0c3ec] border border-[#a0c3ec]/20 font-medium">
            {total} Total Recorded
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Auto Refresh Toggle */}
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`h-7 px-2.5 rounded-md text-xs font-mono border flex items-center gap-1.5 transition ${
              autoRefresh
                ? isLight
                  ? 'bg-emerald-50 border-emerald-300 text-emerald-700'
                  : 'bg-emerald-950/40 border-emerald-500/40 text-emerald-400'
                : isLight
                ? 'border-slate-300 text-slate-500'
                : 'border-white/10 text-slate-400'
            }`}
            title="Toggle Live Auto-Sync"
          >
            <span className={`w-1.5 h-1.5 rounded-full ${autoRefresh ? 'bg-emerald-400 animate-pulse' : 'bg-slate-400'}`} />
            <span>{autoRefresh ? 'Live Sync Active' : 'Sync Paused'}</span>
          </button>

          {/* Manual Refresh */}
          <button
            onClick={loadData}
            disabled={loading}
            className={`h-7 px-2.5 rounded-md text-xs font-mono border flex items-center gap-1.5 transition ${
              isLight
                ? 'border-slate-300 hover:bg-slate-100 text-slate-700'
                : 'border-white/10 hover:bg-white/10 text-slate-300'
            }`}
            title="Refresh list now"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* =================================================================== */}
      {/* 2. SUMMARY METRICS RIBBON                                           */}
      {/* =================================================================== */}
      <div className={`h-11 px-6 border-b grid grid-cols-4 font-mono text-xs divide-x shrink-0 transition-colors ${
        isLight ? 'bg-white border-[#e2e8f0] divide-slate-200' : 'bg-[#0f1115] border-[#212327] divide-[#212327]'
      }`}>
        <div className="flex items-center justify-between px-3">
          <span className="text-[#7d8187] uppercase text-[11px]">Filtered Entries</span>
          <span className="font-bold text-sm">{total}</span>
        </div>
        <div className="flex items-center justify-between px-3">
          <span className="text-[#7d8187] uppercase text-[11px]">Flagged Mules</span>
          <span className="font-bold text-sm text-rose-500">{muleCount}</span>
        </div>
        <div className="flex items-center justify-between px-3">
          <span className="text-[#7d8187] uppercase text-[11px]">Batch Volume</span>
          <span className="font-bold text-sm">
            ${Math.round(totalVolume).toLocaleString()}
          </span>
        </div>
        <div className="flex items-center justify-between px-3">
          <span className="text-[#7d8187] uppercase text-[11px]">SLA Compliance (&lt;20ms)</span>
          <span className="font-bold text-sm text-emerald-400">{slaPct}%</span>
        </div>
      </div>

      {/* =================================================================== */}
      {/* 3. FILTERS & SEARCH CONTROLS                                       */}
      {/* =================================================================== */}
      <div className={`p-4 border-b flex flex-wrap items-center justify-between gap-3 shrink-0 transition-colors ${
        isLight ? 'bg-slate-50 border-[#e2e8f0]' : 'bg-[#12151b] border-[#212327]'
      }`}>
        {/* Search Input */}
        <div className="relative flex-1 min-w-[260px] max-w-md">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-[#7d8187]" />
          <input
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            placeholder="Search by Tx ID, Account, Device, or Subnet..."
            className={`w-full h-9 pl-9 pr-4 rounded-lg text-xs font-mono border focus:outline-none focus:ring-1 transition ${
              isLight
                ? 'bg-white border-slate-300 text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:ring-indigo-500'
                : 'bg-[#090b0e] border-white/10 text-white placeholder:text-slate-500 focus:border-[#a0c3ec] focus:ring-[#a0c3ec]'
            }`}
          />
        </div>

        {/* Filter Dropdowns */}
        <div className="flex items-center gap-2.5 flex-wrap">
          {/* Decision Filter */}
          <div className="flex items-center gap-1.5 text-xs font-mono">
            <span className="text-[#7d8187]">Decision:</span>
            <select
              value={decision}
              onChange={(e) => {
                setDecision(e.target.value);
                setPage(1);
              }}
              className={`h-9 px-2.5 rounded-lg border text-xs font-mono transition cursor-pointer ${
                isLight
                  ? 'bg-white border-slate-300 text-slate-800'
                  : 'bg-[#14171d] border-white/15 text-white'
              }`}
            >
              <option value="ALL">All Decisions</option>
              <option value="HELD_FOR_REVIEW">Held for Review (Mules)</option>
              <option value="AUTO_APPROVE">Auto-Approved</option>
              <option value="FLAGGED_FOR_MONITORING">Flagged for Monitoring</option>
            </select>
          </div>

          {/* Typology Filter */}
          <div className="flex items-center gap-1.5 text-xs font-mono">
            <span className="text-[#7d8187]">Typology:</span>
            <select
              value={typology}
              onChange={(e) => {
                setTypology(e.target.value);
                setPage(1);
              }}
              className={`h-9 px-2.5 rounded-lg border text-xs font-mono transition cursor-pointer ${
                isLight
                  ? 'bg-white border-slate-300 text-slate-800'
                  : 'bg-[#14171d] border-white/15 text-white'
              }`}
            >
              <option value="ALL">All Typologies</option>
              <option value="scatter_gather">Scatter-Gather</option>
              <option value="cycle">Cycle Ring</option>
              <option value="peeling_chain">Peeling Chain</option>
              <option value="fan_out">Fan-Out</option>
              <option value="mule_ring">Mule Ring</option>
              <option value="p2p">Normal P2P</option>
              <option value="merchant">Merchant</option>
            </select>
          </div>

          {/* Sort By Filter */}
          <div className="flex items-center gap-1.5 text-xs font-mono">
            <span className="text-[#7d8187]">Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => {
                setSortBy(e.target.value);
                setPage(1);
              }}
              className={`h-9 px-2.5 rounded-lg border text-xs font-mono transition cursor-pointer ${
                isLight
                  ? 'bg-white border-slate-300 text-slate-800'
                  : 'bg-[#14171d] border-white/15 text-white'
              }`}
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="highest_risk">Highest Risk</option>
              <option value="lowest_risk">Lowest Risk</option>
              <option value="highest_amount">Highest Amount</option>
              <option value="fastest_latency">Fastest Latency</option>
            </select>
          </div>
        </div>
      </div>

      {/* =================================================================== */}
      {/* 4. TRANSACTIONS TABLE                                               */}
      {/* =================================================================== */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-left text-xs font-mono border-collapse">
          <thead className={`sticky top-0 z-10 border-b text-[11px] uppercase tracking-wider transition-colors ${
            isLight
              ? 'bg-slate-100 text-slate-600 border-slate-200'
              : 'bg-[#14171d] text-[#7d8187] border-[#212327]'
          }`}>
            <tr>
              <th className="py-2.5 px-4 font-medium">Tx ID & Timestamp</th>
              <th className="py-2.5 px-4 font-medium">Typology</th>
              <th className="py-2.5 px-4 font-medium">Originator &rarr; Beneficiary</th>
              <th className="py-2.5 px-4 font-medium text-right">Amount</th>
              <th className="py-2.5 px-4 font-medium">Channel</th>
              <th className="py-2.5 px-4 font-medium">Mule Risk</th>
              <th className="py-2.5 px-4 font-medium">Decision</th>
              <th className="py-2.5 px-4 font-medium text-right">Latency</th>
              <th className="py-2.5 px-4 font-medium text-center">Action</th>
            </tr>
          </thead>
          <tbody className={`divide-y transition-colors ${
            isLight ? 'divide-slate-200' : 'divide-[#1c1f26]'
          }`}>
            {loading && transactions.length === 0 ? (
              <tr>
                <td colSpan="9" className="py-12 text-center text-slate-500">
                  <RefreshCw className="w-5 h-5 mx-auto mb-2 animate-spin text-[#a0c3ec]" />
                  Loading transactions history...
                </td>
              </tr>
            ) : transactions.length === 0 ? (
              <tr>
                <td colSpan="9" className="py-12 text-center text-slate-500">
                  No transactions match the current filter criteria.
                </td>
              </tr>
            ) : (
              transactions.map((tx) => {
                const isHeld = tx.decision === 'HELD_FOR_REVIEW';
                const riskPct = ((tx.risk_score || 0) * 100).toFixed(1);
                const typoStyle = typologyColors[tx.typology] || typologyColors.p2p;

                return (
                  <tr
                    key={tx.tx_id}
                    onClick={() => onSelectTransaction(tx)}
                    className={`cursor-pointer transition-colors group ${
                      isLight
                        ? 'hover:bg-slate-100/80'
                        : 'hover:bg-[#15181f]'
                    }`}
                  >
                    {/* Tx ID & Time */}
                    <td className="py-3 px-4">
                      <div className="font-semibold text-[#a0c3ec] group-hover:underline flex items-center gap-1.5">
                        <span>{tx.tx_id}</span>
                      </div>
                      <div className="text-[10px] text-[#7d8187] mt-0.5">
                        {tx.formatted_time || 'Just now'}
                      </div>
                    </td>

                    {/* Typology */}
                    <td className="py-3 px-4">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-mono uppercase tracking-wider border font-medium ${typoStyle}`}>
                        {tx.typology || 'P2P'}
                      </span>
                    </td>

                    {/* Parties */}
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-1.5 text-xs">
                        <span className="text-emerald-400 font-medium">
                          {tx.src_account}
                        </span>
                        <ArrowRight className="w-3 h-3 text-[#7d8187] shrink-0" />
                        <span className={isHeld ? 'text-rose-400 font-medium' : 'text-[#a0c3ec] font-medium'}>
                          {tx.dst_account}
                        </span>
                      </div>
                      <div className="text-[10px] text-[#7d8187] mt-0.5">
                        IP: {tx.ip_subnet || '192.168.1.0/24'}
                      </div>
                    </td>

                    {/* Amount */}
                    <td className="py-3 px-4 text-right font-semibold">
                      ${Number(tx.amount || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>

                    {/* Channel */}
                    <td className="py-3 px-4">
                      <span className={`text-[11px] px-1.5 py-0.5 rounded border ${
                        isLight ? 'bg-slate-100 border-slate-300 text-slate-700' : 'bg-white/5 border-white/10 text-slate-300'
                      }`}>
                        {tx.payment_channel || 'WIRE'}
                      </span>
                    </td>

                    {/* Risk Bar & Score */}
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full ${
                              isHeld ? 'bg-rose-500' : (tx.risk_score || 0) > 0.4 ? 'bg-amber-400' : 'bg-emerald-400'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(5, (tx.risk_score || 0) * 100))}%` }}
                          />
                        </div>
                        <span className={`font-bold ${
                          isHeld ? 'text-rose-500' : (tx.risk_score || 0) > 0.4 ? 'text-amber-400' : 'text-emerald-400'
                        }`}>
                          {riskPct}%
                        </span>
                      </div>
                    </td>

                    {/* Decision Badge */}
                    <td className="py-3 px-4">
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border inline-flex items-center gap-1 ${
                        isHeld
                          ? isLight
                            ? 'bg-rose-100 text-rose-800 border-rose-300'
                            : 'bg-rose-950/60 text-rose-300 border-rose-600/40'
                          : isLight
                          ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                          : 'bg-emerald-950/60 text-emerald-300 border-emerald-600/40'
                      }`}>
                        {isHeld ? <ShieldAlert className="w-2.5 h-2.5" /> : <ShieldCheck className="w-2.5 h-2.5" />}
                        {isHeld ? 'HELD' : 'APPROVED'}
                      </span>
                    </td>

                    {/* Latency */}
                    <td className="py-3 px-4 text-right">
                      <span className="text-emerald-400 font-medium">
                        {tx.latency_ms?.total_ms || 11.8} ms
                      </span>
                    </td>

                    {/* Action */}
                    <td className="py-3 px-4 text-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectTransaction(tx);
                        }}
                        className={`p-1.5 rounded-lg border transition ${
                          isLight
                            ? 'border-slate-300 hover:bg-slate-200 text-slate-700'
                            : 'border-white/10 hover:bg-white/10 text-slate-300'
                        }`}
                        title="View all details of this transaction"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* =================================================================== */}
      {/* 5. PAGINATION FOOTER                                                */}
      {/* =================================================================== */}
      <div className={`p-4 border-t flex items-center justify-between shrink-0 font-mono text-xs transition-colors ${
        isLight ? 'bg-white border-[#e2e8f0]' : 'bg-[#0e1014] border-[#212327]'
      }`}>
        <div className="text-[#7d8187]">
          Showing {Math.min((page - 1) * pageSize + 1, total)} to {Math.min(page * pageSize, total)} of {total} transactions
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-[#7d8187]">
            <span>Rows per page:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setPage(1);
              }}
              className={`h-7 px-2 rounded border text-xs font-mono ${
                isLight ? 'bg-white border-slate-300 text-slate-800' : 'bg-[#14171d] border-white/15 text-white'
              }`}
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className={`p-1.5 rounded border transition disabled:opacity-30 disabled:cursor-not-allowed ${
                isLight ? 'border-slate-300 hover:bg-slate-100' : 'border-white/10 hover:bg-white/10'
              }`}
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span className="px-2">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className={`p-1.5 rounded border transition disabled:opacity-30 disabled:cursor-not-allowed ${
                isLight ? 'border-slate-300 hover:bg-slate-100' : 'border-white/10 hover:bg-white/10'
              }`}
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}


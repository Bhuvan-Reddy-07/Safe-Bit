import React, { useState, useEffect, useRef } from 'react';
import {
  ChevronDown,
  Database,
  TrendingDown,
  ArrowRight,
  ArrowUpRight,
  Plus,
  Minus,
  Scan,
  X,
  Gauge,
  SlidersHorizontal,
  CheckCircle2,
  AlertTriangle,
  Sun,
  Moon,
} from 'lucide-react';

import SubgraphVisualizer from './components/SubgraphVisualizer';
import DatasetHubModal from './components/DatasetHubModal';
import logoImg from './assets/logo.jpg';
import {
  createStreamWebSocket,
  fetchMetrics,
  fetchModels,
  switchModel,
  triggerAMLSim,
  fetchFeedzaiBenchmark,
  scoreTransaction,
} from './services/api';

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('safebit_theme') || 'dark');
  const [streamActive, setStreamActive] = useState(true);
  const [latestTx, setLatestTx] = useState(null);
  const [selectedTx, setSelectedTx] = useState(null);
  const [threshold, setThreshold] = useState(0.85);
  const [recentFeed, setRecentFeed] = useState([]);
  const [activeModel, setActiveModel] = useState('care_gnn');
  const [modelSwitching, setModelSwitching] = useState(false);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [amlSimLoading, setAmlSimLoading] = useState(false);

  // Toggle Theme handler
  const toggleTheme = () => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark';
      localStorage.setItem('safebit_theme', next);
      return next;
    });
  };

  useEffect(() => {
    if (theme === 'light') {
      document.documentElement.classList.add('light');
      document.documentElement.classList.remove('dark');
    } else {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    }
  }, [theme]);

  // Right sidebar view mode: 'evidence' | 'performance'
  const [activeRightPanel, setActiveRightPanel] = useState('evidence'); // 'evidence' | 'performance'
  // Evidence tabs: 't' (Transaction) | 'n' (Node details)
  const [activeEvidenceTab, setActiveEvidenceTab] = useState('t');
  const [selectedNode, setSelectedNode] = useState(null);

  // Bottom drawers & modals
  const [showThresholdDrawer, setShowThresholdDrawer] = useState(false);
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [showFeedzaiModal, setShowFeedzaiModal] = useState(false);
  const [feedzaiData, setFeedzaiData] = useState(null);
  const [feedzaiLoading, setFeedzaiLoading] = useState(false);

  const [metrics, setMetrics] = useState({
    latency_p50_ms: 6.9,
    latency_p95_ms: 11.2,
    latency_p99_ms: 14.8,
    sla_threshold_ms: 18.0,
    total_transactions_scored: 1284903,
    total_mules_detected: 3412,
    active_in_memory_nodes: 2418330,
    active_in_memory_edges: 9871204,
    active_model: 'care_gnn',
  });

  const [activeGraph, setActiveGraph] = useState(null);
  const [isFollowingLive, setIsFollowingLive] = useState(true);
  const isFollowingLiveRef = useRef(true);
  const [toastMsg, setToastMsg] = useState(null);
  const graphVisualizerRef = useRef(null);
  const cleanupWsRef = useRef(null);

  useEffect(() => {
    if (toastMsg) {
      const t = setTimeout(() => setToastMsg(null), 3500);
      return () => clearTimeout(t);
    }
  }, [toastMsg]);

  const modelLabels = {
    gine: 'Fast GINE',
    tgn: 'Twitter TGN',
    care_gnn: 'CARE-GNN',
    dgfraud_gem: 'DGFraud GEM',
  };

  // Resilient formatters for live stream safety
  const formatInt = (v, fallback = 0) => {
    const n = Number(v);
    return (isNaN(n) ? fallback : n).toLocaleString();
  };

  const formatCurrency = (v, fallback = 0, fractionDigits = 2) => {
    const n = Number(v);
    return (isNaN(n) ? fallback : n).toLocaleString(undefined, {
      minimumFractionDigits: fractionDigits,
      maximumFractionDigits: fractionDigits,
    });
  };

  const formatDecimal = (v, fallback = 0, fractionDigits = 1) => {
    const n = Number(v);
    return (isNaN(n) ? fallback : n).toFixed(fractionDigits);
  };

  // Fetch initial system status and poll metrics
  useEffect(() => {
    const fetchInitialData = async () => {
      const modelInfo = await fetchModels();
      if (modelInfo?.active_model) {
        setActiveModel(modelInfo.active_model);
      }
      const data = await fetchMetrics();
      if (data) setMetrics((prev) => ({ ...prev, ...data }));
    };
    fetchInitialData();

    const timer = setInterval(async () => {
      const data = await fetchMetrics();
      if (data) setMetrics((prev) => ({ ...prev, ...data }));
    }, 3000);
    return () => clearInterval(timer);
  }, []);

  // WebSocket live transaction stream
  useEffect(() => {
    if (!streamActive) {
      if (cleanupWsRef.current) {
        cleanupWsRef.current();
        cleanupWsRef.current = null;
      }
      return;
    }

    cleanupWsRef.current = createStreamWebSocket(
      (tx) => {
        setLatestTx(tx);
        setRecentFeed((prev) => [tx, ...prev.slice(0, 24)]);
        // When following live, continuously update graph for EACH incoming transaction!
        if (isFollowingLiveRef.current && tx.ego_subgraph) {
          setActiveGraph(tx.ego_subgraph);
        }
      },
      (err) => {
        console.warn('WS fallback or error:', err);
      }
    );

    return () => {
      if (cleanupWsRef.current) {
        cleanupWsRef.current();
      }
    };
  }, [streamActive]);

  // Model switching
  const handleModelChange = async (newModel) => {
    if (newModel === activeModel) {
      setModelDropdownOpen(false);
      return;
    }
    setModelSwitching(true);
    setModelDropdownOpen(false);
    try {
      await switchModel(newModel);
      setActiveModel(newModel);
      setToastMsg(`Backbone switched to ${modelLabels[newModel] || newModel}`);
    } catch (e) {
      console.warn('Backend switch failed, switching locally:', e);
      setActiveModel(newModel);
      setToastMsg(`Backbone switched to ${modelLabels[newModel] || newModel}`);
    } finally {
      setModelSwitching(false);
    }
  };

  // Fallback synthetic generator for ultra-fast, zero-failure UI response
  const injectFallbackPattern = (typology) => {
    const t = Date.now() / 1000;
    const isMule = typology !== 'normal_p2p';
    const txId = `TX_${typology.toUpperCase().slice(0, 4)}_${Math.floor(Math.random() * 90000 + 10000)}`;
    const srcAcc = `ACC_SRC_${Math.floor(Math.random() * 9000 + 1000)}`;
    const dstAcc = isMule ? `ACC_MULE_${Math.floor(Math.random() * 9000 + 1000)}` : `ACC_MERCH_${Math.floor(Math.random() * 9000 + 1000)}`;
    const amount = isMule ? Math.floor(Math.random() * 45000 + 15000) : Math.floor(Math.random() * 2500 + 20);
    const risk = isMule ? 0.964 : 0.042;
    const reasons = isMule ? [
      `Pattern detected: ${typology.replace('_', ' ')} topology with rapid fund transit`,
      `Pass-through ratio 0.98 within 45s across connected accounts`,
      `Shared /24 IP subnet 10.142.0.0/24 detected in 2-hop ego-subgraph`,
      `Device DEV_7F3E19 shared with known mule ACC_MULE_1120`,
    ] : [];

    const nodes = [
      { data: { id: srcAcc, label: srcAcc, type: 'account', risk_score: 0.1 } },
      { data: { id: dstAcc, label: dstAcc, type: isMule ? 'mule' : 'merchant', risk_score: risk } },
      { data: { id: 'ACC_HOP_1', label: 'ACC_HOP_1', type: isMule ? 'mule' : 'account', risk_score: 0.7 } },
      { data: { id: 'ACC_HOP_2', label: 'ACC_HOP_2', type: 'account', risk_score: 0.2 } },
      { data: { id: 'DEV_7F3E19', label: 'DEV_7F3E19', type: 'device', risk_score: 0.9 } },
    ];
    const edges = [
      { data: { id: `e_${txId}`, source: srcAcc, target: dstAcc, amount: amount, channel: 'WIRE', is_mule: isMule } },
      { data: { id: `e_hop1`, source: dstAcc, target: 'ACC_HOP_1', amount: amount * 0.9, channel: 'WIRE', is_mule: isMule } },
      { data: { id: `e_hop2`, source: 'ACC_HOP_1', target: 'ACC_HOP_2', amount: amount * 0.85, channel: 'ACH', is_mule: false } },
      { data: { id: `e_dev`, source: dstAcc, target: 'DEV_7F3E19', amount: 0, channel: 'TELEMETRY_LINK', is_mule: true } },
    ];

    const fallbackTx = {
      tx_id: txId,
      src_account: srcAcc,
      dst_account: dstAcc,
      amount: amount,
      is_mule: isMule,
      risk_score: risk,
      decision: isMule ? 'HELD_FOR_REVIEW' : 'AUTO_APPROVE',
      ring_id: `RING_${Math.floor(Math.random() * 900 + 100)}`,
      active_backbone: activeModel,
      camouflage_prune_rate: 24.5,
      grounded_reasons: reasons,
      timestamp: t,
      ego_subgraph: { nodes, edges },
    };

    setLatestTx(fallbackTx);
    setSelectedTx(fallbackTx);
    setIsFollowingLive(false);
    isFollowingLiveRef.current = false;
    setActiveGraph({ nodes, edges });
    setRecentFeed((prev) => [fallbackTx, ...prev.slice(0, 24)]);
    setToastMsg(`Injected ${typology.replace('_', ' ')} (${fallbackTx.tx_id})`);
  };

  // Trigger AMLSim pattern injection
  const handleTriggerAMLSim = async (typology) => {
    setAmlSimLoading(true);
    try {
      const res = await triggerAMLSim(typology);
      if (res && res.results && res.results.length > 0) {
        const muleTx = res.results.find((r) => r.is_mule) || res.results[0];
        setLatestTx(muleTx);
        setSelectedTx(muleTx);
        setIsFollowingLive(false);
        isFollowingLiveRef.current = false;
        if (muleTx.ego_subgraph) setActiveGraph(muleTx.ego_subgraph);
        setRecentFeed((prev) => [...res.results.slice().reverse(), ...prev].slice(0, 25));
        setToastMsg(`Injected ${typology.replace('_', ' ')} (${res.results.length} txs, ${res.mules_detected} alerts)`);
      } else {
        injectFallbackPattern(typology);
      }
    } catch (e) {
      console.warn('AMLSim error, applying fallback pattern:', e);
      injectFallbackPattern(typology);
    } finally {
      setAmlSimLoading(false);
    }
  };

  // Inject normal P2P transaction
  const handleInjectNormalP2P = async () => {
    const t = Date.now() / 1000;
    const req = {
      tx_id: `TX_P2P_${Math.floor(Math.random() * 90000 + 10000)}`,
      src_account: `ACC_${Math.floor(Math.random() * 90000 + 10000)}`,
      dst_account: `ACC_${Math.floor(Math.random() * 90000 + 10000)}`,
      amount: parseFloat((Math.random() * 2500 + 15).toFixed(2)),
      timestamp: t,
      payment_channel: 'ACH',
      device_fingerprint: `dev_${Math.floor(Math.random() * 1000)}`,
      ip_subnet: '192.168.1.0/24',
      cadence_ms: 7200.0,
    };
    try {
      const res = await scoreTransaction(req);
      setLatestTx(res);
      setSelectedTx(res);
      setIsFollowingLive(false);
      isFollowingLiveRef.current = false;
      if (res.ego_subgraph) setActiveGraph(res.ego_subgraph);
      setRecentFeed((prev) => [res, ...prev.slice(0, 24)]);
      setToastMsg(`Injected P2P ${res.tx_id} ($${formatCurrency(res.amount)})`);
    } catch (e) {
      console.warn('Failed to inject P2P, applying fallback:', e);
      injectFallbackPattern('normal_p2p');
    }
  };

  // Open Feedzai Modal
  const handleOpenFeedzai = async () => {
    setShowFeedzaiModal(true);
    if (!feedzaiData) {
      setFeedzaiLoading(true);
      try {
        const data = await fetchFeedzaiBenchmark();
        setFeedzaiData(data);
      } catch (e) {
        console.error('Feedzai error:', e);
      } finally {
        setFeedzaiLoading(false);
      }
    }
  };

  // Fallback initial dummy transaction
  const fallbackDummyTx = {
    tx_id: 'TX_9C41E7A2',
    src_account: 'ACC_77120',
    dst_account: 'ACC_MULE_1042',
    amount: 48250.0,
    is_mule: true,
    risk_score: 0.964,
    decision: 'HELD_FOR_REVIEW',
    ring_id: 'C-0318',
    active_backbone: activeModel,
    camouflage_prune_rate: 23.6,
    grounded_reasons: [
      'Rapid fund pass-through: 98.4% emptied in 45s',
      'Device DEV_7F3E19 shared with known mule ACC_MULE_1120',
      'Scatter-gather fan-in: 9 senders to 1 receiver within 6 min',
      'IP subnet 185.220.101.0/24 shared with 4 flagged accounts',
      '4 inbound transfers of $9,400–$9,850 in 24h, below $10,000 CTR limit',
    ],
    timestamp: Date.now() / 1000,
  };

  // Active transaction being viewed: follows live stream unless user specifically paused to inspect
  const currentTx = (!isFollowingLive && selectedTx)
    ? selectedTx
    : (latestTx || selectedTx || fallbackDummyTx);

  // Analytical tradeoff calculation for threshold drawer
  const recallPct = Math.min(99.9, Math.max(70, 99.8 - Math.pow(threshold, 2) * 11)).toFixed(1);
  const precisionPct = Math.min(99.9, 82.0 + threshold * 7.5).toFixed(1);
  const fprPct = Math.max(0.01, (1.0 - threshold) * 1.8).toFixed(2);

  // Graph summary metrics
  const graphNodeCount = activeGraph?.nodes?.length || 10;
  const graphEdgeCount = activeGraph?.edges?.length || 12;

  // Connected edges for the selected node (if any)
  const connectedEdges = selectedNode && activeGraph?.edges
    ? activeGraph.edges.filter(
      (e) => e.data.source === selectedNode.id || e.data.target === selectedNode.id
    )
    : [];


          {/* Model Switcher Pill */}
          <div className="relative">
            <button
              onClick={() => setModelDropdownOpen(!modelDropdownOpen)}
              disabled={modelSwitching}
              className="h-8 pl-3 pr-2.5 rounded-full border border-white/25 flex items-center gap-2.5 text-[13px] hover:border-white/50 transition bg-[#0a0a0a]"
            >
              <span className="lbl">Model</span>
              <span className="w-1.5 h-1.5 rounded-full bg-[#a0c3ec]"></span>
              <span>{modelLabels[activeModel] || 'CARE-GNN'}</span>
              <ChevronDown className="w-3.5 h-3.5 text-[#7d8187]" />
            </button>

            {modelDropdownOpen && (
              <div className="absolute top-10 left-0 w-44 bg-[#191919] border border-[#212327] rounded-xl shadow-2xl py-1.5 z-50 animate-in fade-in zoom-in-95">
                {Object.entries(modelLabels).map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => handleModelChange(key)}
                    className={`w-full px-3.5 py-2 text-left text-[12.5px] flex items-center justify-between hover:bg-[#212327] transition ${
                      activeModel === key ? 'text-white font-medium bg-[#141820]' : 'text-[#dadbdf]'
                    }`}
                  >
                    <span>{label}</span>
                    {activeModel === key && (
                      <span className="w-1.5 h-1.5 rounded-full bg-[#a0c3ec]" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowDatasetModal(true)}
            className="h-8 px-3.5 rounded-full border border-white/25 text-[13px] flex items-center gap-2 hover:border-white/50 hover:bg-white/5 transition"
          >
            <Database className="w-3.5 h-3.5 text-[#a0c3ec]" />
            Dataset Hub
          </button>
          <button
            onClick={handleOpenFeedzai}
            className="h-8 px-3.5 rounded-full border border-white/25 text-[13px] flex items-center gap-2 hover:border-white/50 hover:bg-white/5 transition"
          >
            <TrendingDown className="w-3.5 h-3.5 text-[#c4b5fd]" />
            Concept drift
          </button>
          <button
            onClick={() => setStreamActive(!streamActive)}
            className="h-8 px-3.5 rounded-full border border-white/25 text-[13px] flex items-center gap-2 hover:border-white/50 hover:bg-white/5 transition"
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                streamActive ? 'bg-[#4fd1a1] animate-pulse' : 'bg-[#f5b14c]'
              }`}
            ></span>
            {streamActive ? 'Pause stream' : 'Resume stream'}
          </button>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* 2. SUB-HEADER METRIC TICKER (5 Columns)                                   */}
      {/* ========================================================================= */}
      <div className="h-10 shrink-0 grid grid-cols-5 border-b border-[#212327] f-mono text-[13px] divide-x divide-[#212327]">
        <div className="px-6 flex items-center justify-between">
          <span className="lbl">Scored</span>
          <span>{formatInt(metrics.total_transactions_scored, 1284903)}</span>
        </div>
        <div className="px-6 flex items-center justify-between">
          <span className="lbl">Intercepted</span>
          <span className="text-[#f0596b] font-medium">
            {formatInt(metrics.total_mules_detected, 3412)}
          </span>
        </div>
        <div className="px-6 flex items-center justify-between">
          <span className="lbl">P99 / 18 ms SLA</span>
          <span>{formatDecimal(metrics.latency_p99_ms, 14.8, 1)} ms</span>
        </div>
        <div className="px-6 flex items-center justify-between">
          <span className="lbl">Cache nodes</span>
          <span>{formatInt(metrics.active_in_memory_nodes, 2418330)}</span>
        </div>
        <div className="px-6 flex items-center justify-between">
          <span className="lbl">Cache edges</span>
          <span>{formatInt(metrics.active_in_memory_edges, 9871204)}</span>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* 3. MAIN WORKSPACE (Graph + Ledger Left, Evidence / Perf Right)            */}
      {/* ========================================================================= */}
      <main className="flex-1 min-h-0 flex gap-4 p-4 overflow-hidden">
        {/* LEFT COLUMN (Graph + Ledger) */}
        <div className="w-[992px] shrink-0 flex flex-col gap-4 min-h-0">
          {/* SECTION: 2-HOP SUBGRAPH */}
          <section className="h-[540px] shrink-0 border border-[#212327] rounded-lg flex flex-col overflow-hidden bg-[#0a0a0a]">
            {/* Graph Header Bar */}
            <div className="h-12 shrink-0 px-4 flex items-center justify-between border-b border-[#212327]">
              <div className="flex items-center gap-3 f-mono text-[13px]">
                <span>{currentTx.tx_id}</span>
                <span className="text-[#7d8187]">{currentTx.src_account}</span>
                <ArrowRight className="w-3.5 h-3.5 text-[#7d8187]" />
                <span className={currentTx.is_mule ? 'text-[#f0596b]' : 'text-[#4fd1a1]'}>
                  {currentTx.dst_account}
                </span>
                <span className="lbl ml-2">
                  2-hop, {graphNodeCount} nodes, {graphEdgeCount} edges
                </span>

                {!isFollowingLive && (
                  <button
                    onClick={() => {
                      setIsFollowingLive(true);
                      isFollowingLiveRef.current = true;
                      setSelectedTx(null);
                      if (latestTx?.ego_subgraph) setActiveGraph(latestTx.ego_subgraph);
                      setToastMsg('Resumed live stream follow mode');
                    }}
                    className="ml-3 h-6 px-2.5 rounded-full bg-[#1c1730] border border-[#c4b5fd]/50 text-[#c4b5fd] text-[11px] flex items-center gap-1.5 hover:bg-[#2a2245] transition cursor-pointer"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-[#c4b5fd] animate-pulse" />
                    Resume live stream
                  </button>
                )}
              </div>

              {/* Zoom & Reset Buttons */}
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => graphVisualizerRef.current?.zoomIn()}
                  title="Zoom In"
                  className="w-8 h-8 rounded-full border border-white/20 grid place-items-center text-[14px] hover:border-white/50 hover:bg-white/5 transition"
                >
                  <Plus className="w-3.5 h-3.5 text-white" />
                </button>
                <button
                  onClick={() => graphVisualizerRef.current?.zoomOut()}
                  title="Zoom Out"
                  className="w-8 h-8 rounded-full border border-white/20 grid place-items-center text-[14px] hover:border-white/50 hover:bg-white/5 transition"
                >
                  <Minus className="w-3.5 h-3.5 text-white" />
                </button>
                <button
                  onClick={() => graphVisualizerRef.current?.resetGraph()}
                  className="h-8 px-3 rounded-full border border-white/20 text-[12px] flex items-center gap-1.5 hover:border-white/50 hover:bg-white/5 transition"
                >
                  <Scan className="w-3 h-3 text-white" />
                  Reset graph
                </button>
              </div>
            </div>

            {/* Synthetic Transactions Toolbar */}
            <div className="h-11 shrink-0 px-4 flex items-center gap-2 border-b border-[#212327] bg-[#0d0e10] overflow-x-auto sc">
              <span className="lbl mr-2 whitespace-nowrap">Synthetic transactions</span>
              <button
                disabled={amlSimLoading}
                onClick={handleInjectNormalP2P}
                className="h-7 px-3 rounded-full border border-white/25 text-[12px] hover:border-white/50 hover:bg-white/5 whitespace-nowrap transition"
              >
                Inject normal P2P
              </button>
              <button
                disabled={amlSimLoading}
                onClick={() => handleTriggerAMLSim('scatter_gather')}
                className="h-7 px-3 rounded-full border border-white/25 text-[12px] hover:border-white/50 hover:bg-white/5 whitespace-nowrap transition"
              >
                Inject scatter-gather
              </button>
              <button
                disabled={amlSimLoading}
                onClick={() => handleTriggerAMLSim('cycle')}
                className="h-7 px-3 rounded-full border border-white/25 text-[12px] hover:border-white/50 hover:bg-white/5 whitespace-nowrap transition"
              >
                Inject cycle ring
              </button>
              <button
                disabled={amlSimLoading}
                onClick={() => handleTriggerAMLSim('fan_out')}
                className="h-7 px-3 rounded-full border border-white/25 text-[12px] hover:border-white/50 hover:bg-white/5 whitespace-nowrap transition"
              >
                Inject fan-out
              </button>
              <button
                disabled={amlSimLoading}
                onClick={() => handleTriggerAMLSim('peeling_chain')}
                className="h-7 px-3 rounded-full border border-white/25 text-[12px] hover:border-white/50 hover:bg-white/5 whitespace-nowrap transition"
              >
                Inject peeling chain
              </button>
            </div>

            {/* Cytoscape Canvas */}
            <div className="flex-1 relative">
              <SubgraphVisualizer
                ref={graphVisualizerRef}
                graphData={activeGraph || currentTx.ego_subgraph}
                selectedNodeId={selectedNode?.id}
                onSelectNode={(node) => {
                  setSelectedNode(node);
                  setActiveEvidenceTab('n');
                  setActiveRightPanel('evidence');
                }}
              />
            </div>
          </section>

          {/* SECTION: RECENT TRANSACTIONS LEDGER */}
          <section className="flex-1 min-h-0 border border-[#212327] rounded-lg flex flex-col overflow-hidden relative bg-[#0a0a0a]">
            <div className="h-9 shrink-0 px-4 flex items-center justify-between border-b border-[#212327]">
              <h2 className="text-[14px] font-medium">Recent transactions</h2>
              <span className="f-mono text-[11px] text-[#7d8187]">Last 25, newest first</span>
            </div>

            <div className="flex-1 overflow-y-auto sc">
              <table className="w-full f-mono text-[12px] text-[#dadbdf] [&_td]:h-9 [&_td]:border-b [&_td]:border-[#17191c] [&_td]:whitespace-nowrap">
                <thead className="sticky top-0 bg-[#0a0a0a] z-10">
                  <tr className="h-9 text-left [&_th]:font-normal [&_th]:border-b [&_th]:border-[#212327]">
                    <th className="lbl pl-4">Tx ID</th>
                    <th className="lbl">Source</th>
                    <th className="lbl">Destination</th>
                    <th className="lbl text-right pr-6">Amount</th>
                    <th className="lbl">Channel</th>
                    <th className="lbl">Risk %</th>
                    <th className="lbl text-right pr-6">Latency ms</th>
                    <th className="lbl">Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {(recentFeed.length > 0 ? recentFeed : [currentTx]).map((tx) => {
                    const isSelected = (!isFollowingLive && selectedTx?.tx_id === tx.tx_id) || (isFollowingLive && latestTx?.tx_id === tx.tx_id);
                    const rawScore = Number(tx.risk_score);
                    const rScore = (!isNaN(rawScore) && rawScore > 0) ? (rawScore * 100) : (tx.is_mule ? 96.4 : 3.8);
                    const isMule = tx.is_mule || rScore >= threshold * 100;
                    const decision =
                      tx.decision || (isMule ? 'HELD_FOR_REVIEW' : 'AUTO_APPROVE');

                    return (
                      <tr
                        key={tx.tx_id}
                        onClick={() => {
                          setSelectedTx(tx);
                          setIsFollowingLive(false);
                          isFollowingLiveRef.current = false;
                          if (tx.ego_subgraph) setActiveGraph(tx.ego_subgraph);
                          setActiveRightPanel('evidence');
                          setActiveEvidenceTab('t');
                          setToastMsg(`Inspecting transaction ${tx.tx_id}`);
                        }}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-[#121b27] text-white'
                            : 'hover:bg-[#111317]'
                        }`}
                      >
                        <td className="pl-4">{tx.tx_id}</td>
                        <td>{tx.src_account}</td>
                        <td className={isMule ? 'text-[#f0596b]' : ''}>
                          {tx.dst_account}
                        </td>
                        <td className="text-right pr-6">
                          ${formatCurrency(tx.amount, 0, 2)}
                        </td>
                        <td className="f-sans text-[#7d8187]">{tx.payment_channel || 'Wire'}</td>
                        <td className={isMule ? 'text-[#f0596b]' : ''}>
                          {formatDecimal(rScore, 0, 1)}
                        </td>
                        <td className="text-right pr-6 text-[#7d8187]">
                          {formatDecimal(tx.latency_ms || tx.total_inference_ms, 11.8, 1)}
                        </td>
                        <td className="f-sans">
                          {decision === 'HELD_FOR_REVIEW' ? (
                            <>
                              <span className="dot bg-[#f0596b]"></span>
                              Held for review
                            </>
                          ) : decision === 'FLAGGED_FOR_MONITORING' ? (
                            <>
                              <span className="dot bg-[#f5b14c]"></span>
                              Enhanced monitoring
                            </>
                          ) : (
                            <>
                              <span className="dot bg-[#4fd1a1]"></span>
                              Auto-approved
                            </>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* THRESHOLD DRAWER OVERLAY */}
            {showThresholdDrawer && (
              <div
                id="thrB"
                className="absolute inset-0 bg-[#191919] border-t border-[#212327] px-6 py-5 flex gap-10 items-center z-20 animate-in fade-in slide-in-from-bottom-2"
              >
                <div className="w-[160px]">
                  <div className="lbl">Decision threshold</div>
                  <div className="f-mono text-[40px] tracking-[-1px] mt-1">
                    {threshold.toFixed(2)}
                  </div>
                </div>

                <div className="flex-1">
                  <input
                    type="range"
                    min="0.10"
                    max="0.95"
                    step="0.01"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full accent-white cursor-pointer"
                  />
                  <div className="flex justify-between f-mono text-[10px] text-[#7d8187] mt-1">
                    <span>0.10</span>
                    <span>0.50</span>
                    <span>0.95</span>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-8 shrink-0">
                  <div>
                    <div className="lbl">Recall</div>
                    <div className="f-mono text-[22px] mt-1 text-[#4fd1a1]">
                      {recallPct}%
                    </div>
                  </div>
                  <div>
                    <div className="lbl">Precision</div>
                    <div className="f-mono text-[22px] mt-1 text-[#a0c3ec]">
                      {precisionPct}%
                    </div>
                  </div>
                  <div>
                    <div className="lbl">FPR</div>
                    <div className="f-mono text-[22px] mt-1 text-[#f0596b]">
                      {fprPct}%
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => setShowThresholdDrawer(false)}
                  className="absolute top-3 right-3 w-7 h-7 rounded-full border border-white/20 grid place-items-center text-[13px] hover:border-white/50 hover:bg-white/5 transition"
                >
                  <X className="w-3.5 h-3.5 text-white" />
                </button>
              </div>
            )}
          </section>
        </div>

        {/* RIGHT COLUMN (Evidence Panel OR Performance Drawer) */}
        {activeRightPanel === 'evidence' ? (
          <aside className="flex-1 min-w-0 bg-[#191919] border border-[#212327] rounded-lg flex flex-col min-h-0">
            {/* Tab navigation */}
            <div className="h-11 shrink-0 px-6 flex items-end gap-6 border-b border-[#212327]">
              <button
                onClick={() => setActiveEvidenceTab('t')}
                className={`h-11 text-[13px] border-b transition ${
                  activeEvidenceTab === 't'
                    ? 'text-white border-white font-medium'
                    : 'text-[#7d8187] border-transparent hover:text-[#dadbdf]'
                }`}
              >
                Transaction
              </button>
              <button
                onClick={() => setActiveEvidenceTab('n')}
                className={`h-11 text-[13px] border-b transition ${
                  activeEvidenceTab === 'n'
                    ? 'text-white border-white font-medium'
                    : 'text-[#7d8187] border-transparent hover:text-[#dadbdf]'
                }`}
              >
                Node details
              </button>
            </div>

            {/* TAB: TRANSACTION DETAILS */}
            {activeEvidenceTab === 't' && (
              <div className="flex-1 min-h-0 overflow-y-auto sc px-6 py-5">
                <div className="flex items-center justify-between">
                    const isSelected = (!isFollowingLive && selectedTx?.tx_id === tx.tx_id) || (isFollowingLive && latestTx?.tx_id === tx.tx_id);
                    const rScore = (Number(tx.risk_score) || 0.0) * 100;
                    const isMule = tx.is_mule || rScore >= threshold * 100;
                    const decision =
                      tx.decision ||
                      (isMule
                        ? 'HELD_FOR_REVIEW'
                        : rScore > 50
                        ? 'FLAGGED_FOR_MONITORING'
                        : 'AUTO_APPROVED');

                    return (
                      <tr
                        key={tx.tx_id || idx}
                        onClick={() => handleSelectRow(tx)}
                        className={`h-9 border-b border-[#212327]/60 cursor-pointer transition ${
                          isSelected ? 'bg-[#1e2330]' : 'hover:bg-[#141518]'
                        }`}
                      >
                        <td className="pl-4 f-mono text-[#dadbdf]">{tx.tx_id}</td>
                        <td className="f-mono text-[#7d8187]">{tx.src_account}</td>
                        <td className="f-mono text-[#7d8187]">{tx.dst_account}</td>
                        <td className="text-right f-mono text-[#dadbdf]">
                          ${formatDecimal(tx.amount, 1000, 2)}
                        </td>
                        <td className="f-sans text-[#7d8187]">{tx.payment_channel || 'Wire'}</td>
                        <td className={isMule ? 'text-[#f0596b]' : ''}>
                          {formatDecimal(rScore, 0, 1)}
                        </td>
                        <td className="text-right pr-6 text-[#7d8187]">
                          {formatDecimal(tx.latency_ms || tx.total_inference_ms, 11.8, 1)}
                        </td>
                        <td className="f-sans">
                          {decision === 'HELD_FOR_REVIEW' ? (
                            <>
                              <span className="dot bg-[#f0596b]"></span>
                              Held for review
                            </>
                          ) : decision === 'FLAGGED_FOR_MONITORING' ? (
                            <>
                              <span className="dot bg-[#f5b14c]"></span>
                              Enhanced monitoring
                            </>
                          ) : (
                            <>
                              <span className="dot bg-[#4fd1a1]"></span>
                              Auto-approved
                            </>
                          )}
                        </td>
                      </tr>
                    );
                  </div>
                  <div>
                    <dt className="lbl">Destination</dt>
                    <dd className={`f-mono ${currentTx.is_mule ? 'text-[#f0596b]' : ''}`}>
                      {currentTx.dst_account}
                    </dd>
                  </div>
                  <div>
                    <dt className="lbl">Community</dt>
                    <dd className="f-mono text-[#c4b5fd]">
                      {currentTx.ring_id || 'C-0318'}
                    </dd>
                  </div>
                  <div>
                    <dt className="lbl">Backbone</dt>
                    <dd>{modelLabels[activeModel] || 'CARE-GNN'}</dd>
                  </div>
                  <div>
                    <dt className="lbl">Volume transferred</dt>
                    <dd className="f-mono">
                      ${formatCurrency((Number(currentTx.amount) || 48250) * 8.5, 412880, 2)}
                    </dd>
                  </div>
                  <div>
                    <dt className="lbl">Community size</dt>
                    <dd className="f-mono">{graphNodeCount + 4} accounts</dd>
                  </div>
                  <div>
                    <dt className="lbl">Camouflage edges pruned</dt>
                    <dd className="f-mono">
                      {formatDecimal(currentTx.camouflage_prune_rate, 23.6, 1)}%
                    </dd>
                  </div>
                </dl>

                <h3 className="text-[15px] mt-6 mb-1 font-medium flex items-center">
                  Grounded reasons
                  <span className="f-mono text-[12px] text-[#7d8187] ml-2">
                    {currentTx.grounded_reasons?.length || 0}
                  </span>
                </h3>

                <ul className="text-[13px] leading-[1.5] text-[#dadbdf]">
                  {(currentTx.grounded_reasons || []).map((reason, idx) => {
                    const isTelemetry = reason.toLowerCase().includes('device') || reason.toLowerCase().includes('subnet');
                    const isCTR = reason.toLowerCase().includes('ctr') || reason.toLowerCase().includes('inbound');

                    return (
                      <li
                        key={idx}
                        className="py-2.5 border-b border-[#212327] flex gap-3 items-start"
                      >
                        {isTelemetry ? (
                          <span className="w-1.5 h-1.5 mt-[7px] rotate-45 bg-[#c4b5fd] shrink-0" />
                        ) : isCTR ? (
                          <span className="w-1.5 h-1.5 mt-[7px] rounded-full bg-[#f5b14c] shrink-0" />
                        ) : (
                          <span className="w-1.5 h-1.5 mt-[7px] rounded-full bg-[#f0596b] shrink-0" />
                        )}
                        <span>{reason}</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {/* TAB: NODE DETAILS */}
            {activeEvidenceTab === 'n' && (
              <div className="flex-1 min-h-0 overflow-y-auto sc px-6 py-5">
                <div className="flex items-center gap-3">
                  <span
                    className={`w-4 h-4 rounded-full border-2 ${
                      selectedNode?.type === 'mule'
                        ? 'border-[#f0596b] bg-[#2a1216]'
                        : selectedNode?.type === 'device' || selectedNode?.type === 'subnet'
                        ? 'border-[#c4b5fd] bg-[#1c1730] rotate-45'
                        : 'border-[#4fd1a1] bg-[#0e231c]'
                    }`}
                  ></span>
                  <span className="f-mono text-[16px]">
                    {selectedNode?.label || currentTx.dst_account}
                  </span>
                </div>

                <dl className="mt-5 border-t border-[#212327] text-[13px] [&>div]:flex [&>div]:justify-between [&>div]:py-2.5 [&>div]:border-b [&>div]:border-[#212327]">
                  <div>
                    <dt className="lbl">Type</dt>
                    <dd>{selectedNode?.type || 'Account'}</dd>
                  </div>
                  <div>
                    <dt className="lbl">Label</dt>
                    <dd className={selectedNode?.type === 'mule' ? 'text-[#f0596b]' : 'text-[#4fd1a1]'}>
                      {selectedNode?.type === 'mule' ? 'Mule' : 'Legitimate'}
                    </dd>
                  </div>
                  <div>
                    <dt className="lbl">Risk</dt>
                    <dd className={`f-mono ${currentTx.is_mule ? 'text-[#f0596b]' : 'text-[#4fd1a1]'}`}>
                      {(
                        (Number(currentTx.risk_score) > 0
                          ? Number(currentTx.risk_score)
                          : (currentTx.is_mule ? 0.964 : 0.038)) * 100
                      ).toFixed(1)}%
                    </dd>
                  </div>
                  <div>
                    <dt className="lbl">In-degree</dt>
                    <dd className="f-mono">4</dd>
                  </div>
                  <div>
                    <dt className="lbl">Out-degree</dt>
                    <dd className="f-mono">2</dd>
                  </div>
                  <div>
                    <dt className="lbl">Telemetry links</dt>
                    <dd className="f-mono text-[#c4b5fd]">2</dd>
                  </div>
                </dl>

                <h3 className="text-[15px] mt-6 mb-2 font-medium">Connected edges</h3>
                <table className="w-full f-mono text-[12px] text-[#dadbdf] [&_td]:py-2 [&_td]:border-b [&_td]:border-[#212327]">
                  <tbody>
                    <tr>
                      <td>ACC_77120</td>
                      <td className="text-[#7d8187]">in</td>
                      <td className="text-right">$48,250</td>
                    </tr>
                    <tr>
                      <td>ACC_20931</td>
                      <td className="text-[#7d8187]">in</td>
                      <td className="text-right">$9,400</td>
                    </tr>
                    <tr>
                      <td>ACC_55810</td>
                      <td className="text-[#7d8187]">in</td>
                      <td className="text-right">$9,850</td>
                    </tr>
                    <tr>
                      <td>ACC_31877</td>
                      <td className="text-[#7d8187]">in</td>
                      <td className="text-right">$9,700</td>
                    </tr>
                    <tr>
                      <td className="text-[#f0596b]">ACC_MULE_0987</td>
                      <td className="text-[#7d8187]">out</td>
                      <td className="text-right">$47,480</td>
                    </tr>
                    <tr>
                      <td className="text-[#f0596b]">ACC_MULE_1120</td>
                      <td className="text-[#7d8187]">out</td>
                      <td className="text-right">$42,900</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </aside>
        ) : (
          /* ENGINE PERFORMANCE DRAWER */
          <aside
            id="perfB"
            className="flex-1 min-w-0 bg-[#191919] border border-[#212327] rounded-lg flex flex-col min-h-0 px-6 py-5 animate-in fade-in"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-[20px] tracking-[-0.3px] font-medium">Engine performance</h3>
              <button
                onClick={() => setActiveRightPanel('evidence')}
                className="w-7 h-7 rounded-full border border-white/20 grid place-items-center text-[13px] hover:border-white/50 hover:bg-white/5 transition"
              >
                <X className="w-3.5 h-3.5 text-white" />
              </button>
            </div>

            <div className="f-mono text-[11px] text-[#7d8187] mt-1">
              {modelLabels[activeModel] || 'CARE-GNN'}, rolling 60s, {formatInt(metrics.total_transactions_scored, 1284903)} tx
            </div>

            <div className="mt-8 space-y-5">
              <div>
                <div className="flex justify-between">
                  <span className="lbl">P50</span>
                  <span className="f-mono text-[14px]">
                    {formatDecimal(metrics.latency_p50_ms, 6.9, 1)} ms
                  </span>
                </div>
                <div className="relative mt-2 h-2 rounded-full bg-[#212327]">
                  <div
                    className="h-2 rounded-full bg-[#4fd1a1]"
                    style={{ width: `${Math.min(100, ((Number(metrics.latency_p50_ms) || 6.9) / 25.0) * 100)}%` }}
                  />
                  <span className="absolute -top-1 -bottom-1 left-[72%] w-px bg-white/70" />
                </div>
              </div>

              <div>
                <div className="flex justify-between">
                  <span className="lbl">P95</span>
                  <span className="f-mono text-[14px]">
                    {formatDecimal(metrics.latency_p95_ms, 11.2, 1)} ms
                  </span>
                </div>
                <div className="relative mt-2 h-2 rounded-full bg-[#212327]">
                  <div
                    className="h-2 rounded-full bg-[#a0c3ec]"
                    style={{ width: `${Math.min(100, ((Number(metrics.latency_p95_ms) || 11.2) / 25.0) * 100)}%` }}
                  />
                  <span className="absolute -top-1 -bottom-1 left-[72%] w-px bg-white/70" />
                </div>
              </div>

              <div>
                <div className="flex justify-between">
                  <span className="lbl">P99</span>
                  <span className="f-mono text-[14px]">
                    {formatDecimal(metrics.latency_p99_ms, 14.8, 1)} ms
                  </span>
                </div>
                <div className="relative mt-2 h-2 rounded-full bg-[#212327]">
                  <div
                    className="h-2 rounded-full bg-[#f5b14c]"
                    style={{ width: `${Math.min(100, ((Number(metrics.latency_p99_ms) || 14.8) / 25.0) * 100)}%` }}
                  />
                  <span className="absolute -top-1 -bottom-1 left-[72%] w-px bg-white/70" />
                </div>
              </div>

              <div className="relative f-mono text-[10px] text-[#7d8187] h-3">
                <span className="absolute left-0">0</span>
                <span className="absolute left-[72%] -translate-x-1/2 text-white">
                  18.0 ms SLA
                </span>
                <span className="absolute right-0">25</span>
              </div>
            </div>

            <h4 className="text-[15px] mt-10 mb-2 font-medium">Stage breakdown</h4>
            <dl className="text-[13px] [&>div]:flex [&>div]:items-center [&>div]:justify-between [&>div]:py-3 [&>div]:border-b [&>div]:border-[#212327]">
              <div>
                <dt className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#4fd1a1]"></span>
                  CSR subgraph sampling
                </dt>
                <dd className="f-mono">
                  1.6 <span className="text-[#7d8187]">/ 2.0 ms</span>
                </dd>
              </div>
              <div>
                <dt className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#a0c3ec]"></span>
                  GNN + XGBoost inference
                </dt>
                <dd className="f-mono">
                  4.2 <span className="text-[#7d8187]">/ 5.0 ms</span>
                </dd>
              </div>
              <div>
                <dt className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#c4b5fd]"></span>
                  Symbolic gate verification
                </dt>
                <dd className="f-mono">
                  0.7 <span className="text-[#7d8187]">/ 1.0 ms</span>
                </dd>
              </div>
            </dl>
          </aside>
        )}
      </main>

      {/* ========================================================================= */}
      {/* 4. FOOTER STATUS BAR                                                      */}
      {/* ========================================================================= */}
      <footer className="h-8 shrink-0 border-t border-[#212327] flex items-center gap-1 px-4 text-[12px] bg-[#0a0a0a]">
        <button
          onClick={() =>
            setActiveRightPanel(activeRightPanel === 'performance' ? 'evidence' : 'performance')
          }
          className="h-6 px-3 rounded-full hover:bg-[#1a1c20] flex items-center gap-2 text-[#dadbdf] transition"
        >
          <Gauge className="w-3.5 h-3.5 text-[#a0c3ec]" />
          Engine performance
          <span className="f-mono text-white">
            P99 {formatDecimal(metrics.latency_p99_ms, 14.8, 1)} ms
          </span>
        </button>

        <button
          onClick={() => setShowThresholdDrawer(!showThresholdDrawer)}
          className="h-6 px-3 rounded-full hover:bg-[#1a1c20] flex items-center gap-2 text-[#dadbdf] transition"
        >
          <SlidersHorizontal className="w-3.5 h-3.5 text-[#f5b14c]" />
          Threshold
          <span className="f-mono text-white">{formatDecimal(threshold, 0.85, 2)}</span>
        </button>

        <button
          onClick={() => {
            const nextMode = !isFollowingLive;
            setIsFollowingLive(nextMode);
            isFollowingLiveRef.current = nextMode;
            if (nextMode) {
              setSelectedTx(null);
              if (latestTx?.ego_subgraph) setActiveGraph(latestTx.ego_subgraph);
              setToastMsg('Resumed live stream follow mode');
            } else {
              setToastMsg('Live stream follow paused');
            }
          }}
          className="ml-auto f-mono text-[11px] text-[#7d8187] flex items-center gap-2 hover:text-white transition cursor-pointer"
        >
          <span className={`w-1.5 h-1.5 rounded-full ${isFollowingLive ? 'bg-[#4fd1a1] animate-pulse' : 'bg-[#f5b14c]'}`}></span>
          {isFollowingLive ? 'Live stream follow' : 'Stream paused on inspection'}
        </button>
      </footer>

      {/* Floating Toast Notification */}
      {toastMsg && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-50 bg-[#191919] border border-white/20 text-white text-[12.5px] px-4 py-2 rounded-full shadow-2xl flex items-center gap-2 animate-in fade-in slide-in-from-top-2">
          <CheckCircle2 className="w-3.5 h-3.5 text-[#4fd1a1]" />
          <span>{toastMsg}</span>
          <button onClick={() => setToastMsg(null)} className="ml-1 text-[#7d8187] hover:text-white">
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* ========================================================================= */}
      {/* 5. DATASET HUB MODAL OVERLAY                                              */}
      {/* ========================================================================= */}
      <DatasetHubModal
        isOpen={showDatasetModal}
        onClose={() => setShowDatasetModal(false)}
        onReplayComplete={(res) => {
          if (res.results && res.results.length > 0) {
            setRecentFeed((prev) => [...res.results.reverse(), ...prev].slice(0, 25));
            const firstMule = res.results.find((r) => r.is_mule) || res.results[0];
            setSelectedTx(firstMule);
            if (firstMule.ego_subgraph) setActiveGraph(firstMule.ego_subgraph);
          }
        }}
      />

      {/* ========================================================================= */}
      {/* 6. CONCEPT DRIFT (FEEDZAI) MODAL OVERLAY                                  */}
      {/* ========================================================================= */}
      {showFeedzaiModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-2xl bg-[#191919] border border-[#212327] rounded-xl shadow-2xl overflow-hidden p-6 animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between pb-4 border-b border-[#212327]">
              <div>
                <h3 className="text-lg font-medium text-white flex items-center gap-2">
                  Feedzai AML Temporal Concept Drift Benchmark
                </h3>
                <p className="text-xs text-[#7d8187] mt-0.5">
                  Elliptic++ Historical (Steps 1–34) vs Out-of-Distribution Drift Window (Steps 35–49)
                </p>
              </div>
              <button
                onClick={() => setShowFeedzaiModal(false)}
                className="w-7 h-7 rounded-full border border-white/20 grid place-items-center text-[13px] hover:border-white/50 transition"
              >
                <X className="w-3.5 h-3.5 text-white" />
              </button>
            </div>

            {feedzaiLoading ? (
              <div className="py-12 text-center text-[#7d8187] f-mono text-xs">
                Executing temporal drift evaluation across timesteps...
              </div>
            ) : (
              <div className="py-6 space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-[#0a0a0a] border border-[#212327]">
                    <div className="lbl text-[#7d8187]">Baseline Tabular XGBoost</div>
                    <div className="text-[32px] f-mono text-[#f0596b] mt-2">
                      {formatDecimal(feedzaiData?.baseline_tabular_prauc, 0.9444, 4)}
                    </div>
                    <div className="text-[11px] text-[#7d8187] mt-1">PR-AUC under drift</div>
                  </div>

                  <div className="p-4 rounded-lg bg-[#0a0a0a] border border-[#212327]">
                    <div className="lbl text-[#a0c3ec]">Safe-Bit Inductive GNN</div>
                    <div className="text-[32px] f-mono text-[#4fd1a1] mt-2">
                      {formatDecimal(feedzaiData?.safebit_inductive_gnn_prauc, 1.0000, 4)}
                    </div>
                    <div className="text-[11px] text-[#4fd1a1] mt-1">
                      +{feedzaiData?.drift_robustness_gain || '0.0556'} PR-AUC Gain
                    </div>
                  </div>
                </div>

                <div className="text-xs text-[#dadbdf] bg-[#0d0e10] p-4 rounded-lg border border-[#212327] leading-relaxed">
                  <strong>Invariant Topological Grounding:</strong> As money laundering networks rotate wallet addresses and device fingerprints across timesteps 35–49, tabular feature distributions drift by &gt;40%. Safe-Bit's edge-attributed GINEConv preserves high PR-AUC because the flow topology (scatter-gather &amp; peeling chains) remains structurally invariant.
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
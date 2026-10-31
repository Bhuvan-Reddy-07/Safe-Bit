import React from 'react';
import { Gauge, CheckCircle2, AlertOctagon, Activity } from 'lucide-react';

export default function LatencyMeter({ metrics, currentLatency }) {
  const p50 = metrics?.latency_p50_ms || 4.5;
  const p95 = metrics?.latency_p95_ms || 7.2;
  const p99 = metrics?.latency_p99_ms || 9.4;
  const slaMax = metrics?.sla_threshold_ms || 20.0;

  // Breakdown from latest scored transaction
  const sampling = currentLatency?.sampling_ms || 1.1;
  const inference = currentLatency?.inference_ms || 3.4;
  const verification = currentLatency?.verification_ms || 0.8;
  const total = currentLatency?.total_ms || 5.3;

  const isUnderSLA = p99 <= 18.0;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col gap-4">
      {/* Title & SLA Compliance Badge */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-100">
            Latency Monitor & 20ms SLA Gauge
          </h3>
        </div>

        <div className="flex items-center gap-1.5 text-xs font-mono font-medium">
          {isUnderSLA ? (
            <span className="flex items-center gap-1 text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 rounded-full">
              <CheckCircle2 className="w-3.5 h-3.5" />
              SLA Compliant (P99 &lt; 18ms)
            </span>
          ) : (
            <span className="flex items-center gap-1 text-rose-400 bg-rose-500/10 border border-rose-500/30 px-2.5 py-0.5 rounded-full">
              <AlertOctagon className="w-3.5 h-3.5" />
              SLA Breach Warning
            </span>
          )}
        </div>
      </div>

      {/* Percentiles Bar Meters */}
      <div className="space-y-3">
        {/* P50 */}
        <div>
          <div className="flex justify-between text-xs font-mono mb-1">
            <span className="text-slate-400">P50 Latency (Median)</span>
            <span className="text-emerald-400 font-bold">{p50.toFixed(2)} ms</span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div
              className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, (p50 / slaMax) * 100)}%` }}
            />
          </div>
        </div>

        {/* P95 */}
        <div>
          <div className="flex justify-between text-xs font-mono mb-1">
            <span className="text-slate-400">P95 Latency</span>
            <span className="text-sky-400 font-bold">{p95.toFixed(2)} ms</span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div
              className="bg-sky-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, (p95 / slaMax) * 100)}%` }}
            />
          </div>
        </div>

        {/* P99 */}
        <div>
          <div className="flex justify-between text-xs font-mono mb-1">
            <span className="text-slate-400">P99 Latency (Strict SLA: 18.0 ms)</span>
            <span className="text-amber-400 font-bold">{p99.toFixed(2)} ms</span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800 relative">
            {/* 18ms SLA marker */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-10"
              style={{ left: `${(18.0 / slaMax) * 100}%` }}
              title="18ms SLA Line"
            />
            <div
              className={`h-2 rounded-full transition-all duration-300 ${
                p99 <= 18.0 ? 'bg-amber-400' : 'bg-rose-500'
              }`}
              style={{ width: `${Math.min(100, (p99 / slaMax) * 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Sub-stage Execution Breakdown for latest transaction */}
      <div className="pt-2 border-t border-slate-800/80">
        <div className="text-[11px] uppercase tracking-wider text-slate-400 font-mono mb-2 flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-highlight" />
          Last Transaction Stage Breakdown (Total: {total.toFixed(2)} ms)
        </div>

        <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
          <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
            <div className="text-[10px] text-slate-400">CSR Sampling</div>
            <div className="font-semibold text-emerald-400 mt-0.5">{sampling.toFixed(2)} ms</div>
            <div className="text-[9px] text-slate-400">&lt; 2.0ms SLA</div>
          </div>
          <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
            <div className="text-[10px] text-slate-400">GNN + XGBoost</div>
            <div className="font-semibold text-sky-400 mt-0.5">{inference.toFixed(2)} ms</div>
            <div className="text-[9px] text-slate-400">&lt; 5.0ms SLA</div>
          </div>
          <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
            <div className="text-[10px] text-slate-400">Symbolic Gate</div>
            <div className="font-semibold text-purple-400 mt-0.5">{verification.toFixed(2)} ms</div>
            <div className="text-[9px] text-slate-400">&lt; 1.0ms SLA</div>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import React from "react";
import { Gauge, AlertTriangle, PauseCircle } from "lucide-react";
import { ApiUsage, ToolSelectionStats } from "@/types/aira";

interface ApiUsagePanelProps {
  usage?: ApiUsage | null;
  toolStats?: ToolSelectionStats;
  isRunning: boolean;
}

const PROVIDER_LABELS: Record<string, string> = {
  gemini: "Gemini",
  groq: "Groq",
  nvidia: "NVIDIA",
  openalex: "OpenAlex",
  crossref: "Crossref",
  semantic_scholar: "Semantic Scholar",
};

const label = (name: string) => PROVIDER_LABELS[name] ?? name;

/** A number that turns amber when it is not zero (rate limits, failures, timeouts). */
const Warn: React.FC<{ value: number; text: string }> = ({ value, text }) => (
  <span className={value > 0 ? "text-amber-300 font-semibold" : "text-slate-500"}>
    {value} {text}
  </span>
);

const Row: React.FC<{ name: string; children: React.ReactNode; highlight: boolean }> = ({ name, children, highlight }) => (
  <div
    className={`flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-3 py-2 rounded-xl border text-xs font-mono ${
      highlight ? "border-amber-500/30 bg-amber-500/5" : "border-slate-800 bg-slate-950/60"
    }`}
  >
    <span className="text-slate-200 font-semibold">{name}</span>
    <span className="flex flex-wrap gap-x-3 gap-y-1">{children}</span>
  </div>
);

export const ApiUsagePanel: React.FC<ApiUsagePanelProps> = ({ usage, toolStats, isRunning }) => {
  const stats = toolStats ?? {};
  const accepted = stats.accepted ?? 0;
  const rejected = stats.rejected ?? 0;
  const fallback = stats.fallback ?? 0;
  const automatic = stats.single_option ?? 0;
  const paused = stats.paused ?? 0;
  const hasToolStats = accepted + rejected + fallback + automatic + paused > 0;

  const llmEntries = usage ? Object.entries(usage.llm).sort(([a], [b]) => a.localeCompare(b)) : [];
  const academicEntries = usage ? Object.entries(usage.academic).sort(([a], [b]) => a.localeCompare(b)) : [];
  const totals = usage?.llm_totals;
  const embeddings = usage?.embeddings;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-2xl bg-cyan-500/10 border border-cyan-500/25 flex items-center justify-center">
            <Gauge className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-tight">API Usage</h3>
            <p className="text-[11px] text-slate-400">
              Requests this session actually sent{isRunning ? " (updating while the run is active)" : ""}
            </p>
          </div>
        </div>
        {totals && (
          <span className="text-[11px] font-mono text-slate-300 bg-slate-950 px-2.5 py-1 rounded-full border border-slate-800">
            {totals.calls} LLM call{totals.calls === 1 ? "" : "s"}
            {usage && usage.fallbacks > 0 ? ` · ${usage.fallbacks} fallback${usage.fallbacks === 1 ? "" : "s"}` : ""}
          </span>
        )}
      </div>

      {!usage ? (
        <p className="text-xs text-slate-500 italic">Usage data not available for this session.</p>
      ) : (
        <div className="space-y-2">
          {llmEntries.length === 0 && (
            <p className="text-xs text-slate-500">No LLM requests yet.</p>
          )}
          {llmEntries.map(([name, p]) => (
            <Row key={name} name={label(name)} highlight={p.rate_limited + p.failures + p.timeouts > 0}>
              <span className="text-slate-300">{p.calls} calls</span>
              <span className="text-emerald-400/80">{p.successes} ok</span>
              <Warn value={p.failures} text="failed" />
              <Warn value={p.rate_limited} text="rate-limited (429)" />
              <Warn value={p.timeouts} text="timeouts" />
              {p.cooldowns > 0 && <span className="text-amber-300">{p.cooldowns} pauses</span>}
              <span className="text-slate-500">{(p.prompt_tokens + p.completion_tokens).toLocaleString()} tokens</span>
            </Row>
          ))}

          {embeddings && (
            <Row name="Embeddings" highlight={embeddings.failures + embeddings.rate_limited > 0}>
              <span className="text-slate-300">{embeddings.calls} calls</span>
              <span className="text-slate-500">{embeddings.texts} texts</span>
              <Warn value={embeddings.failures} text="failed" />
              <Warn value={embeddings.rate_limited} text="rate-limited" />
            </Row>
          )}

          {academicEntries.map(([name, a]) => (
            <Row key={name} name={label(name)} highlight={a.rate_limited + a.failures > 0}>
              <span className="text-slate-300">{a.requests} requests</span>
              <Warn value={a.failures} text="failed" />
              <Warn value={a.rate_limited} text="rate-limited (429)" />
            </Row>
          ))}
        </div>
      )}

      {hasToolStats && (
        <div className="pt-3 border-t border-slate-800/80 space-y-2">
          <p className="text-xs font-mono text-slate-300">
            LLM tool choices: <span className="text-emerald-400">{accepted} accepted</span> ·{" "}
            <span className={rejected > 0 ? "text-amber-300" : "text-slate-400"}>{rejected} rejected</span> ·{" "}
            <span className={fallback > 0 ? "text-amber-300" : "text-slate-400"}>{fallback} fallback</span> ·{" "}
            <span className="text-slate-400">{automatic} automatic</span>
          </p>
          {paused > 0 && (
            <p className="text-[11px] text-amber-300 flex items-center gap-1.5">
              <PauseCircle className="w-3.5 h-3.5" />
              LLM proposals were paused after repeated failures; {paused} later decision{paused === 1 ? "" : "s"} used the rules.
            </p>
          )}
          {fallback > 0 && paused === 0 && (
            <p className="text-[11px] text-slate-400 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              Fallback means the LLM was unavailable or too slow, so the rule-based choice was used.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

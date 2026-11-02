"use client";

import React from "react";
import { 
  PlusCircle, 
  Search, 
  FileText, 
  Layers, 
  Cpu, 
  CheckCircle2, 
  ArrowRight,
  TrendingUp,
  BookOpen,
  Sparkles,
} from "lucide-react";
import { SystemHealth } from "@/types/aira";

interface DashboardViewProps {
  onStartNew: () => void;
  onOpenWorkspace: () => void;
  onSelectSession: (id: string) => void;
  historySessions: Array<{ id: string; query: string; status: string; date: string }>;
  indexedPapersCount: number;
  reportsCount: number;
  health: SystemHealth | null;
  onSelectQueryPreset: (query: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onStartNew,
  onOpenWorkspace,
  onSelectSession,
  historySessions,
  indexedPapersCount,
  reportsCount,
  health,
  onSelectQueryPreset,
}) => {
  const completedSessions = historySessions.filter(
    (s) => s.status === "GOAL_ACHIEVED" || s.status === "COMPLETED"
  ).length;

  const quickPresets = [
    {
      title: "Amharic Sentiment Analysis",
      category: "Natural Language Processing",
      query: "Compare recent approaches to Amharic sentiment analysis and identify limitations in datasets and models",
    },
    {
      title: "Speech Recognition Baselines",
      category: "Acoustic Architectures",
      query: "Evaluate Conformer acoustic architecture against Transformer baselines for speech recognition on LibriSpeech",
    },
    {
      title: "Low-Resource Named Entity Recognition",
      category: "Cross-Lingual Transfer",
      query: "Low-resource named entity recognition techniques for Ethiopian languages",
    },
    {
      title: "Ethiopic Script Tokenization",
      category: "Language Modeling",
      query: "Analysis of morphological tokenization challenges in Ethiopic script language modeling",
    },
  ];

  return (
    <div className="space-y-8 max-w-6xl mx-auto py-2">
      {/* Welcome & Hero Section */}
      <div className="relative rounded-3xl bg-gradient-to-br from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 p-8 shadow-2xl overflow-hidden">
        {/* Glow Effects */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-3 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              <span>Autonomous Academic Research Platform</span>
            </div>

            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
              Evidence-Backed Academic Synthesis with LangGraph
            </h1>

            <p className="text-slate-300 text-sm leading-relaxed">
              AIRA orchestrates multi-agent literature search, independent DOI verification, deep PDF extraction, verifiable metric comparison, and gap discovery without hallucinated citations.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button
                onClick={onStartNew}
                className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs py-2.5 px-5 rounded-xl shadow-lg shadow-indigo-600/25 transition-all active:scale-[0.98]"
              >
                <PlusCircle className="w-4 h-4" />
                Start New Investigation
              </button>

              <button
                onClick={onOpenWorkspace}
                className="inline-flex items-center gap-2 bg-slate-800/80 hover:bg-slate-800 text-slate-200 border border-slate-700 font-medium text-xs py-2.5 px-4 rounded-xl transition-all"
              >
                <Search className="w-4 h-4 text-cyan-400" />
                Go to Workspace
              </button>
            </div>
          </div>

          {/* Quick Metrics Cards */}
          <div className="grid grid-cols-2 gap-3 sm:w-80 shrink-0">
            <div className="bg-slate-900/90 border border-slate-800/80 p-4 rounded-2xl shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Investigations</span>
                <Search className="w-4 h-4 text-indigo-400" />
              </div>
              <div className="text-2xl font-bold text-white font-mono">{historySessions.length}</div>
              <div className="text-[10px] text-emerald-400 mt-1 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> {completedSessions} Completed
              </div>
            </div>

            <div className="bg-slate-900/90 border border-slate-800/80 p-4 rounded-2xl shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>RAG Corpus</span>
                <Layers className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-2xl font-bold text-white font-mono">{indexedPapersCount}</div>
              <div className="text-[10px] text-slate-400 mt-1">Indexed Papers</div>
            </div>

            <div className="bg-slate-900/90 border border-slate-800/80 p-4 rounded-2xl shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Reports</span>
                <FileText className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="text-2xl font-bold text-white font-mono">{reportsCount}</div>
              <div className="text-[10px] text-slate-400 mt-1">Comparative Reports</div>
            </div>

            <div className="bg-slate-900/90 border border-slate-800/80 p-4 rounded-2xl shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Kernel State</span>
                <Cpu className="w-4 h-4 text-purple-400" />
              </div>
              <div className="text-xs font-semibold text-emerald-400 font-mono mt-1">
                {health?.status === "healthy" || health?.status === "ok" ? "ACTIVE" : "CONNECTING"}
              </div>
              <div className="text-[10px] text-slate-500 mt-1 truncate">
                {health?.services?.llm_primary || "Gemini Router"}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Suggested Academic Topics Spotlight */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-indigo-400" />
              Recommended Academic Research Inquiries
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Curated benchmark questions testing multi-source discovery, metric extraction, and gap identification
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {quickPresets.map((preset, idx) => (
            <div
              key={idx}
              onClick={() => onSelectQueryPreset(preset.query)}
              className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-indigo-500/50 hover:bg-slate-850 cursor-pointer transition-all group flex flex-col justify-between"
            >
              <div className="space-y-1.5">
                <span className="text-[10px] font-mono uppercase font-semibold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                  {preset.category}
                </span>
                <h3 className="text-sm font-semibold text-white group-hover:text-indigo-300 transition-colors">
                  {preset.title}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed line-clamp-2">
                  {preset.query}
                </p>
              </div>

              <div className="flex items-center justify-end text-xs text-indigo-400 font-medium pt-3 mt-2 border-t border-slate-800/60 group-hover:translate-x-0.5 transition-transform">
                <span className="flex items-center gap-1">
                  Launch Investigation <ArrowRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Investigations Table / Cards */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              Recent Research Activity
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Inspect session traces, checkpoints, or resume paused investigations
            </p>
          </div>
        </div>

        {historySessions.length === 0 ? (
          <div className="text-center py-12 bg-slate-900/40 rounded-2xl border border-slate-800/80 text-slate-500 space-y-2">
            <Search className="w-8 h-8 mx-auto opacity-30 text-indigo-400" />
            <p className="text-sm font-medium text-slate-400">No research investigations started yet.</p>
            <p className="text-xs text-slate-500">Formulate a query above or choose a preset topic to test AIRA.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-2.5">
            {historySessions.slice(0, 5).map((s) => (
              <div
                key={s.id}
                onClick={() => onSelectSession(s.id)}
                className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 hover:bg-slate-850 cursor-pointer transition-all flex items-center justify-between gap-4"
              >
                <div className="min-w-0 space-y-1">
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono text-[10px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">
                      {s.id.slice(0, 10)}
                    </span>
                    <span className="text-xs text-slate-500">• {s.date}</span>
                  </div>
                  <h4 className="text-sm font-medium text-slate-200 truncate">{s.query}</h4>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold ${
                      s.status === "GOAL_ACHIEVED" || s.status === "COMPLETED"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        : s.status === "USER_CLARIFICATION_REQUIRED"
                        ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        : "bg-indigo-500/10 text-indigo-400 border-indigo-500/20"
                    }`}
                  >
                    {s.status}
                  </span>
                  <ArrowRight className="w-4 h-4 text-slate-500" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

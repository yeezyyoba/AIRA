"use client";

import React from "react";
import { 
  BookOpen, 
  Search, 
  History, 
  FileText, 
  ShieldCheck, 
  Sparkles, 
  PlusCircle, 
  CheckCircle2, 
  AlertTriangle 
} from "lucide-react";
import { SystemHealth } from "@/types/aira";

interface SidebarProps {
  currentTab: string;
  onSelectTab: (tab: string) => void;
  onNewSession: () => void;
  health: SystemHealth | null;
  historySessions: Array<{ id: string; query: string; status: string; date: string }>;
  activeSessionId?: string;
  onSelectSession: (id: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  onNewSession,
  health,
  historySessions,
  activeSessionId,
  onSelectSession,
}) => {
  return (
    <aside className="w-72 bg-slate-900 border-r border-slate-800 text-slate-200 flex flex-col h-screen shrink-0 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-white tracking-tight flex items-center gap-1.5">
              AIRA <span className="text-[10px] font-semibold bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-500/30">v0.1</span>
            </h1>
            <p className="text-xs text-slate-400 font-medium">Research Assistant</p>
          </div>
        </div>
      </div>

      {/* New Research Button */}
      <div className="p-4">
        <button
          onClick={onNewSession}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/25 active:scale-[0.98]"
        >
          <PlusCircle className="w-4 h-4" />
          New Investigation
        </button>
      </div>

      {/* Navigation Sections */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-6 scrollbar-thin scrollbar-thumb-slate-800">
        <div>
          <div className="px-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Workspace
          </div>
          <nav className="space-y-1">
            <button
              onClick={() => onSelectTab("workspace")}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                currentTab === "workspace"
                  ? "bg-slate-800 text-white border-l-2 border-indigo-500"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <Search className="w-4 h-4 text-cyan-400" />
              Active Workspace
            </button>
            <button
              onClick={() => onSelectTab("reports")}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                currentTab === "reports"
                  ? "bg-slate-800 text-white border-l-2 border-indigo-500"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <FileText className="w-4 h-4 text-indigo-400" />
              Comparative Reports
            </button>
            <button
              onClick={() => onSelectTab("repository")}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                currentTab === "repository"
                  ? "bg-slate-800 text-white border-l-2 border-indigo-500"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              Indexed Papers & RAG
            </button>
          </nav>
        </div>

        {/* Previous Sessions */}
        <div>
          <div className="px-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center justify-between">
            <span>Recent Sessions</span>
            <History className="w-3.5 h-3.5" />
          </div>
          {historySessions.length === 0 ? (
            <p className="px-3 text-xs text-slate-500 italic">No previous investigations</p>
          ) : (
            <div className="space-y-1">
              {historySessions.map((s) => (
                <button
                  key={s.id}
                  onClick={() => onSelectSession(s.id)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors truncate block ${
                    activeSessionId === s.id
                      ? "bg-slate-800/90 text-indigo-300 border border-slate-700"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                  }`}
                >
                  <p className="font-medium truncate">{s.query}</p>
                  <div className="flex items-center gap-2 mt-1 text-[10px] text-slate-500">
                    <span className="font-mono">{s.id.slice(0, 8)}</span>
                    <span>•</span>
                    <span className={s.status === "GOAL_ACHIEVED" ? "text-emerald-400" : "text-amber-400"}>
                      {s.status}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* System Health / Footer */}
      <div className="p-3 border-t border-slate-800 bg-slate-950/40 text-xs">
        <div className="flex items-center justify-between text-slate-400 mb-1">
          <span className="font-medium text-slate-300 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            Backend Status
          </span>
          {health?.status === "healthy" || health?.status === "ok" ? (
            <span className="flex items-center gap-1 text-emerald-400 font-mono text-[10px]">
              <CheckCircle2 className="w-3 h-3" /> ONLINE
            </span>
          ) : (
            <span className="flex items-center gap-1 text-amber-400 font-mono text-[10px]">
              <AlertTriangle className="w-3 h-3" /> CONNECTING
            </span>
          )}
        </div>
        <div className="text-[10px] text-slate-500 space-y-0.5 font-mono">
          <p className="truncate">Primary: {health?.services?.llm_primary || health?.primary_llm || "Gemini"}</p>
          <p className="truncate">Fallback: {health?.services?.llm_fallback || health?.fallback_llm || "Groq"}</p>
          {(health?.services?.llm_router?.includes("Degraded: True") || health?.degraded_mode) && (
            <p className="text-amber-400 font-semibold">DEGRADED MODE ACTIVE</p>
          )}
        </div>
      </div>
    </aside>
  );
};

"use client";

import React, { useState } from "react";
import {
  BookOpen,
  Search,
  History,
  FileText,
  Layers,
  PlusCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  X,
  Cpu
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
  onRemoveSession: (id: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  onNewSession,
  health,
  historySessions,
  activeSessionId,
  onSelectSession,
  onRemoveSession,
}) => {
  const [collapsed, setCollapsed] = useState(false);

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "workspace", label: "Investigation Workspace", icon: Search },
    { id: "reports", label: "Comparative Reports", icon: FileText },
    { id: "repository", label: "Indexed Papers & RAG", icon: Layers },
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "GOAL_ACHIEVED":
      case "COMPLETED":
        return <span className="text-[9px] font-medium text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-1.5 py-0.5 rounded">Completed</span>;
      case "USER_CLARIFICATION_REQUIRED":
        return <span className="text-[9px] font-medium text-amber-400 bg-amber-950/40 border border-amber-800/40 px-1.5 py-0.5 rounded">Clarification</span>;
      case "IN_PROGRESS":
      case "RUNNING":
        return <span className="text-[9px] font-medium text-indigo-400 bg-indigo-950/40 border border-indigo-800/40 px-1.5 py-0.5 rounded animate-pulse">Running</span>;
      case "FAILED":
      case "SAFETY_STOP":
        return <span className="text-[9px] font-medium text-rose-400 bg-rose-950/40 border border-rose-800/40 px-1.5 py-0.5 rounded">Failed</span>;
      default:
        return <span className="text-[9px] font-medium text-slate-400 bg-slate-800/50 border border-slate-700/50 px-1.5 py-0.5 rounded">{status}</span>;
    }
  };

  return (
    <aside
      className={`bg-slate-900/95 border-r border-slate-800/80 text-slate-200 flex flex-col h-screen shrink-0 select-none transition-all duration-300 relative z-30 ${
        collapsed ? "w-20" : "w-72"
      }`}
    >
      {/* Brand Header */}
      <div className="p-4 border-b border-slate-800/80 flex items-center justify-between">
        <div className="flex items-center space-x-3 overflow-hidden">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-700 to-cyan-500 flex items-center justify-center shadow-lg shadow-indigo-600/20 shrink-0">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          {!collapsed && (
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-bold text-base text-white tracking-tight">AIRA</span>
                <span className="text-[10px] font-mono font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 px-1.5 py-0.5 rounded">
                  v0.1
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium truncate">Intelligent Research Assistant</p>
            </div>
          )}
        </div>

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800/60 transition-colors"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Primary Action: New Investigation */}
      <div className="p-3">
        <button
          onClick={onNewSession}
          className={`w-full flex items-center justify-center gap-2.5 py-2.5 px-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs tracking-wide transition-all shadow-md shadow-indigo-600/20 active:scale-[0.98] ${
            collapsed ? "px-0" : ""
          }`}
          title="Start New Investigation"
        >
          <PlusCircle className="w-4 h-4 shrink-0" />
          {!collapsed && <span>New Investigation</span>}
        </button>
      </div>

      {/* Navigation Links */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-6 scrollbar-thin scrollbar-thumb-slate-800">
        <div>
          {!collapsed && (
            <div className="px-3 text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
              Platform Navigation
            </div>
          )}
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  title={collapsed ? item.label : undefined}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-medium transition-all ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-300 border border-indigo-500/30 font-semibold"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                  } ${collapsed ? "justify-center px-0" : ""}`}
                >
                  <Icon className={`w-4 h-4 shrink-0 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Previous Investigation Sessions */}
        {!collapsed && (
          <div>
            <div className="px-3 text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center justify-between">
              <span>Recent Inquiries</span>
              <History className="w-3.5 h-3.5 text-slate-500" />
            </div>

            {historySessions.length === 0 ? (
              <div className="px-3 py-3 rounded-lg border border-dashed border-slate-800/80 text-center">
                <p className="text-[11px] text-slate-500 italic">No investigations recorded yet</p>
              </div>
            ) : (
              <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                {historySessions.map((s) => {
                  const isSelected = activeSessionId === s.id;
                  return (
                    <div key={s.id} className="group relative">
                      <button
                        onClick={() => onSelectSession(s.id)}
                        className={`w-full text-left p-2.5 pr-8 rounded-xl text-xs transition-all border block ${
                          isSelected
                            ? "bg-slate-800/90 text-indigo-200 border-indigo-500/40 shadow-sm"
                            : "bg-slate-900/60 hover:bg-slate-800/50 text-slate-400 hover:text-slate-200 border-slate-800/70"
                        }`}
                      >
                        <p className="font-medium text-slate-200 truncate leading-snug">{s.query}</p>
                        <div className="flex items-center justify-between gap-1.5 mt-2">
                          <span className="font-mono text-[9px] text-slate-500">{s.id.slice(0, 10)}</span>
                          {getStatusBadge(s.status)}
                        </div>
                      </button>
                      <button
                        onClick={() => onRemoveSession(s.id)}
                        className="absolute top-2 right-2 p-1 rounded-md text-slate-500 hover:text-rose-300 hover:bg-rose-500/10 opacity-60 group-hover:opacity-100 focus:opacity-100"
                        title="Remove from recent inquiries"
                        aria-label={`Remove "${s.query}" from recent inquiries`}
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* System Health / Footer */}
      <div className={`border-t border-slate-800/80 bg-slate-950/60 p-3 text-xs ${collapsed ? "text-center" : ""}`}>
        {!collapsed ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-slate-300">
              <span className="font-medium flex items-center gap-1.5 text-[11px]">
                <Cpu className="w-3.5 h-3.5 text-indigo-400" />
                LangGraph Kernel
              </span>
              {health?.status === "healthy" || health?.status === "ok" ? (
                <span className="flex items-center gap-1 text-emerald-400 font-mono text-[10px] font-semibold">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  ONLINE
                </span>
              ) : (
                <span className="flex items-center gap-1 text-amber-400 font-mono text-[10px]">
                  <AlertTriangle className="w-3 h-3" />
                  CONNECTING
                </span>
              )}
            </div>

            <div className="text-[10px] text-slate-500 space-y-0.5 font-mono bg-slate-900/80 p-2 rounded-lg border border-slate-800/80">
              <div className="flex items-center justify-between">
                <span>LLM Primary:</span>
                <span className="text-slate-300">{health?.services?.llm_primary || health?.primary_llm || "Gemini 2.5"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Fallback:</span>
                <span className="text-slate-300">{health?.services?.llm_fallback || health?.fallback_llm || "Groq Llama-3"}</span>
              </div>
              {(health?.services?.llm_router?.includes("Degraded: True") || health?.degraded_mode) && (
                <div className="text-amber-400 font-semibold pt-1 text-[9px]">DEGRADED MODE ACTIVE</div>
              )}
            </div>
          </div>
        ) : (
          <div className="flex justify-center" title="System Online">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 ring-4 ring-emerald-500/20" />
          </div>
        )}
      </div>
    </aside>
  );
};

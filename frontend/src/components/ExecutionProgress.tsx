"use client";

import React, { useState } from "react";
import { CheckCircle2, Clock, Terminal, ChevronDown, ChevronUp, Cpu, RefreshCw } from "lucide-react";
import { ResearchPlanTask, ObservableLog } from "@/types/aira";

interface ExecutionProgressProps {
  plan?: ResearchPlanTask[] | null;
  logs?: ObservableLog[];
  status: string;
}

export const ExecutionProgress: React.FC<ExecutionProgressProps> = ({
  plan,
  logs = [],
  status,
}) => {
  const [showLogs, setShowLogs] = useState(true);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <Cpu className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white tracking-tight">
              Adaptive Research Plan & Agent Execution
            </h3>
            <p className="text-xs text-slate-400">
              LangGraph stateful orchestration with observable tool calls
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`text-xs px-2.5 py-1 rounded-full font-mono font-medium border ${
              status === "GOAL_ACHIEVED"
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                : status === "USER_CLARIFICATION_REQUIRED"
                ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                : status === "MAX_STEPS_REACHED"
                ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                : "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
            }`}
          >
            {status}
          </span>
        </div>
      </div>

      {/* Plan Steps Grid */}
      {plan && plan.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Staged Sub-Tasks
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {plan.map((task, idx) => {
              const isCompleted = task.status === "COMPLETED";
              const isFailed = task.status === "FAILED";
              const isAdapted = task.status === "ADAPTED";

              return (
                <div
                  key={task.task_id || idx}
                  className={`p-3.5 rounded-xl border transition-all ${
                    isCompleted
                      ? "bg-slate-950/60 border-emerald-900/40 text-slate-200"
                      : isFailed
                      ? "bg-rose-950/20 border-rose-800/40 text-rose-200"
                      : isAdapted
                      ? "bg-amber-950/20 border-amber-800/40 text-amber-200"
                      : "bg-slate-950/40 border-slate-800 text-slate-400"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <span className="text-xs font-semibold tracking-tight text-white flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-slate-800 text-[10px] flex items-center justify-center font-mono text-slate-400">
                        {idx + 1}
                      </span>
                      {task.name}
                    </span>
                    {isCompleted ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    ) : isAdapted ? (
                      <RefreshCw className="w-4 h-4 text-amber-400 shrink-0" />
                    ) : (
                      <Clock className="w-4 h-4 text-slate-500 shrink-0" />
                    )}
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed ml-7">
                    {task.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Observable Event Stream */}
      <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-950">
        <button
          onClick={() => setShowLogs(!showLogs)}
          className="w-full px-4 py-2.5 bg-slate-900/60 border-b border-slate-800/80 flex items-center justify-between text-xs font-semibold text-slate-300 hover:bg-slate-900 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            <span>Observable Tool Logs & Node Transitions ({logs.length})</span>
          </div>
          {showLogs ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showLogs && (
          <div className="p-3 max-h-60 overflow-y-auto font-mono text-[11px] space-y-2 scrollbar-thin scrollbar-thumb-slate-800">
            {logs.length === 0 ? (
              <p className="text-slate-600 italic">No transition events recorded yet.</p>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="flex items-start gap-2 border-l border-slate-800 pl-2">
                  <span className="text-slate-500 shrink-0 text-[10px]">
                    [{log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : `Step ${log.step}`}]
                  </span>
                  <span className="text-indigo-400 shrink-0 font-bold">{log.node}:</span>
                  <span className="text-slate-300">{log.decision}</span>
                  {log.tool && (
                    <span className="ml-auto shrink-0 bg-slate-800 text-slate-400 text-[9px] px-1.5 py-0.5 rounded">
                      tool: {log.tool}
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
};

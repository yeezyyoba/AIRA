"use client";

import React, { useState } from "react";
import {
  CheckCircle2,
  Clock,
  Terminal,
  ChevronDown,
  ChevronUp,
  Cpu,
  RefreshCw,
  AlertCircle,
  Wrench,
  BrainCircuit,
} from "lucide-react";
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
  const [logFilter, setLogFilter] = useState<"ALL" | "TOOLS" | "VALIDATION" | "REFLECTIONS">("ALL");
  const [expandedLogIdx, setExpandedLogIdx] = useState<number | null>(null);
  const [showLogs, setShowLogs] = useState(true);

  // Filter logs based on category
  const filteredLogs = logs.filter((log) => {
    if (logFilter === "TOOLS") return !!log.tool || log.node === "execute_tools" || log.node === "select_tools";
    if (logFilter === "VALIDATION") return log.node === "validate_tool_outputs" || log.node === "retry_handler";
    if (logFilter === "REFLECTIONS") return log.node === "reflect_and_revise" || log.node === "create_plan" || log.node === "receive_request";
    return true;
  });

  // Calculate task counts
  const totalTasks = plan?.length || 0;
  const completedTasks = plan?.filter((t) => t.status === "COMPLETED").length || 0;
  const adaptedTasks = plan?.filter((t) => t.status === "ADAPTED").length || 0;

  const getStatusBadge = () => {
    switch (status) {
      case "GOAL_ACHIEVED":
      case "COMPLETED":
        return <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold flex items-center gap-1.5"><CheckCircle2 className="w-3.5 h-3.5" /> GOAL ACHIEVED</span>;
      case "USER_CLARIFICATION_REQUIRED":
        return <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold flex items-center gap-1.5"><AlertCircle className="w-3.5 h-3.5" /> CLARIFICATION NEEDED</span>;
      case "MAX_STEPS_REACHED":
        return <span className="bg-rose-500/10 text-rose-400 border border-rose-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold">MAX STEPS REACHED</span>;
      case "PARTIAL":
        return <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold flex items-center gap-1.5"><AlertCircle className="w-3.5 h-3.5" /> PARTIAL RESULT</span>;
      case "INSUFFICIENT_EVIDENCE":
        return <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold flex items-center gap-1.5"><AlertCircle className="w-3.5 h-3.5" /> INSUFFICIENT EVIDENCE</span>;
      case "OUT_OF_SCOPE":
        return <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold flex items-center gap-1.5"><AlertCircle className="w-3.5 h-3.5" /> OUT OF SCOPE</span>;
      case "UNACCEPTABLE_RISK":
        return <span className="bg-rose-500/10 text-rose-400 border border-rose-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold flex items-center gap-1.5"><AlertCircle className="w-3.5 h-3.5" /> REFUSED (SAFETY)</span>;
      case "RETRY_REQUESTED":
        return <span className="bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold flex items-center gap-1.5 animate-pulse"><RefreshCw className="w-3.5 h-3.5" /> RETRY IN PROGRESS</span>;
      default:
        return <span className="bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-xs px-3 py-1 rounded-full font-mono font-semibold flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5 animate-spin" /> {status}</span>;
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-7 shadow-xl space-y-7 relative overflow-hidden">
      {/* Header with Agent State */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-slate-800/80 gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-indigo-500/10 border border-indigo-500/25 flex items-center justify-center shadow-inner">
            <BrainCircuit className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold text-white tracking-tight">
                LangGraph Agent Execution & State Observability
              </h3>
            </div>
            <p className="text-xs text-slate-400">
              Structured multi-task plan execution with autonomous reflection and tool validation
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          {getStatusBadge()}
        </div>
      </div>

      {/* Plan Section */}
      {plan && plan.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Research Plan Progress
              </span>
              <span className="text-[10px] font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded-full border border-slate-800">
                {completedTasks}/{totalTasks} phases completed
              </span>
              {adaptedTasks > 0 && (
                <span className="text-[10px] font-mono text-amber-400 bg-amber-950/40 px-2 py-0.5 rounded-full border border-amber-800/40">
                  {adaptedTasks} adapted
                </span>
              )}
            </div>

            {/* Micro progress bar */}
            <div className="hidden sm:flex items-center gap-2 w-36">
              <div className="w-full bg-slate-950 rounded-full h-1.5 border border-slate-800 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-indigo-500 to-emerald-400 h-1.5 rounded-full transition-all duration-500"
                  style={{ width: `${(completedTasks / totalTasks) * 100}%` }}
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {plan.map((task, idx) => {
              const isCompleted = task.status === "COMPLETED";
              const isFailed = task.status === "FAILED";
              const isAdapted = task.status === "ADAPTED";
              const isRunning = task.status === "RUNNING";

              return (
                <div
                  key={task.task_id || idx}
                  className={`p-4 rounded-2xl border transition-all flex flex-col justify-between ${
                    isCompleted
                      ? "bg-slate-950/70 border-emerald-900/40 text-slate-200"
                      : isFailed
                      ? "bg-rose-950/20 border-rose-800/40 text-rose-200"
                      : isAdapted
                      ? "bg-amber-950/20 border-amber-800/40 text-amber-200"
                      : isRunning
                      ? "bg-indigo-950/20 border-indigo-500/40 text-indigo-100 shadow-md shadow-indigo-500/5 ring-1 ring-indigo-500/20"
                      : "bg-slate-950/40 border-slate-800/80 text-slate-400"
                  }`}
                >
                  <div className="space-y-1.5">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className={`w-5 h-5 rounded-full text-[10px] flex items-center justify-center font-mono font-bold shrink-0 ${
                          isCompleted
                            ? "bg-emerald-500/20 text-emerald-400"
                            : isRunning
                            ? "bg-indigo-500 text-white animate-pulse"
                            : "bg-slate-800 text-slate-400"
                        }`}>
                          {idx + 1}
                        </span>
                        <h4 className="text-xs font-semibold text-white tracking-tight line-clamp-1">
                          {task.name}
                        </h4>
                      </div>

                      {isCompleted ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      ) : isAdapted ? (
                        <RefreshCw className="w-4 h-4 text-amber-400 shrink-0" />
                      ) : isRunning ? (
                        <Cpu className="w-4 h-4 text-indigo-400 shrink-0 animate-spin" />
                      ) : (
                        <Clock className="w-4 h-4 text-slate-600 shrink-0" />
                      )}
                    </div>

                    <p className="text-[11px] text-slate-400 leading-relaxed line-clamp-2 pl-7">
                      {task.description}
                    </p>
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono">
                    <span className="text-slate-500">{task.task_id}</span>
                    <span className={
                      isCompleted ? "text-emerald-400 font-semibold" :
                      isRunning ? "text-indigo-400 font-semibold" :
                      isAdapted ? "text-amber-400 font-semibold" :
                      "text-slate-500"
                    }>
                      {task.status}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Observable Event Stream & Tool Call Timeline */}
      <div className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-950/90 shadow-md">
        {/* Stream Header & Filters */}
        <div className="px-5 py-3.5 bg-slate-900/90 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <Terminal className="w-4 h-4 text-indigo-400" />
            <span className="text-xs font-bold text-white tracking-tight">
              Agent Activity Timeline & Tool Traces
            </span>
            <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded-full">
              {logs.length} events
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center bg-slate-950 p-0.5 rounded-xl border border-slate-800 text-[10px] font-medium">
              <button
                onClick={() => setLogFilter("ALL")}
                className={`px-2.5 py-1 rounded-lg transition-colors ${logFilter === "ALL" ? "bg-indigo-600 text-white font-semibold" : "text-slate-400 hover:text-slate-200"}`}
              >
                All
              </button>
              <button
                onClick={() => setLogFilter("TOOLS")}
                className={`px-2.5 py-1 rounded-lg transition-colors ${logFilter === "TOOLS" ? "bg-indigo-600 text-white font-semibold" : "text-slate-400 hover:text-slate-200"}`}
              >
                Tools
              </button>
              <button
                onClick={() => setLogFilter("VALIDATION")}
                className={`px-2.5 py-1 rounded-lg transition-colors ${logFilter === "VALIDATION" ? "bg-indigo-600 text-white font-semibold" : "text-slate-400 hover:text-slate-200"}`}
              >
                Validation
              </button>
              <button
                onClick={() => setLogFilter("REFLECTIONS")}
                className={`px-2.5 py-1 rounded-lg transition-colors ${logFilter === "REFLECTIONS" ? "bg-indigo-600 text-white font-semibold" : "text-slate-400 hover:text-slate-200"}`}
              >
                Reflections
              </button>
            </div>

            <button
              onClick={() => setShowLogs(!showLogs)}
              className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
              title={showLogs ? "Collapse stream" : "Expand stream"}
            >
              {showLogs ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Stream List */}
        {showLogs && (
          <div className="p-4 max-h-80 overflow-y-auto space-y-2.5 text-xs font-mono scrollbar-thin scrollbar-thumb-slate-800">
            {filteredLogs.length === 0 ? (
              <div className="text-center py-8 text-slate-500 italic">
                No events matching the selected filter.
              </div>
            ) : (
              filteredLogs.map((log, i) => {
                const isExpanded = expandedLogIdx === i;
                const isSuccess = log.outcome === "SUCCESS" || log.outcome === "VALID" || log.outcome === "COMPLETE";
                const isRetry = log.outcome?.includes("RETRY");
                const isError = log.outcome === "FAILED" || log.outcome === "ERROR" || log.outcome === "INVALID";

                return (
                  <div
                    key={i}
                    onClick={() => setExpandedLogIdx(isExpanded ? null : i)}
                    className={`p-3 rounded-xl border transition-all cursor-pointer ${
                      isExpanded
                        ? "bg-slate-900 border-indigo-500/40 shadow-sm"
                        : "bg-slate-900/40 border-slate-800/80 hover:bg-slate-900/80 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-2 shrink-0">
                        <span className="text-[10px] text-slate-500 font-mono">
                          {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : `Step ${log.step}`}
                        </span>

                        <span className="text-[11px] font-bold text-indigo-400 bg-indigo-500/10 px-1.5 py-0.5 rounded border border-indigo-500/20">
                          {log.node}
                        </span>
                      </div>

                      <div className="flex items-center gap-2 ml-auto shrink-0">
                        {log.tool && (
                          <span className="text-[10px] font-semibold bg-cyan-950/60 text-cyan-300 border border-cyan-800/40 px-2 py-0.5 rounded flex items-center gap-1">
                            <Wrench className="w-3 h-3" />
                            {log.tool}
                          </span>
                        )}

                        <span className={`text-[10px] px-2 py-0.5 rounded font-semibold border ${
                          isSuccess
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                            : isRetry
                            ? "bg-amber-500/10 text-amber-400 border-amber-500/20 animate-pulse"
                            : isError
                            ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                            : "bg-slate-800 text-slate-400 border-slate-700"
                        }`}>
                          {log.outcome}
                        </span>
                      </div>
                    </div>

                    <p className="text-slate-300 font-sans text-xs mt-2 leading-relaxed">
                      {log.decision}
                    </p>

                    {/* Expandable technical details */}
                    {isExpanded && log.details && (
                      <div className="mt-3 pt-2.5 border-t border-slate-800/80">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                          Node Payload Details:
                        </span>
                        <pre className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 text-[10px] text-slate-300 overflow-x-auto whitespace-pre-wrap">
                          {JSON.stringify(log.details, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
};

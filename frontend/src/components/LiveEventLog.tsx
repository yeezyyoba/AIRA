"use client";

import React, { useEffect, useRef, useState } from "react";
import { Activity, ChevronDown, ChevronUp, ArrowDownToLine } from "lucide-react";
import { airaApi } from "@/services/api";
import { SessionEvent } from "@/types/aira";

interface LiveEventLogProps {
  sessionId: string;
  /** True while the backend run is active; a new run (e.g. after resume) reconnects the stream. */
  isRunning: boolean;
}

const levelStyles: Record<string, string> = {
  ERROR: "text-rose-300 bg-rose-500/10 border-rose-500/30",
  CRITICAL: "text-rose-300 bg-rose-500/10 border-rose-500/30",
  WARNING: "text-amber-300 bg-amber-500/10 border-amber-500/30",
  INFO: "text-slate-400 bg-slate-800/60 border-slate-700/60",
  DEBUG: "text-slate-500 bg-slate-900 border-slate-800",
};

const kindStyles: Record<string, string> = {
  node: "text-indigo-200 font-semibold",
  decision: "text-slate-200",
  status: "text-emerald-300 font-semibold",
  log: "text-slate-400",
};

const formatTime = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};

const formatElapsed = (ms: number) => {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
};

/**
 * Streams a research session's live event log over Server-Sent Events.
 * Mount with `key={sessionId}` so switching sessions starts from a clean log.
 */
export const LiveEventLog: React.FC<LiveEventLogProps> = ({ sessionId, isRunning }) => {
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [stepsOnly, setStepsOnly] = useState(false);
  const [stickToBottom, setStickToBottom] = useState(true);
  const [now, setNow] = useState(() => Date.now());
  const lastSeqRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Open (or reopen after a resume) the event stream.
  useEffect(() => {
    const source = new EventSource(`${airaApi.getSessionEventsUrl(sessionId)}?after=${lastSeqRef.current}`);
    source.onopen = () => setConnected(true);
    source.addEventListener("log", (msg) => {
      try {
        const event = JSON.parse((msg as MessageEvent).data) as SessionEvent;
        if (event.seq <= lastSeqRef.current) return;
        lastSeqRef.current = event.seq;
        setEvents((prev) => [...prev, event].slice(-2000));
      } catch {}
    });
    source.addEventListener("end", () => {
      setConnected(false);
      source.close();
    });
    source.onerror = () => {
      // EventSource retries on its own (resuming via Last-Event-ID); just reflect the state.
      setConnected(false);
    };
    return () => source.close();
  }, [sessionId, isRunning]);

  // Tick the elapsed-time counter while running.
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [isRunning]);

  const visible = stepsOnly ? events.filter((e) => e.kind !== "log") : events;

  // Follow new events unless the user scrolled up to read.
  useEffect(() => {
    if (stickToBottom && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visible.length, stickToBottom, collapsed]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    setStickToBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
  };

  // A resumed session keeps earlier runs' events; time the current run from
  // the first event after the previous run's closing status event.
  const lastStatusIdx = events.map((e) => e.kind).lastIndexOf("status");
  const currentRunFirst = events[lastStatusIdx + 1];
  const startedAt = currentRunFirst ? new Date(currentRunFirst.timestamp).getTime() : null;
  const warnings = events.filter((e) => e.level === "WARNING").length;
  const errors = events.filter((e) => e.level === "ERROR" || e.level === "CRITICAL").length;
  const latestStep = [...events].reverse().find((e) => e.kind === "node");

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-3xl shadow-xl overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 bg-slate-950/60">
        <div className="flex items-center gap-3 min-w-0">
          <Activity className={`w-5 h-5 shrink-0 ${isRunning ? "text-indigo-400" : "text-slate-500"}`} />
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              <h3 className="text-base font-semibold text-white">Live Event Log</h3>
              {isRunning ? (
                <span className="flex items-center gap-1.5 text-[11px] font-mono font-semibold text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                  <span className={`w-1.5 h-1.5 rounded-full bg-emerald-400 ${connected ? "animate-pulse" : "opacity-40"}`} />
                  {connected ? "LIVE" : "RECONNECTING"}
                </span>
              ) : (
                <span className="text-[11px] font-mono text-slate-400 bg-slate-800 border border-slate-700 px-2 py-0.5 rounded-full">
                  ENDED
                </span>
              )}
            </div>
            <p className="text-sm text-slate-400 truncate">
              {isRunning && latestStep ? `Now: ${latestStep.message}` : `${events.length} events`}
              {isRunning && startedAt ? ` · running for ${formatElapsed(now - startedAt)}` : ""}
              {warnings ? ` · ${warnings} warning${warnings === 1 ? "" : "s"}` : ""}
              {errors ? ` · ${errors} error${errors === 1 ? "" : "s"}` : ""}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setStepsOnly((v) => !v)}
            className={`text-xs font-medium px-3 py-1.5 rounded-lg border transition-colors ${
              stepsOnly
                ? "bg-indigo-600/20 text-indigo-200 border-indigo-500/40"
                : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
            }`}
            title="Hide low-level backend log lines and show only workflow steps"
          >
            {stepsOnly ? "Steps only" : "All events"}
          </button>
          <button
            onClick={() => setCollapsed((v) => !v)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-700"
            title={collapsed ? "Expand log" : "Collapse log"}
          >
            {collapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Body */}
      {!collapsed && (
        <div className="relative">
          <div
            ref={scrollRef}
            onScroll={handleScroll}
            className="max-h-96 overflow-y-auto px-4 py-3 font-mono text-[13px] leading-6"
          >
            {visible.length === 0 ? (
              <p className="text-slate-500 italic px-2 py-6 text-center font-sans text-sm">
                {isRunning ? "Waiting for the first events…" : "No live events recorded for this session (the log is kept in memory and resets when the backend restarts)."}
              </p>
            ) : (
              <ul className="space-y-0.5">
                {visible.map((e) => (
                  <li
                    key={e.seq}
                    className={`flex items-start gap-3 px-2 py-1 rounded-md hover:bg-slate-800/40 ${
                      e.kind === "node" ? "mt-2 first:mt-0" : ""
                    }`}
                  >
                    <span className="text-slate-500 shrink-0 tabular-nums">{formatTime(e.timestamp)}</span>
                    <span
                      className={`shrink-0 text-[10px] font-bold px-1.5 rounded border uppercase leading-5 mt-0.5 ${
                        levelStyles[e.level] || levelStyles.INFO
                      }`}
                    >
                      {e.level === "WARNING" ? "WARN" : e.level}
                    </span>
                    {e.source && (
                      <span className="text-cyan-400/80 shrink-0 max-w-[11rem] truncate" title={e.source}>
                        {e.source}
                      </span>
                    )}
                    <span className={`min-w-0 break-words ${
                      e.level === "ERROR" ? "text-rose-200" : e.level === "WARNING" ? "text-amber-200" : kindStyles[e.kind] || kindStyles.log
                    }`}>
                      {e.message}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {!stickToBottom && visible.length > 0 && (
            <button
              onClick={() => setStickToBottom(true)}
              className="absolute bottom-3 right-5 flex items-center gap-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-full shadow-lg"
            >
              <ArrowDownToLine className="w-3.5 h-3.5" />
              Jump to latest
            </button>
          )}
        </div>
      )}
    </div>
  );
};

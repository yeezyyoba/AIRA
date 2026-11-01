"use client";

import React, { useEffect, useRef, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { ResearchQueryInput } from "@/components/ResearchQueryInput";
import { ClarificationBanner } from "@/components/ClarificationBanner";
import { ExecutionProgress } from "@/components/ExecutionProgress";
import { ResearchResultsView } from "@/components/ResearchResultsView";
import { airaApi, ApiClientError } from "@/services/api";
import { ResearchResponse, SystemHealth } from "@/types/aira";
import { AlertCircle, RefreshCw, Layers, FileText } from "lucide-react";

export default function Home() {
  const [currentTab, setCurrentTab] = useState("workspace");
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [currentSession, setCurrentSession] = useState<ResearchResponse | null>(null);
  const [historySessions, setHistorySessions] = useState<Array<{ id: string; query: string; status: string; date: string }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Repository & Reports tabs data
  const [indexedPapers, setIndexedPapers] = useState<any[]>([]);
  const [savedReports, setSavedReports] = useState<any[]>([]);

  // Polling guards
  const isPollingRef = useRef(false);
  const failureCountRef = useRef(0);

  // Restore session history & active session from localStorage on initial load
  useEffect(() => {
    fetchHealth();
    loadReportsList();
    loadPapersList();

    try {
      const savedHistory = localStorage.getItem("aira_history_sessions");
      if (savedHistory) {
        setHistorySessions(JSON.parse(savedHistory));
      }
      const lastSessionId = localStorage.getItem("aira_active_session_id");
      if (lastSessionId) {
        restoreSession(lastSessionId);
      }
    } catch {
      // Storage unavailable or disabled
    }
  }, []);

  // Sync history to localStorage whenever it changes
  useEffect(() => {
    try {
      if (historySessions.length > 0) {
        localStorage.setItem("aira_history_sessions", JSON.stringify(historySessions));
      }
    } catch {}
  }, [historySessions]);

  // Sync activeSessionId to localStorage
  useEffect(() => {
    try {
      if (currentSession?.session_id) {
        localStorage.setItem("aira_active_session_id", currentSession.session_id);
      }
    } catch {}
  }, [currentSession?.session_id]);

  // Non-overlapping polling for active running sessions
  useEffect(() => {
    const isTerminal = (status?: string) => {
      if (!status) return true;
      return [
        "COMPLETED",
        "GOAL_ACHIEVED",
        "MAX_STEPS_REACHED",
        "USER_CLARIFICATION_REQUIRED",
        "INSUFFICIENT_EVIDENCE",
        "SAFETY_STOP",
        "EXPIRED",
        "FAILED",
      ].includes(status);
    };

    if (!currentSession?.session_id || isTerminal(currentSession.status)) {
      return;
    }

    const intervalId = setInterval(async () => {
      if (isPollingRef.current) return;
      isPollingRef.current = true;

      try {
        const updated = await airaApi.getSession(currentSession.session_id);
        failureCountRef.current = 0;
        setCurrentSession(updated);

        // Update status in history
        setHistorySessions((prev) =>
          prev.map((s) => (s.id === updated.session_id ? { ...s, status: updated.status } : s))
        );

        if (isTerminal(updated.status)) {
          clearInterval(intervalId);
          loadReportsList();
          loadPapersList();
        }
      } catch (err: any) {
        failureCountRef.current += 1;
        if (failureCountRef.current >= 3) {
          clearInterval(intervalId);
          setErrorMessage(
            `Connection issue while tracking session '${currentSession.session_id}'. Please check backend connectivity or refresh.`
          );
        }
      } finally {
        isPollingRef.current = false;
      }
    }, 2500);

    return () => {
      clearInterval(intervalId);
    };
  }, [currentSession?.session_id, currentSession?.status]);

  const restoreSession = async (sessionId: string) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await airaApi.getSession(sessionId);
      setCurrentSession(res);
    } catch (err: any) {
      setErrorMessage(
        `Session '${sessionId}' could not be restored because the backend checkpoint expired or was cleared. You can start a new research session.`
      );
      setHistorySessions((prev) =>
        prev.map((s) => (s.id === sessionId ? { ...s, status: "EXPIRED" } : s))
      );
      try {
        localStorage.removeItem("aira_active_session_id");
      } catch {}
    } finally {
      setIsLoading(false);
    }
  };

  const fetchHealth = async () => {
    try {
      const data = await airaApi.getHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  };

  const loadReportsList = async () => {
    try {
      const data = await airaApi.listReports();
      setSavedReports(data.reports || []);
    } catch {}
  };

  const loadPapersList = async () => {
    try {
      const data = await airaApi.listPapers();
      setIndexedPapers(data.extracted_papers || data.papers || []);
    } catch {}
  };

  // Start new session
  const handleStartSession = async (query: string) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await airaApi.startSession(query);
      setCurrentSession(res);

      // Add to session history
      setHistorySessions((prev) => [
        {
          id: res.session_id,
          query: res.research_goal || query,
          status: res.status,
          date: new Date().toLocaleTimeString(),
        },
        ...prev.filter((p) => p.id !== res.session_id),
      ]);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to start research investigation.");
    } finally {
      setIsLoading(false);
      loadReportsList();
      loadPapersList();
    }
  };

  // Resume paused session
  const handleResumeSession = async (clarification: string) => {
    if (!currentSession?.session_id) return;
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await airaApi.resumeSession(currentSession.session_id, clarification);
      setCurrentSession(res);

      setHistorySessions((prev) =>
        prev.map((s) => (s.id === res.session_id ? { ...s, status: res.status } : s))
      );
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to resume session.");
    } finally {
      setIsLoading(false);
      loadReportsList();
      loadPapersList();
    }
  };

  // Select historical session
  const handleSelectSession = async (sessionId: string) => {
    setIsLoading(true);
    setErrorMessage(null);
    setCurrentTab("workspace");
    try {
      const res = await airaApi.getSession(sessionId);
      setCurrentSession(res);
    } catch (err: any) {
      setErrorMessage(
        `Could not retrieve session '${sessionId}' (checkpoint may have expired after backend restart).`
      );
      setHistorySessions((prev) =>
        prev.map((s) => (s.id === sessionId ? { ...s, status: "EXPIRED" } : s))
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        onNewSession={() => {
          setCurrentSession(null);
          setCurrentTab("workspace");
          setErrorMessage(null);
        }}
        health={health}
        historySessions={historySessions}
        activeSessionId={currentSession?.session_id}
        onSelectSession={handleSelectSession}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 overflow-y-auto p-8 space-y-6 scrollbar-thin scrollbar-thumb-slate-800">
        {/* Error notification banner */}
        {errorMessage && (
          <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-200 text-sm flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <button
                onClick={() => {
                  setCurrentSession(null);
                  setCurrentTab("workspace");
                  setErrorMessage(null);
                  try {
                    localStorage.removeItem("aira_active_session_id");
                  } catch {}
                }}
                className="text-xs bg-rose-900/60 hover:bg-rose-900 text-rose-200 px-2.5 py-1 rounded border border-rose-700/50 transition-colors"
              >
                Start New Session
              </button>
              <button
                onClick={() => setErrorMessage(null)}
                className="text-xs text-rose-400 hover:text-rose-200 underline"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* Tab 1: Workspace */}
        {currentTab === "workspace" && (
          <div className="space-y-6 max-w-6xl mx-auto">
            {/* Top Formulation Input */}
            <ResearchQueryInput
              onSubmit={handleStartSession}
              isLoading={isLoading}
            />

            {/* Clarification prompt if paused */}
            {currentSession?.status === "USER_CLARIFICATION_REQUIRED" && (
              <ClarificationBanner
                question={currentSession.clarification_question || "Please clarify your research scope."}
                missingInformation={currentSession.missing_information}
                onResume={handleResumeSession}
                isLoading={isLoading}
              />
            )}

            {/* Execution Plan & Observable Logs */}
            {currentSession && (
              <ExecutionProgress
                plan={currentSession.plan}
                logs={currentSession.observable_logs}
                status={currentSession.status}
              />
            )}

            {/* Results, Provenance and Evidence Viewer */}
            {currentSession && (
              <ResearchResultsView
                report={currentSession.final_report}
                discoveredPapers={currentSession.discovered_papers}
                validatedPapers={currentSession.validated_papers}
                extractedPapers={currentSession.extracted_papers}
                metrics={currentSession.metric_comparisons}
                gaps={currentSession.identified_gaps}
                citations={currentSession.citations}
                isDegraded={currentSession.is_degraded}
              />
            )}
          </div>
        )}

        {/* Tab 2: Comparative Reports Archive */}
        {currentTab === "reports" && (
          <div className="space-y-6 max-w-5xl mx-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
                  <FileText className="w-5 h-5 text-indigo-400" />
                  Comparative Research Reports Archive
                </h2>
                <p className="text-xs text-slate-400">
                  Published multi-paper comparative synthesis documents
                </p>
              </div>
              <button
                onClick={loadReportsList}
                className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Refresh
              </button>
            </div>

            {savedReports.length === 0 ? (
              <div className="text-center py-16 bg-slate-900/40 rounded-2xl border border-slate-800 text-slate-500">
                <FileText className="w-8 h-8 mx-auto mb-2 opacity-40" />
                <p>No comparative reports archived yet. Run an investigation in the workspace to generate reports.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {savedReports.map((r) => (
                  <div key={r.report_id} className="p-5 rounded-2xl bg-slate-900 border border-slate-800 flex items-start justify-between gap-4">
                    <div className="space-y-1.5">
                      <h3 className="text-sm font-bold text-white">{r.research_question}</h3>
                      <p className="text-xs text-slate-400">
                        Analyzed: {r.papers_analyzed?.join(", ") || "Corpus"} • Created: {new Date(r.created_at).toLocaleString()}
                      </p>
                      <span className="text-[10px] font-mono text-indigo-400 block">ID: {r.report_id}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <a
                        href={airaApi.getReportExportUrl(r.report_id, "markdown")}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 font-medium"
                      >
                        Markdown
                      </a>
                      <a
                        href={airaApi.getReportExportUrl(r.report_id, "json")}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 font-medium"
                      >
                        JSON
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Indexed Papers & RAG Repository */}
        {currentTab === "repository" && (
          <div className="space-y-6 max-w-5xl mx-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
                  <Layers className="w-5 h-5 text-emerald-400" />
                  Indexed Knowledge Base & RAG Corpus
                </h2>
                <p className="text-xs text-slate-400">
                  Extracted academic papers and semantic search chunks stored in ChromaDB
                </p>
              </div>
              <button
                onClick={loadPapersList}
                className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Refresh
              </button>
            </div>

            {indexedPapers.length === 0 ? (
              <div className="text-center py-16 bg-slate-900/40 rounded-2xl border border-slate-800 text-slate-500">
                <Layers className="w-8 h-8 mx-auto mb-2 opacity-40" />
                <p>No papers extracted or indexed yet in the persistent repository.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {indexedPapers.map((paperId, i) => (
                  <div key={i} className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-mono font-semibold text-white block">{paperId}</span>
                      <span className="text-[10px] text-emerald-400 font-medium">Extracted & Vector Indexed</span>
                    </div>
                    <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono">
                      PyMuPDF
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

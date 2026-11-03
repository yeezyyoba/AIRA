"use client";

import React, { useEffect, useRef, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { DashboardView } from "@/components/DashboardView";
import { ResearchQueryInput } from "@/components/ResearchQueryInput";
import { ClarificationBanner } from "@/components/ClarificationBanner";
import { ExecutionProgress } from "@/components/ExecutionProgress";
import { ResearchResultsView } from "@/components/ResearchResultsView";
import { ReportsView, type ReportSummary } from "@/components/ReportsView";
import { RepositoryView } from "@/components/RepositoryView";
import { LiveEventLog } from "@/components/LiveEventLog";
import { airaApi } from "@/services/api";
import { ResearchResponse, SystemHealth } from "@/types/aira";
import { AlertCircle } from "lucide-react";

export default function Home() {
  const [currentTab, setCurrentTab] = useState("dashboard");
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [currentSession, setCurrentSession] = useState<ResearchResponse | null>(null);
  const [historySessions, setHistorySessions] = useState<Array<{ id: string; query: string; status: string; date: string }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeQueryDraft, setActiveQueryDraft] = useState<string>("");

  // Repository & Reports tabs data
  const [indexedPapers, setIndexedPapers] = useState<string[]>([]);
  const [savedReports, setSavedReports] = useState<ReportSummary[]>([]);

  // Polling guards
  const isPollingRef = useRef(false);
  const failureCountRef = useRef(0);

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
      setSavedReports((data.reports || []) as ReportSummary[]);
    } catch {}
  };

  const loadPapersList = async () => {
    try {
      const data = await airaApi.listPapers();
      setIndexedPapers(data.extracted_papers || data.papers || []);
    } catch {}
  };

  const restoreSession = async (sessionId: string) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await airaApi.getSession(sessionId);
      setCurrentSession(res);
    } catch {
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

  // Restore session history & active session from localStorage on initial load
  useEffect(() => {
    queueMicrotask(() => {
      fetchHealth();
      loadReportsList();
      loadPapersList();
    });

    try {
      const savedHistory = localStorage.getItem("aira_history_sessions");
      if (savedHistory) {
        const parsedHistory = JSON.parse(savedHistory);
        queueMicrotask(() => setHistorySessions(parsedHistory));
      }
      const lastSessionId = localStorage.getItem("aira_active_session_id");
      if (lastSessionId) {
        queueMicrotask(() => restoreSession(lastSessionId));
      }
    } catch {
      // Storage unavailable or disabled
    }
  }, []);

  // Sync history to localStorage whenever it changes. Skip the very first
  // render (empty list) so it does not overwrite saved history before it loads.
  const historyLoadedRef = useRef(false);
  useEffect(() => {
    if (!historyLoadedRef.current) {
      historyLoadedRef.current = true;
      if (historySessions.length === 0) return;
    }
    try {
      localStorage.setItem("aira_history_sessions", JSON.stringify(historySessions));
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
        "PARTIAL",
        "OUT_OF_SCOPE",
        "UNACCEPTABLE_RISK",
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
          if (updated.status === "FAILED") {
            setErrorMessage(updated.message || `Research session '${updated.session_id}' failed. See the live event log for details.`);
          }
          loadReportsList();
          loadPapersList();
        }
      } catch {
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
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to start research investigation.");
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
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to resume session.");
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
    } catch {
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

  // Remove an inquiry from the local history list (the backend session is untouched).
  const handleRemoveSession = (sessionId: string) => {
    setHistorySessions((prev) => prev.filter((s) => s.id !== sessionId));
    if (currentSession?.session_id === sessionId) {
      setCurrentSession(null);
      setErrorMessage(null);
      try {
        localStorage.removeItem("aira_active_session_id");
      } catch {}
    }
  };

  const isRunning = currentSession?.status === "IN_PROGRESS" || currentSession?.status === "RUNNING";

  const handleLaunchPresetFromDashboard = (presetQuery: string) => {
    setActiveQueryDraft(presetQuery);
    setCurrentTab("workspace");
  };

  return (
    <div className="flex h-screen bg-[#090d16] text-slate-100 font-sans overflow-hidden">
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        onNewSession={() => {
          setCurrentSession(null);
          setCurrentTab("workspace");
          setErrorMessage(null);
          setActiveQueryDraft("");
        }}
        health={health}
        historySessions={historySessions}
        activeSessionId={currentSession?.session_id}
        onSelectSession={handleSelectSession}
        onRemoveSession={handleRemoveSession}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6 scrollbar-thin scrollbar-thumb-slate-800">
        {/* Error notification banner */}
        {errorMessage && (
          <div className="p-4 rounded-2xl bg-rose-950/40 border border-rose-500/40 text-rose-200 text-sm flex items-center justify-between gap-4 max-w-6xl mx-auto shadow-lg">
            <div className="flex items-center gap-2.5">
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
                className="text-xs bg-rose-900/60 hover:bg-rose-900 text-rose-200 px-3 py-1.5 rounded-xl border border-rose-700/50 transition-colors font-medium"
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

        {/* Tab 0: Dashboard */}
        {currentTab === "dashboard" && (
          <DashboardView
            onStartNew={() => {
              setCurrentSession(null);
              setCurrentTab("workspace");
              setActiveQueryDraft("");
            }}
            onOpenWorkspace={() => setCurrentTab("workspace")}
            onSelectSession={handleSelectSession}
            historySessions={historySessions}
            indexedPapersCount={indexedPapers.length}
            reportsCount={savedReports.length}
            health={health}
            onSelectQueryPreset={handleLaunchPresetFromDashboard}
          />
        )}

        {/* Tab 1: Investigation Workspace */}
        {currentTab === "workspace" && (
          <div className="space-y-6 max-w-6xl mx-auto">
            {/* Top Formulation Input */}
            <ResearchQueryInput
              key={activeQueryDraft}
              initialQuery={activeQueryDraft}
              onSubmit={handleStartSession}
              isLoading={isLoading || isRunning}
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

            {/* Refused request: explain why instead of showing an empty run */}
            {(currentSession?.status === "OUT_OF_SCOPE" || currentSession?.status === "UNACCEPTABLE_RISK") && (
              <div
                role="alert"
                className={`rounded-2xl border p-5 text-sm leading-relaxed ${
                  currentSession.status === "OUT_OF_SCOPE"
                    ? "bg-amber-500/10 border-amber-500/30 text-amber-100"
                    : "bg-rose-500/10 border-rose-500/30 text-rose-100"
                }`}
              >
                <p className="font-semibold mb-1">
                  {currentSession.status === "OUT_OF_SCOPE" ? "Outside AIRA's research scope" : "Request refused for safety"}
                </p>
                <p>
                  {currentSession.message ||
                    "AIRA researches language technology: NLP, speech, machine translation, and Amharic and other low-resource languages."}
                </p>
              </div>
            )}

            {/* Execution Plan & Observable Logs */}
            {currentSession && (
              <ExecutionProgress
                plan={currentSession.plan}
                logs={currentSession.observable_logs}
                status={currentSession.status}
              />
            )}

            {/* Live event log streamed from the backend while the run executes */}
            {currentSession?.session_id && (
              <LiveEventLog
                key={currentSession.session_id}
                sessionId={currentSession.session_id}
                isRunning={isRunning}
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
          <ReportsView
            savedReports={savedReports}
            onRefresh={loadReportsList}
          />
        )}

        {/* Tab 3: Indexed Papers & RAG Repository */}
        {currentTab === "repository" && (
          <RepositoryView
            indexedPapers={indexedPapers}
            onRefresh={loadPapersList}
          />
        )}
      </main>
    </div>
  );
}

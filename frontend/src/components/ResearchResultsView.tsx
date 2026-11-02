"use client";

import React, { useState } from "react";
import {
  FileText,
  Download,
  CheckCircle,
  ExternalLink,
  Scale,
  Layers,
  AlertCircle,
  Copy,
  Check,
  Quote,
  ShieldCheck,
  FileCheck2,
} from "lucide-react";
import {
  ResearchComparisonReport,
  DiscoveredPaper,
  ValidatedPaper,
  MetricComparison,
  ResearchGap
} from "@/types/aira";
import { airaApi } from "@/services/api";

interface ResearchResultsViewProps {
  report?: ResearchComparisonReport | null;
  discoveredPapers?: DiscoveredPaper[];
  validatedPapers?: ValidatedPaper[];
  extractedPapers?: string[];
  metrics?: MetricComparison[];
  gaps?: ResearchGap[];
  citations?: Record<string, string>;
  isDegraded?: boolean;
}

export const ResearchResultsView: React.FC<ResearchResultsViewProps> = ({
  report,
  discoveredPapers = [],
  validatedPapers = [],
  extractedPapers = [],
  metrics = [],
  gaps = [],
  citations = {},
  isDegraded = false,
}) => {
  const [activeTab, setActiveTab] = useState<"report" | "papers" | "metrics" | "gaps" | "citations">("report");
  const [copiedCitation, setCopiedCitation] = useState<string | null>(null);

  const handleCopyCitation = async (text: string, id: string) => {
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
      }
      setCopiedCitation(id);
      setTimeout(() => setCopiedCitation(null), 2000);
    } catch {}
  };

  const handleDownload = (format: "markdown" | "json") => {
    if (!report?.report_id) return;
    const url = airaApi.getReportExportUrl(report.report_id, format);
    const link = document.createElement("a");
    link.href = url;
    link.download = `aira_report_${report.report_id}.${format === "markdown" ? "md" : "json"}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const tabs = [
    { id: "report", label: "Comparative Synthesis", icon: FileText, count: null },
    { id: "papers", label: "Corpus & Metadata", icon: Layers, count: discoveredPapers.length || extractedPapers.length },
    { id: "metrics", label: "Experimental Benchmarks", icon: Scale, count: metrics.length || (report?.metric_tables?.length || 0) },
    { id: "gaps", label: "Identified Research Gaps", icon: AlertCircle, count: gaps.length || (report?.research_gaps?.length || 0) },
    { id: "citations", label: "Verifiable Citations", icon: Quote, count: Object.keys(citations).length || (report?.references?.length || 0) },
  ] as const;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-3xl shadow-2xl overflow-hidden space-y-0">
      {/* Evidence Tier / Provenance Header */}
      <div className="px-6 py-3 bg-slate-950/80 border-b border-slate-800 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-3">
        <div className="flex flex-wrap items-center gap-4">
          <span className="font-bold text-slate-300 uppercase tracking-wider text-[10px] flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
            Provenance Standard:
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <strong className="text-emerald-300">Direct Extraction</strong> (Verbatim passage)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <strong className="text-cyan-300">Structured Comparison</strong> (Cross-methodology)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-indigo-400" />
            <strong className="text-indigo-300">Cross-Paper Inference</strong> (Extrapolated)
          </span>
        </div>

        {isDegraded && (
          <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 px-2 py-0.5 rounded-full font-mono text-[10px]">
            Degraded Deterministic Mode
          </span>
        )}
      </div>

      {/* Tabs Navigation */}
      <div className="flex border-b border-slate-800 bg-slate-900/60 px-6 pt-3 overflow-x-auto gap-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`pb-3 px-3 text-xs font-semibold transition-all border-b-2 flex items-center gap-2 whitespace-nowrap ${
                isActive
                  ? "border-indigo-500 text-white font-bold"
                  : "border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700"
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? "text-indigo-400" : "text-slate-500"}`} />
              <span>{tab.label}</span>
              {tab.count !== null && (
                <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded-full ${
                  isActive ? "bg-indigo-500/20 text-indigo-300" : "bg-slate-800 text-slate-400"
                }`}>
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab Panels */}
      <div className="p-6 sm:p-7">
        {/* 1. Final Report Tab */}
        {activeTab === "report" && (
          <div className="space-y-7">
            {/* Report Header */}
            <div className="flex flex-col sm:flex-row sm:items-start justify-between pb-5 border-b border-slate-800 gap-4">
              <div className="space-y-1.5 max-w-3xl">
                <div className="inline-flex items-center gap-1.5 text-[10px] font-mono uppercase font-semibold text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded-full">
                  <FileCheck2 className="w-3 h-3" /> Synthesis Document
                </div>
                <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight leading-snug">
                  {report?.title || (report?.research_question ? `Academic Synthesis: ${report.research_question}` : "Synthesized Investigation Findings")}
                </h3>
                <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400 font-mono">
                  <span>Report ID: {report?.report_id || "session_report"}</span>
                  <span>•</span>
                  <span>Analyzed Papers: {report?.papers_analyzed?.length || extractedPapers.length || 0}</span>
                  {report?.created_at && (
                    <>
                      <span>•</span>
                      <span>Generated: {new Date(report.created_at).toLocaleDateString()}</span>
                    </>
                  )}
                </div>
              </div>

              {report?.report_id && (
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => handleDownload("markdown")}
                    className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-2 px-3.5 rounded-xl border border-slate-700 transition-all shadow-sm"
                  >
                    <Download className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Markdown</span>
                  </button>
                  <button
                    onClick={() => handleDownload("json")}
                    className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-2 px-3.5 rounded-xl border border-slate-700 transition-all shadow-sm"
                  >
                    <Download className="w-3.5 h-3.5 text-cyan-400" />
                    <span>JSON</span>
                  </button>
                </div>
              )}
            </div>

            {report ? (
              <div className="space-y-6 text-sm leading-relaxed text-slate-300">
                {/* Executive Summary */}
                {report.executive_summary && (
                  <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 border border-indigo-500/20 shadow-md space-y-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 block">
                      Executive Summary
                    </span>
                    <p className="text-slate-100 leading-relaxed font-sans text-sm">
                      {report.executive_summary}
                    </p>
                  </div>
                )}

                {/* Objectives & Methodology */}
                {report.objectives_and_methodology_comparison && (
                  <div className="space-y-2.5">
                    <h4 className="text-sm font-bold text-cyan-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-cyan-400" />
                      Comparative Methodology & Objectives
                    </h4>
                    <div className="bg-slate-950/70 p-5 rounded-2xl border border-slate-800/80 whitespace-pre-line text-slate-300 leading-relaxed font-sans text-sm">
                      {report.objectives_and_methodology_comparison}
                    </div>
                  </div>
                )}

                {/* Dataset Comparison */}
                {report.dataset_comparison && (
                  <div className="space-y-2.5">
                    <h4 className="text-sm font-bold text-emerald-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      Dataset Benchmarks & Data Partitions
                    </h4>
                    <div className="bg-slate-950/70 p-5 rounded-2xl border border-slate-800/80 whitespace-pre-line text-slate-300 leading-relaxed font-sans text-sm">
                      {report.dataset_comparison}
                    </div>
                  </div>
                )}

                {/* Model Architecture */}
                {report.model_architecture_comparison && (
                  <div className="space-y-2.5">
                    <h4 className="text-sm font-bold text-indigo-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-indigo-400" />
                      Model & Neural Architectures
                    </h4>
                    <div className="bg-slate-950/70 p-5 rounded-2xl border border-slate-800/80 whitespace-pre-line text-slate-300 leading-relaxed font-sans text-sm">
                      {report.model_architecture_comparison}
                    </div>
                  </div>
                )}

                {/* Results Comparison */}
                {report.experimental_results_comparison && (
                  <div className="space-y-2.5">
                    <h4 className="text-sm font-bold text-cyan-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-cyan-400" />
                      Experimental Results & Benchmark Evaluation
                    </h4>
                    <div className="bg-slate-950/70 p-5 rounded-2xl border border-slate-800/80 whitespace-pre-line text-slate-300 leading-relaxed font-sans text-sm">
                      {report.experimental_results_comparison}
                    </div>
                  </div>
                )}

                {/* Strengths & Limitations */}
                {report.strengths_and_limitations && (
                  <div className="space-y-2.5">
                    <h4 className="text-sm font-bold text-amber-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-amber-400" />
                      Empirical Strengths & Methodological Limitations
                    </h4>
                    <div className="bg-slate-950/70 p-5 rounded-2xl border border-slate-800/80 whitespace-pre-line text-slate-300 leading-relaxed font-sans text-sm">
                      {report.strengths_and_limitations}
                    </div>
                  </div>
                )}

                {/* Evidence Backed Conclusions */}
                {report.evidence_backed_conclusions && (
                  <div className="space-y-2.5">
                    <h4 className="text-sm font-bold text-emerald-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      Evidence-Backed Synthesis & Conclusions
                    </h4>
                    <div className="bg-slate-950/70 p-5 rounded-2xl border border-slate-800/80 whitespace-pre-line text-slate-300 leading-relaxed font-sans text-sm">
                      {report.evidence_backed_conclusions}
                    </div>
                  </div>
                )}

                {/* Scope Boundaries */}
                {report.limitations_of_analysis && (
                  <div className="space-y-2.5">
                    <h4 className="text-sm font-bold text-amber-400 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-amber-400" />
                      Scope Boundaries & Limitations of the Analysis
                    </h4>
                    <div className="bg-amber-950/20 p-5 rounded-2xl border border-amber-800/40 whitespace-pre-line text-amber-200/90 text-xs leading-relaxed font-sans">
                      {report.limitations_of_analysis}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-16 bg-slate-950/40 rounded-2xl border border-slate-800/60 text-slate-500 space-y-2">
                <FileText className="w-8 h-8 mx-auto opacity-30 text-indigo-400" />
                <p className="text-sm font-medium text-slate-400">Synthesis in progress or awaiting tool completion.</p>
                <p className="text-xs text-slate-500">The agent is searching registries, extracting PDFs, and validating findings.</p>
              </div>
            )}
          </div>
        )}

        {/* 2. Papers & Corpus Tab */}
        {activeTab === "papers" && (
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white tracking-tight">
                  Academic Literature Corpus & Metadata Validation
                </h3>
                <p className="text-xs text-slate-400">
                  Papers retrieved from OpenAlex and Semantic Scholar, cross-verified against Crossref DOIs
                </p>
              </div>
            </div>

            {discoveredPapers.length === 0 && extractedPapers.length === 0 ? (
              <div className="text-center py-16 bg-slate-950/40 rounded-2xl border border-slate-800/60 text-slate-500">
                <Layers className="w-8 h-8 mx-auto mb-2 opacity-30 text-emerald-400" />
                <p className="text-sm text-slate-400">No academic papers discovered in this session yet.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-3.5">
                {(discoveredPapers.length > 0
                  ? discoveredPapers
                  : extractedPapers.map((pid): DiscoveredPaper => ({ paper_id: pid, title: pid }))
                ).map((paper, i) => {
                  const pid = paper.paper_id || paper.id || `paper_${i}`;
                  const val = validatedPapers.find((v) => v.paper_id === pid);

                  return (
                    <div key={i} className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3 shadow-sm hover:border-slate-700 transition-colors">
                      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                        <div className="space-y-1">
                          <h4 className="text-sm font-bold text-white tracking-tight leading-snug">
                            {paper.title}
                          </h4>
                          <p className="text-xs text-slate-400">
                            {paper.authors?.join(", ") || "Authors not listed"} • Year: {paper.year || "n.d."} {paper.venue ? `• ${paper.venue}` : ""}
                          </p>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          {val?.status === "VERIFIED" ? (
                            <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[10px] font-mono font-semibold px-2.5 py-0.5 rounded-full flex items-center gap-1">
                              <CheckCircle className="w-3 h-3" /> DOI VERIFIED
                            </span>
                          ) : val?.status === "CONFLICTING" ? (
                            <span className="bg-rose-500/10 text-rose-400 border border-rose-500/30 text-[10px] font-mono font-semibold px-2.5 py-0.5 rounded-full flex items-center gap-1">
                              <AlertCircle className="w-3 h-3" /> METADATA CONFLICT
                            </span>
                          ) : (
                            <span className="bg-slate-800 text-slate-400 text-[10px] font-mono px-2.5 py-0.5 rounded-full border border-slate-700">
                              {val?.status || "INDEXED"}
                            </span>
                          )}
                        </div>
                      </div>

                      {paper.abstract && (
                        <p className="text-xs text-slate-400 line-clamp-3 leading-relaxed border-t border-slate-900 pt-2.5">
                          {paper.abstract}
                        </p>
                      )}

                      <div className="flex flex-wrap items-center justify-between gap-3 pt-1 text-[11px] font-mono text-slate-500">
                        <span className="text-slate-400">ID: {pid}</span>
                        {paper.doi && (
                          <a
                            href={paper.doi.startsWith("http") ? paper.doi : `https://doi.org/${paper.doi}`}
                            target="_blank"
                            rel="noreferrer"
                            className="text-indigo-400 hover:text-indigo-300 hover:underline flex items-center gap-1"
                          >
                            <span>DOI: {paper.doi}</span>
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* 3. Metrics Comparison Tab */}
        {activeTab === "metrics" && (
          <div className="space-y-5">
            <div>
              <h3 className="text-sm font-bold text-white tracking-tight">
                Comparable Metric Ranking & Empirical Results
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Strict metric validation ensures only identical benchmark datasets, splits, and units are ranked together.
              </p>
            </div>

            {(metrics.length > 0 ? metrics : (report?.metric_tables || [])).length === 0 ? (
              <div className="text-center py-16 bg-slate-950/40 rounded-2xl border border-slate-800/60 text-slate-500">
                <Scale className="w-8 h-8 mx-auto mb-2 opacity-30 text-cyan-400" />
                <p className="text-sm text-slate-400">No empirical metrics extracted or compared yet.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {(metrics.length > 0 ? metrics : (report?.metric_tables || [])).map((m, idx) => {
                  const isCompatible = m.compatibility === "COMPATIBLE" || m.is_comparable === true;
                  const models = m.ranked_models && m.ranked_models.length > 0
                    ? m.ranked_models
                    : (m.summary_table || []);

                  return (
                    <div key={idx} className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4 shadow-sm">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
                        <div>
                          <div className="flex items-center gap-2">
                            <h4 className="text-sm font-bold text-white uppercase tracking-tight">
                              {m.metric_name}
                            </h4>
                            {m.goal && (
                              <span className="text-[10px] font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                                {m.goal.toLowerCase().replace(/_/g, " ")}
                              </span>
                            )}
                          </div>
                          {m.dataset && (
                            <p className="text-xs text-slate-400 mt-1">
                              Benchmark Dataset: <span className="font-mono text-indigo-300 font-semibold">{m.dataset}</span> {m.split ? `(${m.split})` : ""}
                            </p>
                          )}
                        </div>

                        <span className={`text-[10px] font-mono px-2.5 py-0.5 rounded-full border font-semibold self-start sm:self-auto ${
                          isCompatible
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                            : "bg-amber-500/10 text-amber-400 border-amber-500/30"
                        }`}>
                          {isCompatible ? "COMPATIBLE EVALUATION PROTOCOL" : "INCOMPATIBLE BENCHMARK"}
                        </span>
                      </div>

                      {!isCompatible && m.incompatibility_reason && (
                        <div className="p-3 rounded-xl bg-amber-950/20 border border-amber-800/40 text-xs text-amber-300 flex items-start gap-2">
                          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
                          <span>{m.incompatibility_reason}</span>
                        </div>
                      )}

                      {models.length > 0 ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                          {models.map((item, mIdx) => (
                            <div key={mIdx} className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-2">
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-xs font-bold text-white truncate">
                                  {item.model_name || item.paper_id}
                                </span>
                                {item.rank !== undefined && (
                                  <span className="text-[10px] font-mono bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded-full font-bold">
                                    Rank #{item.rank}
                                  </span>
                                )}
                              </div>

                              <div className="flex items-baseline justify-between pt-1">
                                <span className="text-lg font-extrabold text-emerald-400 font-mono">
                                  {String(item.normalized_value ?? item.raw_value)}
                                </span>
                                {item.page_number && (
                                  <span className="text-[10px] font-mono text-slate-500">
                                    Page {item.page_number}
                                  </span>
                                )}
                              </div>

                              <div className="text-[10px] text-slate-400 truncate border-t border-slate-900 pt-1.5">
                                {item.dataset_name ? `${item.dataset_name}${item.dataset_split ? ` (${item.dataset_split})` : ""}` : `Paper: ${item.paper_id}`}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : m.paper_values && Object.keys(m.paper_values).length > 0 ? (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                          {Object.entries(m.paper_values).map(([pid, val]: [string, unknown]) => (
                            <div key={pid} className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                              <span className="text-[10px] font-mono text-slate-400 block truncate">{pid}</span>
                              <span className="text-base font-bold text-slate-100 font-mono">{String(val)}</span>
                            </div>
                          ))}
                        </div>
                      ) : null}

                      {m.missing_data_warnings && m.missing_data_warnings.length > 0 && (
                        <div className="text-[10px] text-slate-500 italic pt-1">
                          {m.missing_data_warnings.join(" • ")}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* 4. Research Gaps Tab */}
        {activeTab === "gaps" && (
          <div className="space-y-5">
            <div>
              <h3 className="text-sm font-bold text-white tracking-tight">
                Empirical Research Gaps & Future Directions
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Synthesized from explicitly stated author limitations and cross-paper dataset shortcomings
              </p>
            </div>

            {(gaps.length > 0 ? gaps : (report?.research_gaps || [])).length === 0 ? (
              <div className="text-center py-16 bg-slate-950/40 rounded-2xl border border-slate-800/60 text-slate-500">
                <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-30 text-amber-400" />
                <p className="text-sm text-slate-400">No research gaps established yet.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {(gaps.length > 0 ? gaps : (report?.research_gaps || [])).map((g, idx) => {
                  const gapTypeLabel = g.gap_type === "AUTHOR_ACKNOWLEDGED_GAP"
                    ? "Author-Acknowledged Limitation"
                    : g.gap_type === "CROSS_PAPER_GAP"
                    ? "Cross-Paper Empirical Gap"
                    : g.gap_type === "POTENTIAL_RESEARCH_DIRECTION"
                    ? "Future Direction"
                    : g.category || "Open Empirical Problem";

                  const gapBadgeClass = g.gap_type === "AUTHOR_ACKNOWLEDGED_GAP"
                    ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                    : g.gap_type === "CROSS_PAPER_GAP"
                    ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/30"
                    : "bg-indigo-500/10 text-indigo-400 border-indigo-500/30";

                  return (
                    <div key={idx} className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3.5 shadow-sm">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border uppercase tracking-wider ${gapBadgeClass}`}>
                          {gapTypeLabel}
                        </span>

                        {g.supporting_papers && g.supporting_papers.length > 0 && (
                          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
                            <span>Corpus:</span>
                            {g.supporting_papers.map((p) => (
                              <span key={p} className="bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-slate-300">
                                {p}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {g.title && (
                        <h4 className="text-sm font-bold text-white tracking-tight">
                          {g.title}
                        </h4>
                      )}

                      <p className="text-xs text-slate-200 leading-relaxed font-sans">
                        {g.description}
                      </p>

                      {/* Direct Verbatim Evidence Passages */}
                      {g.evidence && g.evidence.length > 0 && (
                        <div className="space-y-2 pt-1">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
                            Direct Passage Evidence:
                          </span>
                          <div className="space-y-2">
                            {g.evidence.map((ev, eIdx) => (
                              <div key={eIdx} className="bg-slate-900/80 p-3 rounded-xl border border-slate-800 text-xs space-y-1.5">
                                <p className="italic text-slate-300 font-serif leading-relaxed">
                                  &ldquo;{ev.passage}&rdquo;
                                </p>
                                <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400 pt-0.5 border-t border-slate-800/60">
                                  <span className="text-emerald-400 font-semibold">{ev.paper_id}</span>
                                  {ev.page_number && <span>• Page {ev.page_number}</span>}
                                  {ev.section_heading && <span className="truncate">• {ev.section_heading}</span>}
                                  <span className="ml-auto text-[9px] bg-slate-950 px-2 py-0.5 rounded border border-slate-800 text-slate-400">
                                    {ev.evidence_type}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {(g.suggested_direction || g.suggested_future_work) && (
                        <div className="text-xs text-cyan-300 bg-cyan-950/20 border border-cyan-800/30 p-3 rounded-xl">
                          <strong className="text-cyan-200">Recommended Research Direction:</strong> {g.suggested_direction || g.suggested_future_work}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* 5. Citations Tab */}
        {activeTab === "citations" && (
          <div className="space-y-5">
            <div>
              <h3 className="text-sm font-bold text-white tracking-tight">
                Traceable Academic Citations & Bibliography
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Standards-compliant APA 7, IEEE, and BibTeX references formatted directly from verified metadata
              </p>
            </div>

            {Object.keys(citations).length === 0 && (!report?.references || report.references.length === 0) ? (
              <div className="text-center py-16 bg-slate-950/40 rounded-2xl border border-slate-800/60 text-slate-500">
                <Quote className="w-8 h-8 mx-auto mb-2 opacity-30 text-purple-400" />
                <p className="text-sm text-slate-400">No citations compiled for this session yet.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {Object.entries(citations).map(([pid, citText]) => (
                  <div key={pid} className="p-4 rounded-2xl bg-slate-950 border border-slate-800 flex items-start justify-between gap-4 hover:border-slate-700 transition-colors">
                    <div className="space-y-1">
                      <span className="text-[10px] font-mono text-indigo-400 font-bold block">{pid}</span>
                      <p className="text-xs text-slate-200 font-serif leading-relaxed">{citText}</p>
                    </div>

                    <button
                      onClick={() => handleCopyCitation(citText, pid)}
                      className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-all shrink-0 border border-slate-800"
                      title="Copy Citation"
                    >
                      {copiedCitation === pid ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

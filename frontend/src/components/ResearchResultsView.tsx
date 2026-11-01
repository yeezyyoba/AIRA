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
  Check
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
    } catch {
      // Fallback
    }
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

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl shadow-xl overflow-hidden">
      {/* Evidence Tier Legend Bar */}
      <div className="px-6 py-2.5 bg-slate-950/70 border-b border-slate-800 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-3">
        <div className="flex items-center gap-4">
          <span className="font-semibold text-slate-300 uppercase tracking-wider">Provenance Legend:</span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <strong className="text-emerald-300">Direct Extraction</strong> (Verbatim/OCR)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <strong className="text-cyan-300">LLM Synthesis</strong> (Structured comparison)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-indigo-400" />
            <strong className="text-indigo-300">Cross-Paper Inference</strong> (Extrapolated)
          </span>
        </div>

        {isDegraded && (
          <span className="bg-amber-500/10 text-amber-400 border border-amber-500/20 px-2 py-0.5 rounded font-mono text-[10px]">
            Degraded Deterministic Mode
          </span>
        )}
      </div>

      {/* Navigation Tabs */}
      <div className="flex border-b border-slate-800 bg-slate-900/60 px-6 pt-3 overflow-x-auto gap-4">
        <button
          onClick={() => setActiveTab("report")}
          className={`pb-3 text-sm font-semibold transition-colors border-b-2 flex items-center gap-2 ${
            activeTab === "report"
              ? "border-indigo-500 text-white"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <FileText className="w-4 h-4 text-indigo-400" />
          Comparative Report
        </button>

        <button
          onClick={() => setActiveTab("papers")}
          className={`pb-3 text-sm font-semibold transition-colors border-b-2 flex items-center gap-2 ${
            activeTab === "papers"
              ? "border-indigo-500 text-white"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Layers className="w-4 h-4 text-emerald-400" />
          Corpus & Metadata ({discoveredPapers.length || extractedPapers.length})
        </button>

        <button
          onClick={() => setActiveTab("metrics")}
          className={`pb-3 text-sm font-semibold transition-colors border-b-2 flex items-center gap-2 ${
            activeTab === "metrics"
              ? "border-indigo-500 text-white"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Scale className="w-4 h-4 text-cyan-400" />
          Metric Comparisons ({metrics.length})
        </button>

        <button
          onClick={() => setActiveTab("gaps")}
          className={`pb-3 text-sm font-semibold transition-colors border-b-2 flex items-center gap-2 ${
            activeTab === "gaps"
              ? "border-indigo-500 text-white"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <AlertCircle className="w-4 h-4 text-amber-400" />
          Research Gaps ({gaps.length || (report?.research_gaps?.length || 0)})
        </button>

        <button
          onClick={() => setActiveTab("citations")}
          className={`pb-3 text-sm font-semibold transition-colors border-b-2 flex items-center gap-2 ${
            activeTab === "citations"
              ? "border-indigo-500 text-white"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <CheckCircle className="w-4 h-4 text-purple-400" />
          Traceable Citations
        </button>
      </div>

      {/* Tab Contents */}
      <div className="p-6">
        {/* 1. Final Report Tab */}
        {activeTab === "report" && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
              <div>
                <h3 className="text-lg font-bold text-white tracking-tight">
                  {report?.title || `Academic Comparison: ${report?.research_question || "Investigation"}`}
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Report ID: <span className="font-mono text-slate-300">{report?.report_id || "pending"}</span> • Analyzed Papers: {report?.papers_analyzed?.length || extractedPapers.length}
                </p>
              </div>

              {report?.report_id && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleDownload("markdown")}
                    className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-1.5 px-3 rounded-lg border border-slate-700 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" /> Markdown
                  </button>
                  <button
                    onClick={() => handleDownload("json")}
                    className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-1.5 px-3 rounded-lg border border-slate-700 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" /> JSON
                  </button>
                </div>
              )}
            </div>

            {report ? (
              <div className="space-y-6 text-sm leading-relaxed text-slate-300">
                {/* Executive Summary */}
                {report.executive_summary && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                      Executive Summary
                    </h4>
                    <p className="text-slate-200">{report.executive_summary}</p>
                  </div>
                )}

                {/* Objectives & Methodology */}
                {report.objectives_and_methodology_comparison && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold text-cyan-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-cyan-400" />
                      Objectives & Methodology Analysis
                    </h4>
                    <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800/80 whitespace-pre-line text-slate-300">
                      {report.objectives_and_methodology_comparison}
                    </div>
                  </div>
                )}

                {/* Dataset Comparison */}
                {report.dataset_comparison && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold text-emerald-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      Dataset Benchmarks & Splits
                    </h4>
                    <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800/80 whitespace-pre-line text-slate-300">
                      {report.dataset_comparison}
                    </div>
                  </div>
                )}

                {/* Model Architecture */}
                {report.model_architecture_comparison && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold text-indigo-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-indigo-400" />
                      Model & Neural Architectures
                    </h4>
                    <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800/80 whitespace-pre-line text-slate-300">
                      {report.model_architecture_comparison}
                    </div>
                  </div>
                )}

                {/* Results Comparison */}
                {report.experimental_results_comparison && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold text-cyan-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-cyan-400" />
                      Experimental Results & Evaluation
                    </h4>
                    <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800/80 whitespace-pre-line text-slate-300">
                      {report.experimental_results_comparison}
                    </div>
                  </div>
                )}

                {/* Strengths & Limitations */}
                {report.strengths_and_limitations && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold text-amber-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-amber-400" />
                      Empirical Strengths & Methodological Limitations
                    </h4>
                    <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800/80 whitespace-pre-line text-slate-300">
                      {report.strengths_and_limitations}
                    </div>
                  </div>
                )}

                {/* Evidence Backed Conclusions */}
                {report.evidence_backed_conclusions && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold text-white flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      Evidence-Backed Conclusions
                    </h4>
                    <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800/80 whitespace-pre-line text-slate-300">
                      {report.evidence_backed_conclusions}
                    </div>
                  </div>
                )}

                {/* Limitations of Analysis & Scope Boundaries */}
                {report.limitations_of_analysis && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold text-amber-400 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-amber-400" />
                      Scope Boundaries & Limitations of the Analysis
                    </h4>
                    <div className="bg-amber-950/20 p-4 rounded-xl border border-amber-800/40 whitespace-pre-line text-amber-200/90 text-xs leading-relaxed">
                      {report.limitations_of_analysis}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-slate-500">
                <FileText className="w-8 h-8 mx-auto mb-2 opacity-40" />
                <p>Report synthesis in progress or awaiting tool completion.</p>
              </div>
            )}
          </div>
        )}

        {/* 2. Papers & Corpus Tab */}
        {activeTab === "papers" && (
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-white tracking-tight">
              Discovered and Verified Academic Corpus
            </h3>

            {discoveredPapers.length === 0 && extractedPapers.length === 0 ? (
              <p className="text-xs text-slate-500 italic py-6 text-center">No academic papers discovered yet.</p>
            ) : (
              <div className="grid grid-cols-1 gap-3">
                {(discoveredPapers.length > 0
                  ? discoveredPapers
                  : extractedPapers.map((pid): DiscoveredPaper => ({ paper_id: pid, title: pid }))
                ).map((paper, i) => {
                  const pid = paper.paper_id || paper.id || `paper_${i}`;
                  const val = validatedPapers.find(v => v.paper_id === pid);

                  return (
                    <div key={i} className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <h4 className="text-sm font-semibold text-white">{paper.title}</h4>
                          <p className="text-xs text-slate-400 mt-0.5">
                            Authors: {paper.authors?.join(", ") || "Unknown"} • Year: {paper.year || "n.d."} {paper.venue ? `• ${paper.venue}` : ""}
                          </p>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          {val?.status === "VERIFIED" ? (
                            <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] font-mono font-semibold px-2 py-0.5 rounded">
                              DOI VERIFIED
                            </span>
                          ) : val?.status === "CONFLICTING" ? (
                            <span className="bg-rose-500/10 text-rose-400 border border-rose-500/20 text-[10px] font-mono font-semibold px-2 py-0.5 rounded">
                              METADATA CONFLICT
                            </span>
                          ) : (
                            <span className="bg-slate-800 text-slate-400 text-[10px] font-mono px-2 py-0.5 rounded">
                              {val?.status || "INDEXED"}
                            </span>
                          )}
                        </div>
                      </div>

                      {paper.abstract && (
                        <p className="text-xs text-slate-400 line-clamp-3 leading-relaxed border-t border-slate-900 pt-2">
                          {paper.abstract}
                        </p>
                      )}

                      <div className="flex items-center gap-4 pt-1 text-[11px] text-slate-500 font-mono">
                        <span>ID: {pid}</span>
                        {paper.doi && (
                          <a
                            href={paper.doi.startsWith("http") ? paper.doi : `https://doi.org/${paper.doi}`}
                            target="_blank"
                            rel="noreferrer"
                            className="text-cyan-400 hover:underline flex items-center gap-1"
                          >
                            DOI: {paper.doi} <ExternalLink className="w-3 h-3" />
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
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-white tracking-tight">
              Standardized Experimental Result Tables
            </h3>
            <p className="text-xs text-slate-400">
              Only metrics sharing identical benchmark datasets, evaluation protocols, and scale are ranked. Incomparable results are explicitly segregated.
            </p>

            {(metrics.length > 0 ? metrics : (report?.metric_tables || [])).length === 0 ? (
              <p className="text-xs text-slate-500 italic py-6 text-center">No metric comparisons extracted yet.</p>
            ) : (
              <div className="space-y-4">
                {(metrics.length > 0 ? metrics : (report?.metric_tables || [])).map((m, idx) => {
                  const isCompatible = m.compatibility === "COMPATIBLE" || m.is_comparable === true;
                  const models = m.ranked_models && m.ranked_models.length > 0 
                    ? m.ranked_models 
                    : (m.summary_table || []);

                  return (
                    <div key={idx} className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <h4 className="text-sm font-semibold text-white capitalize flex items-center gap-2">
                            {m.metric_name}
                            {m.goal && (
                              <span className="text-[10px] font-mono text-slate-400 font-normal">
                                ({m.goal.toLowerCase().replace(/_/g, " ")})
                              </span>
                            )}
                          </h4>
                          {m.dataset && (
                            <p className="text-xs text-slate-400 mt-0.5">
                              Dataset: <span className="font-mono text-slate-300">{m.dataset}</span> {m.split ? `• Split: ${m.split}` : ""}
                            </p>
                          )}
                        </div>

                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold shrink-0 ${
                          isCompatible 
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                            : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        }`}>
                          {isCompatible ? "COMPATIBLE BENCHMARK" : "INCOMPATIBLE COMPARISON"}
                        </span>
                      </div>

                      {/* Incompatibility explanation if segregated */}
                      {!isCompatible && m.incompatibility_reason && (
                        <div className="p-2.5 rounded-lg bg-amber-950/20 border border-amber-800/40 text-[11px] text-amber-300 flex items-start gap-2">
                          <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5 text-amber-400" />
                          <span>{m.incompatibility_reason}</span>
                        </div>
                      )}

                      {/* Measurements table / cards */}
                      {models.length > 0 ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5 pt-1">
                          {models.map((item, mIdx) => (
                            <div key={mIdx} className="bg-slate-900/90 p-3 rounded-lg border border-slate-800 space-y-1">
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-xs font-semibold text-white truncate">
                                  {item.model_name || item.paper_id}
                                </span>
                                {item.rank !== undefined && (
                                  <span className="text-[10px] font-mono bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded">
                                    Rank #{item.rank}
                                  </span>
                                )}
                              </div>

                              <div className="flex items-baseline justify-between pt-1">
                                <span className="text-base font-bold text-emerald-400 font-mono">
                                  {String(item.normalized_value ?? item.raw_value)}
                                </span>
                                {item.page_number && (
                                  <span className="text-[10px] font-mono text-slate-500">
                                    PDF p. {item.page_number}
                                  </span>
                                )}
                              </div>

                              <div className="text-[10px] text-slate-400 truncate">
                                {item.dataset_name ? `${item.dataset_name}${item.dataset_split ? ` (${item.dataset_split})` : ""}` : `Paper: ${item.paper_id}`}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : m.paper_values && Object.keys(m.paper_values).length > 0 ? (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                          {Object.entries(m.paper_values).map(([pid, val]: [string, any]) => (
                            <div key={pid} className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                              <span className="text-[10px] font-mono text-slate-400 block truncate">{pid}</span>
                              <span className="text-sm font-bold text-slate-100">{String(val)}</span>
                            </div>
                          ))}
                        </div>
                      ) : null}

                      {/* Missing data warnings */}
                      {m.missing_data_warnings && m.missing_data_warnings.length > 0 && (
                        <div className="text-[10px] text-slate-400 italic pt-1">
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
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-white tracking-tight">
              Evidence-Supported Research Gaps
            </h3>
            <p className="text-xs text-slate-400">
              Identified research gaps derived directly from author limitations and cross-paper empirical analysis.
            </p>

            {(gaps.length > 0 ? gaps : (report?.research_gaps || [])).length === 0 ? (
              <p className="text-xs text-slate-500 italic py-6 text-center">No research gaps established yet.</p>
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
                    ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                    : g.gap_type === "CROSS_PAPER_GAP"
                    ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
                    : "bg-indigo-500/10 text-indigo-400 border-indigo-500/20";

                  return (
                    <div key={idx} className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border uppercase tracking-wider ${gapBadgeClass}`}>
                          {gapTypeLabel}
                        </span>
                        {g.supporting_papers && g.supporting_papers.length > 0 && (
                          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
                            <span>Corpus:</span>
                            {g.supporting_papers.map(p => (
                              <span key={p} className="bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-slate-300">
                                {p}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {g.title && (
                        <h4 className="text-sm font-semibold text-white">
                          {g.title}
                        </h4>
                      )}

                      <p className="text-xs text-slate-200 leading-relaxed">
                        {g.description}
                      </p>

                      {/* Verbatim Evidence Citations */}
                      {g.evidence && g.evidence.length > 0 && (
                        <div className="space-y-1.5 pt-1">
                          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block">
                            Direct Passage Evidence:
                          </span>
                          <div className="space-y-1.5">
                            {g.evidence.map((ev, eIdx) => (
                              <div key={eIdx} className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800 text-xs space-y-1">
                                <p className="italic text-slate-300">"{ev.passage}"</p>
                                <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400 pt-0.5">
                                  <span className="text-emerald-400">
                                    {ev.paper_id}
                                  </span>
                                  {ev.page_number && (
                                    <span>• Page {ev.page_number}</span>
                                  )}
                                  {ev.section_heading && (
                                    <span className="truncate">• {ev.section_heading}</span>
                                  )}
                                  <span className={`ml-auto px-1.5 py-0.5 rounded text-[9px] ${
                                    ev.evidence_type === "DIRECT_EXTRACT"
                                      ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                      : ev.evidence_type === "SUMMARY"
                                      ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                                      : "bg-indigo-500/10 text-indigo-400 border border-indigo-500/20"
                                  }`}>
                                    {ev.evidence_type}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {(g.suggested_direction || g.suggested_future_work) && (
                        <div className="text-[11px] text-cyan-300 bg-cyan-950/20 border border-cyan-800/30 p-2.5 rounded-lg">
                          <strong>Proposed Direction:</strong> {g.suggested_direction || g.suggested_future_work}
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
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-white tracking-tight">
              Traceable Academic References
            </h3>
            <p className="text-xs text-slate-400">
              Deterministic citations formatted in APA 7, IEEE, and BibTeX without hallucination.
            </p>

            {Object.keys(citations).length === 0 && (!report?.references || report.references.length === 0) ? (
              <p className="text-xs text-slate-500 italic py-6 text-center">No citations generated yet.</p>
            ) : (
              <div className="space-y-3">
                {Object.entries(citations).map(([pid, citText]) => (
                  <div key={pid} className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <span className="text-[10px] font-mono text-indigo-400 font-semibold">{pid}</span>
                      <p className="text-xs text-slate-200 font-serif leading-relaxed">{citText}</p>
                    </div>

                    <button
                      onClick={() => handleCopyCitation(citText, pid)}
                      className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors shrink-0"
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

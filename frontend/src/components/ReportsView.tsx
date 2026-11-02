
"use client";

import React, { useState } from "react";
import {
  FileText,
  RefreshCw,
  Download,
  Search,
  Calendar,
} from "lucide-react";
import { airaApi } from "@/services/api";

export interface ReportSummary {
  report_id: string;
  research_question?: string;
  title?: string;
  papers_analyzed?: string[];
  created_at?: string;
}

interface ReportsViewProps {
  savedReports: ReportSummary[];
  onRefresh: () => void;
}

export const ReportsView: React.FC<ReportsViewProps> = ({
  savedReports,
  onRefresh,
}) => {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredReports = savedReports.filter((report) => {
    const question = (
      report.research_question || report.title || ""
    ).toLowerCase();

    const paperIds = (report.papers_analyzed || [])
      .join(" ")
      .toLowerCase();

    const reportId = report.report_id.toLowerCase();
    const query = searchQuery.toLowerCase();

    return (
      question.includes(query) ||
      paperIds.includes(query) ||
      reportId.includes(query)
    );
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto py-2">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800/80 gap-3">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/25 flex items-center justify-center">
              <FileText className="w-4 h-4 text-indigo-400" />
            </div>

            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Comparative Research Reports Archive
              </h2>
              <p className="text-xs text-slate-400">
                Published multi-paper comparative synthesis documents and
                structured evaluations
              </p>
            </div>
          </div>
        </div>

        <button
          onClick={onRefresh}
          className="flex items-center gap-2 text-xs text-slate-300 hover:text-white px-3.5 py-2 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 transition-all shadow-sm self-start sm:self-auto"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Reports</span>
        </button>
      </div>

      {/* Search & Statistics */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-slate-900/60 p-3 rounded-2xl border border-slate-800/80">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />

          <input
            type="text"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search reports by topic, research question, or paper ID..."
            className="w-full bg-slate-950 border border-slate-800/80 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none focus:border-indigo-500"
          />
        </div>

        <div className="text-xs font-mono text-slate-400 px-2 shrink-0">
          Showing {filteredReports.length} of {savedReports.length} reports
        </div>
      </div>

      {/* Reports List */}
      {filteredReports.length === 0 ? (
        <div className="text-center py-16 bg-slate-900/40 rounded-3xl border border-slate-800/80 text-slate-500 space-y-3">
          <FileText className="w-10 h-10 mx-auto opacity-30 text-indigo-400" />

          <div className="space-y-1">
            <p className="text-sm font-medium text-slate-300">
              No comparative reports match your filter.
            </p>

            <p className="text-xs text-slate-500">
              {savedReports.length === 0
                ? "Run an investigation in the workspace to automatically generate and archive reports."
                : "Try adjusting your search query."}
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3.5">
          {filteredReports.map((report) => {
            const papers = report.papers_analyzed || [];

            const dateString = report.created_at
              ? new Date(report.created_at).toLocaleDateString()
              : "Recent";

            return (
              <div
                key={report.report_id}
                className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-sm"
              >
                <div className="space-y-2 min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] font-mono text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded-full font-semibold">
                      {report.report_id}
                    </span>

                    <span className="text-xs text-slate-400 flex items-center gap-1 font-mono">
                      <Calendar className="w-3 h-3 text-slate-500" />
                      {dateString}
                    </span>
                  </div>

                  <h3 className="text-sm font-bold text-white tracking-tight leading-snug">
                    {report.research_question ||
                      report.title ||
                      "Academic Synthesis Report"}
                  </h3>

                  {papers.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 pt-1">
                      <span className="text-[10px] text-slate-500 font-mono">
                        Analyzed Corpus:
                      </span>

                      {papers.map((paperId) => (
                        <span
                          key={paperId}
                          className="text-[10px] font-mono bg-slate-950 border border-slate-800 text-slate-300 px-2 py-0.5 rounded-lg"
                        >
                          {paperId}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-2 shrink-0 pt-2 md:pt-0 border-t md:border-t-0 border-slate-800/60">
                  <a
                    href={airaApi.getReportExportUrl(
                      report.report_id,
                      "markdown"
                    )}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3.5 py-2 rounded-xl border border-slate-700 font-semibold transition-all flex items-center gap-1.5 shadow-sm"
                  >
                    <Download className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Markdown</span>
                  </a>

                  <a
                    href={airaApi.getReportExportUrl(
                      report.report_id,
                      "json"
                    )}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3.5 py-2 rounded-xl border border-slate-700 font-semibold transition-all flex items-center gap-1.5 shadow-sm"
                  >
                    <Download className="w-3.5 h-3.5 text-cyan-400" />
                    <span>JSON</span>
                  </a>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

"use client";

import React, { useState } from "react";
import { 
  Layers, 
  RefreshCw, 
  Search, 
  Database, 
  CheckCircle, 
} from "lucide-react";

interface RepositoryViewProps {
  indexedPapers: string[];
  onRefresh: () => void;
}

export const RepositoryView: React.FC<RepositoryViewProps> = ({
  indexedPapers,
  onRefresh,
}) => {
  const [searchQuery, setSearchQuery] = useState("");

  const filtered = indexedPapers.filter((paperId) =>
    paperId.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6 max-w-6xl mx-auto py-2">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800/80 gap-3">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center">
              <Layers className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Indexed Knowledge Base & RAG Corpus
              </h2>
              <p className="text-xs text-slate-400">
                Extracted academic literature and verified semantic passages stored in ChromaDB
              </p>
            </div>
          </div>
        </div>

        <button
          onClick={onRefresh}
          className="flex items-center gap-2 text-xs text-slate-300 hover:text-white px-3.5 py-2 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 transition-all shadow-sm self-start sm:self-auto"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Knowledge Base</span>
        </button>
      </div>

      {/* Search & Stats */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-slate-900/60 p-3 rounded-2xl border border-slate-800/80">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search indexed literature by paper ID or title..."
            className="w-full bg-slate-950 border border-slate-800/80 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none focus:border-indigo-500"
          />
        </div>

        <div className="text-xs font-mono text-slate-400 px-2 shrink-0">
          {filtered.length} indexed documents
        </div>
      </div>

      {/* Grid of Indexed Papers */}
      {filtered.length === 0 ? (
        <div className="text-center py-16 bg-slate-900/40 rounded-3xl border border-slate-800/80 text-slate-500 space-y-3">
          <Database className="w-10 h-10 mx-auto opacity-30 text-emerald-400" />
          <div className="space-y-1">
            <p className="text-sm font-medium text-slate-300">No papers indexed matching your filter.</p>
            <p className="text-xs text-slate-500">
              {indexedPapers.length === 0
                ? "Run an investigation in the workspace to discover and index papers automatically."
                : "Try adjusting your search criteria."}
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {filtered.map((paperId, idx) => {
            const title = paperId.replace(/_/g, " ");

            return (
              <div
                key={idx}
                className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between space-y-3 shadow-sm"
              >
                <div className="space-y-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[10px] font-mono font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> EXTRACTED & EMBEDDED
                    </span>
                    <span className="text-[10px] font-mono text-slate-500 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                      PyMuPDF + Chroma
                    </span>
                  </div>

                  <h3 className="text-sm font-bold text-white capitalize tracking-tight pt-1">
                    {title}
                  </h3>
                </div>

                <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span className="text-slate-500 truncate max-w-xs">ID: {paperId}</span>
                  <span className="text-indigo-400 text-[10px]">Dense Chunks Active</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

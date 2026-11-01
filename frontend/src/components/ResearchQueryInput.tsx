"use client";

import React, { useState } from "react";
import { Search, Loader2, Sparkles, HelpCircle } from "lucide-react";

interface ResearchQueryInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
}

export const ResearchQueryInput: React.FC<ResearchQueryInputProps> = ({
  onSubmit,
  isLoading,
}) => {
  const [query, setQuery] = useState("");

  const presets = [
    "Compare recent approaches to Amharic sentiment analysis and identify limitations in datasets and models",
    "Evaluate Conformer acoustic architecture against Transformer baselines for speech recognition on LibriSpeech",
    "Low-resource named entity recognition techniques for Ethiopian languages",
    "Analysis of morphological tokenization challenges in Ethiopic script language modeling",
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    onSubmit(query.trim());
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
      {/* Decorative background glow */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />

      <div className="flex items-center gap-2 mb-3">
        <Sparkles className="w-5 h-5 text-cyan-400" />
        <h2 className="text-base font-semibold text-white tracking-tight">
          Formulate Research Objective
        </h2>
      </div>

      <p className="text-xs text-slate-400 mb-4 max-w-2xl leading-relaxed">
        Submit a research inquiry. AIRA will parse objectives, plan execution, query academic registries, cross-validate DOIs, extract empirical findings, and synthesize verifiable reports.
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
            placeholder="e.g. Compare recent approaches to Amharic sentiment analysis and evaluate benchmark datasets..."
            rows={3}
            className="w-full bg-slate-950/80 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none resize-none transition-all disabled:opacity-50"
          />
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-1">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <HelpCircle className="w-4 h-4 text-slate-500" />
            <span>Try sample academic queries below</span>
          </div>

          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-medium text-sm py-2 px-6 rounded-xl shadow-lg shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-[0.98]"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Orchestrating Workflow...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Investigate
              </>
            )}
          </button>
        </div>
      </form>

      {/* Preset pills */}
      <div className="mt-5 pt-4 border-t border-slate-800/80">
        <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block mb-2">
          Suggested Academic Benchmarks
        </span>
        <div className="flex flex-wrap gap-2">
          {presets.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              disabled={isLoading}
              onClick={() => setQuery(preset)}
              className="text-left text-xs bg-slate-800/60 hover:bg-slate-800 hover:text-slate-200 border border-slate-700/60 text-slate-300 px-3 py-1.5 rounded-lg transition-colors truncate max-w-md active:scale-[0.99] disabled:opacity-50"
            >
              {preset}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

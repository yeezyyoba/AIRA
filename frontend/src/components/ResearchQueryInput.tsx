"use client";

import React, { useState } from "react";
import { Search, Loader2, BookOpen, ArrowRight, Lightbulb, Compass } from "lucide-react";

interface ResearchQueryInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  initialQuery?: string;
}

export const ResearchQueryInput: React.FC<ResearchQueryInputProps> = ({
  onSubmit,
  isLoading,
  initialQuery = "",
}) => {
  const [query, setQuery] = useState(initialQuery);

  const presets = [
    {
      label: "Amharic Sentiment Analysis",
      query: "Compare recent approaches to Amharic sentiment analysis and identify limitations in datasets and models",
    },
    {
      label: "Conformer vs Transformer ASR",
      query: "Evaluate Conformer acoustic architecture against Transformer baselines for speech recognition on LibriSpeech",
    },
    {
      label: "Low-Resource NER",
      query: "Low-resource named entity recognition techniques for Ethiopian languages",
    },
    {
      label: "Morphological Tokenization",
      query: "Analysis of morphological tokenization challenges in Ethiopic script language modeling",
    },
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    onSubmit(query.trim());
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-7 shadow-xl relative overflow-hidden">
      {/* Decorative background glow */}
      <div className="absolute top-0 right-0 w-80 h-80 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <Compass className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">
              Formulate Research Investigation
            </h2>
            <p className="text-xs text-slate-400">
              Agentic planning, multi-registry search, and verifiable comparative synthesis
            </p>
          </div>
        </div>

        <span className="text-[10px] font-mono font-medium text-slate-400 bg-slate-950 px-2.5 py-1 rounded-full border border-slate-800 self-start sm:self-auto">
          Multi-Paper Evidence Protocol
        </span>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 mt-4">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
            placeholder="State your academic research question, domain, or comparative objectives (e.g. Compare recent approaches to Amharic sentiment analysis and evaluate benchmark datasets)..."
            rows={3}
            className="w-full bg-slate-950/90 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-2xl px-4 py-3.5 text-sm text-slate-100 placeholder-slate-500 outline-none resize-none transition-all disabled:opacity-50 leading-relaxed font-sans"
          />
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-1">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Lightbulb className="w-4 h-4 text-amber-400/80 shrink-0" />
            <span className="text-[11px] leading-tight">
              AIRA validates DOIs, cross-examines methodologies, and cites verbatim excerpts.
            </span>
          </div>

          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs py-2.5 px-6 rounded-xl shadow-lg shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-[0.98] shrink-0"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Orchestrating Workflow...</span>
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                <span>Start Investigation</span>
                <ArrowRight className="w-3.5 h-3.5 opacity-70" />
              </>
            )}
          </button>
        </div>
      </form>

      {/* Preset pills */}
      <div className="mt-5 pt-4 border-t border-slate-800/80">
        <div className="flex items-center gap-2 mb-2">
          <BookOpen className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Benchmark Queries
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {presets.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              disabled={isLoading}
              onClick={() => setQuery(preset.query)}
              className="text-xs bg-slate-950 hover:bg-slate-800/80 hover:text-indigo-300 border border-slate-800 hover:border-indigo-500/30 text-slate-300 px-3 py-1.5 rounded-xl transition-all active:scale-[0.99] disabled:opacity-50"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

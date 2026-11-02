"use client";

import React, { useState } from "react";
import { HelpCircle, ArrowRight, Loader2, AlertCircle } from "lucide-react";

interface ClarificationBannerProps {
  question: string;
  missingInformation?: string[] | null;
  onResume: (response: string) => void;
  isLoading: boolean;
}

export const ClarificationBanner: React.FC<ClarificationBannerProps> = ({
  question,
  missingInformation,
  onResume,
  isLoading,
}) => {
  const [answer, setAnswer] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!answer.trim() || isLoading) return;
    onResume(answer.trim());
  };

  return (
    <div className="bg-gradient-to-r from-amber-950/40 via-amber-900/20 to-slate-900 border border-amber-500/40 rounded-3xl p-6 sm:p-7 shadow-2xl relative overflow-hidden">
      {/* Soft background glow */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-amber-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10 flex flex-col md:flex-row md:items-start gap-5">
        <div className="w-12 h-12 rounded-2xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center shrink-0 shadow-inner">
          <HelpCircle className="w-6 h-6 text-amber-400" />
        </div>

        <div className="flex-1 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <span className="text-[10px] font-mono font-semibold tracking-wider uppercase text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2.5 py-0.5 rounded-full">
                Interactive Agent Clarification
              </span>
              <h3 className="text-base font-bold text-white tracking-tight mt-1">
                The agent requests additional context to refine execution
              </h3>
            </div>

            <span className="text-[10px] font-mono text-amber-300 bg-amber-950/70 border border-amber-700/50 px-2.5 py-1 rounded-lg">
              State: PAUSED (Waiting for User)
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-950/80 border border-amber-900/40 space-y-2">
            <p className="text-sm font-medium text-slate-100 leading-relaxed">
              {question}
            </p>

            {missingInformation && missingInformation.length > 0 && (
              <div className="pt-2 border-t border-slate-800/80 text-xs">
                <span className="text-amber-400/90 font-medium flex items-center gap-1.5 mb-1.5">
                  <AlertCircle className="w-3.5 h-3.5" /> Underspecified Scope Boundaries:
                </span>
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 pl-2">
                  {missingInformation.map((item, idx) => (
                    <li key={idx} className="text-slate-300 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-2.5 pt-1">
            <input
              type="text"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              disabled={isLoading}
              placeholder="e.g. Focus on Ethiopic character embeddings and the SemEval benchmark..."
              className="flex-1 bg-slate-950 border border-slate-700/80 focus:border-amber-400 focus:ring-1 focus:ring-amber-400 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none transition-all disabled:opacity-50 font-sans"
            />
            <button
              type="submit"
              disabled={!answer.trim() || isLoading}
              className="flex items-center justify-center gap-2 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs py-3 px-6 rounded-xl shadow-lg shadow-amber-500/20 transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-slate-950" />
                  <span>Resuming Graph...</span>
                </>
              ) : (
                <>
                  <span>Resume Investigation</span>
                  <ArrowRight className="w-4 h-4 text-slate-950" />
                </>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

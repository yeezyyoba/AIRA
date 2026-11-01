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
    <div className="bg-amber-950/30 border border-amber-500/40 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm">
      <div className="flex items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center shrink-0 mt-0.5">
          <HelpCircle className="w-5 h-5 text-amber-400" />
        </div>

        <div className="flex-1 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-amber-300 uppercase tracking-wider flex items-center gap-2">
              <span>Clarification Required</span>
              <span className="text-[10px] bg-amber-500/20 border border-amber-500/30 text-amber-400 px-2 py-0.5 rounded-full font-mono">
                Workflow Paused
              </span>
            </h3>
          </div>

          <p className="text-sm text-slate-200 font-medium leading-relaxed">
            {question}
          </p>

          {missingInformation && missingInformation.length > 0 && (
            <div className="space-y-1 bg-amber-950/40 border border-amber-800/30 rounded-lg p-3 text-xs text-amber-200/80">
              <span className="font-semibold block text-amber-300 mb-1 flex items-center gap-1.5">
                <AlertCircle className="w-3.5 h-3.5" /> Underspecified Areas:
              </span>
              <ul className="list-disc list-inside space-y-0.5">
                {missingInformation.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          <form onSubmit={handleSubmit} className="pt-2">
            <div className="flex gap-2">
              <input
                type="text"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                disabled={isLoading}
                placeholder="Provide specific models, datasets, or constraints to resume..."
                className="flex-1 bg-slate-950 border border-slate-700 focus:border-amber-400 focus:ring-1 focus:ring-amber-400 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 outline-none transition-all disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={!answer.trim() || isLoading}
                className="flex items-center gap-2 bg-amber-600 hover:bg-amber-500 text-slate-950 font-semibold text-sm px-5 py-2.5 rounded-xl shadow-lg shadow-amber-600/20 transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-slate-950" />
                    Resuming...
                  </>
                ) : (
                  <>
                    <span>Resume</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

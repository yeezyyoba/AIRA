# AIRA — One-Page Evaluation Report

AIRA is a LangGraph research agent for Amharic and low-resource language technology. It was evaluated with 10 live runs (real Gemini, Groq, OpenAlex, Crossref, Semantic Scholar), offline end-to-end scenarios and component benchmarks. Last updated 4 Nov 2026.

## 1. Success rate

| Evaluation | Result |
| --- | --- |
| Research runs that completed all six phases with a saved, cited report | **8 / 8** (2–5 papers extracted per run) |
| Runs stopped early before any research (clarification or scope/risk check), as designed | **2 / 2** (no report is written for these) |
| Research runs with a known quality defect | 2 / 8 (runs 2 and 3, see §2; both fixed) · final status saved as `GOAL_ACHIEVED` for runs 1, 3, 7 |
| Offline end-to-end scenarios (`tests/test_graph.py`) | **18 / 18 passed** |
| Full regression suite (offline, no network) | **464 / 464 passed** |
| Component benchmarks | Search nDCG 0.85, MRR 0.87, Recall@5 1.00, P@5 0.62 · PDF field accuracy 0.89 · RAG correct paper 100 %, correct page 50 % · analysis 5/5 · reliability 4/4 |

Research runs (from the saved reports in `data/reports/`):

| Run | Date | Query | Outcome |
| --- | --- | --- | --- |
| 1 | 2 Nov | Amharic sentiment analysis with transformer models | `GOAL_ACHIEVED`, 149 steps; 20 found (19 VERIFIED), 4 extracted |
| 2 | 2 Nov | Compare transformer models for Amharic sentiment on AfriSenti | Full report, but only 1 of 5 extracted papers was about Amharic |
| 3 | 2 Nov | Same query, real Gemini embeddings | `GOAL_ACHIEVED`, 98 steps; 10/10 VERIFIED, 5 extracted |
| 4 | 3 Nov | Literature review: transformer models for Amharic NLP | 3 extracted; an injection pattern on page 19 of one paper was flagged and treated as data |
| 5 | 3 Nov | "what is amharic" | 2 extracted, no metrics or gaps: a definition question fits a comparison agent poorly |
| 6 | 3 Nov | Compare approaches to Amharic sentiment and identify dataset/model limitations | 5 extracted, 5 metric tables, 2 gaps |
| 7 | 3 Nov | Compare transformer models for Amharic sentiment on AfriSenti (LLM function calling on) | `GOAL_ACHIEVED` in degraded mode; tool proposals: 1 accepted, 3 rejected, 8 fell back to rules |
| 8 | 3 Nov | Same query as run 6, **after the rate-limit fixes** | 4 extracted; tool proposals **13 accepted, 0 rejected, 0 fallback** |

Runs 9 and 10 stopped before research (clarification or refusal), so no tools ran and no report was written.

## 2. Failures found (all fixed; details in `AUDIT_REPORT.md`)

- **PDF downloads (run 1):** 18 of 22 failed (landing pages, 404/202) and retries nearly used up the step budget. Fix: skip permanent errors, rank direct PDF links first.
- **Off-topic evidence and report bloat (run 2):** extraction followed citation order and the report was 1,374 lines. Fix: relevance ranking, bounded excerpts.
- **Wrong claim (run 3):** a "best WER of 6.0" came from a regex reading "checkpoints⁶". Fix: word-boundary metric patterns.
- **Rate limits (run 7):** 7 proposals timed out, Gemini/Groq 429s and 503s, 26 Semantic Scholar 429s, and 3 valid proposals rejected over `{"": {}}` arguments. Fix: budgeted proposals, circuit breaker, provider cooldowns, Semantic Scholar pacing, no-argument handling; run 8 then had 0 fallbacks. NVIDIA was added as a third provider.
- **Scope:** off-topic questions used to run the whole workflow; they now stop with `OUT_OF_SCOPE`.

## 3. Agentic behaviour shown

- **Planning and replanning:** a six-phase plan; run 1 revised it once (`PLAN_REVISED`) to expand the search.
- **Autonomous tool choice:** the LLM proposes calls via function calling and code validates them: 13 of 13 accepted in run 8; in run 7, invalid ones were rejected with a logged reason.
- **Self-correction:** run 1 made 12 retry requests and recorded 6 cases where retries were exhausted, then moved on. Empty searches are reformulated.
- **Validation:** every tool output is labelled VALID/EMPTY/INVALID/ERROR; VERIFIED needs two registries; injected text in papers is flagged and never followed (run 4).
- **Critique and stopping:** the critic approved runs 1, 3 and 7; runs 9 and 10 stopped at the ask-user step. Offline tests cover every stopping status and degraded mode.
- **Explained decisions:** each step logs a one-line reason, streamed live to the UI with per-session API usage.

## 4. Limitations

- 10 live runs, mostly on Amharic sentiment analysis. Success rates are indicative, not statistically meaningful.
- The final status was saved for only 3 of the 8 research runs; the rest are judged from their saved reports.
- The rate-limit fixes were confirmed in one live run (run 8); the NVIDIA fallback has been tested offline only.
- Gap/metric extraction and section detection are heuristic; no OCR; the RAG benchmark used mock embeddings (correct page 50 %).
- Sessions are in-memory and lost on restart.

## 5. Improvements

1. Run varied queries (machine translation, speech, NER, other low-resource languages) and save each run's final status with its report.
2. Improve PDF section parsing and claim-level grounding checks.
3. Add OCR, a durable SQLite checkpointer, and a real-embedding RAG benchmark with Amharic questions.

**Details:** `evaluation/evaluation_report.md` · `evaluation/evaluation_results.json` · `AUDIT_REPORT.md`

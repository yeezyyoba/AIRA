# AIRA — One-Page Evaluation Report

AIRA is a LangGraph research agent for language technology (Amharic and low-resource NLP). It searches registries, validates metadata, reads PDFs, retrieves evidence (RAG), compares results and writes a cited report. It was evaluated in three ways: live end-to-end runs with real Gemini, Groq, OpenAlex, Crossref and Semantic Scholar; offline end-to-end graph scenarios; and component benchmarks. Last updated 3 Nov 2026.

## 1. Success rate

| Evaluation | Result |
| --- | --- |
| Live runs, final status recorded | **3 / 3 `GOAL_ACHIEVED`** (a 4th live run produced a full report; its status was not recorded) |
| Live runs without a substantive quality defect | **2 / 4** (one extracted mostly off-topic papers; one stated a wrong metric, see §2) |
| Offline end-to-end scenarios (`tests/test_graph.py`) | **18 / 18 passed** |
| Full regression suite (offline, no network) | **447 / 447 passed** |
| Component benchmarks | Search nDCG 0.85, MRR 0.87, Recall@5 1.00, P@5 0.62 · PDF field accuracy 0.89 · RAG correct paper 100 %, correct page 50 % · analysis 5/5 · reliability 4/4 |

Live runs:

| Run | Query | Outcome |
| --- | --- | --- |
| 1 (2 Nov) | Amharic sentiment analysis with transformer models | `GOAL_ACHIEVED` in 149/150 steps: 20 papers found (19 VERIFIED), 4 extracted, 8 RAG passages, 3 metric comparisons, 2 gaps, 20 citations |
| 2 (2 Nov) | Compare transformer models for Amharic sentiment on AfriSenti | Full report, but only 1 of 5 extracted papers was about Amharic |
| 3 (2 Nov) | Same query, real Gemini embeddings | `GOAL_ACHIEVED` in 98 steps: 10/10 VERIFIED, 5 extracted, 8 RAG passages, 3 gaps, 10 citations |
| 4 (3 Nov) | Same query, LLM function calling on | `GOAL_ACHIEVED`; LLM tool proposals: 1 accepted, 3 rejected, 8 fell back to rules, 4 single-option steps |

## 2. Failures found (all fixed; details in `AUDIT_REPORT.md`)

- **PDF downloads:** 18 of 22 downloads failed in run 1 (doi.org landing pages, HTTP 404/202). Permanent errors were retried and nearly used up the step budget. Fix: permanent errors are skipped, and direct PDF links are ranked first.
- **Off-topic evidence (run 2):** extraction followed citation order. Fix: candidates are ranked by query relevance.
- **Wrong claim (run 3):** the conclusions reported a "best WER of 6.0" for a sentiment paper because a regex read "checkpoints⁶" as a WER value. Fix: metric patterns are anchored at word boundaries.
- **Report bloat (run 2):** the report was 1,374 lines of raw PDF text. Fix: excerpts are bounded.
- **Rate limits (run 4):** 7 tool proposals timed out and kept running in the background; there were Gemini 503/429 and Groq 429 errors and 26 Semantic Scholar 429s. 3 valid proposals were rejected because Groq sent `{"": {}}` as arguments. Fix: budgeted single-shot proposals, a circuit breaker, provider cooldowns, Semantic Scholar pacing, and no-argument handling.
- **Scope:** "who won the World Cup" used to run the whole workflow. It now stops with `OUT_OF_SCOPE`.

## 3. Agentic behaviour shown

- **Planning and replanning:** a six-phase plan; run 1 revised it once (`PLAN_REVISED`) to expand the search.
- **Autonomous tool choice:** the LLM proposes calls via function calling. Code accepted one proposal that differed from the rule choice and rejected invalid ones with a logged reason.
- **Self-correction:** run 1 made 12 retry requests and recorded 6 cases where retries were exhausted, then moved on. Empty searches are reformulated.
- **Validation:** every tool output is classified (VALID, EMPTY, INVALID or ERROR). VERIFIED needs two registries, and prompt-injection text inside papers is contained (tested).
- **Critique and stopping:** the critic approved runs 1, 3 and 4. Offline tests cover every stopping status: clarification pause and resume, `UNACCEPTABLE_RISK`, `OUT_OF_SCOPE`, `MAX_STEPS_REACHED`, and degraded mode with both LLMs down.
- **Explained decisions:** each step logs a one-line reason, streamed live to the UI together with per-session API usage.

## 4. Limitations

- Only 4 live runs, all on Amharic sentiment. Success rates are indicative, not statistically meaningful.
- The rate-limit fixes from 3 Nov are verified offline only; they have not yet been confirmed in a live run.
- Gap and metric extraction and PDF section detection are heuristic. Conclusions still need expert review.
- No OCR for scanned PDFs. The RAG benchmark used mock embeddings (correct-page rate 50 %).
- Sessions live in in-memory checkpoints and are lost on restart. Cooldowns and usage counters are per process.

## 5. Improvements

1. Repeat live runs on 5–10 varied queries (MT, speech, NER, low-resource languages) and report success with the API Usage numbers.
2. Improve PDF section parsing and claim-level grounding checks.
3. Add OCR, a durable SQLite checkpointer, and a real-embedding RAG benchmark with Amharic questions.

**Details:** `evaluation/evaluation_report.md` · `evaluation/evaluation_results.json` · `AUDIT_REPORT.md`

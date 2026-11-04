# AIRA Codebase Audit — November 2026

Scope: the full backend (`agents/`, `app/`, `llm/`, `services/`, `tools/`, `memory/`,
`validation/`, `evaluation/`, `tests/`). The Next.js frontend was inspected only for
API compatibility and was not modified.

Verification: `python -m pytest tests` → **254 passed** (252 confirmed on the developer's Mac) on Python 3.11.18 and 3.13
(baseline before the audit: 199 tests, 6 genuine failures). The backend was started with
uvicorn and `/health`, `/api/v1/health`, session creation, the risk stop, the
`/papers/read` guard and `/reports/` were exercised over HTTP.

---

## 1. Confirmed bugs (fixed)

| # | Severity | Location | Problem | Fix |
|---|---|---|---|---|
| 1 | Critical | `agents/nodes.py` `execute_tools` | `v_db.add_chunks(chunks)` omitted the required `embeddings` argument → `TypeError` on every fresh extraction. The retry then "succeeded" only because the paper had been cached, so the failure was hidden and **no paper was ever indexed**. | Extraction now calls `index_research_paper` (chunk → embed → store); indexing outcome is recorded per paper and only successful indexing populates `indexed_papers`. |
| 2 | Critical | `agents/nodes.py` | RAG tool never selected by the agent; its execution branch passed `vector_store=` (real parameter is `vector_db`) → would raise. `retrieved_evidence` was always empty. | RAG retrieval is a real action in `compare_and_analyze`; results feed state, API and report. |
| 3 | Critical | `tests/test_frontend_api_journey.py` (root cause of the slow test) | Search/validate/read were mocked, but comparison and reporting used the global `ModelRouter`, which made real Gemini/Groq calls whenever `.env` had keys. Embeddings could also hit Gemini, and the repository wrote into `data/processed`. | The test injects an offline router and an isolated repository; `tests/conftest.py` blanks keys, mocks embeddings, isolates data dirs and **blocks outbound sockets** for the whole suite. |
| 4 | High | `nodes.py` identity handling | The validator derived its own `paper_id` (DOI or `"unknown_paper"`) while citations were keyed by the original record id. Mismatched keys made citation selection repeat until the step limit; papers without id/DOI all collided on `"unknown_paper"`. | One identity function (`_paper_key`, title-hash fallback) is passed to the validator and used for validation, extraction, citations and retries. |
| 5 | High | `nodes.py` `validate_tool_outputs` / `store_evidence` | The same `str(enum).upper()` bug as the original ValidationStatus issue, but for `ExtractionStatus`: `FAILED` became `"EXTRACTIONSTATUS.FAILED"`, so failed extractions could count as evidence. | `_sanitize_data` converts every Enum to its value at the state boundary; `_status_value` normalises everywhere statuses are compared. |
| 6 | High | `nodes.py` `retry_handler` | Retry counters were keyed by tool name only, so two retries on paper A exhausted the budget for paper B, and exhaustion **blocked the entire extraction phase**. | Per-item retry keys and failure markers; an exhausted paper is skipped and the others continue. |
| 7 | High | `app/api/v1/endpoints/papers.py` | `/papers/read` accepted any string; the reader treats non-URLs as local paths → the server could be made to read PDFs from its own disk. | Request schema requires `http(s)://`; the agent also refuses non-HTTP sources. |
| 8 | High | `requirements.txt` | Missing `python-multipart`; a clean install cannot import the app (upload endpoint). | Added. |
| 9 | High | `.env` / `app/core/config.py` | `MAX_AGENT_STEPS` was ignored (graph hardcoded 120). The `.env` value 20 would allow ~3 tool calls once enforced. | Setting is wired through the API; default 150; `.env` and `.env.example` updated. |
| 10 | High | `tests/test_phase9_evaluation.py` | Every test run overwrote the committed `evaluation_report.md` / `evaluation_results.json`. | `run_all_evaluations(output_dir=...)`; the test writes to a temp dir. |
| 11 | High | `research.py` `_register_report_if_present` | Filled missing report sections with invented text ("Conclusions established.", "Experimental measurements extracted."); partial reports failed validation and the error was swallowed, so they never reached the archive. | Reports are validated as-is; failures are logged; partial reports are schema-valid and say "Not established". |
| 12 | Medium | `services/report_service.py` | Conclusion was a hardcoded generic claim ("models demonstrate domain-specific efficacy"). The service also re-ran every analysis (repeating all LLM calls). | Conclusions are built from the computed metric tables; precomputed agent analyses are reused. |
| 13 | Medium | `tools/gap_identifier.py` | "Potential direction" gap always claimed noise, domain transfer and compute limits, whichever keyword matched. | Description lists the themes actually found per paper. |
| 14 | Medium | `nodes.py` `analyze_goal` / `check_information` | After clarification the domain stayed stale (first-run "Machine Learning"); the search query contained "(Clarification: …)"; a short query plus clarification was still "missing information" forever. | Domain/questions re-derived on resume; clean `search_query`; clarification counts toward sufficiency. |
| 15 | Medium | `nodes.py` `create_plan` | Pre-completed tasks stayed `PENDING` in `phase_statuses`. | Synchronised. |
| 16 | Medium | `nodes.py` `generate_report_node` | Report generation was unguarded; an exception crashed the graph after retries. | Falls back to a disclosed PARTIAL report. |
| 17 | Medium | `nodes.py` stopping rules | "Unacceptable risk" was not implemented; max-step stops were reported as PARTIAL. | Risk detection → `UNACCEPTABLE_RISK` before any tool runs; explicit `MAX_STEPS_REACHED`. |
| 18 | Medium | `nodes.py` | Methodology analysis ran on the first paper only; one bad metadata record failed validation for all papers; validation of 20 papers ran sequentially. | All papers analysed; per-record rejection; bounded concurrency. |
| 19 | Medium | `llm/*` | No LLM request timeout; Groq SDK retried internally on top of router retries (up to 9 calls); `INVALID_ARGUMENT` classified as an auth error; status codes matched as substrings ("4013 tokens" → 401); degraded mode and Gemini quota state were permanent until restart; unexpected primary errors skipped the fallback. | Timeouts from settings; SDK retries off; `BadRequestError` (not retried, falls back); word-boundary matching; cooldown recovery; generic fallback path; JSON-fence tolerant parsing. |
| 20 | Medium | `tools/paper_comparator.py` | Delimiter escaping matched exact lowercase tags only; LLM call blocked the event loop. | All `<`/`>` escaped; call runs in a worker thread. |
| 21 | Medium | `services/paper_repository.py` | `Path(id).name` made DOIs with equal suffixes share a file. | Full id is sanitised; legacy files still readable. |
| 22 | Medium | `reports.py` | Each report listed twice (report id and session id); archive lost on restart. | One entry per report; reports persisted by the File Manager and loaded from disk. |
| 23 | Medium | `graph.py` routing | Crashed on string plan entries (agentic evaluator); empty initial search was never retried. | Tolerant normalisation; empty initial search → retry. |
| 24 | Low | `/health` | Claimed "SQLite (Configured)" though nothing uses SQLite. | Reports actual stores and key presence (never values). |
| 25 | Low | logs | Step numbers in logs were off by one / stale on resume. | Each entry records the step it executed at. |

## 1b. Findings from the first live run (real Gemini/Groq/OpenAlex, 2 Nov 2026)

Query "Amharic sentiment analysis with transformer models": **GOAL_ACHIEVED** in
8.5 minutes, 149/150 steps, 20 papers found (19 VERIFIED), 4 extracted and indexed,
8 RAG passages, 3 metric comparisons, 2 gaps, 20 citations, no degraded mode.
The run also exposed four defects, all fixed:

| # | Problem | Fix |
|---|---|---|
| 26 | Permanent PDF failures (HTTP 404/202, "not a valid PDF") were retried twice each; 18 of 22 downloads failed, almost exhausting the step budget. | Only transient failures (timeouts, network, 5xx, 408/429) are retried; permanent ones are recorded once and skipped. |
| 27 | Report references used titles parsed from PDFs (e.g. "Proceedings of the …") instead of registry-verified titles. | References prefer validated metadata (title, authors, year, DOI). |
| 28 | "Incompatible datasets" reasons embedded pages of raw PDF text in the conclusions. | Dataset names and reasons are truncated and whitespace-normalised. |
| 29 | Reflection message claimed a phase was unresolved when only one paper had failed. | Message describes item-level failure accurately. |
| 30 | Second live run ("Amharic sentiment … AfriSenti"): only 1 of 5 extracted papers was about Amharic, because extraction followed the search API's citation order. | Extraction candidates are ranked by query-keyword relevance (title + abstract), then validation confidence. |
| 31 | The same report was 1,374 lines: whole raw PDF sections were pasted into comparison sections. | Report sections show bounded excerpts (~450 chars); full text stays in the repository/JSON. |
| 32 | Third live run (98 steps, real Gemini embeddings): the conclusions claimed a "best WER of 6.0" for a sentiment paper. The metric regexes had no word boundaries, so "checkpoints⁶ were" was read as "6 WER" (also "lower", "answer", "cancer"). | Metric names and values are anchored at word boundaries; "BLEU score of X" phrasing is now supported. Verified on the cached papers from that run. |
| 33 | No research-scope check: "who won the World Cup", "tell me a joke" or "write my cover letter" ran the full ~100-step research workflow (and were all labelled "Machine Learning"). | Keyword-based scope check in `meta_tools/ask_user.py` before any tool runs: language research proceeds; AI/ML topics without a language angle get a clarification question; anything else stops with the new status `OUT_OF_SCOPE` and an explanatory `message`. |
| 34 | Frontend: `PARTIAL` and `UNACCEPTABLE_RISK` were missing from the terminal-status list, so the UI polled those sessions forever and showed a spinning "in progress" badge. | Added them (and `OUT_OF_SCOPE`) as terminal statuses with their own badges, plus a banner explaining refused requests. |
| 35 | Verifiable Citations tab crashed (citations are objects since the audit) and papers were identified only by DOI. | Citations normalised to text; papers shown as "First author et al., year" with the full title, in the UI and the Markdown report. |
| 36 | Tool selection was purely rule based, although the course brief (slides 10-14) asks for function calling: "the model proposes the route; the application decides what to trust and execute". | Hybrid selection: `select_tools` lists the legal options; the new `propose_tool_call` node lets Gemini/Groq propose a call with native function calling; code validates the tool, arguments (Pydantic), paper ids and limits, and falls back to the rule choice when the proposal is invalid or the LLM is unavailable. Decisions are logged and counted (`tool_selection_stats`). Tested offline in `tests/test_tool_calling.py`, including a paper title that tries to inject a tool call. |
| 37 | Live run (event log, 3 Nov): six tool proposals timed out at 20 s, but `asyncio.wait_for` cannot stop the worker thread, so each orphaned call kept retrying and falling back to Groq in the background, burning both providers' rate limits. | Proposals are single-shot with a time budget: one attempt per provider, the remaining budget is passed as the provider's HTTP timeout (Gemini per-request `http_options`, Groq `timeout`), and no fallback starts once the budget is spent. After `TOOL_SELECTION_MAX_CONSECUTIVE_FAILURES` (2) consecutive failures, proposals are paused for the rest of the run ("LLM proposals paused for this run after repeated failures; using rules"), counted as `paused`. |
| 38 | Same run: Groq answered no-argument tool calls with `{"": {}}`, which failed `extra="forbid"` ("Extra inputs are not permitted"), so three valid choices were rejected. | Empty-named keys are dropped; for tools that take no LLM arguments, other arguments are ignored and noted (URLs and code-owned fields are still rejected first). Gemini no longer receives an empty parameter schema for those tools. |
| 39 | Gemini 503 "high demand" / 504 / 429 and Groq 429 were retried against the same provider within 0.5–2 s, which mostly earned more 429s. | No same-provider retry for 429, 503 overload or timeouts: the provider is put in a cooldown (its own "retry in Ns" hint, capped at 30 s; defaults 20 s for 429, 15 s for 503), the router falls back immediately and skips cooling providers. A Gemini quota error with a short hint pauses Gemini only for that hint; without one (daily quota) for the full cooldown. When every provider is cooling down the call fails fast without entering the 2-minute Degraded Mode. Connection resets are still retried. |
| 40 | 26 Semantic Scholar 429s in one run: every DOI was looked up in all three registries, concurrently. | A process-wide pacer spaces Semantic Scholar requests (`SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS`, 1.1 s) and a 429's `Retry-After` delays all callers. S2 is skipped when Crossref and OpenAlex both return records that match the input (the two registries VERIFIED requires); it is no longer listed in `sources_checked` then. Crossref and OpenAlex are queried in parallel. |
| 41 | PDF downloads often failed on doi.org landing pages (HTTP 202, "not a valid PDF") although a direct PDF link was known. | Links are ranked: direct `.pdf` / arXiv / OpenReview PDFs first, doi.org landing pages last; arXiv `abs` and ACL Anthology pages map to their PDF addresses. |
| 42 | No visibility into how many API requests a run made. | Per-session `api_usage` in the API response (per LLM provider: calls, successes, failures, 429s, timeouts, pauses, tokens; fallbacks; real embedding calls; academic API requests and 429s), an end-of-run "API usage: …" log line and live event, and an **API Usage** panel in the UI with the LLM tool-choice counts. Tested in `tests/test_rate_limits.py` (46 tests, all offline). |
| 43 | Only two LLM providers: when Gemini and Groq were both rate limited, every LLM step fell back to rules. | Optional third provider, NVIDIA (`llm/nvidia_provider.py`, OpenAI-compatible endpoint, no extra SDK). The router tries Gemini → Groq → NVIDIA → degraded mode, skips providers that are cooling down, stays within a tool proposal's time budget, and enters Degraded Mode only when every provider fails for a reason other than a short rate limit. Tested offline in `tests/test_nvidia_provider.py` (17 tests). |

## 2. Test expectations corrected (with reasons)

* `test_scenario_09`: asked for `GOAL_ACHIEVED` from **one** extracted paper. A comparative
  review cannot be achieved with one paper. The test now uses two papers (one injected)
  and asserts the payload stays escaped inside the untrusted block.
* `test_scenario_16`: the mocked papers had no PDF URL, so the honest result is
  `INSUFFICIENT_EVIDENCE`. The test now supplies PDFs and a mocked reader and asserts
  `GOAL_ACHIEVED`.
* `test_empty_analysis_result_is_valid`: kept as written; the implementation was fixed.

New tests: `tests/test_reliability.py` (29), `tests/test_prompt_injection.py` (12),
`tests/test_tools.py` (17), replacing three `assert True` placeholders.

## 3. Assignment requirements

| Requirement | Status |
|---|---|
| Stop: goal achieved | `GOAL_ACHIEVED` only when critique approves ≥2 extracted papers |
| Stop: essential information missing | `USER_CLARIFICATION_REQUIRED` (pause + resume) |
| Stop: unacceptable risk | `UNACCEPTABLE_RISK` (instruction override, secret/destructive requests, fabrication); off-topic requests stop with `OUT_OF_SCOPE` |
| Stop: max steps | `MAX_STEPS_REACHED` with a bounded partial report |
| Max two retries, correct counting/exhaustion | Per tool/item; `MAX_RETRIES=2`; tested |
| Unreliable input tests | Missing, conflicting, invalid, timeouts, injection, invalid tool output, PDF failures |
| Agentic logs | Plan, tool reasons, results, plan revisions (`PLAN_REVISED`), reflections, retries, stopping reason |
| Function calling | Tool specs (name, when-to-use description, JSON schema) for all 10 execution tools; native Gemini/Groq function calling proposes the next call, and code validates and executes it, with the rules as fallback |
| ≥5 execution tools, integrated | Search, metadata validation (function calling), PDF reader, RAG retriever, result analyzer (compute), citation generator, file manager (system) — all executed in the end-to-end test |

## 4. Not implemented / remaining limitations

* Sessions live in LangGraph's in-memory checkpointer: a restart loses paused sessions
  (reports survive). A SQLite checkpointer needs `langgraph-checkpoint-sqlite`.
* ~~`memory/database.py`, `meta_tools/*`, `validation/*_validation.py` remain stubs.~~
  Resolved: the meta-tools (planner, plan updater, ask user, critic, finish) and the
  tool-output / evidence validators now live in `meta_tools/` and `validation/` as pure
  functions called by the graph nodes; routing moved to `agents/routing.py`. Unused
  placeholders (`memory/database.py`, `memory/repositories.py`,
  `services/embedding_service.py`, `services/evidence_service.py`) were removed.
* The risk filter is pattern-based; it can be bypassed by paraphrase. Retrieved content
  is still always treated as data.
* Gap identification, metric extraction and **PDF section detection** are heuristic
  (regex/keyword based). Section boundaries can be wrong (e.g. a "Methodology"
  excerpt starting mid-word) and a methods sentence mentioning "limitation" can be
  reported as an author-acknowledged gap. Improving the section parser
  (`tools/paper_reader.py`) is the highest-value next quality improvement.
* `/research/sessions` runs the whole workflow inside the request; long runs need a
  generous client timeout (a background-job API would be better).
* Backup files (`agents/nodes.py.bak*`, `evaluation/*.backup`) were left untouched.
* LLM tool selection was exercised in one live run (findings 37-39). The fixes for
  those findings are tested with mocked clients only; a new live run should confirm
  the accepted/rejected/fallback counts in the API Usage panel.
  Each proposal costs one LLM call and one graph step (about 6 per run, within
  `MAX_AGENT_STEPS=150`).
* Cooldowns, the Semantic Scholar pacer and session usage are per server process
  (not shared across several workers). Gemini rejects request deadlines under ~10 s, so
  a Gemini proposal can outlive a shorter budget by a few seconds; its result is then
  discarded, and no fallback is started.
* Live provider behaviour (real Gemini/Groq/OpenAlex) was not exercised in this audit;
  use `scripts/manual_test_*.py` for that.

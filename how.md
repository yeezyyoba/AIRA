# AIRA — How It Works, Start to Finish

*Amharic Intelligent Research Assistant · updated version (3 Nov 2026) · colour legend at the end of the document*

## Part A — How AIRA works

### 1. The request comes in

- **Frontend:** you type a query in the Next.js app, for example “Compare transformer models for Amharic sentiment analysis on AfriSenti”. The app sends `POST /api/v1/research/sessions` with `run_in_background: true`. Through the API you can also pass a result limit (up to 100) and a year range.
- **Backend (`research.py`):**
  - It creates a `session_id`, starts the LangGraph workflow as a background task and replies at once with 🟢`IN_PROGRESS`.
  - LangGraph's MemorySaver saves the state after every node, keyed by the `session_id`, so a paused session can be resumed.
  - The frontend checks `GET /sessions/{id}` every few seconds and opens a live event stream (`/sessions/{id}/events`) that shows each step as it happens.
  - All API calls made during the run are counted for this session. This feeds the API Usage panel.

### 2. Understanding the request (runs once)

- 🟣`receive_request`: sets up the shared state: budgets (`max_steps = 150`, search results ≤ 100), empty evidence lists, the log and the tool-selection counters. On a resumed session it keeps the evidence gathered so far.
- 🟣`analyze_goal`: builds a clean search query, the domain (from keywords) and three default research questions (methods, datasets/metrics, gaps), using simple rules without the LLM.
- 🟣`check_information` calls the 🔴**ask_user meta-tool**, which can stop or pause the run:
  - **Unacceptable risk:** the request matches a risk pattern (“ignore previous instructions”, “reveal API keys”, “fabricate citations”). The run ends with 🟢`UNACCEPTABLE_RISK`; no tool ever runs.
  - **Out of scope:** the request isn't about language research (e.g. “who won the World Cup”). The run stops with 🟢`OUT_OF_SCOPE` and the UI explains why.
  - **AI topic without a language angle:** the agent asks a clarification question instead of guessing.
  - **Missing information:** the topic is empty or too broad (e.g. “papers”). 🟣`ask_user` pauses the run with 🟢`USER_CLARIFICATION_REQUIRED`. Your answer goes to `/resume` and the graph restarts with the clarification merged into the search query.
- 🟣`create_plan` calls the 🔴**planner meta-tool**, which creates the six-phase research plan below. Each phase starts as 🟢`PENDING`; on a resumed session, phases already done are recognised and skipped.

| \#  | Phase (task id)       | Goal                                              | Execution tools used                                                                                                           |
|-----|-----------------------|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| 1   | `search_and_validate` | Find candidate papers and check their metadata    | 🔵`search_academic_papers`, 🔵`validate_metadata_integrity`                                                                    |
| 2   | `extract_and_index`   | Read up to 5 relevant PDFs and index them for RAG | 🔵`read_research_paper` (+ indexing into 🟠**ChromaDB**)                                                                       |
| 3   | `compare_and_analyze` | Retrieve evidence and compare the papers          | 🔵`retrieve_research_context`, 🔵`compare_research_papers`, 🔵`analyze_research_methodology`, 🔵`analyze_experimental_results` |
| 4   | `identify_gaps`       | Find limitations and open problems                | 🔵`identify_research_gaps`                                                                                                     |
| 5   | `generate_citations`  | Cite every paper used                             | 🔵`generate_citations`                                                                                                         |
| 6   | `synthesize_report`   | Write the cited comparison report                 | 🔵`generate_research_comparison_report`                                                                                        |

The 🔴**planner** only *creates* these phases. The tool loop in section 3 *carries them out*, and the 🔴**plan_updater** only *changes their status* (done, blocked, or reopened for an expanded search).

### 3. The tool loop (repeats until all phases are done)

Each phase from the plan is carried out by this loop, one tool call per cycle, in phase order.

**Loop:** 🟣`select_tools` → 🟣`propose_tool_call` → 🟣`execute_tools` → 🟣`validate_tool_outputs` → (🟣`retry_handler`) → 🟣`store_evidence` → 🟣`reflect_and_revise` → back to 🟣`select_tools`

- 🟣`select_tools` — the 🔴**tool-selection meta-tool**, rule part: lists the legal next calls for the first unfinished phase (in Phase 2, one option per paper that can still be read). The first option is the rule-based choice. A call that already succeeded with the same inputs is never offered again.
- 🟣`propose_tool_call` — the 🔴**tool-selection meta-tool**, LLM part (function calling):
  - The 🟠**LLM** receives only the legal tools, each with a name, a when-to-use description and a JSON schema, and proposes one call through native function calling: 🟠**Gemini** first, 🟠**Groq** as fallback.
  - Paper content goes into the prompt marked as untrusted data, so text inside a paper cannot change which tool runs.
  - **Code validates the proposal:** is the tool known and legal in this phase; are the arguments valid (Pydantic, unknown fields rejected); is the LLM trying to supply URLs, file paths or code-owned fields; is the paper ID one of the session's real candidates; does the search query pass the scope and risk check; are years and limits within bounds (clamped); has this call already succeeded?
  - **Outcomes:** 🟢`ACCEPTED` (the LLM's choice runs), 🟢`REJECTED` (reason logged, rule choice runs), 🟢`FALLBACK` (LLM timed out or failed, rule choice runs), 🟢`SINGLE_OPTION` (only one legal call, no LLM call made).
  - **Budget:** 20 seconds per proposal, one attempt per provider; the HTTP timeout is the time remaining, so nothing keeps running after AIRA stops waiting.
  - **Circuit breaker:** after 2 failures in a row, LLM proposals pause for the rest of the run (🟢`PAUSED`): “LLM proposals paused for this run after repeated failures; using rules.”
- 🟣`execute_tools`: runs the chosen 🔵`execution tool` and records the result or the error.
- 🟣`validate_tool_outputs`: checks the result's shape (`validation/tool_validation.py`) and labels it 🟢`VALID`, 🟢`EMPTY`, 🟢`INVALID` or 🟢`ERROR`.
- 🟣`retry_handler`: runs after 🟢`INVALID` or 🟢`ERROR`, or after an empty first search.
  - At most 2 retries, counted per tool and per paper, so one bad PDF doesn't use up the retries for the others.
  - Only temporary errors are retried (timeouts, network errors, HTTP 5xx, 408, 429). Permanent errors (404, “not a valid PDF”) are recorded once and skipped.
  - A failed search is retried with a reformulated query.
  - When retries run out, the paper is marked failed, or the whole phase is marked 🟢`BLOCKED`, and the agent moves on.
- 🟣`store_evidence`: applies the evidence rules (`validation/evidence_validation.py`), so output from failed calls is never stored. It saves the result into the state and ticks off phases through the 🔴**plan_updater meta-tool**.
- 🟣`reflect_and_revise`: logs a reflection and enforces the step budget. If too little evidence was gathered, it revises the plan through the 🔴**plan_updater meta-tool** (see Phase 2).

### How the loop carries out each phase of the plan

The phases were created by the 🔴**planner** in 🟣`create_plan` (section 2). Below is what the loop does in each one.

#### Phase 1 — Search & validate

- 🔵`search_academic_papers` queries 🟠**OpenAlex**, 🟠**Semantic Scholar** and 🟠**Crossref**. It merges duplicates (same DOI, or a title ≥ 85% similar with the same year) and continues if one provider fails.

  > 💡 **In simple words:** AIRA searches three big online catalogues of research papers at once. The same paper often shows up in more than one of them, so AIRA combines copies into one. Two results count as the same paper if they have the same DOI (a paper's unique ID number), or if their titles are almost identical and they're from the same year. If one catalogue is down, AIRA uses the other two.
  

- Semantic Scholar requests are spaced 1.1 seconds apart across the whole app; if it sends a 429 with a wait time, all callers wait.

  > 💡 **In simple words:** Semantic Scholar only allows about one request per second. AIRA waits at least 1.1 seconds between requests to it. If Semantic Scholar answers "too many requests" (error 429) and says "wait X seconds", every part of AIRA waits, not just the request that got the error.

- 🔵`validate_metadata_integrity` cross-checks title, authors, year and DOI. 🟠**Crossref** and 🟠**OpenAlex** are queried in parallel; 🟠**Semantic Scholar** is only asked if those two don't already agree.

  > 💡 **In simple words:** For each paper, AIRA checks that the title, authors, year and DOI are real by looking the paper up in the catalogues. It asks Crossref and OpenAlex at the same time to save time. It only asks Semantic Scholar if those two disagree or one is missing. This avoids using up Semantic Scholar's strict limit.

- Results: 🟢`VERIFIED` (requires two registries), 🟢`PARTIALLY_VERIFIED`, 🟢`UNVERIFIED`, 🟢`CONFLICTING`. 🟢`CONFLICTING` papers are never used.

  > 💡 **In simple words:** Each paper gets one label. 🟢`VERIFIED`: two catalogues confirm the details. 🟢`PARTIALLY_VERIFIED`: only one confirms them. 🟢`UNVERIFIED`: the paper wasn't found, which doesn't mean it's fake. 🟢`CONFLICTING`: the catalogues contradict the details, so AIRA won't use the paper at all.

#### Phase 2 — Extract & index (one paper per cycle, up to 5)

- Candidates are ranked by relevance to your query (title and abstract). The 🟠**LLM** may propose a different paper from the list, never one outside it.

  > 💡 **In simple words:** AIRA sorts the papers by how well their title and summary match your question, and plans to read the best match first. The LLM may suggest reading another paper first, but only one already on the list. It can't add a paper of its own.

- **PDF link ranking:** direct PDF links (`.pdf`, arXiv, ACL Anthology, OpenReview) first, doi.org landing pages last. arXiv abstract pages and ACL Anthology pages are converted to their PDF addresses.

  > 💡 **In simple words:** A paper can have several links. Some go straight to the PDF file; others go to a web page about the paper, which often blocks downloads. AIRA tries the direct PDF links first and the web pages last. For arXiv and ACL Anthology, it knows how to turn the page address into the PDF address.

- **PDF download:** only http(s) links; private and internal addresses blocked, including through redirects (SSRF protection); size limit; the file must start with `%PDF-`.

  > 💡 **In simple words:** AIRA downloads files safely: it only follows normal web links; it refuses links that point inside your own computer or network, even if a website redirects there (this blocks a type of attack called SSRF); it refuses files that are too big; and it checks that the file really is a PDF, because every real PDF starts with `%PDF-`.

- 🔵`read_research_paper`: extracts text with PyMuPDF (pdfplumber as fallback), flags scanned PDFs as needing OCR, detects sections from the page layout, keeps page numbers for citations, flags prompt-injection text inside the PDF and never follows it, and saves the result as JSON.

  > 💡 **In simple words:** AIRA reads the PDF's text, using PyMuPDF first and pdfplumber if PyMuPDF fails. It then marks scanned PDFs (just pictures of pages) as unreadable for now; finds the sections, such as Abstract, Methods and Results, from the page layout; remembers which page each piece of text came from, so it can cite the page later; flags any hidden instructions in the PDF trying to trick the AI (prompt injection) and never obeys them; and saves the result as a JSON file.

- **Indexing:** section-aware chunks of 1,000 characters with a 150-character overlap, embedded with the 🟠**Gemini embedding model** and stored in 🟠**ChromaDB**.

  > 💡 **In simple words:** AIRA cuts the paper's text into small pieces of about 1,000 characters, without mixing sections. Neighbouring pieces share 150 characters so no sentence gets lost at a cut. Gemini turns each piece into a list of numbers that captures its meaning (an "embedding"). The pieces are stored in ChromaDB, a database that finds text by meaning.

- **Plan revision:** if fewer than 2 papers produced usable text, the 🔴**plan_updater meta-tool** revises the plan once: Phase 1 reopens with an expanded search asking for twice as many results.

  > 💡 **In simple words:** A comparison needs at least two papers. If fewer than two could be read, AIRA changes its plan once: it goes back to Phase 1 and searches again, asking for twice as many papers, to find more it can read.

#### Phase 3 — Compare & analyse (one action per cycle; the LLM can choose the order)

- 🔵`retrieve_research_context` (RAG): returns the 8 chunks most similar to the question, each with its paper, section, page and similarity score.

  > 💡 **In simple words:** AIRA searches the stored pieces for the 8 that best match your question. Each piece comes with where it's from (paper, section, page) and how close a match it is. This is RAG: finding evidence before answering.

- 🔵`compare_research_papers`: code first builds a comparison table across seven dimensions; then the 🟠**LLM** writes a narrative comparison from that table.

  > 💡 **In simple words:** AIRA's code first fills in a table comparing the papers on seven points: goals, method, datasets, models, experiment setup, results and limitations. Then the LLM writes a readable comparison based on that table.

- Every claim in the narrative is checked against the papers (do the cited pages exist, do the numbers appear in the text, do rankings match the metrics); claims that fail are marked unverified.

  > 💡 **In simple words:** AIRA doesn't trust the LLM's text. It checks each claim: does the cited page exist, does the number actually appear in the paper, and if it says "model A is best", do the numbers agree? Claims that fail are labelled unverified.

- 🔵`analyze_research_methodology`: design, data, architecture, training and evaluation for each paper; each item labelled as a direct extract or an inference.

  > 💡 **In simple words:** For each paper, AIRA writes down how the research was done: the design, the data, the model, how it was trained and how it was tested. Each item says whether it was copied directly from the paper or worked out by AIRA.

- 🔵`analyze_experimental_results`: extracts metrics with regex and pandas (anchored at word boundaries, which fixed the false “WER 6.0” claim), normalises units, accounts for direction (higher F1 is better, lower WER is better) and flags unfair comparisons instead of ranking them.

  > 💡 **In simple words:** AIRA finds the result numbers in each paper, such as accuracy, F1 and WER. It only matches whole words (earlier it mistook text like "checkpoints⁶" for "WER 6", which caused a wrong claim). It puts numbers on the same scale, so 85% and 0.85 are treated as the same value. It knows which direction is better: higher F1 is better, lower error rate (WER) is better. If two numbers can't fairly be compared (different datasets or metric types), it says so instead of picking a winner.

#### Phase 4 — Research gaps

- 🔵`identify_research_gaps` reads only the Limitations, Discussion, Conclusion and Future Work sections, labels each gap (acknowledged by the authors, cross-paper gap, or possible research direction) and adds a disclaimer that the gaps hold only within the analysed papers.

  > 💡 **In simple words:** To find what's still missing in the research, AIRA only reads the sections where authors discuss weaknesses and future work. It labels each gap by where it came from: the authors admitted it themselves; AIRA noticed it by comparing several papers; or it's an idea for future research. It also notes that these gaps only apply to the papers it read, not to the whole field.

#### Phase 5 — Citations

- 🔵`generate_citations` produces APA 7, IEEE or BibTeX from the validated metadata. It never invents missing fields; the LLM is not involved.

  > 💡 **In simple words:** AIRA writes the reference list in standard formats (APA 7, IEEE or BibTeX), using the checked details from Phase 1. If something is missing, such as the year, it leaves it out rather than making it up. Only code does this, never the LLM.

#### Phase 6 — Report

- 🔵`generate_research_comparison_report` builds an 11-section report with metric tables. It reuses the analyses already computed, labels papers “First author et al., year”, keeps excerpts short, uses registry-verified references, and writes “Not established” for anything not supported.

  > 💡 **In simple words:** AIRA puts everything into a final report with 11 sections and result tables. It reuses the work it already did instead of redoing it; names papers in a readable way, for example "Tela et al., 2020", instead of by ID numbers; only quotes short pieces of text; uses the checked details from the catalogues for its references; and where the evidence doesn't support an answer, writes "Not established" instead of guessing.

### 4. Wrap-up

- 🟣`critique_findings` calls the 🔴**critic meta-tool**: 🟢`INSUFFICIENT_EVIDENCE` (fewer than 2 papers extracted), 🟢`NEEDS_REVISION` (a phase is missing or blocked; loops back to 🟣`select_tools` if work remains and steps are left), or 🟢`APPROVED`.
- 🟣`generate_report`: writes a clearly labelled 🟢`PARTIAL` report if the evidence is thin, attaches phase statuses, citations, security flags and the degraded flag, and saves `.md` and `.json` files.
- 🟣`finish_task` calls the 🔴**finish meta-tool**: it keeps an earlier specific stop (🟢`MAX_STEPS_REACHED`, 🟢`UNACCEPTABLE_RISK`, 🟢`OUT_OF_SCOPE`, 🟢`USER_CLARIFICATION_REQUIRED`); otherwise the run ends with 🟢`GOAL_ACHIEVED`, 🟢`PARTIAL` or 🟢`INSUFFICIENT_EVIDENCE`.
- **API and UI:** the final response includes the status, plan, logs, papers, metrics, gaps, citations, evidence, `tool_selection_stats` and `api_usage`. An “API usage: …” line closes the live log. The UI shows the results, the citations tab, the API Usage panel and the reports archive with Markdown/JSON export.

### 5. Where the LLM is used

1.  **Choosing the next tool:** the 🟠**LLM** proposes through native function calling; code decides (🟣`propose_tool_call`).
2.  **The narrative comparison** in 🔵`compare_research_papers`, checked claim by claim afterwards.
3.  **Embeddings for RAG** (🟠**Gemini embedding model**): similarity search, not reasoning.

**How LLM calls are protected:**

- 🟠**Gemini** (`gemini-3.8-flash`) is used first and 🟠**Groq** (`openai/gpt-oss-120b`) as fallback.
- A 429 or 503 isn't retried against the same provider. That provider is paused for its own suggested wait time (capped at 30 s) and requests go to the other one.
- If both providers are down, AIRA enters degraded mode: tool choice uses the rules, the comparison falls back to a deterministic summary, the report is flagged as degraded, and everything else keeps working.

**Still done entirely by code:** goal analysis, planning, the legal tool options, validation, retries, critique, metrics, gaps and citations.

## Part B — The assignment brief, point by point

> **Brief:** *“Build an agent for a real problem of your choice. It must show agentic behavior: autonomy, planning, tools, memory, self-correction.”*

AIRA tackles a real problem: researchers working on Amharic and other low-resource languages spend hours finding, checking and comparing papers. AIRA does this and shows each step.

- **Autonomy:** after one query, AIRA runs to a final status without further input. It decides which tool to call (🟠**LLM** proposal plus rules), which papers to read, whether to retry, whether to revise its plan and when to stop. It asks you only when information is missing.
- **Planning:** the 🔴**planner** creates a six-phase plan; the 🔴**plan_updater** ticks off, blocks or reopens phases. In live run 1 the plan was actually revised to expand the search.
- **Tools:** 10 🔵`execution tools` (see requirement 3).
- **Memory:** within a run, the research state, saved by LangGraph after every node (this is what makes pause and resume work); beyond a run, the paper chunks in 🟠**ChromaDB**, the extracted papers as JSON, and the reports archive.
- **Self-correction:** bounded retries; failed searches retried with a reformulated query; plan revision when evidence is thin; the 🔴**critic** sending the agent back to work; invalid LLM proposals replaced by the rule choice; LLM claims checked against the papers.

### Core requirements

> **Brief:** *1. “Agentic system on any practical domain with a clear goal.”*

- **Domain:** academic research on language technology.
- **Goal:** a validated, cited comparison report that answers the user's research question.
- “Done” is defined in code: 🟢`GOAL_ACHIEVED` only when the 🔴**critic** approves and at least 2 papers were extracted.
- The scope check in the 🔴**ask_user meta-tool** keeps the agent on that goal; off-topic requests get 🟢`OUT_OF_SCOPE`.

> **Brief:** *2. “Use LangGraph: branches, loops and state, no hard-coded linear steps.”*

- **Branches:** after 🟣`check_information` (plan, ask the user, or refuse); after 🟣`validate_tool_outputs` (retry or store); after 🟣`reflect_and_revise` (next tool or critique); after 🟣`critique_findings` (loop back or write the report).
- **Loops:** the tool cycle; 🟣`retry_handler` → 🟣`execute_tools`; 🟣`critique_findings` → 🟣`select_tools`; the expanded-search revision.
- **State:** the typed `ResearchState`, checkpointed per session.
- **Not linear:** the path depends on what happens. A run with failing PDFs takes a different route and a different number of steps: run 1 took 149 steps, run 3 took 98.

> **Brief:** *3. “At least 5 tools from execution-tool categories (function calling, search, code & compute, RAG, system & environment).”*

AIRA has 10 🔵`execution tools`, covering every category:

- **Function calling:** all 10 tools have function-calling specs and are chosen through native 🟠**Gemini**/🟠**Groq** function calling; 🔵`validate_metadata_integrity` is the one listed under this category.
- **Search:** 🔵`search_academic_papers` (🟠**OpenAlex**, 🟠**Semantic Scholar**, 🟠**Crossref**).
- **Code & compute:** 🔵`analyze_experimental_results`, 🔵`compare_research_papers`, 🔵`analyze_research_methodology`, 🔵`identify_research_gaps`, 🔵`generate_citations`.
- **RAG:** 🔵`retrieve_research_context` (🟠**ChromaDB**).
- **System & environment:** 🔵`read_research_paper` (downloads and reads files, with SSRF protection) and 🔵`generate_research_comparison_report` (saves files through the file manager).

> **Brief:** *4. “Include a meta-tool: plan/to-do, ask user, critique, tool selection or delegation.”*

AIRA includes every type the brief lists except delegation:

- **Plan / to-do:** 🔴**planner** and 🔴**plan_updater**.
- **Ask user:** 🔴**ask_user** — pause and resume, plus the scope and risk refusals.
- **Critique:** 🔴**critic**.
- **Tool selection:** 🔴**tool selection** — 🟣`select_tools` combined with 🟣`propose_tool_call`.
- The 🔴**finish** meta-tool sets the final status.

> **Brief:** *5. “Keep task state, validate every tool output, cite evidence, briefly explain each decision.”*

- **Task state:** `ResearchState` holds the plan, phase statuses, step count, retry counts, evidence and tool-selection stats.
- **Validate every tool output:** each result passes through 🟣`validate_tool_outputs` (🟢`VALID`/🟢`EMPTY`/🟢`INVALID`/🟢`ERROR`) before 🟣`store_evidence` admits it. LLM tool proposals are also validated before they run.
- **Cite evidence:** RAG passages carry the paper, section and page; report conclusions point to their source; citations come from registry-verified metadata; metadata needs two registries to count as 🟢`VERIFIED`.
- **Explain each decision:** every node logs a one-line reason, e.g. “LLM proposed read_research_paper(paper_id=…): accepted. (Rule would have chosen …)” or “Retrying with reformulated query …”. These lines stream live into the UI.

### Deliverables

> **Brief:** *1. “Runnable source code + clear setup instructions.”*

- The README covers setup (Python 3.11 virtual environment, `pip install -r requirements.txt`, `.env` from `.env.example`, `npm ci`) and how to run it (`python run.py --server`, `npm run dev`).
- 447 tests run fully offline.

> **Brief:** *2. “README: workflow, agent pattern, tools, state, stopping rules, retries, fallbacks.”*

All of these are in README §1–2: the agent pattern (plan-and-execute with reflection and self-correction), a table of every node, the state fields, the seven stopping statuses, the retry rules, a fallback table, the 10 tools and the meta-tools.

> **Brief:** *3. “Architecture diagram (planner → tools → memory → validation → decision).”*

In README §1.2 and `ARCHITECTURE.md` §2, drawn left to right in exactly that order: **Planner** (plan, select, propose) → **Tools** (the 10 tools) → **Memory** (state, ChromaDB, JSON) → **Validation** (validate and retry) → **Decision** (reflect, critique, report, finish), with the retry, “phases left” and “needs revision” loops.

> **Brief:** *4. “One-page evaluation report (success rate, failures, limitations, improvements, agentic behaviour shown).”*

`evaluation/ONE_PAGE_SUMMARY.md` covers each item:

- **Success rate:** 3 of 3 live runs with a recorded final status reached 🟢`GOAL_ACHIEVED`; only 2 of 4 live runs had no quality problems; 18 of 18 offline end-to-end scenarios and 447 of 447 tests pass.
- **Failures:** PDF download failures, off-topic papers, the wrong WER claim, rate limits — each listed with its fix.
- **Limitations:** a small sample, rate-limit fixes not yet confirmed live, heuristic extraction, no OCR.
- **Improvements:** more varied live runs, better section parsing, OCR, durable checkpoints.
- **Agentic behaviour shown:** real numbers from the runs.

### Two things your instructor may ask about

- **“Does the LLM really choose the tools?”** Yes. It proposes each call through native function calling, but only from the legal options, and code validates it (“the model proposes; the application disposes”). The rules remain as validator and fallback, so a wrong or injected proposal can't skip phases, read arbitrary URLs or invent citations. The API Usage panel shows the counts (accepted / rejected / fallback / automatic).
- **“How do you handle API limits?”** Single-shot, time-limited proposals; a circuit breaker; per-provider pauses that honour the provider's own wait times; Semantic Scholar pacing; per-session usage counts in the UI. These fixes have so far been tested offline only, until the next live run.

<div class="legend">

## Legend

GitHub Markdown cannot colour text, so each category is marked with a coloured symbol. The PDF version uses the matching text colour.

| Marker | Category | Meaning |
|---|---|---|
| 🔴 **bold** | Meta-tools | Act on the agent's own process: planner, plan_updater, ask_user, tool selection, critic, finish |
| 🔵 `code` | Execution tools (normal functions) | Act on the outside world: search, metadata validation, PDF reader, RAG retrieval, comparison, methodology, results analysis, gaps, citations, report |
| 🟣 `code` | LangGraph nodes | The steps of the graph, e.g. 🟣`select_tools`, 🟣`execute_tools`, 🟣`retry_handler` |
| 🟢 `code` | Statuses and outcomes | Stopping reasons (🟢`GOAL_ACHIEVED`), validation labels (🟢`VALID`, 🟢`VERIFIED`), tool-choice outcomes (🟢`ACCEPTED`, 🟢`FALLBACK`) |
| 🟠 **bold** | LLMs and external services | Gemini, Groq, the embedding model, OpenAlex, Semantic Scholar, Crossref, ChromaDB |
| 💡 quote | In simple words | A plain-language explanation of the sentence just above it |
| `code` (no marker) | Files, settings, API endpoints | e.g. `research.py`, `max_steps`, `/resume` |
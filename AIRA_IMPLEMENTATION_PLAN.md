# AIRA --- Amharic Intelligent Research Assistant

## Updated Technical Implementation Plan

**Project type:** Agentic AI research assistant\
**Primary framework:** LangGraph\
**LLM providers:** Gemini (primary) + independent fallback provider\
**Programming language:** Python\
**Interface:** Streamlit\
**Database:** SQLite + ChromaDB\
**Development environment:** Google Antigravity\
**Target hardware:** Apple Silicon Mac; no dedicated GPU required

------------------------------------------------------------------------

# 1. Project Overview

AIRA (Amharic Intelligent Research Assistant) is an autonomous research
assistant designed to support academic research in Amharic NLP and other
low-resource languages.

The system accepts a research question, creates a research plan,
searches academic sources, retrieves and analyzes papers, compares
experimental results, identifies potential research gaps, and generates
a structured report with traceable citations.

AIRA uses LangGraph to manage its workflow, state, memory, tool
selection, conditional branches, loops, reflection, and stopping
decisions.

A major design requirement is API resilience: exhausting one LLM
provider's quota must not cause the entire application to fail.

### Example research request

> Compare recent approaches to Amharic sentiment analysis and identify
> limitations in existing datasets and models.

AIRA should independently: 1. Understand the research objective. 2.
Create a research plan. 3. Search academic databases. 4. Validate paper
metadata. 5. Read and extract relevant papers. 6. Store extracted
research in its knowledge base. 7. Retrieve relevant evidence. 8.
Compare methods and reported results. 9. Identify potential research
gaps. 10. Critique its findings. 11. Revise the research plan if
necessary. 12. Generate a cited research report.

# 2. Main Objectives

-   Build a genuinely agentic system using LangGraph.
-   Demonstrate autonomous planning and tool selection.
-   Integrate at least seven execution tools.
-   Implement persistent memory and RAG.
-   Support multiple LLM providers.
-   Recover from API quota exhaustion and tool failures.
-   Validate tool outputs and academic evidence.
-   Demonstrate conditional branches, loops, and self-correction.
-   Produce evidence-based research reports.
-   Maintain detailed execution logs.

# 3. Project Scope

## Initial research domains

-   Amharic sentiment analysis.
-   Amharic named entity recognition.
-   Amharic machine translation.
-   Amharic handwritten text recognition.
-   Low-resource language modeling.

The architecture should allow additional languages such as Afaan Oromo
and Tigrinya in the future.

## Out of scope for the first version

-   Training large language models from scratch.
-   Running expensive GPU-based experiments.
-   Claiming that an identified research gap is scientifically proven.
-   Automatically submitting research papers.
-   Generating fabricated academic citations.

AIRA will initially analyze existing research and user-provided
experimental results.

# 4. Technology Stack

  -----------------------------------------------------------------------
  Component               Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Programming             Python                  Main development
                                                  language

  Agent orchestration     LangGraph               State, branches, loops
                                                  and decisions

  LLM integration         LangChain               Model and tool
                                                  integration

  Primary LLM             Gemini API              Reasoning and
                                                  generation

  Fallback LLM            Groq or OpenRouter      Alternative model
                                                  provider

  Embeddings              Gemini embeddings       Vector representations

  Vector database         ChromaDB                Research knowledge
                                                  retrieval

  Database                SQLite                  Persistent application
                                                  data

  Academic search         OpenAlex                Paper discovery

  Academic search         Semantic Scholar        Paper discovery

  Metadata verification   Crossref                DOI and publication
                                                  validation

  PDF extraction          PyMuPDF                 Extract paper content

  PDF fallback            pdfplumber              Alternative extraction

  Data analysis           Pandas, NumPy           Numerical processing

  ML metrics              scikit-learn            Model evaluation

  Visualization           Matplotlib              Charts

  UI                      Streamlit               User interface

  Validation              Pydantic                Input/output validation

  Testing                 pytest                  Automated testing

  Logging                 Python logging          Execution records

  Observability           LangSmith               Optional tracing
  -----------------------------------------------------------------------

Use compatible, currently supported library versions. Verify current SDK
and model availability during implementation rather than assuming model
names or API behavior.

# 5. LLM Provider Architecture --- Critical Requirement

AIRA must never depend exclusively on Gemini.

## 5.1 Primary provider

Gemini API will handle: - Research planning. - Reasoning. - Evidence
synthesis. - Paper summarization. - Research gap analysis. - Criticism
and reflection. - Report generation.

Use the official Google Gen AI SDK through a LangChain-compatible
integration where appropriate.

## 5.2 Fallback provider

Configure a second independent LLM provider.

Recommended options: - Groq. - OpenRouter.

Select a supported model with a suitable context window and
structured-output capabilities.

Do not assume either service has unlimited free usage.

## 5.3 Model router

Create a centralized model router.

Responsibilities: - Select the primary model. - Detect quota
exhaustion. - Detect provider unavailability. - Detect rate limits. -
Switch to the fallback model. - Record provider changes. - Prevent
unnecessary repeated requests. - Preserve the original task and state.

Suggested provider statuses: - `AVAILABLE` - `RATE_LIMITED` -
`QUOTA_EXHAUSTED` - `TEMPORARILY_UNAVAILABLE` - `AUTHENTICATION_ERROR`

Authentication errors should not trigger endless provider switching.

## 5.4 Provider fallback workflow

1.  An agent node requests LLM reasoning.
2.  The model router calls Gemini.
3.  If Gemini succeeds, continue normally.
4.  If Gemini reaches a quota limit, classify the error.
5.  Switch to the fallback provider.
6.  Retry the same logical task with the fallback model.
7.  Validate the fallback response.
8.  Record the provider change.
9.  Continue the LangGraph workflow.

If the fallback also fails, enter degraded mode.

## 5.5 Degraded mode

When no LLM provider is available, AIRA should continue operations that
do not require LLM reasoning.

Examples: - Search academic APIs. - Download papers. - Extract PDF
text. - Verify DOI metadata. - Generate citations. - Calculate
metrics. - Retrieve previously indexed documents. - Save research
progress. - Export available results.

The system should clearly mark which tasks require unavailable LLM
reasoning.

It must not pretend that deterministic calculation or simple retrieval
is equivalent to LLM reasoning.

If the task cannot safely continue, checkpoint the state and pause for
recovery.

# 6. Seven Execution Tools

At least five execution-tool categories must be represented. Implement
the following seven tools.

## Tool 1: `search_academic_papers`

**Category:** Search

Search: - OpenAlex. - Semantic Scholar. - Crossref.

Inputs: - `query` - `year_from` - `year_to` - `sources` - `max_results`

Outputs: - Paper title. - Authors. - Publication year. - DOI. -
Abstract. - Source. - URL. - Paper identifier.

Requirements: - Merge duplicate papers. - Preserve source information. -
Respect API rate limits. - Handle unavailable sources. - Do not invent
missing abstracts or metadata.

## Tool 2: `read_research_paper`

**Category:** System & Environment

Responsibilities: - Download a research paper. - Extract text from
PDF. - Preserve page numbers. - Extract sections where possible. -
Detect empty or incomplete extraction. - Identify scanned PDFs that may
need OCR.

Libraries: - PyMuPDF. - pdfplumber as fallback.

## Tool 3: `retrieve_research_context`

**Category:** RAG

Responsibilities: - Retrieve relevant paper chunks. - Search the local
ChromaDB collection. - Filter by research topic or paper. - Return
source metadata. - Include page references. - Return relevance scores.

Use: - LangChain text splitters. - ChromaDB. - Gemini embeddings
initially.

Important: implement an embedding-provider abstraction. If Gemini
embeddings are unavailable, use a separately configured local embedding
model or continue querying previously indexed vectors. Never attempt to
compare vectors produced by incompatible embedding models without
re-embedding the collection.

## Tool 4: `analyze_model_results`

**Category:** Code & Compute

Responsibilities: - Calculate accuracy. - Calculate precision. -
Calculate recall. - Calculate macro, micro and weighted F1. - Calculate
CER and WER where applicable. - Compare models. - Generate tables and
charts.

Libraries: - Pandas. - NumPy. - scikit-learn. - Matplotlib.

Important: - Validate input data. - Never compare incompatible
experimental setups as if they were directly comparable. - Distinguish
results reported by papers from results calculated by AIRA. - Do not
invent missing experimental values.

## Tool 5: `validate_paper_metadata`

**Category:** Function Calling

Responsibilities: - Verify DOI. - Check title. - Check authors. - Check
publication year. - Compare information across sources.

Possible statuses: - `VERIFIED` - `PARTIALLY_VERIFIED` - `CONFLICTING` -
`UNVERIFIED`

## Tool 6: `manage_research_files`

**Category:** System & Environment

Actions: - Save paper. - Retrieve paper. - List papers. - Organize
documents. - Save generated reports. - Retrieve previous research
sessions.

Security: - Restrict file operations to designated project
directories. - Validate filenames. - Prevent arbitrary path access.

## Tool 7: `generate_citation`

**Category:** Function Calling

Supported styles: - APA 7. - IEEE. - BibTeX.

Use verified metadata wherever possible.

Never generate missing DOI numbers, author names, publication dates, or
titles.

# 7. Meta-Tools

These tools help the agent manage its workflow rather than directly
retrieve research information.

## `create_research_plan`

Creates an initial task plan based on the research objective.

## `update_research_plan`

Updates the plan when: - A tool fails. - Evidence is insufficient. - A
new research direction is discovered. - Conflicting findings require
further investigation.

The agent must explain why it changed the plan.

## `critique_findings`

Evaluates: - Evidence quality. - Citation validity. - Research
relevance. - Contradictions. - Unsupported conclusions. - Completeness.

Possible results: - `APPROVED` - `NEEDS_REVISION` -
`INSUFFICIENT_EVIDENCE`

## `ask_user`

Requests clarification when essential information is missing.

The workflow should pause and resume after the user responds.

## `finish_research`

Evaluates whether the objective is complete, partially complete,
blocked, or unsafe to continue.

# 8. LangGraph Architecture

Implement a stateful graph with conditional branches and loops.

## Main workflow

1.  `receive_request`
2.  `analyze_research_goal`
3.  `create_plan`
4.  `check_information`
5.  `select_tools`
6.  `execute_tools`
7.  `validate_tool_outputs`
8.  `store_evidence`
9.  `reflect_on_results`
10. `revise_plan`
11. `critique_findings`
12. `generate_report`
13. `final_validation`
14. `finish_task`

## Conditional branches

After checking information: - Missing essential information →
`ask_user`. - Sufficient information → create or continue plan.

After tool execution: - Valid result → store evidence. - Invalid result
→ retry or select alternative. - Tool unavailable → choose another
source or enter degraded mode.

After reflection: - Insufficient evidence → revise plan. - Conflicting
evidence → investigate further. - Sufficient evidence → critique
findings.

After critique: - `APPROVED` → generate report. - `NEEDS_REVISION` →
return to investigation. - `INSUFFICIENT_EVIDENCE` → investigate or
produce a qualified partial report.

At the end: - Complete → finish. - Maximum steps reached → save partial
report. - High-risk situation → stop safely. - Provider unavailable →
checkpoint and pause if reasoning is essential.

The graph must make decisions dynamically. Do not implement a fixed
sequence of seven tool calls.

# 9. State Management

Create a central Pydantic-compatible state schema.

Suggested fields: - `research_goal` - `conversation_history` -
`research_plan` - `completed_tasks` - `current_task` -
`selected_tools` - `papers` - `extracted_documents` -
`metadata_validation_results` - `retrieved_evidence` - `findings` -
`model_comparisons` - `potential_research_gaps` - `tool_results` -
`validation_results` - `errors` - `reflections` - `retry_count` -
`step_count` - `missing_information` - `clarification_status` -
`current_llm_provider` - `provider_status` - `api_failure_history` -
`degraded_mode` - `task_status` - `final_report`

All graph nodes must update state deliberately.

Do not allow tool outputs to silently overwrite validated evidence.

# 10. Memory Architecture

Implement three forms of memory.

## Working memory

Stored in LangGraph state.

Contains: - Current plan. - Current task. - Tool results. - Evidence. -
Retry count. - Current execution status.

## Persistent application memory

Stored in SQLite.

Contains: - Research sessions. - User requests. - Saved papers. -
Metadata. - Generated reports. - Execution summaries.

## Vector memory

Stored in ChromaDB.

Contains: - Extracted paper chunks. - Embeddings. - Paper identifiers. -
Page numbers. - Section names. - Source metadata.

Use LangGraph checkpointing with SQLite so an interrupted research task
can resume.

Persistent memory must not treat old research results as automatically
current. Reverify important metadata and time-sensitive claims.

# 11. Evidence and Citation Management

Every important research finding should have traceable evidence.

Store: - Paper identifier. - Title. - Authors. - Year. - DOI. - Source
URL. - Page number. - Section. - Relevant passage. - Metadata
verification status.

Distinguish three types of information:

1.  **VERIFIED PAPER EVIDENCE** --- directly supported by an accessible
    source.
2.  **COMPUTED RESULT** --- calculated from supplied data.
3.  **AGENT INFERENCE** --- a conclusion generated by reasoning over
    evidence.

Potential research gaps must be described as potential gaps, not proven
gaps.

If evidence is unavailable, explicitly state that.

# 12. Reliability and Recovery Requirements

This section is mandatory.

## Retry policy

-   Maximum two retries per failed operation.
-   Each retry must use a corrected query, alternative source, or
    different valid approach.
-   Do not repeatedly execute an identical failed request.
-   Record every retry and its reason.

## API failure handling

Handle: - Quota exhaustion. - Rate limits. - Authentication errors. -
Network timeouts. - Provider outages. - Invalid model responses. -
Malformed structured outputs.

Use bounded exponential backoff for temporary failures.

Do not retry authentication errors or exhausted quotas as though they
were temporary network failures.

## API fallback

-   Gemini is the primary provider.
-   A second provider is configured independently.
-   Model switching is centralized.
-   Fallback outputs are validated.
-   Provider usage is logged.

## Tool fallback

Examples: - OpenAlex unavailable → try Semantic Scholar. - Semantic
Scholar unavailable → try Crossref where suitable. - PyMuPDF extraction
fails → try pdfplumber. - Primary embedding provider unavailable → use
configured local embeddings or pause new indexing. - LLM provider
unavailable → use fallback or degraded mode.

## Stopping rules

Stop when: - The research goal is sufficiently complete. - Essential
information is missing. - Evidence cannot be reliably established. -
Risk is too high. - Maximum graph steps are reached. - All suitable LLM
providers are unavailable and reasoning is necessary.

Suggested initial limits: - Maximum graph steps: 20. - Maximum retries
per operation: 2. - Maximum papers per search: 20. - Configurable
execution timeout.

Do not interpret these limits as permission to generate unsupported
conclusions.

# 13. Failure Simulation and Testing

AIRA must be tested against deliberately unreliable conditions.

## Test 1: Gemini quota exhaustion

Simulate a quota error.

Expected: - Error is classified. - Fallback provider is selected. - Task
continues. - Provider switch is logged.

## Test 2: Both providers unavailable

Expected: - Enter degraded mode. - Continue deterministic operations. -
Save progress. - Pause if LLM reasoning is essential.

## Test 3: Missing research information

Expected: - Ask the user for clarification. - Do not invent missing
details.

## Test 4: Conflicting academic metadata

Expected: - Compare sources. - Mark conflict. - Avoid presenting
unverified information as fact.

## Test 5: Outdated research

Expected: - Detect that the requested date range is not covered. -
Revise the search. - Include publication dates in the final report.

## Test 6: Tool timeout

Expected: - Retry within the defined limit. - Try an alternative source
if available. - Record the failure.

## Test 7: Prompt injection in a PDF

Expected: - Treat paper content as untrusted data. - Never follow
instructions embedded in a paper. - Continue extracting relevant
academic content safely.

## Test 8: Invalid numerical data

Expected: - Reject malformed input. - Explain the validation failure. -
Avoid producing misleading metrics.

Use pytest and mocked API responses for reproducible testing. Do not
depend on exhausting real API quotas during a presentation.

# 14. Execution Logging and Observability

Every important agent decision must be recorded.

Log fields: - Timestamp. - Session ID. - Graph node. - Research
objective. - Current plan. - Tool selected. - Tool selection reason. -
Tool arguments, with secrets removed. - Tool result summary. -
Validation result. - Retry count. - Reflection. - State update. -
Current LLM provider. - Provider switch reason. - Next decision. - Stop
reason.

Use LangSmith optionally for visual tracing.

The logs must prove: - The agent created a plan. - The agent selected
tools. - The agent revised its plan. - The agent recovered from a
failure. - The agent updated its state. - The agent made a stopping
decision.

Never log API keys or other secrets.

# 15. User Interface

Build a Streamlit application.

## Main sections

### Research Workspace

-   Enter research question.
-   Configure date range.
-   Select research topic.
-   Start research.
-   Display final report.

### Execution Monitor

-   Current graph node.
-   Current plan.
-   Completed tasks.
-   Selected tools.
-   Current LLM provider.
-   Retry information.
-   API failure and recovery events.
-   Degraded mode indicator.

### Research Library

-   Saved papers.
-   Paper metadata.
-   Search previous papers.
-   View extracted text.

### Analysis

-   Model comparison tables.
-   Metric calculations.
-   Charts.

### Research Gap Analysis

-   Identified limitations.
-   Supporting evidence.
-   Confidence and evidence status.
-   Potential research directions.

### Reports

-   Research summary.
-   Methodology comparison.
-   Findings.
-   Limitations.
-   Citations.
-   Export to Markdown and PDF.

# 16. Project Directory Structure

``` text
aira/
│
├── app/
│   ├── main.py
│   └── ui/
│       ├── workspace.py
│       ├── execution_monitor.py
│       ├── research_library.py
│       └── reports.py
│
├── agents/
│   ├── graph.py
│   ├── nodes.py
│   ├── routing.py
│   ├── state.py
│   └── prompts.py
│
├── llm/
│   ├── model_router.py
│   ├── gemini_provider.py
│   ├── fallback_provider.py
│   ├── provider_errors.py
│   └── usage_tracker.py
│
├── tools/
│   ├── academic_search.py
│   ├── paper_reader.py
│   ├── rag_retriever.py
│   ├── result_analyzer.py
│   ├── metadata_validator.py
│   ├── file_manager.py
│   └── citation_generator.py
│
├── meta_tools/
│   ├── planner.py
│   ├── plan_updater.py
│   ├── critic.py
│   ├── ask_user.py
│   └── finish.py
│
├── memory/
│   ├── database.py
│   ├── checkpointing.py
│   ├── vector_store.py
│   └── repositories.py
│
├── services/
│   ├── evidence_service.py
│   ├── report_service.py
│   └── embedding_service.py
│
├── validation/
│   ├── tool_validation.py
│   ├── evidence_validation.py
│   └── schemas.py
│
├── logging_config/
│   └── logger.py
│
├── data/
│   ├── papers/
│   ├── processed/
│   ├── reports/
│   └── chroma/
│
├── tests/
│   ├── test_tools.py
│   ├── test_graph.py
│   ├── test_model_router.py
│   ├── test_api_fallback.py
│   ├── test_memory.py
│   ├── test_reliability.py
│   └── test_prompt_injection.py
│
├── evaluation/
│   ├── test_cases.json
│   └── evaluation_report.md
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── run.py
```

# 17. Implementation Phases

Build and test the project phase by phase.

## Phase 1: Project foundation

-   Create project structure.
-   Create virtual environment.
-   Install dependencies.
-   Configure environment variables.
-   Create initial README.
-   Verify Python environment.

**Deliverable:** application starts successfully.

## Phase 2: LLM provider integration

-   Integrate Gemini.
-   Integrate fallback provider.
-   Implement centralized model router.
-   Classify provider errors.
-   Implement quota detection.
-   Implement provider switching.
-   Create mocked failure tests.

**Deliverable:** working multi-provider LLM service.

## Phase 3: Academic search tools

-   Integrate OpenAlex.
-   Integrate Semantic Scholar.
-   Integrate Crossref.
-   Implement paper deduplication.
-   Validate metadata.
-   Test API failures.

**Deliverable:** working academic discovery and metadata validation.

## Phase 4: Paper reading and citation tools

-   Download papers.
-   Extract PDF text.
-   Preserve page numbers.
-   Implement extraction fallback.
-   Generate APA and IEEE citations.
-   Validate citations.

**Deliverable:** working paper-reading and citation tools.

## Phase 5: RAG and persistent memory

-   Implement document chunking.
-   Generate embeddings.
-   Configure ChromaDB.
-   Store paper metadata.
-   Implement retrieval.
-   Configure SQLite.
-   Implement LangGraph checkpointing.

**Deliverable:** searchable persistent research knowledge base.

## Phase 6: Analysis tools

-   Implement numerical metric calculations.
-   Implement model comparison.
-   Add input validation.
-   Add visualizations.
-   Test calculations against known values.

**Deliverable:** working research analysis tools.

## Phase 7: LangGraph agent

-   Define agent state.
-   Implement graph nodes.
-   Implement planner.
-   Implement tool selection.
-   Implement conditional routing.
-   Implement loops.
-   Implement reflection.
-   Implement critic.
-   Implement stopping rules.

**Deliverable:** autonomous research workflow.

## Phase 8: Reliability and recovery

-   Implement retry limits.
-   Implement API fallback.
-   Implement degraded mode.
-   Implement checkpoint recovery.
-   Implement tool fallback.
-   Test all failure scenarios.

**Deliverable:** resilient agent with demonstrated recovery.

## Phase 9: Logging and monitoring

-   Implement structured logs.
-   Record decisions.
-   Record tool selection.
-   Record plan revisions.
-   Record API switches.
-   Record state changes.
-   Integrate optional LangSmith tracing.

**Deliverable:** inspectable agent execution.

## Phase 10: Streamlit interface

-   Build research workspace.
-   Build execution monitor.
-   Build research library.
-   Build analysis interface.
-   Build report export.

**Deliverable:** usable application.

## Phase 11: Evaluation and documentation

-   Run all automated tests.
-   Run failure simulations.
-   Evaluate research accuracy.
-   Evaluate citation validity.
-   Evaluate recovery.
-   Prepare architecture diagram.
-   Write README.
-   Prepare one-page evaluation report.
-   Prepare presentation demo.

**Deliverable:** complete submission package.

# 18. Evaluation Metrics

Do not invent evaluation results. Run experiments and report actual
measurements.

Evaluate:

  -----------------------------------------------------------------------
  Metric                              What it measures
  ----------------------------------- -----------------------------------
  Task success rate                   Percentage of successfully
                                      completed research tasks

  Search success rate                 Successful academic search
                                      operations

  Metadata accuracy                   Correctness of verified paper
                                      metadata

  PDF extraction success              Successful text extraction

  Retrieval relevance                 Relevance of retrieved paper
                                      passages

  Citation validity                   Whether citations point to real
                                      sources

  Numerical correctness               Correctness of calculated metrics

  Fallback success rate               Recovery when primary provider
                                      fails

  Tool recovery rate                  Recovery from tool errors

  Unsupported claim rate              Findings without adequate evidence

  Average execution time              Time required per research task

  Average tool calls                  Number of tool executions per task
  -----------------------------------------------------------------------

Include: - Test conditions. - Number of test cases. - Actual results. -
Failure examples. - Limitations. - Future improvements.

# 19. Demonstration Scenario

Prepare a repeatable presentation scenario.

**Research request:**

> Analyze recent approaches to Amharic sentiment analysis, compare the
> reported models and identify potential research gaps.

Demonstrate:

1.  AIRA receives the research question.
2.  The planner creates a research plan.
3.  The agent selects academic search tools.
4.  Papers are retrieved and metadata is validated.
5.  The paper reader extracts text.
6.  Documents are indexed in ChromaDB.
7.  RAG retrieves relevant evidence.
8.  The analysis tool compares reported metrics.
9.  The critic checks findings and citations.
10. AIRA generates a research report.

Then demonstrate a deliberate failure:

1.  Simulate Gemini quota exhaustion.
2.  Show the model router detecting the failure.
3.  Show the switch to the alternative provider.
4.  Continue the research task.
5.  Show the execution log and preserved state.

Finally, simulate both LLM providers being unavailable:

1.  Enter degraded mode.
2.  Show local retrieval and deterministic tools still working.
3.  Save the checkpoint.
4.  Explain which reasoning-dependent tasks are paused.

Use mocked failures for reliable demonstration rather than deliberately
exhausting real API quotas.

# 20. Security Requirements

-   Store API keys in `.env`.
-   Never commit `.env`.
-   Validate all user inputs.
-   Restrict filesystem operations.
-   Use HTTP timeouts.
-   Respect API rate limits.
-   Treat paper content as untrusted.
-   Prevent prompt injection from controlling tools.
-   Never execute arbitrary LLM-generated shell commands.
-   Do not expose secrets in logs.
-   Validate tool outputs before they enter trusted state.

# 21. Environment Configuration

Create `.env.example`:

``` env
# Primary LLM
GEMINI_API_KEY=
GEMINI_MODEL=

# Fallback LLM
FALLBACK_PROVIDER=groq
GROQ_API_KEY=
GROQ_MODEL=

# Academic APIs
OPENALEX_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
CROSSREF_MAILTO=

# Optional tracing
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=aira-research-agent

# Database
DATABASE_URL=sqlite:///./aira.db
CHROMA_PERSIST_DIRECTORY=./data/chroma

# Agent limits
MAX_AGENT_STEPS=20
MAX_RETRIES=2
MAX_SEARCH_RESULTS=20
```

Only configure credentials for services actually used. Semantic Scholar
and Crossref can initially be used without separate paid subscriptions
or API keys, subject to their access limits and policies.

# 22. Required Deliverables

1.  Runnable Python source code.
2.  Requirements file.
3.  Environment example.
4.  Setup instructions.
5.  README.
6.  LangGraph architecture diagram.
7.  Seven working execution tools.
8.  Meta-tools.
9.  Multi-provider model router.
10. API quota recovery.
11. Degraded mode.
12. Persistent memory.
13. Automated tests.
14. Failure simulation results.
15. One-page evaluation report.
16. Working Streamlit application.
17. Demonstration scenario.

# 23. Instructions for Google Antigravity

Use Antigravity as the primary development agent.

Do not ask it to implement the entire application in one response.

Follow these rules:

1.  First inspect the project requirements and propose a detailed
    architecture.
2.  Create the directory structure.
3.  Implement one phase at a time.
4.  Before moving to the next phase, run the relevant tests.
5.  Fix errors before continuing.
6.  Keep dependencies compatible.
7.  Use current official documentation for library integrations.
8.  Do not invent API endpoints or model names.
9.  Do not hardcode API keys.
10. Do not use fake research results in the final application.
11. Keep the code modular.
12. Explain important code and design decisions.
13. Maintain a list of completed and remaining tasks.
14. Update the README as implementation progresses.
15. Ensure the project can run on an Apple Silicon Mac without a
    dedicated GPU.

For the LLM fallback, use mocked provider failures during automated
testing.

Do not proceed to the next phase until the current phase passes its
tests.

At the end of each phase, provide: - Files created or modified. -
Features implemented. - Commands to run. - Tests executed. - Test
results. - Known issues. - Remaining tasks.

**Most important:** The final application must demonstrate actual
autonomous decisions through LangGraph. It must not be a fixed pipeline
disguised as an agent.

AIRA should be able to plan, select tools, inspect results, reflect,
revise its approach, recover from failures, and decide when to stop.

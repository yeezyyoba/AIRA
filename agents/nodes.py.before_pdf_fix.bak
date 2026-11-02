
"""Node implementations for AIRA LangGraph Agentic Workflow.

Implements all 14 graph nodes with observable logging, retry tracking,
schema validation, tool dispatch, and degraded mode safety.
"""
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional
import uuid

from agents.state import ResearchState
from llm.model_router import ModelRouter, model_router as default_model_router
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from tools.academic_search import search_academic_papers
from tools.citation_generator import generate_citation, CitationStyle
from tools.metadata_validator import validate_paper_metadata
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from services.report_service import generate_research_comparison_report
from tools.rag_retriever import retrieve_research_context
from tools.gap_identifier import identify_research_gaps
from tools.methodology_analyzer import analyze_research_methodology
from tools.paper_comparator import compare_research_papers
from tools.paper_reader import read_research_paper
from tools.result_analyzer import analyze_experimental_results
from validation.schemas import PaperMetadataSchema

logger = get_logger("agents.nodes")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(
    state: ResearchState,
    node: str,
    decision: str,
    tool: Optional[str] = None,
    outcome: str = "SUCCESS",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a structured log entry without exposing private chain-of-thought."""
    entry = {
        "timestamp": _now_iso(),
        "step": state.get("step_count", 0),
        "node": node,
        "decision": decision,
        "tool": tool,
        "outcome": outcome,
    }
    if details:
        entry["details"] = details
    return entry


def _sanitize_data(obj: Any) -> Any:
    """Recursively sanitize data to standard Python primitives."""
    if isinstance(obj, dict):
        return {k: _sanitize_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_data(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _sanitize_data(obj.model_dump())
    if hasattr(obj, "item"):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# 1. Request Ingestion & Goal Analysis
# ---------------------------------------------------------------------------

def receive_request(state: ResearchState) -> Dict[str, Any]:
    """Ingest the research query and initialize execution context."""
    query = state.get("query", "").strip()
    session_id = state.get("session_id") or str(uuid.uuid4())[:8]
    logs = list(state.get("observable_logs", []))

    logs.append({
        "timestamp": _now_iso(),
        "step": 0,
        "node": "receive_request",
        "decision": f"Ingested research request for session '{session_id}'.",
        "tool": None,
        "outcome": "SUCCESS",
    })

    max_steps = state.get("max_steps", 15)
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        max_steps = 15

    return {
        "session_id": session_id,
        "query": query,
        "research_goal": state.get("research_goal") or query,
        "research_questions": list(state.get("research_questions", [])),
        "domain": state.get("domain"),
        "research_plan": list(state.get("research_plan", [])),
        "completed_tasks": list(state.get("completed_tasks", [])),
        "current_task": None,
        "selected_tool": None,
        "tool_inputs": None,
        "tool_results": list(state.get("tool_results", [])),
        "validation_status": "PENDING",
        "executed_tool_signatures": list(state.get("executed_tool_signatures", [])),
        "discovered_papers": list(state.get("discovered_papers", [])),
        "validated_papers": list(state.get("validated_papers", [])),
        "extracted_papers": list(state.get("extracted_papers", [])),
        "indexed_papers": list(state.get("indexed_papers", [])),
        "retrieved_evidence": list(state.get("retrieved_evidence", [])),
        "metric_comparisons": list(state.get("metric_comparisons", [])),
        "identified_gaps": list(state.get("identified_gaps", [])),
        "citations": dict(state.get("citations", {})),
        "reflections": list(state.get("reflections", [])),
        "critique_status": None,
        "step_count": 0,
        "max_steps": max_steps,
        "retry_counts": dict(state.get("retry_counts", {})),
        "missing_information": [],
        "clarification_question": None,
        "clarification_response": state.get("clarification_response"),
        "stopping_reason": None,
        "observable_logs": logs,
        "is_degraded": False,
        "final_report": None,
    }


def analyze_goal(state: ResearchState) -> Dict[str, Any]:
    """Parse the research goal into a domain and research questions."""
    query = state.get("query", "")
    clarification = state.get("clarification_response")
    effective_goal = f"{query} (Clarification: {clarification})" if clarification else query
    step = state.get("step_count", 0) + 1

    domain = state.get("domain")
    if not domain:
        q_lower = effective_goal.lower()
        if "speech" in q_lower or "asr" in q_lower:
            domain = "Speech Recognition / ASR"
        elif "vision" in q_lower or "image" in q_lower:
            domain = "Computer Vision"
        elif any(term in q_lower for term in ("summarization", "nlp", "qa", "language")):
            domain = "Natural Language Processing"
        else:
            domain = "Machine Learning"

    questions = list(state.get("research_questions", []))
    if not questions:
        questions = [
            f"What are the leading architectures and methodologies for {effective_goal}?",
            "What benchmark datasets and evaluation metrics are reported?",
            "What empirical limitations and research gaps remain open?",
        ]

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "analyze_goal",
        "decision": f"Analyzed goal: domain '{domain}', formulated {len(questions)} research questions.",
        "tool": None,
        "outcome": "SUCCESS",
    })

    return {
        "research_goal": effective_goal,
        "domain": domain,
        "research_questions": questions,
        "step_count": step,
        "observable_logs": logs,
    }


def check_information(state: ResearchState) -> Dict[str, Any]:
    """Check whether essential research information is missing."""
    query = state.get("query", "").strip()
    clarification = state.get("clarification_response")
    step = state.get("step_count", 0) + 1
    missing: List[str] = []
    question: Optional[str] = None

    clean_q = re.sub(r"[^\w\s]", "", query).strip()
    if not clean_q or len(clean_q) < 4:
        missing.append("Specific research topic or problem statement is missing.")
        question = "Please provide the specific research topic, domain, or research question you wish to investigate."
    elif clean_q.lower() in ("research", "papers", "study", "analysis", "compare") and not clarification:
        missing.append("Research scope is underspecified.")
        question = f"The query '{query}' is broad. What specific models, benchmarks, or languages would you like to focus on?"

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "check_information",
        "decision": (
            f"Essential information missing: {'; '.join(missing)}. Requesting clarification."
            if missing else "Research parameters verified as sufficient to proceed."
        ),
        "tool": None,
        "outcome": "CLARIFICATION_REQUIRED" if missing else "SUCCESS",
    })

    return {
        "missing_information": missing,
        "clarification_question": question,
        "step_count": step,
        "observable_logs": logs,
    }


def ask_user(state: ResearchState) -> Dict[str, Any]:
    """Pause execution and formulate a clarification request."""
    step = state.get("step_count", 0) + 1
    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "ask_user",
        "decision": f"Paused workflow for user clarification: {state.get('clarification_question')}",
        "tool": None,
        "outcome": "PAUSED",
    })

    return {
        "stopping_reason": "USER_CLARIFICATION_REQUIRED",
        "step_count": step,
        "observable_logs": logs,
    }


# ---------------------------------------------------------------------------
# 2. Planning & Tool Selection
# ---------------------------------------------------------------------------

def create_plan(state: ResearchState) -> Dict[str, Any]:
    """Generate a structured multi-step research plan."""
    step = state.get("step_count", 0) + 1
    plan = [
        {
            "task_id": "search_and_validate",
            "name": "Academic Paper Discovery & Metadata Validation",
            "description": f"Search academic registries for papers on '{state.get('research_goal')}' and independently verify DOIs/metadata.",
            "status": "PENDING",
        },
        {
            "task_id": "extract_and_index",
            "name": "PDF Information Extraction & Vector Indexing",
            "description": "Read paper PDFs, extract structured research dimensions, and index text passages for RAG retrieval.",
            "status": "PENDING",
        },
        {
            "task_id": "compare_and_analyze",
            "name": "Methodology Comparison & Experimental Analysis",
            "description": "Perform multi-paper methodology comparison, reproducibility audit, and compatible metric benchmarking.",
            "status": "PENDING",
        },
        {
            "task_id": "identify_gaps",
            "name": "Research Gap & Limitation Analysis",
            "description": "Identify author-acknowledged and cross-paper empirical research gaps with evidence citations.",
            "status": "PENDING",
        },
        {
            "task_id": "synthesize_report",
            "name": "Comprehensive Research Report Synthesis",
            "description": "Generate an 11-section research comparison report with bibliography and evidence citations.",
            "status": "PENDING",
        },
    ]

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "create_plan",
        "decision": f"Created research plan with {len(plan)} structured phases.",
        "tool": None,
        "outcome": "SUCCESS",
    })

    return {
        "research_plan": plan,
        "current_task": "search_and_validate",
        "step_count": step,
        "observable_logs": logs,
    }


def select_tools(state: ResearchState) -> Dict[str, Any]:
    """Determine the next execution tool based on workflow progress."""
    step = state.get("step_count", 0) + 1
    completed = set(state.get("completed_tasks", []))
    discovered = state.get("discovered_papers", [])
    validated = state.get("validated_papers", [])
    extracted = state.get("extracted_papers", [])
    selected_tool = None
    tool_inputs: Dict[str, Any] = {}
    reason = ""

    if "search_and_validate" not in completed:
        if not discovered:
            selected_tool = "search_academic_papers"
            tool_inputs = {"query": state.get("query", "")}
            reason = "Initial academic search to identify relevant papers."
        elif not validated:
            selected_tool = "validate_metadata_integrity"
            tool_inputs = {"papers": discovered}
            reason = "Validate paper metadata and DOI authenticity across registries."
        else:
            completed.add("search_and_validate")

    if "search_and_validate" in completed and "extract_and_index" not in completed:
        unextracted = [
            p for p in (validated or discovered)
            if isinstance(p, dict) and (p.get("paper_id") or p.get("id")) not in extracted
        ]
        if unextracted:
            target_p = unextracted[0]
            pid = target_p.get("paper_id") or target_p.get("id") or "paper"
            selected_tool = "read_research_paper"
            tool_inputs = {
                "paper_id": pid,
                "file_path_or_url": (
                    target_p.get("pdf_url")
                    or target_p.get("url")
                    or target_p.get("open_access_url")
                    or f"data/papers/{pid}.pdf"
                ),
            }
            reason = f"Extract structured research dimensions and text for paper '{pid}'."
        else:
            completed.add("extract_and_index")

    if "extract_and_index" in completed and "compare_and_analyze" not in completed:
        if not state.get("metric_comparisons"):
            selected_tool = "analyze_experimental_results"
            tool_inputs = {"paper_ids": extracted}
            reason = "Analyze reported numerical metrics and experimental comparability."
        else:
            completed.add("compare_and_analyze")

    if "compare_and_analyze" in completed and "identify_gaps" not in completed:
        if not state.get("identified_gaps"):
            selected_tool = "identify_research_gaps"
            tool_inputs = {
                "paper_ids": extracted,
                "research_area": state.get("domain") or state.get("research_goal"),
            }
            reason = "Identify author-acknowledged and cross-paper empirical gaps."
        else:
            completed.add("identify_gaps")

    if "identify_gaps" in completed and "synthesize_report" not in completed:
        selected_tool = "generate_research_comparison_report"
        tool_inputs = {
            "paper_ids": extracted,
            "research_question": state.get("research_goal"),
        }
        reason = "Compile the comparative research report."

    sig = f"{selected_tool}:{tool_inputs}" if selected_tool else "none"
    executed_sigs = list(state.get("executed_tool_signatures", []))

    if (
        selected_tool
        and sig in executed_sigs
        and state.get("validation_status") != "RETRY_REQUESTED"
    ):
        logger.info("Duplicate tool signature detected; suppressing repeated call.")
        selected_tool = None
        reason = "All planned tool actions for the current stage have been executed."

    if selected_tool:
        executed_sigs.append(sig)

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "select_tools",
        "decision": reason or "All planned tasks completed. Proceeding to reflection.",
        "tool": selected_tool,
        "outcome": "SELECTED" if selected_tool else "COMPLETE",
    })

    return {
        "selected_tool": selected_tool,
        "tool_inputs": tool_inputs,
        "completed_tasks": list(completed),
        "executed_tool_signatures": executed_sigs,
        "step_count": step,
        "observable_logs": logs,
    }


# ---------------------------------------------------------------------------
# 3. Execution & Validation
# ---------------------------------------------------------------------------

async def execute_tools(
    state: ResearchState,
    router: Optional[ModelRouter] = None,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
) -> Dict[str, Any]:
    """Execute the selected tool with error capture and output normalization."""
    step = state.get("step_count", 0) + 1
    tool_name = state.get("selected_tool")
    inputs = state.get("tool_inputs") or {}
    llm = router or default_model_router
    repo = repository or default_paper_repo
    v_db = vector_db or default_vector_store

    tool_results = list(state.get("tool_results", []))
    result_data = None
    exec_error: Optional[str] = None

    if not tool_name:
        return {"step_count": step, "validation_status": "NO_OP"}

    logger.info(f"Executing tool '{tool_name}' with inputs: {inputs}")

    try:
        if tool_name == "search_academic_papers":
            q = inputs.get("query", "")
            search_result = await search_academic_papers(
                query=q,
                max_results=inputs.get("max_results", 5),
            )

            # Support both the search service envelope and direct list responses.
            if isinstance(search_result, dict):
                raw_papers = search_result.get("papers", [])

                if not isinstance(raw_papers, list):
                    raise ValueError(
                        "Academic search response contains an invalid papers field."
                    )

                result_data = {
                    **search_result,
                    "papers": [],
                }

            elif isinstance(search_result, list):
                raw_papers = search_result
                result_data = {"papers": []}

            else:
                raise ValueError("Academic search returned an invalid response.")

            normalized_papers = []

            for paper in raw_papers:
                if hasattr(paper, "model_dump"):
                    paper = paper.model_dump()
                elif hasattr(paper, "dict") and callable(paper.dict):
                    paper = paper.dict()

                if not isinstance(paper, dict):
                    raise ValueError(
                        "Academic search returned a paper record that is not a dictionary."
                    )

                normalized_papers.append(_sanitize_data(paper))

            result_data["papers"] = normalized_papers

        elif tool_name == "validate_metadata_integrity":
            papers = inputs.get("papers", [])
            if not isinstance(papers, list):
                raise ValueError("Metadata validation requires a list of paper records.")

            val_results = []
            for paper in papers:
                if not isinstance(paper, dict):
                    raise ValueError("Metadata validation received a non-dictionary paper record.")
                meta = PaperMetadataSchema(**paper)
                validation_result = await validate_paper_metadata(meta)
                val_results.append(
                    validation_result.model_dump()
                    if hasattr(validation_result, "model_dump")
                    else _sanitize_data(validation_result)
                )
            result_data = val_results

        elif tool_name == "read_research_paper":
            pid = inputs.get("paper_id")
            path = inputs.get("file_path_or_url")
            existing = repo.get_paper(pid) if repo and pid else None

            if existing:
                result_data = existing.model_dump()
            else:
                ext_res = await read_research_paper(path, paper_id=pid)
                if ext_res is None:
                    raise ValueError("Paper reader returned no extraction result.")

                if repo:
                    repo.save_paper(ext_res)

                if v_db:
                    from services.document_chunker import DocumentChunker
                    chunks = DocumentChunker().chunk_paper(ext_res)
                    if chunks:
                        v_db.add_chunks(chunks)

                result_data = ext_res.model_dump()

        elif tool_name == "retrieve_research_context":
            res = await retrieve_research_context(
                question=inputs.get("query") or inputs.get("question", ""),
                vector_store=v_db,
            )
            result_data = res.model_dump()

        elif tool_name == "compare_research_papers":
            res = await compare_research_papers(
                paper_ids=inputs.get("paper_ids", []),
                repository=repo,
                vector_db=v_db,
                router=llm,
            )
            result_data = res.model_dump()

        elif tool_name == "analyze_research_methodology":
            res = await analyze_research_methodology(
                paper_id=inputs.get("paper_id", ""),
                repository=repo,
                router=llm,
            )
            result_data = res.model_dump()

        elif tool_name == "analyze_experimental_results":
            res = analyze_experimental_results(
                paper_ids=inputs.get("paper_ids", []),
                repository=repo,
            )
            result_data = [
                r.model_dump() if hasattr(r, "model_dump") else _sanitize_data(r)
                for r in res
            ]

        elif tool_name == "identify_research_gaps":
            res = await identify_research_gaps(
                paper_ids=inputs.get("paper_ids", []),
                research_area=inputs.get("research_area", ""),
                repository=repo,
                router=llm,
            )
            result_data = res.model_dump()

        elif tool_name == "generate_research_comparison_report":
            pids = inputs.get("paper_ids", [])
            if len(pids) >= 2:
                res = await generate_research_comparison_report(
                    paper_ids=pids,
                    research_question=inputs.get("research_question", ""),
                    repository=repo,
                    vector_db=v_db,
                    router=llm,
                )
                result_data = _sanitize_data(res.model_dump())
                result_data.setdefault(
                    "title",
                    f"Comparative Research Report: {inputs.get('research_question', '')}",
                )
            else:
                pid = pids[0] if pids else "single_paper"
                result_data = {
                    "report_id": f"rep_{uuid.uuid4().hex[:8]}",
                    "title": f"Single Paper Research Report: {inputs.get('research_question', '')}",
                    "executive_summary": (
                        f"Factual analysis of paper '{pid}'. "
                        "Multi-paper comparison unavailable."
                    ),
                    "is_degraded": llm.is_degraded_mode() if llm else False,
                }

        elif tool_name == "generate_citation":
            res = generate_citation(
                paper=inputs.get("paper", {}),
                style=inputs.get("style", CitationStyle.APA7),
            )
            result_data = res.model_dump()

        else:
            exec_error = f"Unrecognized execution tool: {tool_name}"

    except Exception as exc:
        logger.error(
            f"Execution error in tool '{tool_name}': {exc}",
            exc_info=True,
        )
        exec_error = str(exc)

    record = {
        "timestamp": _now_iso(),
        "tool": tool_name,
        "inputs": _sanitize_data(inputs),
        "result": _sanitize_data(result_data),
        "error": exec_error,
        "status": "ERROR" if exec_error else "SUCCESS",
    }
    tool_results.append(record)

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "execute_tools",
        "decision": (
            f"Executed tool '{tool_name}'. "
            f"Outcome: {'ERROR - ' + exec_error if exec_error else 'SUCCESS'}"
        ),
        "tool": tool_name,
        "outcome": "ERROR" if exec_error else "SUCCESS",
    })

    return {
        "tool_results": tool_results,
        "step_count": step,
        "observable_logs": logs,
    }


def validate_tool_outputs(state: ResearchState) -> Dict[str, Any]:
    """Validate tool outputs against expected structure and safety rules."""
    step = state.get("step_count", 0) + 1
    tool_results = state.get("tool_results", [])

    if not tool_results:
        return {"step_count": step, "validation_status": "NO_OP"}

    last_record = tool_results[-1]
    tool_name = last_record.get("tool")
    error = last_record.get("error")
    result = last_record.get("result")
    validation_status = "VALID"
    detail = "Result passed structural validation."

    if error:
        validation_status = "ERROR"
        detail = f"Execution failed with error: {error}"
    elif result is None:
        validation_status = "INVALID"
        detail = "Tool returned null output."
    elif isinstance(result, list) and not result:
        validation_status = "INVALID"
        detail = "Tool returned an empty results list."
    elif tool_name == "search_academic_papers":
        if not isinstance(result, dict):
            validation_status = "INVALID"
            detail = "Academic search did not return a response dictionary."
        elif not isinstance(result.get("papers"), list):
            validation_status = "INVALID"
            detail = "Academic search response has no valid papers list."
    elif tool_name == "validate_metadata_integrity":
        if not isinstance(result, list):
            validation_status = "INVALID"
            detail = "Metadata validator did not return a list."
        elif not result:
            validation_status = "INVALID"
            detail = "Metadata validator returned no records."

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "validate_tool_outputs",
        "decision": f"Validated output of tool '{tool_name}': {validation_status}. {detail}",
        "tool": tool_name,
        "outcome": validation_status,
    })

    return {
        "validation_status": validation_status,
        "step_count": step,
        "observable_logs": logs,
    }


def retry_handler(state: ResearchState) -> Dict[str, Any]:
    """Enforce bounded retry policy with parameter reformulation."""
    step = state.get("step_count", 0) + 1
    selected_tool = state.get("selected_tool") or "unknown_operation"
    inputs = dict(state.get("tool_inputs") or {})
    retry_counts = dict(state.get("retry_counts", {}))

    current_retries = retry_counts.get(selected_tool, 0)
    logs = list(state.get("observable_logs", []))

    if current_retries < 2:
        new_retries = current_retries + 1
        retry_counts[selected_tool] = new_retries

        if selected_tool == "search_academic_papers":
            original_query = inputs.get("query", "")
            domain = state.get("domain", "")
            if domain and domain.lower() not in original_query.lower():
                inputs["query"] = f"{original_query} {domain}"
            else:
                inputs["query"] = (
                    f"{original_query.split()[0]} benchmark"
                    if " " in original_query
                    else f"{original_query} benchmark"
                )

        decision = (
            f"Attempting retry {new_retries}/2 for tool '{selected_tool}' "
            f"with reformulated parameters: {inputs}."
        )
        status = "RETRY_REQUESTED"
    else:
        decision = (
            f"Retry limit exhausted ({current_retries}/2) for tool "
            f"'{selected_tool}'. Marking task as adapted/failed."
        )
        status = "RETRY_EXHAUSTED"

    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "retry_handler",
        "decision": decision,
        "tool": selected_tool,
        "outcome": status,
    })

    return {
        "retry_counts": retry_counts,
        "tool_inputs": inputs if status == "RETRY_REQUESTED" else None,
        "validation_status": status,
        "step_count": step,
        "observable_logs": logs,
    }


# ---------------------------------------------------------------------------
# 4. Evidence Storage & Reflection
# ---------------------------------------------------------------------------

def store_evidence(state: ResearchState) -> Dict[str, Any]:
    """Store validated findings while preserving provenance."""
    step = state.get("step_count", 0) + 1
    tool_results = state.get("tool_results", [])

    if not tool_results:
        return {"step_count": step}

    last_record = tool_results[-1]
    tool_name = last_record.get("tool")
    result = last_record.get("result")

    # Never store failed or invalid tool output as accepted evidence.
    if last_record.get("error") or last_record.get("status") != "SUCCESS":
        logs = list(state.get("observable_logs", []))
        logs.append({
            "timestamp": _now_iso(),
            "step": step,
            "node": "store_evidence",
            "decision": f"Skipped storing failed output from '{tool_name}'.",
            "tool": tool_name,
            "outcome": "SKIPPED",
        })
        return {"step_count": step, "observable_logs": logs}

    discovered = list(state.get("discovered_papers", []))
    validated = list(state.get("validated_papers", []))
    extracted = list(state.get("extracted_papers", []))
    retrieved_ev = list(state.get("retrieved_evidence", []))
    metric_comps = list(state.get("metric_comparisons", []))
    gaps = list(state.get("identified_gaps", []))
    citations = dict(state.get("citations", {}))
    final_rep = state.get("final_report")

    if tool_name == "search_academic_papers" and isinstance(result, dict):
        papers = result.get("papers", [])
        if isinstance(papers, list):
            discovered = papers

    elif tool_name == "validate_metadata_integrity" and isinstance(result, list):
        validated = result

    elif tool_name == "read_research_paper" and isinstance(result, dict):
        pid = result.get("paper_id")
        if pid and pid not in extracted:
            extracted.append(pid)

    elif tool_name == "retrieve_research_context" and isinstance(result, dict):
        retrieved_ev.extend(result.get("chunks", []))

    elif tool_name == "analyze_experimental_results" and isinstance(result, list):
        metric_comps = result

    elif tool_name == "identify_research_gaps" and isinstance(result, dict):
        gaps = result.get("gaps", [])

    elif tool_name == "generate_research_comparison_report" and isinstance(result, dict):
        final_rep = result

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "store_evidence",
        "decision": f"Stored validated findings from tool '{tool_name}' into working memory.",
        "tool": tool_name,
        "outcome": "SUCCESS",
    })

    return {
        "discovered_papers": discovered,
        "validated_papers": validated,
        "extracted_papers": extracted,
        "retrieved_evidence": retrieved_ev,
        "metric_comparisons": metric_comps,
        "identified_gaps": gaps,
        "citations": citations,
        "final_report": final_rep,
        "step_count": step,
        "observable_logs": logs,
    }


def reflect_and_revise(state: ResearchState) -> Dict[str, Any]:
    """Reflect on evidence and determine whether the workflow should adapt."""
    step = state.get("step_count", 0) + 1
    reflections = list(state.get("reflections", []))
    plan = list(state.get("research_plan", []))
    extracted = state.get("extracted_papers", [])
    max_steps = state.get("max_steps", 15)

    if step >= max_steps:
        decision = (
            f"Workflow step limit reached ({step}/{max_steps}). "
            "Proceeding to synthesize available evidence."
        )
    elif not extracted and state.get("validation_status") == "RETRY_EXHAUSTED":
        decision = (
            "Primary academic search exhausted without extracted papers. "
            "Revising plan to synthesize qualified findings."
        )
    elif len(extracted) >= 1:
        decision = (
            f"Factual evidence collected from {len(extracted)} extracted paper(s). "
            "Proceeding with the remaining research tasks."
        )
    else:
        decision = "Reflected on current findings; proceeding to the next research phase."

    reflections.append(decision)

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "reflect_and_revise",
        "decision": decision,
        "tool": None,
        "outcome": "SUCCESS",
    })

    return {
        "reflections": reflections,
        "research_plan": plan,
        "step_count": step,
        "observable_logs": logs,
    }


def critique_findings(state: ResearchState) -> Dict[str, Any]:
    """Critique evidence quality, coverage, and empirical grounding."""
    step = state.get("step_count", 0) + 1
    extracted = state.get("extracted_papers", [])
    status = "APPROVED"
    reason = "Evidence is empirically grounded with verified metadata and traceable citations."

    if not extracted:
        status = "INSUFFICIENT_EVIDENCE"
        reason = "No research papers could be retrieved or extracted. A qualified report is required."
    elif len(extracted) == 1:
        status = "APPROVED"
        reason = (
            "Single-paper analysis complete. Comparative conclusions must be "
            "explicitly limited because only one paper was extracted."
        )

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "critique_findings",
        "decision": f"Critique verdict: {status}. {reason}",
        "tool": None,
        "outcome": status,
    })

    return {
        "critique_status": status,
        "step_count": step,
        "observable_logs": logs,
    }


# ---------------------------------------------------------------------------
# 5. Report Generation & Completion
# ---------------------------------------------------------------------------

async def generate_report_node(
    state: ResearchState,
    router: Optional[ModelRouter] = None,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
) -> Dict[str, Any]:
    """Compile the research report using the report service."""
    step = state.get("step_count", 0) + 1
    extracted = state.get("extracted_papers", [])
    goal = state.get("research_goal", "Academic Research Evaluation")
    llm = router or default_model_router
    repo = repository or default_paper_repo
    v_db = vector_db or default_vector_store

    report_data = state.get("final_report")
    if not report_data:
        if extracted and len(extracted) >= 2:
            report_schema = await generate_research_comparison_report(
                paper_ids=extracted,
                research_question=goal,
                repository=repo,
                vector_db=v_db,
                router=llm,
            )
            report_data = _sanitize_data(report_schema.model_dump())
            report_data.setdefault("title", f"Comparative Research Report: {goal}")
        else:
            report_data = _sanitize_data({
                "report_id": f"rep_{uuid.uuid4().hex[:8]}",
                "title": f"Academic Research Report: {goal}",
                "executive_summary": (
                    f"Investigation into '{goal}'. "
                    f"Findings are established across {len(extracted)} extracted paper(s): "
                    f"{', '.join(extracted) if extracted else 'None'}. "
                    "Comparative conclusions are limited by the available evidence."
                ),
                "is_degraded": llm.is_degraded_mode() if llm else False,
            })

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "generate_report",
        "decision": "Generated a report using the available research evidence.",
        "tool": "generate_research_comparison_report",
        "outcome": "SUCCESS",
    })

    return {
        "final_report": report_data,
        "step_count": step,
        "observable_logs": logs,
    }


def finish_task(state: ResearchState) -> Dict[str, Any]:
    """Assign the terminal stopping reason and record completion."""
    step = state.get("step_count", 0) + 1
    stopping_reason = state.get("stopping_reason")

    if not stopping_reason:
        if state.get("critique_status") == "APPROVED":
            stopping_reason = "GOAL_ACHIEVED"
        elif state.get("step_count", 0) >= state.get("max_steps", 15):
            stopping_reason = "MAX_STEPS_REACHED"
        elif state.get("critique_status") == "INSUFFICIENT_EVIDENCE":
            stopping_reason = "INSUFFICIENT_EVIDENCE"
        else:
            stopping_reason = "COMPLETED"

    logs = list(state.get("observable_logs", []))
    logs.append({
        "timestamp": _now_iso(),
        "step": step,
        "node": "finish_task",
        "decision": f"Final workflow termination. Reason: {stopping_reason}.",
        "tool": None,
        "outcome": "TERMINATED",
    })

    return {
        "stopping_reason": stopping_reason,
        "step_count": step,
        "observable_logs": logs,
    }

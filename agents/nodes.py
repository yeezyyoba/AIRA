"""Node implementations for AIRA's evidence-driven LangGraph workflow.

The workflow distinguishes completed, blocked, and failed phases. Missing papers,
PDFs, metrics, or citations are never silently treated as successful findings.

Design rules enforced here:
* Every paper has exactly one identity key (``_paper_key``) that is used for
  validation, extraction, indexing, citation and retry bookkeeping.
* Retries are bounded (``settings.MAX_RETRIES``, default 2) and scoped per tool
  *and* per item, so one bad PDF cannot exhaust the budget for the others.
* Content returned by tools is data. It is serialised to plain JSON types
  (enums become their values) before it enters the graph state.
* The workflow stops for exactly one of: goal achieved, essential information
  missing, unacceptable risk, max steps reached, or insufficient evidence.
"""
import asyncio
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
import uuid

from agents.state import ResearchState
from app.core.config import settings
from llm.model_router import ModelRouter, model_router as default_model_router
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from services.pdf_downloader import InvalidPDFError, PDFDownloadError, PDFSizeExceededError, SSRFProtectionError
from services.report_service import (
    export_report_markdown,
    generate_research_comparison_report,
)
from tools.academic_search import search_academic_papers
from tools.citation_generator import CitationStyle, generate_citation
from tools.file_manager import save_report
from tools.gap_identifier import identify_research_gaps
from tools.metadata_validator import validate_paper_metadata
from tools.methodology_analyzer import analyze_research_methodology
from tools.paper_comparator import compare_research_papers
from tools.paper_indexer import index_research_paper
from tools.paper_reader import read_research_paper
from tools.rag_retriever import retrieve_research_context
from tools.result_analyzer import analyze_experimental_results
from validation.schemas import PaperMetadataSchema, ResearchComparisonReportSchema

logger = get_logger("agents.nodes")

PHASES = (
    "search_and_validate",
    "extract_and_index",
    "compare_and_analyze",
    "identify_gaps",
    "generate_citations",
    "synthesize_report",
)

PHASE_DESCRIPTIONS = {
    "search_and_validate": (
        "Discover relevant academic papers and validate their metadata against scholarly registries."
    ),
    "extract_and_index": (
        "Extract full text from accessible PDFs and index it for semantic retrieval."
    ),
    "compare_and_analyze": (
        "Retrieve supporting passages, compare papers, analyze methodologies, and inspect reported metrics."
    ),
    "identify_gaps": "Identify evidence-supported limitations and research gaps.",
    "generate_citations": "Generate bibliographic citations for the papers used.",
    "synthesize_report": "Synthesize a report that distinguishes evidence from missing information.",
}

TOOL_PHASES = {
    "search_academic_papers": "search_and_validate",
    "validate_metadata_integrity": "search_and_validate",
    "read_research_paper": "extract_and_index",
    "retrieve_research_context": "compare_and_analyze",
    "compare_research_papers": "compare_and_analyze",
    "analyze_research_methodology": "compare_and_analyze",
    "analyze_experimental_results": "compare_and_analyze",
    "identify_research_gaps": "identify_gaps",
    "generate_citations": "generate_citations",
    "generate_research_comparison_report": "synthesize_report",
}

ANALYSIS_ACTIONS = {
    "retrieve_research_context": "rag_retrieval",
    "compare_research_papers": "paper_comparison",
    "analyze_research_methodology": "methodology",
    "analyze_experimental_results": "experimental_results",
}

RESOLVED_STATUSES = {"COMPLETED", "BLOCKED", "SKIPPED", "FAILED"}
USABLE_VALIDATION_STATUSES = {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
VALIDATION_PRIORITY = {"VERIFIED": 0, "PARTIALLY_VERIFIED": 1, "UNVERIFIED": 2}
FAILED_EXTRACTION_STATUSES = {"FAILED", "SCANNED_NEEDS_OCR"}

# Requests that try to subvert the agent or ask for destructive / secret-leaking
# actions are refused outright (the "unacceptable risk" stopping rule).
RISK_PATTERN = re.compile(
    r"(?i)("
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instructions|rules|prompts?)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|your)\s+(?:instructions|rules)"
    r"|(?:reveal|print|show|leak|dump|exfiltrate)\s+(?:the\s+|your\s+)?(?:system\s+prompt|api[\s_-]*keys?|secrets?|passwords?|credentials|\.env)"
    r"|override\s+(?:the\s+)?system\s+prompt"
    r"|rm\s+-rf"
    r"|drop\s+(?:table|database)"
    r"|delete\s+(?:all\s+)?(?:files|the\s+database|everything)"
    r"|fabricate\s+(?:citations?|results|data|references)"
    r"|(?:invent|make\s+up)\s+(?:citations?|references|results)"
    r")"
)

GENERIC_QUERY_TERMS = {"research", "papers", "paper", "study", "analysis", "compare", "review", "topic"}
_STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "for", "and", "or", "to", "with", "between",
    "about", "what", "which", "how", "are", "is", "do", "does", "compare", "comparison",
    "evaluate", "evaluation", "study", "studies", "research", "papers", "paper", "analysis",
    "clarification", "using", "based", "versus", "vs",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_retries() -> int:
    try:
        return max(0, int(settings.MAX_RETRIES))
    except (TypeError, ValueError):
        return 2


def _log_event(state, node, decision, tool=None, outcome="SUCCESS", details=None, step=None):
    entry = {
        "timestamp": _now_iso(),
        "step": step if step is not None else state.get("step_count", 0),
        "node": node,
        "decision": decision,
        "tool": tool,
        "outcome": outcome,
    }
    if details:
        entry["details"] = _sanitize_data(details)
    return entry


def _sanitize_data(obj):
    """Convert tool output into plain JSON-compatible data for the graph state.

    Enum members become their ``.value`` (``ValidationStatus.VERIFIED`` ->
    ``"VERIFIED"``), Pydantic models are dumped in JSON mode, and paths/dates
    become strings. This is the single normalisation point that prevents the
    ``str(enum).upper()`` class of bugs downstream.
    """
    if isinstance(obj, Enum):
        return _sanitize_data(obj.value)
    if isinstance(obj, dict):
        return {str(k): _sanitize_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_data(v) for v in obj]
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return _sanitize_data(obj.model_dump(mode="json"))
        except TypeError:
            return _sanitize_data(obj.model_dump())
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:  # numpy scalar conversion is best-effort
            return obj
    return obj


def _status_value(raw: Any) -> str:
    """Normalise a status that may be an Enum, a plain string, or missing."""
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().upper()


def _update_plan_task(plan, task_id, status, reason=None):
    updated = []
    found = False
    for task in plan:
        if not isinstance(task, dict):
            task = {"task_id": str(task), "name": str(task), "description": "", "status": "PENDING"}
        item = dict(task)
        if item.get("task_id") == task_id:
            item["status"] = status
            found = True
            if reason:
                item["status_reason"] = reason
            else:
                item.pop("status_reason", None)
        updated.append(item)
    if not found:
        updated.append({
            "task_id": task_id,
            "name": task_id.replace("_", " ").title(),
            "description": PHASE_DESCRIPTIONS.get(task_id, ""),
            "status": status,
            **({"status_reason": reason} if reason else {}),
        })
    return updated


def _resolve_phase(plan, completed, phase, status, reason):
    """Resolve a phase while preserving whether it completed or was blocked."""
    plan = _update_plan_task(plan, phase, status, reason)
    if status in RESOLVED_STATUSES and phase not in completed:
        completed.append(phase)
    return plan, completed


def _reopen_phase(plan, completed, phase, reason):
    plan = _update_plan_task(plan, phase, "IN_PROGRESS", reason)
    completed = [task for task in completed if task != phase]
    return plan, completed


# ---------------------------------------------------------------------------
# Paper identity and metadata helpers
# ---------------------------------------------------------------------------

def _paper_id(paper):
    if not isinstance(paper, dict):
        return None
    return (
        paper.get("paper_id")
        or paper.get("id")
        or paper.get("doi")
        or paper.get("url")
    )


def _paper_key(paper) -> Optional[str]:
    """Stable identity for a raw paper record (falls back to a title hash)."""
    pid = _paper_id(paper)
    if pid:
        return str(pid)
    title = (paper or {}).get("title") if isinstance(paper, dict) else None
    if isinstance(title, str) and title.strip():
        normalized = re.sub(r"\W+", " ", title.lower()).strip()
        return "title_" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return None


def _original_metadata(record):
    """Metadata validator results wrap the source record in original_metadata."""
    if not isinstance(record, dict):
        return {}
    original = record.get("original_metadata")
    return original if isinstance(original, dict) and original else record


def _record_key(record) -> Optional[str]:
    """Identity of a validation record; always equal to the source paper's key."""
    if not isinstance(record, dict):
        return None
    if record.get("paper_id"):
        return str(record["paper_id"])
    return _paper_key(_original_metadata(record))


def _pdf_url(record):
    metadata = _original_metadata(record)
    candidates = [metadata.get("pdf_url"), metadata.get("open_access_url")]
    provider_metadata = metadata.get("provider_metadata") or {}
    if isinstance(provider_metadata, dict):
        for provider in ("semantic_scholar", "openalex", "crossref"):
            item = provider_metadata.get(provider) or {}
            if isinstance(item, dict):
                candidates.extend([item.get("pdf_url"), item.get("open_access_url")])
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith(("https://", "http://")):
            return candidate
    return None


def _validation_usable(record):
    if not isinstance(record, dict):
        return False

    status = _status_value(record.get("status"))

    # Explicit validation outcomes take precedence. CONFLICTING is never usable.
    if status:
        return status in USABLE_VALIDATION_STATUSES

    # Preloaded metadata without a validation status remains usable,
    # but must be treated as unverified rather than verified.
    metadata = _original_metadata(record)
    has_identity = bool(record.get("paper_id") or _paper_key(metadata))
    has_bibliographic_data = bool(metadata.get("title") or metadata.get("doi"))
    return has_identity and has_bibliographic_data


def _usable_validated(validated) -> List[Dict[str, Any]]:
    usable = [p for p in validated if isinstance(p, dict) and _validation_usable(p)]
    return sorted(
        usable,
        key=lambda rec: VALIDATION_PRIORITY.get(_status_value(rec.get("status")), 3),
    )


def _citation_metadata(record) -> Dict[str, Any]:
    return _original_metadata(record) if isinstance(record, dict) else {}


def _merge_discovered(existing: List[Dict[str, Any]], new: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    merged = list(existing)
    seen = {_paper_key(p) for p in merged}
    added = 0
    for paper in new:
        key = _paper_key(paper)
        if key and key not in seen:
            merged.append(paper)
            seen.add(key)
            added += 1
    return merged, added


def _keywords(text: str, limit: int = 8) -> str:
    words = re.findall(r"[\w\-\.]+", text or "", flags=re.UNICODE)
    kept: List[str] = []
    for word in words:
        if word.lower() in _STOPWORDS or len(word) < 2:
            continue
        if word.lower() not in (k.lower() for k in kept):
            kept.append(word)
        if len(kept) >= limit:
            break
    return " ".join(kept)


def _signature(tool: str, inputs: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(_sanitize_data(inputs), sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(inputs)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{tool}:{digest}"


def _retry_key(tool: str, inputs: Dict[str, Any]) -> str:
    """Retry budgets are scoped per item for per-paper tools."""
    if tool == "read_research_paper" and inputs.get("paper_id"):
        return f"{tool}:{inputs['paper_id']}"
    return tool


def _failure_marker(tool: str, inputs: Dict[str, Any]) -> Optional[str]:
    if tool == "read_research_paper" and inputs.get("paper_id"):
        return f"extract:{inputs['paper_id']}"
    if tool in ANALYSIS_ACTIONS:
        return f"action:{ANALYSIS_ACTIONS[tool]}"
    if tool == "generate_citations":
        return "action:citations"
    return None


# ---------------------------------------------------------------------------
# 1. Request ingestion and goal analysis
# ---------------------------------------------------------------------------

def receive_request(state: ResearchState) -> Dict[str, Any]:
    """Ingest a new request or a resumed session without discarding finished work.

    Evidence (papers, extractions, analyses, citations, phase statuses) is
    carried over. Per-run counters (steps, retries) restart, because each
    invocation gets its own bounded budget. On resumption with a clarification,
    the inferred domain and questions are recomputed from the clarified goal.
    """
    query = (state.get("query") or "").strip()
    session_id = state.get("session_id") or str(uuid.uuid4())[:8]
    clarification = state.get("clarification_response")
    logs = list(state.get("observable_logs", []))

    max_steps = state.get("max_steps", settings.MAX_AGENT_STEPS)
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        max_steps = settings.MAX_AGENT_STEPS

    max_results = state.get("max_results") or settings.MAX_SEARCH_RESULTS
    if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
        max_results = settings.MAX_SEARCH_RESULTS

    is_resume = bool(logs)
    logs.append(_log_event(
        state, "receive_request",
        f"{'Resumed' if is_resume else 'Ingested'} research request for session '{session_id}'.",
        step=0,
    ))

    return {
        "session_id": session_id,
        "query": query,
        "research_goal": state.get("research_goal") or query,
        "research_questions": [] if clarification else list(state.get("research_questions", [])),
        "domain": None if clarification else state.get("domain"),
        "max_results": min(max_results, 100),
        "year_from": state.get("year_from"),
        "year_to": state.get("year_to"),
        "research_plan": list(state.get("research_plan", [])),
        "completed_tasks": list(state.get("completed_tasks", [])),
        "current_task": None,
        "selected_tool": None,
        "tool_inputs": None,
        "tool_results": list(state.get("tool_results", [])),
        "validation_status": "PENDING",
        "executed_tool_signatures": list(state.get("executed_tool_signatures", [])),
        "successful_tool_signatures": list(state.get("successful_tool_signatures", [])),
        "failed_items": list(state.get("failed_items", [])),
        "discovered_papers": list(state.get("discovered_papers", [])),
        "validated_papers": list(state.get("validated_papers", [])),
        "extracted_papers": list(state.get("extracted_papers", [])),
        "extraction_results": list(state.get("extraction_results", [])),
        "indexed_papers": list(state.get("indexed_papers", [])),
        "indexing_results": list(state.get("indexing_results", [])),
        "retrieved_evidence": list(state.get("retrieved_evidence", [])),
        "metric_comparisons": list(state.get("metric_comparisons", [])),
        "methodology_analyses": list(state.get("methodology_analyses", [])),
        "paper_comparisons": list(state.get("paper_comparisons", [])),
        "identified_gaps": list(state.get("identified_gaps", [])),
        "citations": dict(state.get("citations", {})),
        "reflections": list(state.get("reflections", [])),
        "critique_status": None,
        "step_count": 0,
        "max_steps": max_steps,
        "retry_counts": {},
        "pending_revision": None,
        "plan_revisions": 0,
        "expanded_search_pending": False,
        "missing_information": [],
        "clarification_question": None,
        "clarification_response": clarification,
        "stopping_reason": None,
        "risk_flags": [],
        "security_flags": list(state.get("security_flags", [])),
        "observable_logs": logs,
        "is_degraded": False,
        "final_report": None,
        "phase_statuses": dict(state.get("phase_statuses", {})),
        "analysis_actions_completed": list(state.get("analysis_actions_completed", [])),
    }


def _infer_domain(text: str) -> str:
    q = text.lower()
    if "speech" in q or re.search(r"\basr\b", q):
        return "Speech Recognition / ASR"
    if "vision" in q or "image" in q:
        return "Computer Vision"
    if any(x in q for x in ("summarization", "nlp", "language", "translation", "sentiment", "amharic", "text")) or re.search(r"\bqa\b", q):
        return "Natural Language Processing"
    return "Machine Learning"


def analyze_goal(state: ResearchState) -> Dict[str, Any]:
    query = (state.get("query") or "").strip()
    clarification = (state.get("clarification_response") or "").strip()
    goal = f"{query} (Clarification: {clarification})" if clarification else query

    # Search engines receive plain keywords, not the bookkeeping text above.
    query_core = re.sub(r"[^\w\s]", " ", query).strip().lower()
    if clarification and (len(query_core) < 4 or query_core in GENERIC_QUERY_TERMS):
        search_query = clarification
    elif clarification:
        search_query = f"{query} {clarification}"
    else:
        search_query = query

    domain = state.get("domain") or _infer_domain(goal)
    questions = list(state.get("research_questions", [])) or [
        f"What architectures and methods have been studied for {search_query}?",
        "Which datasets, experimental settings, and evaluation metrics are reported?",
        "What limitations and research gaps are supported by the literature?",
    ]
    step = state.get("step_count", 0) + 1
    logs = list(state.get("observable_logs", []))
    logs.append(_log_event(
        state, "analyze_goal",
        f"Identified domain '{domain}' and {len(questions)} research questions.",
        step=step,
    ))
    return {
        "research_goal": goal,
        "search_query": search_query.strip(),
        "domain": domain,
        "research_questions": questions,
        "step_count": step,
        "observable_logs": logs,
    }


def check_information(state: ResearchState) -> Dict[str, Any]:
    """Stopping rules 'essential information missing' and 'unacceptable risk'."""
    query = (state.get("query") or "").strip()
    clarification = (state.get("clarification_response") or "").strip()
    combined = f"{query} {clarification}".strip()
    clean = re.sub(r"[^\w\s]", "", combined).strip()
    missing, question, risks = [], None, []

    risk_match = RISK_PATTERN.search(combined)
    if risk_match:
        risks.append(
            f"Request contains an instruction the agent must not follow: '{risk_match.group(0)}'."
        )
    elif len(clean) < 4:
        missing.append("A specific research topic is required.")
        question = "Please provide the specific research topic or question."
    elif clean.lower() in GENERIC_QUERY_TERMS:
        missing.append("The research scope is too broad.")
        question = "Which specific topic, models, datasets, or language should the research focus on?"

    step = state.get("step_count", 0) + 1
    logs = list(state.get("observable_logs", []))
    if risks:
        decision, outcome = "; ".join(risks) + " Stopping for unacceptable risk.", "UNACCEPTABLE_RISK"
    elif missing:
        decision, outcome = "; ".join(missing), "CLARIFICATION_REQUIRED"
    else:
        decision, outcome = "Research request contains sufficient initial information.", "SUCCESS"
    logs.append(_log_event(state, "check_information", decision, outcome=outcome, step=step))

    update: Dict[str, Any] = {
        "missing_information": missing,
        "clarification_question": question,
        "risk_flags": risks,
        "step_count": step,
        "observable_logs": logs,
    }
    if risks:
        update["stopping_reason"] = "UNACCEPTABLE_RISK"
    return update


def ask_user(state: ResearchState) -> Dict[str, Any]:
    step = state.get("step_count", 0) + 1
    logs = list(state.get("observable_logs", []))
    logs.append(_log_event(
        state, "ask_user",
        f"Clarification required: {state.get('clarification_question')}",
        outcome="PAUSED",
        step=step,
    ))
    return {
        "stopping_reason": "USER_CLARIFICATION_REQUIRED",
        "step_count": step,
        "observable_logs": logs,
    }


# ---------------------------------------------------------------------------
# 2. Planning and deterministic tool selection
# ---------------------------------------------------------------------------

def create_plan(state: ResearchState) -> Dict[str, Any]:
    """Build the research plan and recognize evidence already in the state."""
    plan = [{
        "task_id": phase,
        "name": phase.replace("_", " ").title(),
        "description": PHASE_DESCRIPTIONS[phase],
        "status": "PENDING",
    } for phase in PHASES]

    completed = [task for task in state.get("completed_tasks", []) if task in PHASES]
    previous_statuses = dict(state.get("phase_statuses", {}))
    statuses = {phase: previous_statuses.get(phase, "PENDING") for phase in PHASES}
    reasons: Dict[str, str] = {}

    validated = state.get("validated_papers", [])
    discovered = state.get("discovered_papers", [])
    extracted = state.get("extracted_papers", [])
    citations = state.get("citations", {})

    usable = _usable_validated(validated)
    if usable and "search_and_validate" not in completed:
        completed.append("search_and_validate")
    if extracted and "extract_and_index" not in completed:
        completed.append("extract_and_index")

    # Tasks reported complete by the caller (resumption) must not stay PENDING.
    for phase in completed:
        if statuses.get(phase) not in RESOLVED_STATUSES:
            statuses[phase] = "COMPLETED"
            reasons[phase] = "Recognized from evidence already present in the session."

    # Blocked phases from a previous run are retried after new information
    # (e.g. a clarification) rather than being treated as permanently final.
    if state.get("clarification_response"):
        for phase, status in list(statuses.items()):
            if status == "BLOCKED":
                statuses[phase] = "PENDING"
                completed = [task for task in completed if task != phase]

    if not usable and discovered and statuses["search_and_validate"] == "PENDING":
        statuses["search_and_validate"] = "IN_PROGRESS"

    if citations and statuses["generate_citations"] == "PENDING":
        statuses["generate_citations"] = "IN_PROGRESS"

    for task in plan:
        task["status"] = statuses[task["task_id"]]
        if task["task_id"] in reasons:
            task["status_reason"] = reasons[task["task_id"]]

    step = state.get("step_count", 0) + 1
    logs = list(state.get("observable_logs", []))
    logs.append(_log_event(
        state,
        "create_plan",
        (
            f"Created {len(plan)} research phases; recognized "
            f"{len(usable)} usable validated and {len(extracted)} extracted paper(s)."
        ),
        details={"plan": [t["task_id"] for t in plan], "phase_statuses": statuses},
        step=step,
    ))

    return {
        "research_plan": plan,
        "completed_tasks": completed,
        "phase_statuses": statuses,
        "analysis_actions_completed": list(state.get("analysis_actions_completed", [])),
        "step_count": step,
        "observable_logs": logs,
    }


def _extraction_candidates(state, validated, extracted, extraction_results, failed) -> List[Tuple[str, str]]:
    attempted = {
        str(item.get("paper_id"))
        for item in extraction_results
        if isinstance(item, dict) and item.get("paper_id")
    }
    already = set(map(str, extracted))
    candidates = []
    for record in _usable_validated(validated):
        key = _record_key(record)
        url = _pdf_url(record)
        if (
            key and url
            and key not in already
            and key not in attempted
            and f"extract:{key}" not in failed
        ):
            candidates.append((key, url))
    return candidates


def _citation_requests(state, validated, extracted, citations, failed) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    seen = set(map(str, citations))
    for record in _usable_validated(validated):
        key = _record_key(record)
        meta = _citation_metadata(record)
        if key and key not in seen and f"citation:{key}" not in failed and (meta.get("title") or meta.get("doi")):
            requests.append({"key": key, "paper": meta})
            seen.add(key)
    # Extracted papers that never went through metadata validation (e.g. uploads)
    # are cited from their own extracted bibliographic fields.
    for pid in extracted:
        key = str(pid)
        if key not in seen and f"citation:{key}" not in failed:
            requests.append({"key": key, "paper_id": key})
            seen.add(key)
    return requests


def select_tools(state: ResearchState) -> Dict[str, Any]:
    """Choose the next not-yet-completed action; never silently skip a phase."""
    completed = list(state.get("completed_tasks", []))
    plan = list(state.get("research_plan", []))
    phase_statuses = dict(state.get("phase_statuses", {}))
    discovered = list(state.get("discovered_papers", []))
    validated = list(state.get("validated_papers", []))
    extracted = list(state.get("extracted_papers", []))
    extraction_results = list(state.get("extraction_results", []))
    indexed = list(state.get("indexed_papers", []))
    citations = dict(state.get("citations", {}))
    failed = set(state.get("failed_items", []))
    actions_done = set(state.get("analysis_actions_completed", []))
    pending_revision = state.get("pending_revision")
    selected, inputs, task, reason = None, {}, None, ""

    def choose(tool, phase, tool_inputs, why):
        nonlocal selected, inputs, task, reason
        selected, task, inputs, reason = tool, phase, tool_inputs, why

    def resolve(phase, status, why):
        nonlocal plan, completed
        plan, completed = _resolve_phase(plan, completed, phase, status, why)
        phase_statuses[phase] = status

    max_results = state.get("max_results") or settings.MAX_SEARCH_RESULTS
    search_query = state.get("search_query") or state.get("research_goal") or state.get("query")

    # Phase 1: discovery and metadata validation.
    if "search_and_validate" not in completed:
        validated_keys = {_record_key(v) for v in validated}
        pending = [
            p for p in discovered
            if _paper_key(p) not in validated_keys and f"validate:{_paper_key(p)}" not in failed
        ]
        if not discovered or state.get("expanded_search_pending"):
            expanded = bool(state.get("expanded_search_pending"))
            choose(
                "search_academic_papers", "search_and_validate",
                {
                    "query": search_query,
                    "max_results": min(max_results * (2 if expanded else 1), 100),
                    "year_from": state.get("year_from"),
                    "year_to": state.get("year_to"),
                },
                (
                    "Expanded search after plan revision: more candidates are needed for full-text evidence."
                    if expanded else
                    "Discover candidate papers from OpenAlex, Semantic Scholar and Crossref."
                ),
            )
        elif pending:
            choose(
                "validate_metadata_integrity", "search_and_validate",
                {"papers": pending},
                f"Cross-check metadata of {len(pending)} unvalidated candidate(s) against scholarly registries.",
            )
        else:
            usable = _usable_validated(validated)
            if usable:
                resolve(
                    "search_and_validate", "COMPLETED",
                    f"Validated {len(usable)} usable record(s); conflicting records were excluded.",
                )
            else:
                resolve(
                    "search_and_validate", "BLOCKED",
                    "No usable metadata records remained after validation.",
                )

    # Phase 2: full-text extraction and indexing, one paper per action.
    if selected is None and "search_and_validate" in completed and "extract_and_index" not in completed:
        cap = max(1, int(settings.MAX_PAPERS_TO_EXTRACT))
        attempted = len([r for r in extraction_results if isinstance(r, dict)])
        candidates = _extraction_candidates(state, validated, extracted, extraction_results, failed)
        if candidates and len(extracted) < cap and attempted < cap * 2:
            pid, url = candidates[0]
            choose(
                "read_research_paper", "extract_and_index",
                {"paper_id": pid, "file_path_or_url": url},
                f"Extract and index full text of paper {pid} (highest-confidence remaining candidate).",
            )
        elif (
            len(extracted) < 2
            and discovered
            and state.get("plan_revisions", 0) < settings.MAX_PLAN_REVISIONS
            and not pending_revision
        ):
            # Defer to reflection: it decides whether to revise the plan.
            pending_revision = (
                f"Only {len(extracted)} paper(s) yielded full text; a comparative review needs at least two."
            )
            reason = "Extraction candidates exhausted with insufficient evidence; requesting plan revision."
        elif extracted:
            resolve(
                "extract_and_index", "COMPLETED",
                f"Extracted full text from {len(extracted)} paper(s); {len(indexed)} indexed for retrieval.",
            )
        else:
            resolve(
                "extract_and_index", "BLOCKED",
                "No validated paper with an accessible PDF produced usable full text.",
            )

    # Phase 3: RAG retrieval, comparison, methodology and metric analysis.
    if (
        selected is None and not pending_revision
        and "extract_and_index" in completed and "compare_and_analyze" not in completed
    ):
        pids = list(extracted)
        if not pids:
            resolve(
                "compare_and_analyze", "BLOCKED",
                "No successfully extracted papers are available for empirical analysis.",
            )
        else:
            goal = state.get("search_query") or state.get("research_goal") or ""
            indexed_extracted = [p for p in pids if p in indexed]
            plan_actions = []
            if indexed_extracted:
                plan_actions.append((
                    "rag_retrieval", "retrieve_research_context",
                    {"question": goal, "paper_ids": indexed_extracted, "top_k": 8},
                    "Retrieve the passages most relevant to the research question from the indexed papers.",
                ))
            if len(pids) >= 2:
                plan_actions.append((
                    "paper_comparison", "compare_research_papers", {"paper_ids": pids},
                    "Compare the extracted papers across research dimensions.",
                ))
            plan_actions.append((
                "methodology", "analyze_research_methodology", {"paper_ids": pids},
                "Analyze the methodology of every extracted paper.",
            ))
            plan_actions.append((
                "experimental_results", "analyze_experimental_results", {"paper_ids": pids},
                "Inspect reported metrics and determine whether comparisons are compatible.",
            ))
            for action, tool, tool_inputs, why in plan_actions:
                if action not in actions_done and f"action:{action}" not in failed:
                    choose(tool, "compare_and_analyze", tool_inputs, why)
                    break
            if selected is None:
                done = [a for a, *_ in plan_actions if a in actions_done]
                failed_actions = [a for a, *_ in plan_actions if f"action:{a}" in failed]
                if done:
                    note = f"Completed analysis actions: {done}."
                    if failed_actions:
                        note += f" Failed after retries: {failed_actions}."
                    resolve("compare_and_analyze", "COMPLETED", note)
                else:
                    resolve("compare_and_analyze", "BLOCKED", f"All analysis actions failed: {failed_actions}.")

    # Phase 4: research gaps depend on extracted evidence.
    if selected is None and not pending_revision and "compare_and_analyze" in completed and "identify_gaps" not in completed:
        if extracted:
            choose(
                "identify_research_gaps", "identify_gaps",
                {
                    "paper_ids": extracted,
                    "research_area": state.get("search_query") or state.get("domain") or state.get("research_goal"),
                },
                "Identify limitations and research gaps from extracted papers.",
            )
        else:
            resolve(
                "identify_gaps", "BLOCKED",
                "Gap analysis requires extracted full-text evidence; none is available.",
            )

    # Phase 5: citations (one batched action for every eligible paper).
    if selected is None and not pending_revision and "identify_gaps" in completed and "generate_citations" not in completed:
        requests = _citation_requests(state, validated, extracted, citations, failed)
        if requests and "action:citations" not in failed:
            choose(
                "generate_citations", "generate_citations",
                {"requests": requests, "style": CitationStyle.APA7.value},
                f"Generate APA 7 citations for {len(requests)} paper(s) from their validated or extracted metadata.",
            )
        elif citations:
            resolve(
                "generate_citations", "COMPLETED",
                f"Generated citations for {len(citations)} paper(s).",
            )
        else:
            resolve(
                "generate_citations", "BLOCKED",
                "No bibliographic metadata is available for citation generation.",
            )

    # Phase 6: comparative report service requires at least two extracted papers.
    if selected is None and not pending_revision and "generate_citations" in completed and "synthesize_report" not in completed:
        if len(extracted) >= 2:
            choose(
                "generate_research_comparison_report", "synthesize_report",
                {"paper_ids": extracted, "research_question": state.get("research_goal")},
                "Synthesize the comparative report from extracted evidence.",
            )
        else:
            resolve(
                "synthesize_report", "BLOCKED",
                "A comparative report requires at least two extracted papers; a qualified partial report will be produced instead.",
            )

    # Safety net: never re-run a call that already succeeded with identical inputs.
    if selected and _signature(selected, inputs) in set(state.get("successful_tool_signatures", [])):
        marker = _failure_marker(selected, inputs) or f"duplicate:{selected}"
        logger.warning("Suppressed duplicate execution of '%s'; marking %s.", selected, marker)
        failed.add(marker)
        reason = f"Suppressed duplicate call to '{selected}' (identical inputs already succeeded)."
        selected, task, inputs = None, None, {}

    if selected:
        phase_statuses[task] = "IN_PROGRESS"
        plan = _update_plan_task(plan, task, "IN_PROGRESS")

    step = state.get("step_count", 0) + 1
    logs = list(state.get("observable_logs", []))
    logs.append(_log_event(
        state, "select_tools",
        reason or "All research phases are resolved; proceeding to evidence critique.",
        tool=selected,
        outcome="SELECTED" if selected else ("REVISION_REQUESTED" if pending_revision else "NO_ACTION"),
        details={"task": task, "phase_statuses": phase_statuses},
        step=step,
    ))
    return {
        "selected_tool": selected,
        "tool_inputs": inputs if selected else None,
        "current_task": task,
        "completed_tasks": completed,
        "research_plan": plan,
        "phase_statuses": phase_statuses,
        "failed_items": sorted(failed),
        "pending_revision": pending_revision,
        "validation_status": "PENDING",
        "step_count": step,
        "observable_logs": logs,
    }


# ---------------------------------------------------------------------------
# 3. Tool execution and output validation
# ---------------------------------------------------------------------------

async def _validate_batch(papers: List[Any]) -> Dict[str, Any]:
    """Validate metadata concurrently (bounded) and keep every record's identity."""
    semaphore = asyncio.Semaphore(max(1, int(settings.METADATA_VALIDATION_CONCURRENCY)))
    rejected: List[Dict[str, Any]] = []

    async def _one(raw_paper):
        paper = _sanitize_data(raw_paper)
        if not isinstance(paper, dict):
            rejected.append({"paper_id": None, "reason": "Malformed (non-object) paper record."})
            return None
        key = _paper_key(paper)
        try:
            metadata = PaperMetadataSchema(**paper)
        except Exception as exc:
            rejected.append({"paper_id": key, "reason": f"Invalid metadata: {exc}"})
            return None
        async with semaphore:
            try:
                result = await validate_paper_metadata(metadata, paper_id=key)
            except Exception as exc:
                logger.warning("Metadata validation failed for '%s': %s", key, exc)
                rejected.append({"paper_id": key, "reason": f"Validation error: {exc}"})
                return None
        validation = _sanitize_data(result)
        if not isinstance(validation, dict):
            rejected.append({"paper_id": key, "reason": "Validator returned a non-object result."})
            return None
        # The validator's schema defaults original_metadata to {}; keep the
        # source record attached so title/DOI/PDF URL are never lost.
        if not validation.get("original_metadata"):
            validation["original_metadata"] = paper
        validation["paper_id"] = key
        return validation

    results = await asyncio.gather(*(_one(p) for p in papers))
    records = [r for r in results if r is not None]
    if papers and not records and len(rejected) == len(papers) and all(
        r["reason"].startswith("Validation error") for r in rejected
    ):
        raise RuntimeError(f"Metadata validation failed for every paper: {rejected[0]['reason']}")
    return {"records": records, "rejected": rejected}


async def _run_per_paper(paper_ids, fn):
    outputs, errors = [], {}
    for pid in paper_ids:
        try:
            outputs.append(_sanitize_data(await fn(pid)))
        except Exception as exc:
            errors[str(pid)] = str(exc)
    if paper_ids and not outputs:
        raise RuntimeError(f"Analysis failed for every paper: {errors}")
    return outputs, errors


def _citation_from_extraction(repo, paper_id) -> Optional[Dict[str, Any]]:
    record = repo.get_paper(paper_id) if repo else None
    data = record.extracted_data if record else None
    if not data or not data.title:
        return None
    year = None
    created = (record.document_metadata or {}).get("year") if record else None
    if isinstance(created, int):
        year = created
    return {"paper_id": paper_id, "title": data.title, "authors": list(data.authors or []), "year": year}


async def execute_tools(
    state: ResearchState,
    router: Optional[ModelRouter] = None,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
) -> Dict[str, Any]:
    step = state.get("step_count", 0) + 1
    tool_name = state.get("selected_tool")
    inputs = state.get("tool_inputs") or {}
    llm = router or default_model_router
    repo = repository or default_paper_repo
    v_db = vector_db or default_vector_store
    records = list(state.get("tool_results", []))

    if not tool_name:
        return {"step_count": step, "validation_status": "NO_OP"}

    result_data, error, extra = None, None, {}
    logger.info("Executing AIRA tool '%s'.", tool_name)
    try:
        if tool_name == "search_academic_papers":
            raw = await search_academic_papers(
                query=inputs.get("query", ""),
                max_results=inputs.get("max_results", settings.MAX_SEARCH_RESULTS),
                year_from=inputs.get("year_from"),
                year_to=inputs.get("year_to"),
            )
            if isinstance(raw, dict):
                papers, envelope = raw.get("papers", []), dict(raw)
            elif isinstance(raw, list):
                papers, envelope = raw, {}
            else:
                raise ValueError("Academic search returned an unsupported response.")
            if not isinstance(papers, list):
                raise ValueError("Academic search response has an invalid papers list.")
            normalized = []
            for paper in papers:
                paper = _sanitize_data(paper)
                if not isinstance(paper, dict):
                    raise ValueError("Search returned a malformed paper record.")
                normalized.append(paper)
            envelope = _sanitize_data(envelope)
            envelope.pop("papers", None)
            result_data = {**envelope, "papers": normalized, "total_results": len(normalized)}

        elif tool_name == "validate_metadata_integrity":
            papers = inputs.get("papers", [])
            if not isinstance(papers, list):
                raise ValueError("Metadata validation requires a list.")
            result_data = await _validate_batch(papers)

        elif tool_name == "read_research_paper":
            pid = inputs.get("paper_id")
            source = inputs.get("file_path_or_url")
            if not pid or not source:
                raise ValueError("Paper extraction requires paper_id and file_path_or_url.")
            if not str(source).startswith(("https://", "http://")):
                raise ValueError("The agent only reads remote PDFs over HTTP(S); local paths are refused.")
            extraction = repo.get_paper(pid) if repo else None
            if extraction is None:
                extraction = await read_research_paper(source, paper_id=pid)
                if extraction is None:
                    raise ValueError("Paper reader returned no extraction result.")
                if extraction.paper_id != pid:
                    extraction = extraction.model_copy(update={"paper_id": pid})
                if repo and _status_value(extraction.status) not in {"FAILED"}:
                    repo.save_paper(extraction)
            result_data = _sanitize_data(extraction)

            usable = (
                _status_value(extraction.status) not in FAILED_EXTRACTION_STATUSES
                and extraction.extracted_data is not None
            )
            if usable and v_db is not None:
                try:
                    existing = v_db.get_paper_records(pid).get("ids") or []
                    if existing:
                        extra["indexing"] = {"paper_id": pid, "status": "SUCCESS", "chunk_count": len(existing),
                                             "message": "Already indexed."}
                    else:
                        summary = await index_research_paper(
                            pid, extraction_result=extraction, repository=repo, vector_db=v_db,
                        )
                        extra["indexing"] = _sanitize_data(summary)
                except Exception as exc:
                    # Indexing failure does not invalidate the extraction itself.
                    logger.warning("Indexing failed for '%s': %s", pid, exc)
                    extra["indexing"] = {"paper_id": pid, "status": "FAILED", "message": str(exc)}

        elif tool_name == "retrieve_research_context":
            result = await retrieve_research_context(
                question=inputs.get("question") or inputs.get("query", ""),
                paper_ids=inputs.get("paper_ids"),
                top_k=inputs.get("top_k", settings.RAG_DEFAULT_TOP_K),
                vector_db=v_db,
            )
            result_data = _sanitize_data(result)

        elif tool_name == "compare_research_papers":
            result = await compare_research_papers(
                paper_ids=inputs.get("paper_ids", []),
                repository=repo,
                vector_db=v_db,
                router=llm,
            )
            result_data = _sanitize_data(result)

        elif tool_name == "analyze_research_methodology":
            pids = inputs.get("paper_ids") or ([inputs["paper_id"]] if inputs.get("paper_id") else [])
            outputs, errors = await _run_per_paper(
                pids,
                lambda pid: analyze_research_methodology(paper_id=pid, repository=repo, router=llm),
            )
            result_data = {"analyses": outputs, "errors": errors}

        elif tool_name == "analyze_experimental_results":
            result = analyze_experimental_results(
                paper_ids=inputs.get("paper_ids", []),
                repository=repo,
            )
            result_data = _sanitize_data(result)

        elif tool_name == "identify_research_gaps":
            result = await identify_research_gaps(
                paper_ids=inputs.get("paper_ids", []),
                research_area=inputs.get("research_area", ""),
                repository=repo,
                router=llm,
            )
            result_data = _sanitize_data(result)

        elif tool_name == "generate_research_comparison_report":
            result = await generate_research_comparison_report(
                paper_ids=inputs.get("paper_ids", []),
                research_question=inputs.get("research_question", ""),
                repository=repo,
                vector_db=v_db,
                router=llm,
                precomputed=_precomputed_analyses(state),
            )
            result_data = _sanitize_data(result)

        elif tool_name == "generate_citations":
            style = CitationStyle(inputs.get("style", CitationStyle.APA7.value))
            generated, failures = {}, {}
            for request in inputs.get("requests", []):
                key = str(request.get("key"))
                paper = request.get("paper")
                if paper is None and request.get("paper_id"):
                    paper = _citation_from_extraction(repo, request["paper_id"])
                if not paper:
                    failures[key] = "No bibliographic metadata available."
                    continue
                try:
                    citation = generate_citation(paper=paper, style=style)
                    generated[key] = _sanitize_data(citation)
                except Exception as exc:
                    failures[key] = str(exc)
            if inputs.get("requests") and not generated:
                raise RuntimeError(f"Citation generation failed for every paper: {failures}")
            result_data = {"citations": generated, "failed": failures}

        else:
            raise ValueError(f"Unrecognized execution tool: {tool_name}")

    except Exception as exc:
        logger.warning("Tool '%s' failed: %s", tool_name, exc, exc_info=True)
        error = f"{type(exc).__name__}: {exc}"
        extra["retryable"] = _is_retryable(exc)

    record = {
        "timestamp": _now_iso(),
        "tool": tool_name,
        "inputs": _summarize_inputs(inputs),
        "result": _sanitize_data(result_data),
        "error": error,
        "status": "ERROR" if error else "SUCCESS",
        "signature": _signature(tool_name, inputs),
        **extra,
    }
    records.append(record)
    signatures = list(state.get("executed_tool_signatures", []))
    signatures.append(record["signature"])

    logs = list(state.get("observable_logs", []))
    logs.append(_log_event(
        state, "execute_tools",
        f"Tool '{tool_name}' " + (f"failed: {error}" if error else "returned output."),
        tool=tool_name,
        outcome="ERROR" if error else "SUCCESS",
        details={"inputs": _summarize_inputs(inputs)},
        step=step,
    ))
    return {
        "tool_results": records,
        "executed_tool_signatures": signatures,
        "is_degraded": bool(state.get("is_degraded")) or _router_degraded(llm),
        "step_count": step,
        "observable_logs": logs,
    }


def _router_degraded(llm) -> bool:
    try:
        return bool(llm.is_degraded_mode()) if llm is not None else False
    except Exception:
        return False


def _is_retryable(exc: Exception) -> bool:
    """Retrying only helps for transient failures (timeouts, 5xx, rate limits, network).

    A missing PDF (404), a non-PDF response, an oversized file or a blocked URL
    fails identically on every attempt, so it is recorded once and skipped.
    """
    if isinstance(exc, (InvalidPDFError, PDFSizeExceededError, SSRFProtectionError)):
        return False
    if isinstance(exc, PDFDownloadError):
        match = re.search(r"HTTP (\d{3})", str(exc))
        if match:
            code = int(match.group(1))
            return code >= 500 or code in (408, 429)
    return True


def _summarize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep logged inputs small: long paper lists are summarized."""
    summary: Dict[str, Any] = {}
    for key, value in (inputs or {}).items():
        if key in ("papers", "requests") and isinstance(value, list):
            summary[key] = f"<{len(value)} item(s)>"
        else:
            summary[key] = _sanitize_data(value)
    return summary


def _precomputed_analyses(state: ResearchState) -> Dict[str, Any]:
    return {
        "comparison": (state.get("paper_comparisons") or [None])[-1],
        "methodologies": state.get("methodology_analyses") or [],
        "metric_tables": state.get("metric_comparisons"),
        "gaps": state.get("identified_gaps"),
        "citations": state.get("citations") or {},
        "retrieved_evidence": state.get("retrieved_evidence") or [],
        # Registry-verified bibliographic metadata beats titles parsed from PDFs.
        "paper_metadata": {
            _record_key(rec): {
                "title": _original_metadata(rec).get("title"),
                "authors": _original_metadata(rec).get("authors") or [],
                "year": _original_metadata(rec).get("year"),
                "doi": _original_metadata(rec).get("doi"),
            }
            for rec in _usable_validated(state.get("validated_papers") or [])
        },
    }


def validate_tool_outputs(state: ResearchState) -> Dict[str, Any]:
    step = state.get("step_count", 0) + 1
    records = state.get("tool_results", [])
    logs = list(state.get("observable_logs", []))
    if not records:
        logs.append(_log_event(state, "validate_tool_outputs", "No tool output to validate.",
                               outcome="INVALID", step=step))
        return {"step_count": step, "validation_status": "INVALID", "observable_logs": logs}

    record = records[-1]
    tool, result, error = record.get("tool"), record.get("result"), record.get("error")
    status, detail = "VALID", "Output structure is valid."
    if error or record.get("status", "SUCCESS" if not error else "ERROR") != "SUCCESS":
        status, detail = "ERROR", f"Tool execution failed: {error or 'unknown error'}"
    elif result is None:
        status, detail = "INVALID", "Tool returned null."
    elif tool == "search_academic_papers":
        if not isinstance(result, dict) or not isinstance(result.get("papers"), list):
            status, detail = "INVALID", "Search response must contain a papers list."
        elif not result["papers"]:
            status, detail = "EMPTY", "Search ran but found no papers."
    elif tool == "validate_metadata_integrity":
        if not isinstance(result, dict) or not isinstance(result.get("records"), list):
            status, detail = "INVALID", "Metadata validation must return a records list."
        elif not result["records"]:
            status, detail = "EMPTY", "No paper metadata records could be validated."
        elif any(not isinstance(r, dict) or not r.get("paper_id") for r in result["records"]):
            status, detail = "INVALID", "A validation record is missing its paper identity."
    elif tool == "read_research_paper":
        if not isinstance(result, dict) or not result.get("paper_id"):
            status, detail = "INVALID", "Extraction result is missing paper_id."
        else:
            extraction_status = _status_value(result.get("status"))
            if extraction_status in FAILED_EXTRACTION_STATUSES or not result.get("extracted_data"):
                status, detail = "EMPTY", (
                    "Paper extraction did not produce usable structured content "
                    f"(status={extraction_status or 'unknown'})."
                )
    elif tool == "generate_citations":
        if not isinstance(result, dict) or not isinstance(result.get("citations"), dict):
            status, detail = "INVALID", "Citation generator returned no citation map."
    elif tool in {"compare_research_papers", "identify_research_gaps", "retrieve_research_context"}:
        if not isinstance(result, dict):
            status, detail = "INVALID", f"{tool} must return a structured object."
    elif tool == "analyze_research_methodology":
        if not isinstance(result, dict) or not isinstance(result.get("analyses"), list):
            status, detail = "INVALID", "Methodology analysis must return a list of analyses."
    elif tool == "analyze_experimental_results":
        if not isinstance(result, list):
            status, detail = "INVALID", "Experimental analysis must return a list."
        elif not result:
            # A completed analysis that found no comparable metrics is a valid finding.
            detail = "Analysis completed; no comparable experimental metrics were reported."
    elif tool == "generate_research_comparison_report":
        if not isinstance(result, dict) or not result.get("report_id"):
            status, detail = "INVALID", "Report service returned no identifiable report object."

    logs.append(_log_event(
        state, "validate_tool_outputs",
        f"Validated '{tool}': {status}. {detail}",
        tool=tool,
        outcome=status,
        step=step,
    ))
    return {
        "validation_status": status,
        "step_count": step,
        "observable_logs": logs,
    }


def _reformulate_search_query(query: str, domain: Optional[str], attempt: int) -> str:
    keywords = _keywords(query) or query
    if attempt == 1 and domain:
        return f"{keywords} {domain}".strip()
    # Second retry broadens further: fewer, core keywords only.
    return _keywords(query, limit=4) or keywords


def retry_handler(state: ResearchState) -> Dict[str, Any]:
    """Bounded retry policy (max ``settings.MAX_RETRIES`` per tool/item)."""
    step = state.get("step_count", 0) + 1
    tool = state.get("selected_tool") or "unknown_operation"
    inputs = dict(state.get("tool_inputs") or {})
    counts = dict(state.get("retry_counts", {}))
    key = _retry_key(tool, inputs)
    current = counts.get(key, 0)
    logs = list(state.get("observable_logs", []))
    phase = state.get("current_task") or TOOL_PHASES.get(tool)
    limit = _max_retries()
    out_of_steps = step >= state.get("max_steps", settings.MAX_AGENT_STEPS)

    last = (state.get("tool_results") or [{}])[-1]
    permanent = last.get("tool") == tool and last.get("retryable") is False

    if current < limit and not out_of_steps and not permanent:
        current += 1
        counts[key] = current
        adjustment = "same parameters"
        if tool == "search_academic_papers":
            inputs["query"] = _reformulate_search_query(inputs.get("query", ""), state.get("domain"), current)
            adjustment = f"reformulated query '{inputs['query']}'"
        logs.append(_log_event(
            state, "retry_handler",
            f"Retry {current}/{limit} for '{tool}' with {adjustment}.",
            tool=tool, outcome="RETRY_REQUESTED", step=step,
        ))
        return {
            "retry_counts": counts,
            "tool_inputs": inputs,
            "validation_status": "RETRY_REQUESTED",
            "step_count": step,
            "observable_logs": logs,
        }

    completed = list(state.get("completed_tasks", []))
    plan = list(state.get("research_plan", []))
    statuses = dict(state.get("phase_statuses", {}))
    failed = set(state.get("failed_items", []))
    extraction_results = list(state.get("extraction_results", []))
    if permanent:
        why = "a permanent error (retrying cannot succeed)"
    elif out_of_steps:
        why = "the step budget is exhausted"
    else:
        why = f"{limit} retries"
    marker = _failure_marker(tool, inputs)

    if marker:
        # Item-scoped failure: record it so selection moves on to other work.
        failed.add(marker)
        if tool == "read_research_paper":
            extraction_results.append({
                "paper_id": inputs.get("paper_id"),
                "status": "FAILED",
                "error_message": f"Extraction failed after {why}.",
            })
        decision = f"'{tool}' failed after {why} ({marker}); item recorded as failed, other work continues."
    elif tool == "search_academic_papers" and state.get("discovered_papers"):
        # A failed *expanded* search keeps the original evidence.
        decision = "Expanded search failed after retries; continuing with previously discovered papers."
    elif phase:
        plan, completed = _resolve_phase(
            plan, completed, phase, "BLOCKED",
            f"Tool '{tool}' failed after {why}. This phase could not be completed.",
        )
        statuses[phase] = "BLOCKED"
        decision = f"'{tool}' failed after {why}. Phase '{phase}' marked BLOCKED."
    else:
        decision = f"Retry limit reached for '{tool}'."

    logs.append(_log_event(
        state, "retry_handler", decision,
        tool=tool, outcome="RETRY_EXHAUSTED",
        details={"phase": phase, "retry_count": current, "retry_key": key},
        step=step,
    ))
    return {
        "retry_counts": counts,
        "tool_inputs": None,
        "validation_status": "RETRY_EXHAUSTED",
        "completed_tasks": completed,
        "research_plan": plan,
        "phase_statuses": statuses,
        "failed_items": sorted(failed),
        "extraction_results": extraction_results,
        "expanded_search_pending": False,
        "step_count": step,
        "observable_logs": logs,
    }


# ---------------------------------------------------------------------------
# 4. Evidence storage and phase accounting
# ---------------------------------------------------------------------------

def store_evidence(state: ResearchState) -> Dict[str, Any]:
    step = state.get("step_count", 0) + 1
    records = state.get("tool_results", [])
    logs = list(state.get("observable_logs", []))
    if not records:
        return {"step_count": step}

    record = records[-1]
    tool, result = record.get("tool"), record.get("result")

    # Failed or structurally invalid tool calls never alter evidence.
    if record.get("error") or record.get("status") != "SUCCESS":
        logs.append(_log_event(state, "store_evidence", f"Did not store output from failed tool '{tool}'.",
                               tool=tool, outcome="SKIPPED", step=step))
        return {"step_count": step, "observable_logs": logs}
    if state.get("validation_status") in {"ERROR", "INVALID"}:
        logs.append(_log_event(state, "store_evidence", f"Rejected structurally invalid output from '{tool}'.",
                               tool=tool, outcome="REJECTED", step=step))
        return {"step_count": step, "observable_logs": logs}

    completed = list(state.get("completed_tasks", []))
    plan = list(state.get("research_plan", []))
    statuses = dict(state.get("phase_statuses", {}))
    actions = list(state.get("analysis_actions_completed", []))
    discovered = list(state.get("discovered_papers", []))
    validated = list(state.get("validated_papers", []))
    extracted = list(state.get("extracted_papers", []))
    extraction_results = list(state.get("extraction_results", []))
    indexed = list(state.get("indexed_papers", []))
    indexing_results = list(state.get("indexing_results", []))
    retrieved = list(state.get("retrieved_evidence", []))
    metrics = list(state.get("metric_comparisons", []))
    methodologies = list(state.get("methodology_analyses", []))
    comparisons = list(state.get("paper_comparisons", []))
    gaps = list(state.get("identified_gaps", []))
    citations = dict(state.get("citations", {}))
    failed = set(state.get("failed_items", []))
    security_flags = list(state.get("security_flags", []))
    final_report = state.get("final_report")
    successful = list(state.get("successful_tool_signatures", []))
    expanded_pending = bool(state.get("expanded_search_pending"))
    note = f"Stored output from '{tool}'."

    if record.get("signature"):
        successful.append(record["signature"])

    if tool == "search_academic_papers":
        papers = result.get("papers", []) if isinstance(result, dict) else []
        discovered, added = _merge_discovered(discovered, papers)
        expanded_pending = False
        note = f"Search returned {len(papers)} paper(s); {added} new after de-duplication."
        if discovered:
            statuses["search_and_validate"] = "IN_PROGRESS"
        else:
            plan, completed = _resolve_phase(
                plan, completed, "search_and_validate", "BLOCKED",
                "Search returned zero papers. No evidence-based review can proceed.",
            )
            statuses["search_and_validate"] = "BLOCKED"

    elif tool == "validate_metadata_integrity":
        new_records = result.get("records", []) if isinstance(result, dict) else []
        known = {_record_key(v) for v in validated}
        for rec in new_records:
            if _record_key(rec) not in known:
                validated.append(rec)
                known.add(_record_key(rec))
        for rejected in (result.get("rejected", []) if isinstance(result, dict) else []):
            if rejected.get("paper_id"):
                failed.add(f"validate:{rejected['paper_id']}")
        conflicting = [r for r in new_records if _status_value(r.get("status")) == "CONFLICTING"]
        note = (
            f"Validated {len(new_records)} record(s); {len(conflicting)} conflicting record(s) "
            "excluded from evidence."
        )

    elif tool == "read_research_paper":
        pid = result.get("paper_id") if isinstance(result, dict) else None
        extraction_results.append({
            "paper_id": pid,
            "status": _status_value(result.get("status")) if isinstance(result, dict) else "",
            "page_count": result.get("page_count") if isinstance(result, dict) else None,
            "warnings": (result.get("warnings") or [])[:10] if isinstance(result, dict) else [],
        })
        usable = (
            pid and isinstance(result, dict) and result.get("extracted_data")
            and _status_value(result.get("status")) not in FAILED_EXTRACTION_STATUSES
        )
        if usable and pid not in extracted:
            extracted.append(pid)
        if not usable and pid:
            failed.add(f"extract:{pid}")
        injection_warnings = [
            w for w in (result.get("warnings") or []) if "injection" in str(w).lower()
        ] if isinstance(result, dict) else []
        if injection_warnings:
            security_flags.append({"paper_id": pid, "type": "PROMPT_INJECTION_PATTERN",
                                   "details": injection_warnings[:3]})
        indexing = record.get("indexing")
        if isinstance(indexing, dict):
            indexing_results.append(indexing)
            if _status_value(indexing.get("status")) in {"SUCCESS", "PARTIAL"} and pid and pid not in indexed:
                indexed.append(pid)
        note = (
            f"Paper {pid}: {'usable full text extracted' if usable else 'no usable full text'}; "
            f"indexing={_status_value((indexing or {}).get('status')) or 'NOT_RUN'}."
        )

    elif tool == "retrieve_research_context":
        items = result.get("items", []) if isinstance(result, dict) else []
        retrieved = items
        actions.append("rag_retrieval")
        note = f"Retrieved {len(items)} supporting passage(s) from the vector store."
        if isinstance(result, dict) and result.get("warning"):
            note += f" Warning: {result['warning']}"

    elif tool == "compare_research_papers":
        comparisons.append(result)
        actions.append("paper_comparison")

    elif tool == "analyze_research_methodology":
        methodologies = list(result.get("analyses", []))
        actions.append("methodology")
        if result.get("errors"):
            note += f" Per-paper failures: {result['errors']}."

    elif tool == "analyze_experimental_results":
        metrics = list(result) if isinstance(result, list) else []
        actions.append("experimental_results")
        note = f"Experimental-results analysis produced {len(metrics)} metric comparison(s)."

    elif tool == "identify_research_gaps":
        found = result.get("gaps", []) if isinstance(result, dict) else []
        gaps = found if isinstance(found, list) else []
        plan, completed = _resolve_phase(
            plan, completed, "identify_gaps", "COMPLETED",
            f"Gap analysis executed and returned {len(gaps)} candidate gap(s).",
        )
        statuses["identify_gaps"] = "COMPLETED"

    elif tool == "generate_citations":
        citations.update(result.get("citations", {}))
        for key in (result.get("failed") or {}):
            failed.add(f"citation:{key}")
        note = f"Generated {len(result.get('citations', {}))} citation(s); {len(result.get('failed') or {})} failed."

    elif tool == "generate_research_comparison_report":
        final_report = result
        plan, completed = _resolve_phase(
            plan, completed, "synthesize_report", "COMPLETED",
            "Comparative report service returned a report object.",
        )
        statuses["synthesize_report"] = "COMPLETED"

    logs.append(_log_event(
        state, "store_evidence", note,
        tool=tool, outcome="SUCCESS",
        details={"completed_tasks": completed, "phase_statuses": statuses},
        step=step,
    ))
    return {
        "discovered_papers": discovered,
        "validated_papers": validated,
        "extracted_papers": extracted,
        "extraction_results": extraction_results,
        "indexed_papers": indexed,
        "indexing_results": indexing_results,
        "retrieved_evidence": retrieved,
        "metric_comparisons": metrics,
        "methodology_analyses": methodologies,
        "paper_comparisons": comparisons,
        "identified_gaps": gaps,
        "citations": citations,
        "final_report": final_report,
        "completed_tasks": completed,
        "research_plan": plan,
        "phase_statuses": statuses,
        "analysis_actions_completed": list(dict.fromkeys(actions)),
        "failed_items": sorted(failed),
        "security_flags": security_flags,
        "successful_tool_signatures": successful,
        "expanded_search_pending": expanded_pending,
        "step_count": step,
        "observable_logs": logs,
    }


def reflect_and_revise(state: ResearchState) -> Dict[str, Any]:
    """Assess progress, revise the plan when evidence is insufficient, enforce max steps."""
    step = state.get("step_count", 0) + 1
    max_steps = state.get("max_steps", settings.MAX_AGENT_STEPS)
    reflections = list(state.get("reflections", []))
    statuses = dict(state.get("phase_statuses", {}))
    extracted = state.get("extracted_papers", [])
    plan = list(state.get("research_plan", []))
    completed = list(state.get("completed_tasks", []))
    blocked = [k for k, v in statuses.items() if v == "BLOCKED"]
    logs = list(state.get("observable_logs", []))
    update: Dict[str, Any] = {}
    outcome = "SUCCESS"

    if step >= max_steps:
        decision = (
            f"Maximum step budget reached ({step}/{max_steps}). Stopping tool execution; "
            "a report bounded to the evidence gathered so far will be produced."
        )
        update["stopping_reason"] = "MAX_STEPS_REACHED"
        update["pending_revision"] = None
        outcome = "MAX_STEPS_REACHED"
    elif state.get("pending_revision"):
        revisions = state.get("plan_revisions", 0) + 1
        reason = state["pending_revision"]
        plan, completed = _reopen_phase(
            plan, completed, "search_and_validate",
            f"Plan revision {revisions}: {reason} Expanding the search for more open-access candidates.",
        )
        statuses["search_and_validate"] = "IN_PROGRESS"
        plan = _update_plan_task(plan, "extract_and_index", "PENDING", "Awaiting candidates from the expanded search.")
        statuses["extract_and_index"] = "PENDING"
        decision = f"PLAN REVISED ({revisions}): {reason} Re-opening search with a larger candidate pool."
        update.update({
            "research_plan": plan,
            "completed_tasks": completed,
            "phase_statuses": statuses,
            "plan_revisions": revisions,
            "pending_revision": None,
            "expanded_search_pending": True,
        })
        outcome = "PLAN_REVISED"
    elif state.get("validation_status") == "RETRY_EXHAUSTED":
        last_failed = (state.get("failed_items") or [])
        decision = (
            f"Retries exhausted or not applicable for a tool call; {len(last_failed)} item(s) recorded as failed. "
            f"Continuing the plan with the remaining candidates ({len(extracted)} paper(s) extracted so far)."
        )
        outcome = "PARTIAL"
    else:
        decision = (
            f"Evidence checkpoint: {len(extracted)} paper(s) extracted; "
            f"{len(blocked)} phase(s) blocked."
        )

    if blocked and outcome not in ("MAX_STEPS_REACHED", "PLAN_REVISED"):
        decision += " Blocked phases will remain explicitly identified in the final outcome."
        outcome = "PARTIAL"

    reflections.append(decision)
    logs.append(_log_event(state, "reflect_and_revise", decision, outcome=outcome, step=step))
    update.update({"reflections": reflections, "step_count": step, "observable_logs": logs})
    return update


def critique_findings(state: ResearchState) -> Dict[str, Any]:
    extracted = state.get("extracted_papers", [])
    statuses = state.get("phase_statuses", {})
    completed = set(state.get("completed_tasks", []))
    failed = set(state.get("failed_items", []))
    required = {"search_and_validate", "extract_and_index", "compare_and_analyze", "identify_gaps", "generate_citations"}
    missing = sorted(required - completed)
    blocked = sorted(k for k, v in statuses.items() if v == "BLOCKED")
    failed_actions = sorted(item for item in failed if item.startswith("action:"))

    if not extracted:
        verdict = "INSUFFICIENT_EVIDENCE"
        reason = "No papers yielded usable extracted full-text evidence."
    elif len(extracted) < 2:
        verdict = "INSUFFICIENT_EVIDENCE"
        reason = "Only one paper was extracted; comparative claims cannot be supported."
    elif missing or blocked or failed_actions:
        verdict = "NEEDS_REVISION"
        reason = f"Research phases incomplete/blocked or analysis actions failed: {missing or blocked or failed_actions}."
    else:
        verdict = "APPROVED"
        reason = "Required research phases ran; conclusions must remain bounded by the extracted evidence."

    step = state.get("step_count", 0) + 1
    logs = list(state.get("observable_logs", []))
    logs.append(_log_event(
        state, "critique_findings",
        f"Critique verdict: {verdict}. {reason}",
        outcome=verdict,
        details={"extracted_papers": len(extracted), "missing_phases": missing,
                 "blocked_phases": blocked, "failed_actions": failed_actions},
        step=step,
    ))
    return {"critique_status": verdict, "step_count": step, "observable_logs": logs}


# ---------------------------------------------------------------------------
# 5. Report generation and honest termination
# ---------------------------------------------------------------------------

def _references_from_state(state: ResearchState) -> List[Dict[str, Any]]:
    references = []
    for key, citation in (state.get("citations") or {}).items():
        if isinstance(citation, dict):
            references.append({
                "paper_id": key,
                "title": citation.get("title") or key,
                "citation_text": citation.get("citation_text"),
                "doi": citation.get("doi"),
            })
        else:
            references.append({"paper_id": key, "title": key, "citation_text": str(citation)})
    return references


def _partial_report(state: ResearchState, goal: str, extracted: List[str], llm, error: Optional[str]) -> Dict[str, Any]:
    """A clearly labelled PARTIAL report. It states what is unknown; it invents nothing."""
    validated = state.get("validated_papers", [])
    not_established = (
        "Not established: a comparison requires usable full text from at least two papers "
        f"(available: {len(extracted)})."
    )
    limitations = [
        f"Only {len(extracted)} paper(s) yielded usable full text out of {len(validated)} validated metadata record(s).",
        "No benchmark rankings or model-superiority claims were generated.",
    ]
    if error:
        limitations.append(f"Comparative report generation failed: {error}")
    blocked = {k: v for k, v in (state.get("phase_statuses") or {}).items() if v == "BLOCKED"}
    if blocked:
        limitations.append(f"Blocked phases: {', '.join(sorted(blocked))}.")
    return {
        "report_id": f"rep_{uuid.uuid4().hex[:8]}",
        "title": f"Research Investigation (Partial): {goal}",
        "report_status": "PARTIAL",
        "research_question": goal,
        "papers_analyzed": extracted,
        "objectives_and_methodology_comparison": not_established,
        "dataset_comparison": not_established,
        "model_architecture_comparison": not_established,
        "experimental_results_comparison": (
            f"{len(state.get('metric_comparisons', []))} metric comparison(s) were computed; see metric tables."
            if state.get("metric_comparisons") else not_established
        ),
        "strengths_and_limitations": not_established,
        "research_gaps": state.get("identified_gaps", []),
        "evidence_backed_conclusions": (
            "No evidence-backed comparative conclusion can be drawn from the available material."
        ),
        "limitations_of_analysis": " ".join(limitations),
        "references": _references_from_state(state),
        "metric_tables": state.get("metric_comparisons", []),
        "is_degraded": _router_degraded(llm),
        "created_at": _now_iso(),
    }


async def generate_report_node(
    state: ResearchState,
    router: Optional[ModelRouter] = None,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
) -> Dict[str, Any]:
    step = state.get("step_count", 0) + 1
    extracted = list(state.get("extracted_papers", []))
    goal = state.get("research_goal") or "Academic Research Evaluation"
    llm = router or default_model_router
    report = state.get("final_report")
    generation_error = None
    logs = list(state.get("observable_logs", []))

    if state.get("stopping_reason") == "UNACCEPTABLE_RISK":
        logs.append(_log_event(state, "generate_report", "No report produced for a refused request.",
                               outcome="SKIPPED", step=step))
        return {"final_report": None, "step_count": step, "observable_logs": logs}

    if not report and len(extracted) >= 2:
        try:
            schema = await generate_research_comparison_report(
                paper_ids=extracted,
                research_question=goal,
                repository=repository or default_paper_repo,
                vector_db=vector_db or default_vector_store,
                router=llm,
                precomputed=_precomputed_analyses(state),
            )
            report = _sanitize_data(schema)
        except Exception as exc:
            logger.warning("Comparative report generation failed: %s", exc, exc_info=True)
            generation_error = f"{type(exc).__name__}: {exc}"

    if not report:
        report = _partial_report(state, goal, extracted, llm, generation_error)

    report = _sanitize_data(report)
    report.setdefault("title", f"Research Comparison Report: {goal}")
    report.setdefault("report_status", "COMPLETE")
    report["phase_statuses"] = state.get("phase_statuses", {})
    report["citations"] = state.get("citations", {})
    report["extracted_papers"] = extracted
    report["security_flags"] = state.get("security_flags", [])
    report["is_degraded"] = bool(report.get("is_degraded")) or bool(state.get("is_degraded")) or _router_degraded(llm)

    saved_paths: Dict[str, str] = {}
    try:
        markdown = export_report_markdown(ResearchComparisonReportSchema(**report))
    except Exception as exc:
        logger.warning("Report could not be rendered to Markdown: %s", exc)
        markdown = None
    try:
        saved_paths = save_report(report, markdown=markdown)
    except Exception as exc:
        logger.error("Report '%s' could not be saved: %s", report.get("report_id"), exc)

    complete = report.get("report_status") == "COMPLETE" and len(extracted) >= 2
    logs.append(_log_event(
        state, "generate_report",
        "Generated an evidence-bounded report; unavailable evidence is disclosed."
        + (f" Saved to {', '.join(saved_paths)}." if saved_paths else " Report was not saved to disk."),
        tool="generate_research_comparison_report",
        outcome="SUCCESS" if complete else "PARTIAL",
        step=step,
    ))
    return {
        "final_report": report,
        "saved_report_paths": saved_paths,
        "is_degraded": report["is_degraded"],
        "step_count": step,
        "observable_logs": logs,
    }


def finish_task(state: ResearchState) -> Dict[str, Any]:
    step = state.get("step_count", 0) + 1
    stopping = state.get("stopping_reason")
    statuses = state.get("phase_statuses", {})
    extracted = state.get("extracted_papers", [])
    critique = state.get("critique_status")

    if not stopping or stopping in {"COMPLETED", "GOAL_ACHIEVED"}:
        if critique == "APPROVED" and len(extracted) >= 2:
            stopping = "GOAL_ACHIEVED"
        elif critique == "INSUFFICIENT_EVIDENCE" or not extracted:
            stopping = "INSUFFICIENT_EVIDENCE"
        else:
            stopping = "PARTIAL"

    logs = list(state.get("observable_logs", []))
    logs.append(_log_event(
        state, "finish_task",
        f"Workflow terminated with status '{stopping}'.",
        outcome=stopping,
        details={
            "extracted_papers": len(extracted),
            "phase_statuses": statuses,
            "steps_used": step,
            "max_steps": state.get("max_steps"),
        },
        step=step,
    ))
    return {"stopping_reason": stopping, "step_count": step, "observable_logs": logs}

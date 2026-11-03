"""Meta-Tool: Research Planner.

Creates the structured research plan (a to-do list of phases) that the agent
follows and checks off. The planner is a pure function: it reads facts about
the session and returns a plan. It does not touch LangGraph state or logs;
the ``create_plan`` node in ``agents/nodes.py`` handles that.

On a resumed session the planner recognises work that is already done (papers
already validated or extracted), so the agent does not repeat it.
"""
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

PHASES: Tuple[str, ...] = (
    "search_and_validate",
    "extract_and_index",
    "compare_and_analyze",
    "identify_gaps",
    "generate_citations",
    "synthesize_report",
)

PHASE_DESCRIPTIONS: Dict[str, str] = {
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

# A phase in one of these states is finished (successfully or not) and is not
# scheduled again unless the plan is explicitly revised.
RESOLVED_STATUSES = {"COMPLETED", "BLOCKED", "SKIPPED", "FAILED"}


def build_research_plan(
    completed_tasks: Iterable[str],
    previous_statuses: Optional[Mapping[str, str]] = None,
    *,
    has_usable_validated: bool = False,
    has_extracted: bool = False,
    has_discovered: bool = False,
    has_citations: bool = False,
    clarification_received: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, str], Dict[str, str]]:
    """Build the research plan.

    Args:
        completed_tasks: Phases the caller already reports as complete.
        previous_statuses: Phase statuses from an earlier run (resumption).
        has_usable_validated: At least one paper passed metadata validation.
        has_extracted: At least one paper's full text was extracted.
        has_discovered: Search found candidates that are not yet validated.
        has_citations: Some citations already exist.
        clarification_received: The user answered a clarification question,
            so phases that were BLOCKED before get another chance.

    Returns:
        ``(plan, completed, statuses, reasons)``: the task list, the completed
        phase ids, a status per phase, and the reason for any status the
        planner set from existing evidence.
    """
    plan = [{
        "task_id": phase,
        "name": phase.replace("_", " ").title(),
        "description": PHASE_DESCRIPTIONS[phase],
        "status": "PENDING",
    } for phase in PHASES]

    completed = [task for task in completed_tasks if task in PHASES]
    previous = dict(previous_statuses or {})
    statuses = {phase: previous.get(phase, "PENDING") for phase in PHASES}
    reasons: Dict[str, str] = {}

    if has_usable_validated and "search_and_validate" not in completed:
        completed.append("search_and_validate")
    if has_extracted and "extract_and_index" not in completed:
        completed.append("extract_and_index")

    # Tasks reported complete by the caller (resumption) must not stay PENDING.
    for phase in completed:
        if statuses.get(phase) not in RESOLVED_STATUSES:
            statuses[phase] = "COMPLETED"
            reasons[phase] = "Recognized from evidence already present in the session."

    # Blocked phases from a previous run are retried after new information
    # (e.g. a clarification) rather than being treated as permanently final.
    if clarification_received:
        for phase, status in list(statuses.items()):
            if status == "BLOCKED":
                statuses[phase] = "PENDING"
                completed = [task for task in completed if task != phase]

    if not has_usable_validated and has_discovered and statuses["search_and_validate"] == "PENDING":
        statuses["search_and_validate"] = "IN_PROGRESS"

    if has_citations and statuses["generate_citations"] == "PENDING":
        statuses["generate_citations"] = "IN_PROGRESS"

    for task in plan:
        task["status"] = statuses[task["task_id"]]
        if task["task_id"] in reasons:
            task["status_reason"] = reasons[task["task_id"]]

    return plan, completed, statuses, reasons

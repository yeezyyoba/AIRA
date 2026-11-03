"""AIRA Graph Routing and Conditional Edges.

Each ``route_after_*`` function reads the research state after a node runs
and returns the name of the next node. Together they implement the agent's
branches, loops, retries and stopping rules:

* ``check_information`` -> ask the user, refuse (risk), or plan.
* ``select_tools`` -> execute a tool, or reflect when nothing is left to do.
* ``validate_tool_outputs`` -> retry a failed/invalid call, or store evidence.
* ``retry_handler`` -> run the tool again, or give up and reflect.
* ``reflect_and_revise`` -> pick the next tool, or move on to critique.
* ``critique_findings`` -> loop back for revision, or write the report.

The step budget (``MAX_AGENT_STEPS``) is enforced here so the graph always
terminates.
"""
from agents.state import ResearchState
from app.core.config import settings


def _plan_tasks(state: ResearchState):
    """Normalise plan entries; tolerate legacy string task ids."""
    tasks = []
    for task in state.get("research_plan", []) or []:
        if isinstance(task, dict):
            tasks.append(task)
        elif task is not None:
            tasks.append({"task_id": str(task), "status": "PENDING"})
    return tasks


def _step_budget_exhausted(state: ResearchState) -> bool:
    return state.get("step_count", 0) >= state.get("max_steps", settings.MAX_AGENT_STEPS)


def route_after_check_information(state: ResearchState) -> str:
    if state.get("stopping_reason") == "UNACCEPTABLE_RISK" or state.get("risk_flags"):
        return "finish_task"
    if state.get("missing_information"):
        return "ask_user"
    return "create_plan"


def route_after_select_tools(state: ResearchState) -> str:
    if state.get("selected_tool"):
        return "execute_tools"
    return "reflect_and_revise"


def route_after_validation(state: ResearchState) -> str:
    status = state.get("validation_status")
    if status in ("INVALID", "ERROR"):
        return "retry_handler"
    # An initial search that finds nothing is retried with a reformulated
    # query; an empty *expanded* search simply adds no new candidates.
    if (
        status == "EMPTY"
        and state.get("selected_tool") == "search_academic_papers"
        and not state.get("discovered_papers")
    ):
        return "retry_handler"
    return "store_evidence"


def route_after_retry(state: ResearchState) -> str:
    if state.get("validation_status") == "RETRY_REQUESTED":
        return "execute_tools"
    return "reflect_and_revise"


def route_after_reflection(state: ResearchState) -> str:
    if state.get("stopping_reason") or _step_budget_exhausted(state):
        return "critique_findings"

    if state.get("final_report"):
        return "critique_findings"

    plan = _plan_tasks(state)
    completed = set(state.get("completed_tasks", []))
    if plan and all(task.get("task_id") in completed for task in plan):
        return "critique_findings"

    return "select_tools"


def route_after_critique(state: ResearchState) -> str:
    """Loop back only when a genuinely unresolved phase remains and budget allows."""
    if state.get("stopping_reason") or _step_budget_exhausted(state):
        return "generate_report"

    completed = set(state.get("completed_tasks", []))
    unresolved = any(
        task.get("task_id") not in completed
        and task.get("status", "PENDING") in ("PENDING", "IN_PROGRESS")
        for task in _plan_tasks(state)
    )

    if state.get("critique_status") == "NEEDS_REVISION" and unresolved:
        return "select_tools"

    return "generate_report"

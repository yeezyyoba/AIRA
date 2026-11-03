"""Meta-Tool: Plan Updater.

Updates the research plan while the agent works: checks off phases, records
why a phase was blocked, and revises the plan when evidence is insufficient
(for example, re-opening the search when too few PDFs could be read).

All functions are pure. They take the current plan and return new values;
the graph nodes in ``agents/nodes.py`` store the results in state.
"""
from typing import Any, Dict, List, Mapping, Optional, Tuple

from meta_tools.planner import PHASE_DESCRIPTIONS, RESOLVED_STATUSES

Plan = List[Dict[str, Any]]


def update_plan_task(plan: Plan, task_id: str, status: str, reason: Optional[str] = None) -> Plan:
    """Set one task's status (adding the task if it is not in the plan)."""
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


def resolve_phase(plan: Plan, completed: List[str], phase: str, status: str, reason: str) -> Tuple[Plan, List[str]]:
    """Resolve a phase while preserving whether it completed or was blocked."""
    plan = update_plan_task(plan, phase, status, reason)
    if status in RESOLVED_STATUSES and phase not in completed:
        completed.append(phase)
    return plan, completed


def reopen_phase(plan: Plan, completed: List[str], phase: str, reason: str) -> Tuple[Plan, List[str]]:
    """Put a finished phase back in progress."""
    plan = update_plan_task(plan, phase, "IN_PROGRESS", reason)
    completed = [task for task in completed if task != phase]
    return plan, completed


def revise_plan_for_expanded_search(
    plan: Plan,
    completed: List[str],
    statuses: Mapping[str, str],
    revision_number: int,
    reason: str,
) -> Tuple[Plan, List[str], Dict[str, str], str]:
    """Revise the plan when too few papers yielded usable evidence.

    Re-opens ``search_and_validate`` to fetch a larger candidate pool and
    resets ``extract_and_index`` to wait for the new candidates.

    Returns:
        ``(plan, completed, statuses, decision)`` where ``decision`` is a
        one-line explanation for the observable log.
    """
    statuses = dict(statuses)
    plan, completed = reopen_phase(
        plan, completed, "search_and_validate",
        f"Plan revision {revision_number}: {reason} Expanding the search for more open-access candidates.",
    )
    statuses["search_and_validate"] = "IN_PROGRESS"
    plan = update_plan_task(plan, "extract_and_index", "PENDING", "Awaiting candidates from the expanded search.")
    statuses["extract_and_index"] = "PENDING"
    decision = f"PLAN REVISED ({revision_number}): {reason} Re-opening search with a larger candidate pool."
    return plan, completed, statuses, decision

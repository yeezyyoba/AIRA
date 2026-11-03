"""Meta-Tool: Research Critic.

Reviews the agent's own progress before a report is written and returns one
of three verdicts:

* ``APPROVED``: every required phase ran and at least two papers have
  extracted full text, so comparative conclusions can be drawn.
* ``NEEDS_REVISION``: some phase is incomplete, blocked, or an analysis
  action failed. The graph loops back to tool selection if budget allows.
* ``INSUFFICIENT_EVIDENCE``: fewer than two papers have usable full text,
  so a comparison cannot be supported.

The critic is a pure function; the ``critique_findings`` node logs the verdict.
"""
from typing import Any, Dict, Iterable, Mapping

APPROVED = "APPROVED"
NEEDS_REVISION = "NEEDS_REVISION"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

REQUIRED_PHASES = frozenset({
    "search_and_validate",
    "extract_and_index",
    "compare_and_analyze",
    "identify_gaps",
    "generate_citations",
})


def critique_research(
    extracted_papers: Iterable[str],
    phase_statuses: Mapping[str, str],
    completed_tasks: Iterable[str],
    failed_items: Iterable[str],
) -> Dict[str, Any]:
    """Judge whether the gathered evidence supports a final report.

    Returns:
        A dict with ``verdict``, ``reason``, ``missing_phases``,
        ``blocked_phases`` and ``failed_actions``.
    """
    extracted = list(extracted_papers or [])
    completed = set(completed_tasks or [])
    failed = set(failed_items or [])
    missing = sorted(REQUIRED_PHASES - completed)
    blocked = sorted(k for k, v in (phase_statuses or {}).items() if v == "BLOCKED")
    failed_actions = sorted(item for item in failed if item.startswith("action:"))

    if not extracted:
        verdict = INSUFFICIENT_EVIDENCE
        reason = "No papers yielded usable extracted full-text evidence."
    elif len(extracted) < 2:
        verdict = INSUFFICIENT_EVIDENCE
        reason = "Only one paper was extracted; comparative claims cannot be supported."
    elif missing or blocked or failed_actions:
        verdict = NEEDS_REVISION
        reason = f"Research phases incomplete/blocked or analysis actions failed: {missing or blocked or failed_actions}."
    else:
        verdict = APPROVED
        reason = "Required research phases ran; conclusions must remain bounded by the extracted evidence."

    return {
        "verdict": verdict,
        "reason": reason,
        "missing_phases": missing,
        "blocked_phases": blocked,
        "failed_actions": failed_actions,
    }

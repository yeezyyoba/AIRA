"""Meta-Tool: Finish Research.

Declares how the workflow ended. An earlier, specific stop (such as
``MAX_STEPS_REACHED``, ``UNACCEPTABLE_RISK`` or
``USER_CLARIFICATION_REQUIRED``) is kept as is. Otherwise the outcome is
derived from the critic's verdict:

* ``GOAL_ACHIEVED``: the critic approved and at least two papers were extracted.
* ``INSUFFICIENT_EVIDENCE``: the critic said so, or nothing was extracted.
* ``PARTIAL``: anything in between.

``determine_stopping_reason`` is pure; the ``finish_task`` node logs it.
"""
from typing import Optional

GOAL_ACHIEVED = "GOAL_ACHIEVED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
PARTIAL = "PARTIAL"


def determine_stopping_reason(
    stopping_reason: Optional[str],
    critique_status: Optional[str],
    extracted_count: int,
) -> str:
    """Return the final stopping reason for the run."""
    if stopping_reason and stopping_reason not in {"COMPLETED", GOAL_ACHIEVED}:
        return stopping_reason
    if critique_status == "APPROVED" and extracted_count >= 2:
        return GOAL_ACHIEVED
    if critique_status == "INSUFFICIENT_EVIDENCE" or not extracted_count:
        return INSUFFICIENT_EVIDENCE
    return PARTIAL

"""AIRA Evidence Validator.

Decides what is allowed to become evidence in the research state. These are
the rules the ``store_evidence`` node applies before it records anything:

* Output from a failed or structurally invalid tool call never becomes evidence.
* A paper counts as extracted only if it has structured content and its
  extraction did not fail (scanned PDFs that need OCR do not count).
* Text inside a paper that looks like a prompt injection is flagged as a
  security concern, never followed.
"""
from typing import Any, Dict, List, Optional, Tuple

from validation.tool_validation import FAILED_EXTRACTION_STATUSES, status_value


def can_store_tool_output(record: Dict[str, Any], validation_status: Optional[str]) -> Tuple[bool, str, str]:
    """Check whether a tool result may be stored as evidence.

    Returns:
        ``(allowed, outcome, reason)``. ``outcome`` is ``"SUCCESS"`` when
        allowed, otherwise ``"SKIPPED"`` (the tool failed) or ``"REJECTED"``
        (the output failed validation).
    """
    tool = record.get("tool")
    if record.get("error") or record.get("status") != "SUCCESS":
        return False, "SKIPPED", f"Did not store output from failed tool '{tool}'."
    if validation_status in {"ERROR", "INVALID"}:
        return False, "REJECTED", f"Rejected structurally invalid output from '{tool}'."
    return True, "SUCCESS", ""


def is_usable_extraction(result: Any) -> bool:
    """True if a paper-reader result contains usable full-text evidence."""
    return bool(
        isinstance(result, dict)
        and result.get("paper_id")
        and result.get("extracted_data")
        and status_value(result.get("status")) not in FAILED_EXTRACTION_STATUSES
    )


def injection_warnings(result: Any, limit: Optional[int] = None) -> List[Any]:
    """Return the reader's prompt-injection warnings for a paper (if any)."""
    if not isinstance(result, dict):
        return []
    found = [w for w in (result.get("warnings") or []) if "injection" in str(w).lower()]
    return found[:limit] if limit is not None else found

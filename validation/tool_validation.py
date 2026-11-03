"""AIRA Tool Output Validator.

Every tool result is checked here before the agent trusts it. The validator
returns one of:

* ``VALID``: the output has the expected structure.
* ``EMPTY``: the tool ran correctly but found nothing (no papers, no usable
  full text). Empty is a finding, not a crash.
* ``INVALID``: the output is malformed (wrong type, missing identity).
* ``ERROR``: the tool raised or reported a failure.

The graph routes ``INVALID`` and ``ERROR`` to the retry handler. The
``validate_tool_outputs`` node in ``agents/nodes.py`` calls
``validate_tool_output`` and logs the result.
"""
from typing import Any, Optional, Tuple

# Extraction statuses that mean "no usable full text".
FAILED_EXTRACTION_STATUSES = {"FAILED", "SCANNED_NEEDS_OCR"}


def status_value(raw: Any) -> str:
    """Normalise a status that may be an Enum, a plain string, or missing."""
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().upper()


def validate_tool_output(
    tool: Optional[str],
    result: Any,
    error: Optional[str] = None,
    record_status: Optional[str] = "SUCCESS",
) -> Tuple[str, str]:
    """Check one tool result.

    Args:
        tool: Name of the tool that produced the result.
        result: The tool's return value (already converted to plain JSON types).
        error: Error message if the tool failed.
        record_status: ``"SUCCESS"`` or ``"ERROR"`` as recorded by the executor.

    Returns:
        ``(status, detail)`` where ``detail`` explains the status in one sentence.
    """
    status, detail = "VALID", "Output structure is valid."
    if error or record_status != "SUCCESS":
        return "ERROR", f"Tool execution failed: {error or 'unknown error'}"
    if result is None:
        return "INVALID", "Tool returned null."

    if tool == "search_academic_papers":
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
            extraction_status = status_value(result.get("status"))
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

    return status, detail

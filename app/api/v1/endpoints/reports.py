"""Reports API Endpoints.

Enables generating structured comparison reports and exporting to Markdown and JSON.
"""
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Query, Response, status

from app.schemas.reports import ComparisonReportRequest
from services.report_service import (
    export_report_json,
    export_report_markdown,
    generate_research_comparison_report,
)
from logging_config.logger import get_logger
from tools.file_manager import FileManagerError, list_saved_reports, load_report, save_report
from validation.schemas import ResearchComparisonReportSchema

router = APIRouter()
logger = get_logger("api.reports")

# In-process cache of validated reports, keyed by report_id only.
_REPORTS_STORE: Dict[str, ResearchComparisonReportSchema] = {}
# Session id -> report id, so a session's report can be exported by session id.
_SESSION_REPORTS: Dict[str, str] = {}


def _resolve_report(report_id: str) -> Optional[ResearchComparisonReportSchema]:
    """Find a report by id (or session id) in memory, then on disk."""
    report_id = _SESSION_REPORTS.get(report_id, report_id)
    if report_id in _REPORTS_STORE:
        return _REPORTS_STORE[report_id]
    try:
        data = load_report(report_id)
    except FileManagerError as exc:
        logger.warning("Rejected report lookup '%s': %s", report_id, exc)
        return None
    if not data:
        return None
    try:
        schema = ResearchComparisonReportSchema(**data)
    except Exception as exc:
        logger.error("Saved report '%s' does not match the report schema: %s", report_id, exc)
        return None
    _REPORTS_STORE[schema.report_id] = schema
    return schema


@router.get("/", tags=["Reports"])
def list_reports() -> dict:
    """List generated research reports (in-memory and saved on disk), one entry per report."""
    summaries: Dict[str, dict] = {}
    try:
        for item in list_saved_reports():
            summaries[item["report_id"]] = item
    except Exception as exc:
        logger.error("Could not list saved reports: %s", exc)
    for r in _REPORTS_STORE.values():
        summaries[r.report_id] = {
            "report_id": r.report_id,
            "research_question": r.research_question,
            "papers_analyzed": r.papers_analyzed,
            "created_at": r.created_at,
            "is_degraded": r.is_degraded,
            "report_status": r.report_status,
        }
    ordered = sorted(summaries.values(), key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"reports": ordered, "total": len(ordered)}


@router.post("/comparison", response_model=ResearchComparisonReportSchema, tags=["Reports"])
async def create_comparison_report_endpoint(request: ComparisonReportRequest) -> ResearchComparisonReportSchema:
    """Generate a structured 11-section research comparison report."""
    try:
        report = await generate_research_comparison_report(
            paper_ids=request.paper_ids,
            research_question=request.research_question,
        )
        _REPORTS_STORE[report.report_id] = report
        try:
            save_report(report.model_dump(mode="json"), markdown=export_report_markdown(report))
        except Exception as exc:
            logger.error("Report '%s' was generated but could not be saved: %s", report.report_id, exc)
        return report
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Report generation failed: {e}")


@router.get("/{report_id}", response_model=ResearchComparisonReportSchema, tags=["Reports"])
def get_report_endpoint(report_id: str) -> ResearchComparisonReportSchema:
    """Retrieve a previously generated research report by ID."""
    report = _resolve_report(report_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report '{report_id}' not found.")
    return report


@router.get("/{report_id}/export", tags=["Reports"])
def export_report_endpoint(
    report_id: str,
    format: str = Query("markdown", description="Export format: 'markdown' or 'json'"),
) -> Response:
    """Export research report to publication-ready Markdown or JSON format."""
    report = _resolve_report(report_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report '{report_id}' not found.")

    fmt = format.strip().lower()

    if fmt in ("md", "markdown"):
        content = export_report_markdown(report)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="aira_report_{report.report_id}.md"'},
        )
    elif fmt == "json":
        content = export_report_json(report)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="aira_report_{report.report_id}.json"'},
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{format}'. Supported formats: 'markdown', 'json'.",
        )

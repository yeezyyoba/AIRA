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
from validation.schemas import ResearchComparisonReportSchema

router = APIRouter()

# In-memory storage for generated reports in Phase 6
_REPORTS_STORE: Dict[str, ResearchComparisonReportSchema] = {}


@router.get("/", tags=["Reports"])
def list_reports() -> dict:
    """List generated research reports."""
    summaries = [
        {
            "report_id": r.report_id,
            "research_question": r.research_question,
            "papers_analyzed": r.papers_analyzed,
            "created_at": r.created_at,
            "is_degraded": r.is_degraded,
        }
        for r in _REPORTS_STORE.values()
    ]
    return {"reports": summaries, "total": len(summaries)}


@router.post("/comparison", response_model=ResearchComparisonReportSchema, tags=["Reports"])
async def create_comparison_report_endpoint(request: ComparisonReportRequest) -> ResearchComparisonReportSchema:
    """Generate a structured 11-section research comparison report."""
    try:
        report = await generate_research_comparison_report(
            paper_ids=request.paper_ids,
            research_question=request.research_question,
        )
        _REPORTS_STORE[report.report_id] = report
        return report
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Report generation failed: {e}")


@router.get("/{report_id}", response_model=ResearchComparisonReportSchema, tags=["Reports"])
def get_report_endpoint(report_id: str) -> ResearchComparisonReportSchema:
    """Retrieve a previously generated research report by ID."""
    if report_id not in _REPORTS_STORE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report '{report_id}' not found.")
    return _REPORTS_STORE[report_id]


@router.get("/{report_id}/export", tags=["Reports"])
def export_report_endpoint(
    report_id: str,
    format: str = Query("markdown", description="Export format: 'markdown' or 'json'"),
) -> Response:
    """Export research report to publication-ready Markdown or JSON format."""
    if report_id not in _REPORTS_STORE:
        # Fallback check against active session checkpointer
        try:
            from app.api.v1.endpoints.research import _api_checkpointer
            checkpoint = _api_checkpointer.get({"configurable": {"thread_id": report_id}})
            if checkpoint and "channel_values" in checkpoint:
                rep = checkpoint["channel_values"].get("final_report")
                if rep and isinstance(rep, dict):
                    schema = ResearchComparisonReportSchema(**rep)
                    _REPORTS_STORE[report_id] = schema
        except Exception:
            pass

    if report_id not in _REPORTS_STORE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report '{report_id}' not found.")

    report = _REPORTS_STORE[report_id]
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

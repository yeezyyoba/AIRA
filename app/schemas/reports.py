"""Schemas for Report Generation and Export Endpoints."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from validation.schemas import ResearchComparisonReportSchema


class ComparisonReportRequest(BaseModel):
    """Request payload for generating a structured comparison report."""
    paper_ids: List[str] = Field(..., min_length=2, description="List of paper IDs to analyze and synthesize")
    research_question: str = Field(..., min_length=3, description="Central research query guiding report synthesis")

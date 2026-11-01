"""Health Check API Response Schemas."""
from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for system health response."""
    status: str = Field(..., examples=["healthy"])
    version: str = Field(..., examples=["0.1.0"])
    phase: str = Field(..., examples=["Phase 1: Architecture & Project Setup"])
    services: Dict[str, str] = Field(default_factory=dict)

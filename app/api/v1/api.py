"""AIRA API v1 Router Aggregator."""
from fastapi import APIRouter
from app.api.v1.endpoints import health, research, papers, reports

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(research.router, prefix="/research")
api_router.include_router(papers.router, prefix="/papers")
api_router.include_router(reports.router, prefix="/reports")

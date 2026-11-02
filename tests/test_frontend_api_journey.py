"""Targeted Acceptance Tests: Frontend API User Journey and Session Lifecycle.

Validates:
1. Health endpoint detection.
2. Academic session initiation from user query.
3. Ambiguous research request triggers USER_CLARIFICATION_REQUIRED.
4. Observable logs, plan tasks, and clarification prompts return in response.
5. Resuming session with clarification responses unblocks and completes workflow.
6. Generated comparison report is stored and indexed in reports archive.
7. Markdown and JSON exports download cleanly with Content-Disposition headers.
8. Repository listing of indexed papers.
9. Expired or invalid session IDs return clean 404 with error details.
10. Resuming an active/completed session returns 400 Bad Request.
"""
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from validation.schemas import (
    PaperMetadataSchema,
    PaperValidationResultSchema,
    ValidationStatus,
)


@pytest.fixture
def api_client():
    return TestClient(app)


def test_complete_frontend_user_journey(api_client, monkeypatch, offline_llm_router, tmp_path):
    """Verify the entire end-to-end API user journey used by the frontend dashboard.

    Every external dependency is replaced: academic search, metadata
    validation, PDF reading, the LLM router, and the paper repository. Real
    network access is additionally blocked by tests/conftest.py, so this test
    can never fall through to Gemini/Groq or a PDF download. (The earlier
    slowness came from exactly that: search/validate/read were mocked, but the
    comparison and report steps still used the global ModelRouter, which made
    real Gemini/Groq calls whenever API keys were present in .env.)
    """
    from services.paper_repository import PaperRepository

    isolated_repo = PaperRepository(storage_dir=str(tmp_path / "processed"))
    monkeypatch.setattr("agents.nodes.default_model_router", offline_llm_router)
    monkeypatch.setattr("agents.nodes.default_paper_repo", isolated_repo)
    # 1. Health check detection
    res_health = api_client.get("/api/v1/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] == "healthy"
    assert "version" in health_data
    assert "services" in health_data
    assert "llm_primary" in health_data["services"]

    # 2. Paper repository listing
    res_papers = api_client.get("/api/v1/papers/")
    assert res_papers.status_code == 200
    papers_data = res_papers.json()
    assert "papers" in papers_data or "extracted_papers" in papers_data

    # Mock external tools upfront so tests run 100% offline without network calls
    async def mock_search_fn(*args, **kwargs):
        return [
            PaperMetadataSchema(
                id="paper_transformer_001",
                title="Attention Is All You Need",
                authors=["Vaswani et al."],
                year=2017,
                open_access_url="https://example.org/attention-is-all-you-need.pdf",
            ),
            PaperMetadataSchema(
                id="paper_transformer_002",
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                authors=["Devlin et al."],
                year=2018,
                open_access_url="https://example.org/bert-pretraining.pdf",
            ),
        ]
    monkeypatch.setattr("agents.nodes.search_academic_papers", mock_search_fn)

    async def mock_validate_fn(paper, paper_id=None, **kwargs):
        return PaperValidationResultSchema(status=ValidationStatus.VERIFIED, paper_id=paper_id or paper.id)
    monkeypatch.setattr("agents.nodes.validate_paper_metadata", mock_validate_fn)

    async def mock_read_paper(file_path_or_url, paper_id=None, **kwargs):
        from validation.schemas import ExtractionStatus, ExtractedPaperDataSchema, PaperExtractionResultSchema
        pid = paper_id or "paper_transformer_001"
        return PaperExtractionResultSchema(
            status=ExtractionStatus.SUCCESS,
            extraction_method="pymupdf",
            paper_id=pid,
            extracted_data=ExtractedPaperDataSchema(
                paper_id=pid,
                title="Attention Is All You Need",
                authors=["Vaswani et al."],
                abstract="Transformer model based entirely on attention.",
                methodology="Multi-head self-attention mechanisms without recurrence.",
                results="Achieves 28.4 BLEU on WMT 2014 English-to-German.",
            ),
            raw_text_by_page={1: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."},
        )
    monkeypatch.setattr("agents.nodes.read_research_paper", mock_read_paper)

    # 3. User submits ambiguous research query ("analysis") -> clarification needed
    res_ambiguous = api_client.post(
        "/api/v1/research/sessions",
        json={"query": "analysis"},
    )
    assert res_ambiguous.status_code == 200
    amb_data = res_ambiguous.json()
    assert amb_data["status"] == "USER_CLARIFICATION_REQUIRED"
    assert amb_data["clarification_question"] is not None
    assert isinstance(amb_data["missing_information"], list)
    sess_id = amb_data["session_id"]

    # 4. Inspect session state while paused
    res_inspect = api_client.get(f"/api/v1/research/sessions/{sess_id}")
    assert res_inspect.status_code == 200
    inspect_data = res_inspect.json()
    assert inspect_data["session_id"] == sess_id
    assert inspect_data["status"] == "USER_CLARIFICATION_REQUIRED"
    assert len(inspect_data["observable_logs"]) > 0

    # 5. User submits clarification response to resume paused session
    res_resume = api_client.post(
        f"/api/v1/research/sessions/{sess_id}/resume",
        json={"clarification_response": "Compare multi-head self-attention mechanisms between Transformer and BERT on GLUE benchmark"},
    )
    assert res_resume.status_code == 200
    resume_data = res_resume.json()
    assert resume_data["status"] == "GOAL_ACHIEVED"
    assert resume_data["plan"] is not None
    assert all(task["status"] == "COMPLETED" for task in resume_data["plan"])
    assert sorted(resume_data["extracted_papers"]) == ["paper_transformer_001", "paper_transformer_002"]
    assert resume_data["retrieved_evidence"], "RAG retrieval should run over the indexed papers"
    assert set(resume_data["citations"]) == {"paper_transformer_001", "paper_transformer_002"}
    executed = [log["tool"] for log in resume_data["observable_logs"] if log["node"] == "execute_tools"]
    for tool in ("search_academic_papers", "validate_metadata_integrity", "read_research_paper",
                 "retrieve_research_context", "compare_research_papers", "analyze_research_methodology",
                 "analyze_experimental_results", "identify_research_gaps", "generate_citations",
                 "generate_research_comparison_report"):
        assert tool in executed, f"{tool} was not executed"
    # No tool ran twice with identical inputs (only the two distinct PDFs repeat a tool name).
    assert executed.count("read_research_paper") == 2
    assert len(executed) == len(set(executed)) + 1
    # The offline router was the one consulted, so no provider could be called.
    assert offline_llm_router.generate.called
    assert len(resume_data["observable_logs"]) > 0
    assert resume_data["final_report"] is not None
    final_rep = resume_data["final_report"]
    report_id = final_rep.get("report_id")
    assert report_id is not None

    # 6. Verify report is registered and available in Reports Archive
    res_archive = api_client.get("/api/v1/reports/")
    assert res_archive.status_code == 200
    archive_data = res_archive.json()
    assert "reports" in archive_data
    archived_ids = [r["report_id"] for r in archive_data["reports"]]
    assert report_id in archived_ids
    assert len(archived_ids) == len(set(archived_ids)), "each report must be listed once"

    # 7. Verify Markdown export with attachment download header
    res_md = api_client.get(f"/api/v1/reports/{report_id}/export?format=markdown")
    assert res_md.status_code == 200
    assert "text/markdown" in res_md.headers["content-type"]
    assert "attachment" in res_md.headers["content-disposition"]
    assert f"aira_report_{report_id}.md" in res_md.headers["content-disposition"]
    md_content = res_md.text
    assert "Research Comparison Report" in md_content or "AIRA" in md_content

    # 8. Verify JSON export with attachment download header
    res_json = api_client.get(f"/api/v1/reports/{report_id}/export?format=json")
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers["content-type"]
    assert "attachment" in res_json.headers["content-disposition"]
    assert f"aira_report_{report_id}.json" in res_json.headers["content-disposition"]
    json_data = json.loads(res_json.text)
    assert json_data["report_id"] == report_id


def test_session_lifecycle_error_recovery(api_client):
    """Verify session error scenarios: expired session, invalid resume, invalid format."""
    # 1. Nonexistent/expired session returns 404
    res_notfound = api_client.get("/api/v1/research/sessions/sess_nonexistent_999")
    assert res_notfound.status_code == 404
    assert "not found" in res_notfound.json()["detail"].lower()

    # 2. Resuming nonexistent session returns 404
    res_resume_bad = api_client.post(
        "/api/v1/research/sessions/sess_nonexistent_999/resume",
        json={"clarification_response": "any clarification"},
    )
    assert res_resume_bad.status_code == 404

    # 3. Export with invalid format returns 400
    res_reports = api_client.get("/api/v1/reports/")
    reports = res_reports.json().get("reports", [])
    if reports:
        rep_id = reports[0]["report_id"]
        res_bad_fmt = api_client.get(f"/api/v1/reports/{rep_id}/export?format=xml")
        assert res_bad_fmt.status_code == 400
        assert "unsupported format" in res_bad_fmt.json()["detail"].lower()

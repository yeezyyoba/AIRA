"""Tool-level regression tests for fixes made during the audit."""
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from llm.gemini_provider import GeminiProvider, LLMResponse, strip_json_fences
from llm.model_router import ModelRouter
from llm.provider_errors import BadRequestError, ProviderStatus
from services.paper_repository import PaperRepository
from tools.file_manager import FileManagerError, list_saved_reports, load_report, save_report, sanitize_identifier
from validation.schemas import ExtractionStatus, PaperExtractionResultSchema


# --- File manager (System & Environment tool) --------------------------------

def test_file_manager_round_trip_and_listing(tmp_path):
    paths = save_report({"report_id": "rep_1", "research_question": "q", "created_at": "2026-01-01"},
                        markdown="# r", base_dir=str(tmp_path))
    assert set(paths) == {"json", "md"}
    assert load_report("rep_1", base_dir=str(tmp_path))["research_question"] == "q"
    assert [r["report_id"] for r in list_saved_reports(base_dir=str(tmp_path))] == ["rep_1"]


@pytest.mark.parametrize("evil", ["../../etc/passwd", "..", "/abs/path", "a/../../b", "x\\..\\y"])
def test_file_manager_cannot_escape_reports_dir(tmp_path, evil):
    save_report({"report_id": evil}, base_dir=str(tmp_path))
    assert all(p.parent == tmp_path for p in tmp_path.iterdir())
    assert "/" not in sanitize_identifier(evil) and "\\" not in sanitize_identifier(evil)


def test_file_manager_refuses_reports_without_id(tmp_path):
    with pytest.raises(FileManagerError):
        save_report({"title": "no id"}, base_dir=str(tmp_path))


# --- Paper repository identity ------------------------------------------------

def _extraction(pid):
    return PaperExtractionResultSchema(paper_id=pid, status=ExtractionStatus.SUCCESS, extraction_method="pymupdf")


def test_repository_does_not_collide_dois_with_the_same_suffix(tmp_path):
    repo = PaperRepository(storage_dir=str(tmp_path))
    repo.save_paper(_extraction("10.1000/abc"))
    repo.save_paper(_extraction("10.2000/abc"))
    fresh = PaperRepository(storage_dir=str(tmp_path))  # bypass the in-memory cache
    assert fresh.get_paper("10.1000/abc").paper_id == "10.1000/abc"
    assert fresh.get_paper("10.2000/abc").paper_id == "10.2000/abc"
    assert all(p.parent == tmp_path for p in tmp_path.iterdir())


# --- API input validation -----------------------------------------------------

def test_read_endpoint_rejects_local_file_paths():
    client = TestClient(app)
    for url in ("/etc/passwd", "file:///etc/hosts", "../data/processed/x.pdf"):
        response = client.post("/api/v1/papers/read", json={"url": url})
        assert response.status_code == 422, url


def test_research_session_rejects_inverted_year_range():
    client = TestClient(app)
    response = client.post("/api/v1/research/sessions",
                           json={"query": "Amharic ASR", "year_from": 2024, "year_to": 2020})
    assert response.status_code == 422


# --- LLM routing ----------------------------------------------------------------

def _provider(name, **behaviour):
    p = MagicMock()
    p.name, p.model, p.status = name, f"{name}-model", ProviderStatus.AVAILABLE
    for method, value in behaviour.items():
        setattr(p, method, value)
    return p


def test_bad_request_is_not_retried_but_falls_back():
    gemini = _provider("gemini", generate=MagicMock(side_effect=BadRequestError("400 INVALID_ARGUMENT", "gemini")))
    groq = _provider("groq", generate=MagicMock(return_value=LLMResponse(
        content="ok from groq", provider="groq", model="m", latency_seconds=0.0)))
    router = ModelRouter(gemini_provider=gemini, fallback_provider=groq, initial_backoff_seconds=0)
    assert router.generate("hi").content == "ok from groq"
    assert gemini.generate.call_count == 1


def test_unexpected_primary_exception_still_falls_back():
    gemini = _provider("gemini", generate=MagicMock(side_effect=AttributeError("sdk bug")))
    groq = _provider("groq", generate=MagicMock(return_value=LLMResponse(
        content="fallback ok", provider="groq", model="m", latency_seconds=0.0)))
    router = ModelRouter(gemini_provider=gemini, fallback_provider=groq, initial_backoff_seconds=0)
    assert router.generate("hi").content == "fallback ok"


def test_degraded_mode_recovers_after_cooldown():
    gemini = _provider("gemini", generate=MagicMock(side_effect=AttributeError("down")))
    groq = _provider("groq", generate=MagicMock(side_effect=AttributeError("down")))
    router = ModelRouter(gemini_provider=gemini, fallback_provider=groq, initial_backoff_seconds=0)
    with pytest.raises(Exception):
        router.generate("hi")
    assert router.is_degraded_mode()
    router.cooldown_seconds = 0.01
    time.sleep(0.02)
    assert not router.is_degraded_mode()


def test_invalid_argument_is_not_misclassified_as_authentication():
    provider = GeminiProvider(api_key="dummy", client=MagicMock())
    with pytest.raises(BadRequestError):
        provider._classify_and_raise(Exception("400 INVALID_ARGUMENT: response_schema is not supported"))
    assert provider.status != ProviderStatus.AUTHENTICATION_ERROR


def test_status_codes_are_matched_as_whole_numbers():
    provider = GeminiProvider(api_key="dummy", client=MagicMock())
    from llm.provider_errors import ProviderUnavailableError
    with pytest.raises(ProviderUnavailableError):
        # '4013 tokens' must not be read as HTTP 401.
        provider._classify_and_raise(Exception("context of 4013 tokens: connection reset"))


def test_structured_output_tolerates_markdown_fences():
    assert strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_json_fences('{"a": 1}') == '{"a": 1}'


# --- Report grounding ------------------------------------------------------------

@pytest.mark.anyio
async def test_report_conclusions_are_grounded_not_generic(tmp_path):
    from services.report_service import generate_research_comparison_report
    from validation.schemas import ExtractedPaperDataSchema

    repo = PaperRepository(storage_dir=str(tmp_path))
    for pid, wer in (("a", "2.1"), ("b", "3.8")):
        repo.save_paper(PaperExtractionResultSchema(
            paper_id=pid, status=ExtractionStatus.SUCCESS, extraction_method="pymupdf", page_count=4,
            extracted_data=ExtractedPaperDataSchema(
                paper_id=pid, title=f"Paper {pid}", dataset="LibriSpeech",
                results=f"Model-{pid} achieved {wer}% WER on test-clean.",
            ),
        ))
    router = MagicMock()
    router.is_degraded_mode.return_value = True
    report = await generate_research_comparison_report(["a", "b"], "ASR on LibriSpeech", repository=repo, router=router)
    assert "domain-specific efficacy" not in report.evidence_backed_conclusions
    assert report.title and report.report_status == "COMPLETE"


# --- Readable paper labels -------------------------------------------------------

def test_paper_label_formats():
    from services.report_service import paper_label

    assert paper_label("10.1/x", "T", ["Shamsuddeen Hassan Muhammad", "David Adelani", "Sebastian Ruder"], 2022) == "Muhammad et al., 2022"
    assert paper_label("10.1/x", "T", ["Dang Van Thin", "Dai Nguyen"], 2023) == "Thin & Nguyen, 2023"
    assert paper_label("10.1/x", "T", ["Muhammad, S. H."], None) == "Muhammad"
    assert paper_label("10.1/x", "Sentiment Analysis Across Multiple African Languages: A Current Benchmark", [], None).endswith("…")
    assert paper_label("10.1/x", None, [], None) == "10.1/x"


@pytest.mark.anyio
async def test_markdown_report_names_papers_not_dois(tmp_path):
    from services.report_service import export_report_markdown, generate_research_comparison_report
    from validation.schemas import ExtractedPaperDataSchema

    repo = PaperRepository(storage_dir=str(tmp_path))
    meta = {}
    for pid, author, year in (("10.18653/v1/a", "Ada Lovelace", 2021), ("10.18653/v1/b", "Alan Turing", 2022)):
        repo.save_paper(PaperExtractionResultSchema(
            paper_id=pid, status=ExtractionStatus.SUCCESS, extraction_method="pymupdf", page_count=4,
            extracted_data=ExtractedPaperDataSchema(paper_id=pid, title=f"Paper by {author}",
                                                    methodology="We fine-tune a transformer.", dataset="AfriSenti"),
        ))
        meta[pid] = {"title": f"Paper by {author}", "authors": [author], "year": year, "doi": pid}
    router = MagicMock()
    router.is_degraded_mode.return_value = True
    report = await generate_research_comparison_report(
        ["10.18653/v1/a", "10.18653/v1/b"], "transformers", repository=repo, router=router,
        precomputed={"paper_metadata": meta},
    )
    md = export_report_markdown(report)
    assert "**Lovelace, 2021**" in md and "**Turing, 2022**" in md
    body = md.split("## 11.")[0]
    assert "**10.18653/v1/a**" not in body
    assert "https://doi.org/10.18653/v1/a" in md

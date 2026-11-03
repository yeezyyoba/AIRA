"""Rate-limit, timeout and API-usage behaviour (no network: every SDK client is mocked).

Covers:
- provider retry hints ("retry in 37s", "try again in 1m2s") are parsed;
- a 429 / 503 is not retried against the same provider: it is put in a cooldown
  (the provider's hint, capped) and the router falls back at once;
- providers in a cooldown are skipped; when every provider is cooling down the
  router fails fast without entering the long Degraded Mode;
- budgeted single-shot tool proposals: one attempt per provider, the remaining
  budget becomes the HTTP timeout, and no fallback once the budget is spent;
- the tool-proposal circuit breaker pauses LLM proposals after repeated failures;
- no-argument tool calls such as ``{"": {}}`` are accepted (arguments ignored);
- per-session API usage is attributed across worker threads.
"""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents import tool_calling
from agents.nodes import propose_tool_call, select_tools
from app.core.config import settings
from llm.fallback_provider import GroqProvider
from llm.gemini_provider import GeminiProvider, LLMResponse, ToolCall
from llm.model_router import ModelRouter
from llm.provider_errors import (
    DegradedModeError,
    InvalidResponseError,
    ProviderStatus,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    parse_retry_after_seconds,
)
from llm.usage_tracker import get_session_usage, track_session_usage, usage_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(provider, content="ok"):
    return LLMResponse(content=content, provider=provider, model=f"{provider}-model", latency_seconds=0.01)


def _provider(cls, name, side_effect=None, method="generate"):
    mock = MagicMock(spec=cls)
    mock.name = name
    mock.model = f"{name}-model"
    mock.status = ProviderStatus.AVAILABLE
    if side_effect is not None:
        getattr(mock, method).side_effect = side_effect
    else:
        getattr(mock, method).return_value = _ok(name)
    return mock


def _router(gemini_effect=None, groq_effect=None, method="generate", **kwargs):
    gemini = _provider(GeminiProvider, "gemini", gemini_effect, method)
    groq = _provider(GroqProvider, "groq", groq_effect, method)
    router = ModelRouter(gemini_provider=gemini, fallback_provider=groq, max_retries=2,
                         initial_backoff_seconds=0.01, **kwargs)
    return router, gemini, groq


@pytest.fixture(autouse=True)
def _fresh_tracker():
    usage_tracker.reset()
    yield


# ---------------------------------------------------------------------------
# Retry hints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message, expected", [
    ("429 RESOURCE_EXHAUSTED. Please retry in 37.2s.", 37.2),
    ("{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '41s'}", 41.0),
    ("Rate limit reached for model openai/gpt-oss-120b. Please try again in 1m2.5s.", 62.5),
    ("Please try again in 7.3s. Need more tokens?", 7.3),
    ("Please try again in 640ms.", 0.64),
    ("503 UNAVAILABLE: the model is experiencing high demand", None),
])
def test_retry_hint_parsing(message, expected):
    value = parse_retry_after_seconds(Exception(message))
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


def test_retry_after_header_is_used():
    error = Exception("rate limited")
    error.response = SimpleNamespace(headers={"retry-after": "12"})
    assert parse_retry_after_seconds(error) == 12.0


# ---------------------------------------------------------------------------
# Router: 429 / 503 / timeouts
# ---------------------------------------------------------------------------

def test_rate_limit_is_not_retried_and_cools_down_with_the_hint():
    router, gemini, groq = _router(
        gemini_effect=RateLimitError("429 Please retry in 12s.", provider="gemini", retry_after=12.0))
    assert router.generate("q").provider == "groq"
    assert gemini.generate.call_count == 1  # no same-provider retry
    assert 10.0 < router.cooling_remaining("gemini") <= 12.0

    # The next request skips Gemini entirely while it cools down.
    assert router.generate("q2").provider == "groq"
    assert gemini.generate.call_count == 1
    assert groq.generate.call_count == 2
    assert not router.is_degraded_mode()


def test_retry_hint_is_capped(monkeypatch):
    monkeypatch.setattr(settings, "LLM_RETRY_HINT_MAX_SECONDS", 30.0)
    router, _, _ = _router(gemini_effect=RateLimitError("429", provider="gemini", retry_after=600.0))
    router.generate("q")
    assert 25.0 < router.cooling_remaining("gemini") <= 30.0


def test_overloaded_503_falls_back_without_retry():
    router, gemini, groq = _router(gemini_effect=ProviderUnavailableError(
        "503 UNAVAILABLE. The model is overloaded due to high demand.", provider="gemini"))
    assert router.generate("q").provider == "groq"
    assert gemini.generate.call_count == 1
    assert router.cooling_remaining("gemini") > 0


def test_timeout_falls_back_without_retry_and_is_counted():
    router, gemini, groq = _router(gemini_effect=ProviderUnavailableError(
        "504 DEADLINE_EXCEEDED", provider="gemini"))
    with track_session_usage("timeout-session") as usage:
        assert router.generate("q").provider == "groq"
    assert gemini.generate.call_count == 1
    snap = usage.snapshot()
    assert snap["llm"]["gemini"]["timeouts"] == 1
    assert snap["llm"]["gemini"]["failures"] == 1
    assert snap["llm"]["groq"]["successes"] == 1
    assert snap["fallbacks"] == 1


def test_connection_reset_is_still_retried():
    router, gemini, groq = _router(gemini_effect=[
        ProviderUnavailableError("Connection reset by peer", provider="gemini"), _ok("gemini")])
    assert router.generate("q").provider == "gemini"
    assert gemini.generate.call_count == 2
    groq.generate.assert_not_called()


def test_every_provider_cooling_down_fails_fast_without_degraded_mode():
    router, gemini, groq = _router(
        gemini_effect=RateLimitError("429", provider="gemini", retry_after=20.0),
        groq_effect=RateLimitError("429 Please try again in 15s", provider="groq", retry_after=15.0),
    )
    with pytest.raises(DegradedModeError, match="rate limited"):
        router.generate("q")
    assert not router.is_degraded_mode()  # a per-minute limit is not a 2-minute outage

    with pytest.raises(DegradedModeError, match="temporarily paused"):
        router.generate("q2")
    assert gemini.generate.call_count == 1 and groq.generate.call_count == 1  # nothing re-sent

    # Cooldowns expire on their own.
    router._cooling_until = {name: time.monotonic() - 1 for name in router._cooling_until}
    gemini.generate.side_effect = None
    gemini.generate.return_value = _ok("gemini")
    assert router.generate("q3").provider == "gemini"


def test_hard_failure_of_both_providers_still_enters_degraded_mode():
    router, _, _ = _router(
        gemini_effect=QuotaExceededError("quota", provider="gemini"),
        groq_effect=ProviderUnavailableError("500 internal", provider="groq"),
    )
    with pytest.raises(DegradedModeError):
        router.generate("q")
    assert router.is_degraded_mode()


def test_gemini_quota_with_short_hint_pauses_only_for_the_hint():
    router, gemini, _ = _router(
        gemini_effect=QuotaExceededError("RESOURCE_EXHAUSTED retry in 20s", provider="gemini", retry_after=20.0))
    router.generate("q")
    assert gemini.status == ProviderStatus.QUOTA_EXHAUSTED
    assert router._primary_quota_seconds == 20.0

    router._primary_quota_since = time.monotonic() - 21
    gemini.generate.side_effect = None
    gemini.generate.return_value = _ok("gemini")
    assert router.generate("q2").provider == "gemini"


def test_gemini_daily_quota_without_hint_uses_full_cooldown():
    router, gemini, _ = _router(gemini_effect=QuotaExceededError(
        "You exceeded your current quota, please check your plan and billing details.", provider="gemini"))
    router.generate("q")
    assert router._primary_quota_seconds is None  # full LLM_DEGRADED_COOLDOWN_SECONDS
    router.generate("q2")
    assert gemini.generate.call_count == 1


def test_groq_quota_puts_groq_in_cooldown():
    router, gemini, groq = _router(
        gemini_effect=RateLimitError("429", provider="gemini", retry_after=5.0),
        groq_effect=QuotaExceededError("quota exceeded", provider="groq"),
    )
    with pytest.raises(DegradedModeError):
        router.generate("q")
    assert router.cooling_remaining("groq") > 60


# ---------------------------------------------------------------------------
# Router: budgeted single-shot tool proposals
# ---------------------------------------------------------------------------

def _tool_response(provider):
    call = ToolCall(name="compare_research_papers", arguments={}, call_id="c")
    return LLMResponse(content="{}", provider=provider, model="m", latency_seconds=0.0, structured_data=call)


def test_single_shot_proposal_passes_budget_as_timeout_and_does_not_retry():
    router, gemini, groq = _router(
        gemini_effect=InvalidResponseError("no function call", provider="gemini"),
        method="generate_tool_call",
    )
    groq.generate_tool_call.return_value = _tool_response("groq")
    response = router.generate_tool_call("p", [], single_shot=True, timeout=20.0)
    assert response.provider == "groq"
    assert gemini.generate_tool_call.call_count == 1  # single shot: no retry
    gemini_timeout = gemini.generate_tool_call.call_args.kwargs["timeout"]
    groq_timeout = groq.generate_tool_call.call_args.kwargs["timeout"]
    assert 15.0 < gemini_timeout <= 20.0
    assert groq_timeout <= gemini_timeout


def test_spent_budget_skips_the_fallback_and_does_not_enter_degraded_mode():
    def slow_failure(*args, **kwargs):
        time.sleep(0.3)
        raise ProviderUnavailableError("Connection reset by peer", provider="gemini")

    router, gemini, groq = _router(gemini_effect=slow_failure, method="generate_tool_call")
    with pytest.raises(ProviderUnavailableError, match="budget"):
        router.generate_tool_call("p", [], single_shot=True, timeout=3.2)  # 2.9s left < 3s minimum
    groq.generate_tool_call.assert_not_called()
    assert not router.is_degraded_mode()


def test_budgeted_failure_of_both_providers_does_not_enter_degraded_mode():
    router, _, _ = _router(
        gemini_effect=InvalidResponseError("no call", provider="gemini"),
        groq_effect=InvalidResponseError("no call", provider="groq"),
        method="generate_tool_call",
    )
    with pytest.raises(InvalidResponseError):
        router.generate_tool_call("p", [], single_shot=True, timeout=20.0)
    assert not router.is_degraded_mode()


def test_gemini_request_timeout_is_set_per_call():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        function_calls=[SimpleNamespace(name="compare_research_papers", args={}, id="c1")], usage_metadata=None)
    provider = GeminiProvider(api_key="test", client=client)
    provider.generate_tool_call("p", [tool_calling.TOOL_SPECS["compare_research_papers"]], timeout=17.5)
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.http_options.timeout == 17500
    provider.generate_tool_call("p", [tool_calling.TOOL_SPECS["compare_research_papers"]], timeout=2.0)
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.http_options.timeout == 10000  # Gemini's minimum server deadline


def test_groq_request_timeout_is_set_per_call():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            id="c", function=SimpleNamespace(name="compare_research_papers", arguments="{}"))]))],
        usage=None)
    provider = GroqProvider(api_key="test", client=client)
    provider.generate_tool_call("p", [tool_calling.TOOL_SPECS["compare_research_papers"]], timeout=8.0)
    assert client.chat.completions.create.call_args.kwargs["timeout"] == 8.0


# ---------------------------------------------------------------------------
# No-argument tools
# ---------------------------------------------------------------------------

def _analysis_state():
    from tests.test_tool_calling import _extraction_state  # same fixture data
    state = {
        **_extraction_state(),
        "completed_tasks": ["search_and_validate", "extract_and_index"],
        "extracted_papers": ["p1", "p2"], "indexed_papers": ["p1", "p2"],
        "analysis_actions_completed": ["rag_retrieval"],
    }
    return {**state, **select_tools(state)}


def _check(name, arguments, state):
    return tool_calling.validate_proposal(
        name, arguments, state["tool_options"], state,
        max_search_results=40, max_top_k=10,
        already_succeeded=lambda t, i: False, scope_check=lambda q: None,
    )


def test_empty_key_placeholder_is_accepted_for_no_argument_tool():
    state = _analysis_state()
    check = _check("analyze_research_methodology", {"": {}}, state)
    assert check.ok, check.reason
    assert check.tool == "analyze_research_methodology"
    assert any("placeholder" in note for note in check.notes)


def test_stray_arguments_for_no_argument_tool_are_ignored_and_noted():
    state = _analysis_state()
    check = _check("compare_research_papers", {"focus": "all"}, state)
    assert check.ok, check.reason
    assert any("ignored argument" in note for note in check.notes)
    assert "focus" not in check.inputs


def test_no_argument_tool_still_rejects_urls_and_code_owned_fields():
    state = _analysis_state()
    assert not _check("compare_research_papers", {"note": "see https://evil.example/x.pdf"}, state).ok
    owned = next(iter(tool_calling.CODE_OWNED_FIELDS))
    assert not _check("compare_research_papers", {owned: "x"}, state).ok


def test_tools_with_arguments_still_reject_unknown_fields():
    state = _analysis_state()
    check = _check("search_academic_papers", {"query": "Amharic NER", "bogus": 1}, {
        **state, "tool_options": [{"tool": "search_academic_papers", "task": "search_and_validate",
                                   "inputs": {"query": "Amharic NER", "max_results": 20}}]})
    assert not check.ok and "Extra inputs" in check.reason


def test_groq_empty_key_payload_parses_and_validates():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            id="c", function=SimpleNamespace(name="analyze_research_methodology", arguments='{"": {}}'))]))],
        usage=None)
    call = GroqProvider(api_key="test", client=client).generate_tool_call(
        "p", [tool_calling.TOOL_SPECS["analyze_research_methodology"]]).structured_data
    assert call.arguments == {"": {}}
    assert _check(call.name, call.arguments, _analysis_state()).ok


def test_gemini_empty_key_payload_parses_and_validates():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        function_calls=[SimpleNamespace(name="analyze_research_methodology", args={"": {}}, id="c")],
        usage_metadata=None)
    call = GeminiProvider(api_key="test", client=client).generate_tool_call(
        "p", [tool_calling.TOOL_SPECS["analyze_research_methodology"]]).structured_data
    assert _check(call.name, call.arguments, _analysis_state()).ok


def test_gemini_declaration_omits_schema_for_no_argument_tools():
    no_args = GeminiProvider._function_declaration(tool_calling.TOOL_SPECS["compare_research_papers"])
    assert no_args.parameters_json_schema is None
    with_args = GeminiProvider._function_declaration(tool_calling.TOOL_SPECS["read_research_paper"])
    assert with_args.parameters_json_schema["properties"]["paper_id"]


# ---------------------------------------------------------------------------
# Circuit breaker on tool proposals
# ---------------------------------------------------------------------------

def _failing_router(exc):
    router = MagicMock()
    router.is_degraded_mode.return_value = False
    router.generate_tool_call = MagicMock(side_effect=exc)
    return router


@pytest.mark.anyio
async def test_repeated_proposal_failures_pause_llm_proposals(monkeypatch):
    monkeypatch.setattr(settings, "TOOL_SELECTION_MAX_CONSECUTIVE_FAILURES", 2)
    state = _analysis_state()
    router = _failing_router(InvalidResponseError("no call", provider="router"))

    first = await propose_tool_call(state, router=router)
    assert first["tool_selection_failures"] == 1 and not first["llm_proposals_paused"]

    second = await propose_tool_call({**state, **first}, router=router)
    assert second["llm_proposals_paused"] is True
    assert "LLM proposals paused for this run after 2 consecutive failures; using rules." in [
        log["decision"] for log in second["observable_logs"]]

    third = await propose_tool_call({**state, **second}, router=router)
    assert router.generate_tool_call.call_count == 2  # not called once paused
    assert third["tool_selection"]["outcome"] == "PAUSED"
    assert third["selected_tool"] == state["selected_tool"]  # the rule choice
    assert third["tool_selection_stats"]["paused"] == 1
    assert third["tool_selection_stats"]["fallback"] == 2


@pytest.mark.anyio
async def test_a_successful_proposal_resets_the_failure_count():
    state = _analysis_state()
    failing = _failing_router(InvalidResponseError("no call", provider="router"))
    first = await propose_tool_call(state, router=failing)
    ok = MagicMock()
    ok.is_degraded_mode.return_value = False
    ok.generate_tool_call = MagicMock(return_value=_tool_response("gemini"))
    second = await propose_tool_call({**state, **first}, router=ok)
    assert second["tool_selection_failures"] == 0
    assert ok.generate_tool_call.call_args.kwargs["single_shot"] is True
    assert ok.generate_tool_call.call_args.kwargs["timeout"] == float(settings.TOOL_SELECTION_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Per-session usage attribution
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_session_usage_is_attributed_across_threads_and_isolated():
    router, _, _ = _router()
    with track_session_usage("session-a"):
        await asyncio.to_thread(router.generate, "q")
        await asyncio.to_thread(router.generate, "q")
    with track_session_usage("session-b"):
        await asyncio.to_thread(router.generate, "q")
    assert get_session_usage("session-a").snapshot()["llm"]["gemini"]["calls"] == 2
    assert get_session_usage("session-b").snapshot()["llm"]["gemini"]["calls"] == 1
    summary = get_session_usage("session-a").summary_line()
    assert summary.startswith("API usage: gemini 2 calls")


# ---------------------------------------------------------------------------
# Semantic Scholar pacing and lookup skipping
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock  # noqa: E402

import httpx  # noqa: E402

from services.academic_clients import (  # noqa: E402
    CrossrefClient,
    OpenAlexClient,
    RequestPacer,
    SemanticScholarClient,
    semantic_scholar_pacer,
)
from tools.metadata_validator import validate_paper_metadata  # noqa: E402
from validation.schemas import PaperMetadataSchema  # noqa: E402


@pytest.mark.anyio
async def test_pacer_spaces_concurrent_requests():
    pacer = RequestPacer("test", lambda: 0.15)
    start = time.monotonic()
    stamps = []

    async def one():
        await pacer.wait_turn()
        stamps.append(time.monotonic() - start)

    await asyncio.gather(*(one() for _ in range(4)))
    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(gap >= 0.13 for gap in gaps), gaps


@pytest.mark.anyio
async def test_pacer_back_off_delays_everyone():
    pacer = RequestPacer("test", lambda: 0.0)
    pacer.back_off(0.2)
    start = time.monotonic()
    await pacer.wait_turn()
    assert time.monotonic() - start >= 0.18


def _s2_response(status, headers=None, json_body=None):
    return httpx.Response(status_code=status, headers=headers or {}, json=json_body if json_body is not None else {},
                          request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"))


@pytest.mark.anyio
async def test_semantic_scholar_429_retry_after_pushes_the_shared_pacer(monkeypatch):
    monkeypatch.setattr(settings, "SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS", 0.0)
    semantic_scholar_pacer.reset()
    client = SemanticScholarClient()
    http = AsyncMock(spec=httpx.AsyncClient)
    http.request.side_effect = [_s2_response(429, {"Retry-After": "0.3"}), _s2_response(200, json_body={"data": []})]
    start = time.monotonic()
    with track_session_usage("s2-session") as usage:
        assert await client.search("Amharic NER", client=http) == []
    assert time.monotonic() - start >= 0.28  # waited for Retry-After before the retry
    assert usage.snapshot()["academic"]["semantic_scholar"] == {
        "requests": 2, "successes": 1, "failures": 1, "rate_limited": 1}
    semantic_scholar_pacer.reset()


@pytest.mark.anyio
async def test_semantic_scholar_requests_are_paced(monkeypatch):
    monkeypatch.setattr(settings, "SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS", 0.12)
    semantic_scholar_pacer.reset()
    client = SemanticScholarClient()
    http = AsyncMock(spec=httpx.AsyncClient)
    http.request.return_value = _s2_response(200, json_body={"data": []})
    start = time.monotonic()
    await asyncio.gather(*(client.search(f"query {i}", client=http) for i in range(3)))
    assert time.monotonic() - start >= 0.22
    semantic_scholar_pacer.reset()


def _record(source, **overrides):
    data = dict(title="BERT for Amharic Sentiment Classification", authors=["Tirunesh Dibaba"], year=2021,
                doi="10.1007/s10579-021-09550-y", source=source, sources=[source])
    data.update(overrides)
    return PaperMetadataSchema(**data)


def _clients(cr, oa, s2):
    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.get_by_doi = AsyncMock(return_value=cr)
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.get_by_doi = AsyncMock(return_value=oa)
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.get_by_doi = AsyncMock(return_value=s2)
    return dict(crossref_client=mock_cr, openalex_client=mock_oa, semantic_scholar_client=mock_s2), mock_s2


@pytest.mark.anyio
async def test_semantic_scholar_skipped_when_crossref_and_openalex_agree():
    clients, s2 = _clients(_record("crossref"), _record("openalex"), _record("semantic_scholar"))
    result = await validate_paper_metadata(_record("user_upload"), **clients)
    s2.get_by_doi.assert_not_called()
    assert result.status.value == "VERIFIED"
    assert "semantic_scholar" not in result.sources_checked  # not claimed as checked


@pytest.mark.anyio
@pytest.mark.parametrize("cr, oa", [
    (_record("crossref"), None),                                   # only one registry
    (_record("crossref"), _record("openalex", year=2015)),         # registries disagree on year
    (_record("crossref"), _record("openalex", title="A completely different paper about speech")),
])
async def test_semantic_scholar_still_consulted_when_needed(cr, oa):
    clients, s2 = _clients(cr, oa, _record("semantic_scholar"))
    result = await validate_paper_metadata(_record("user_upload"), **clients)
    s2.get_by_doi.assert_awaited_once()
    assert "semantic_scholar" in result.sources_checked


# ---------------------------------------------------------------------------
# PDF link ranking
# ---------------------------------------------------------------------------

from agents.nodes import _pdf_url  # noqa: E402


def _validated(metadata):
    return {"paper_id": "p1", "status": "VERIFIED", "original_metadata": {"id": "p1", "title": "T", **metadata}}


def test_direct_pdf_preferred_over_doi_landing_page():
    record = _validated({
        "open_access_url": "https://doi.org/10.18653/v1/2020.acl-main.1",
        "provider_metadata": {"semantic_scholar": {"open_access_url": "https://aclanthology.org/2020.acl-main.1.pdf"}},
    })
    assert _pdf_url(record) == "https://aclanthology.org/2020.acl-main.1.pdf"


def test_arxiv_and_acl_abstract_pages_map_to_their_pdfs():
    assert _pdf_url(_validated({"open_access_url": "https://arxiv.org/abs/2103.11811"})) == \
        "https://arxiv.org/pdf/2103.11811"
    assert _pdf_url(_validated({"open_access_url": "https://aclanthology.org/2022.lrec-1.5/"})) == \
        "https://aclanthology.org/2022.lrec-1.5.pdf"


def test_doi_landing_page_is_still_used_as_a_last_resort():
    record = _validated({"open_access_url": "https://doi.org/10.1007/s10579-021-09550-y"})
    assert _pdf_url(record) == "https://doi.org/10.1007/s10579-021-09550-y"


def test_plain_landing_url_is_not_offered_as_an_open_access_copy():
    assert _pdf_url(_validated({"url": "https://link.springer.com/article/10.1007/x"})) is None
    assert _pdf_url(_validated({"url": "https://arxiv.org/abs/2103.11811"})) == "https://arxiv.org/pdf/2103.11811"


# ---------------------------------------------------------------------------
# api_usage in the API response, embeddings counting
# ---------------------------------------------------------------------------

def test_api_response_carries_session_api_usage():
    from app.api.v1.endpoints.research import _build_response

    router, _, _ = _router()
    with track_session_usage("session-usage-api"):
        router.generate("q")
    response = _build_response({}, "session-usage-api", "COMPLETED", query="q")
    assert response.api_usage["llm"]["gemini"]["calls"] == 1
    assert response.api_usage["llm_totals"]["calls"] == 1
    assert _build_response({}, "never-ran-here", "COMPLETED", query="q").api_usage is None


@pytest.mark.anyio
async def test_only_real_embedding_providers_are_counted():
    from services.embedding_provider import GeminiEmbeddingProvider, MockEmbeddingProvider

    with track_session_usage("session-embeddings") as usage:
        await MockEmbeddingProvider().embed_documents(["a", "b"])
        assert usage.snapshot()["embeddings"]["calls"] == 0

        provider = GeminiEmbeddingProvider(api_key="test", dimension=4, batch_size=10)

        async def fake(texts, task_type=None):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        provider._generate_embeddings = fake
        await provider.embed_documents(["a", "b", "c"])
    snap = usage.snapshot()["embeddings"]
    assert snap["calls"] == 1 and snap["successes"] == 1 and snap["texts"] == 3

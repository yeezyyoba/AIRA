"""NVIDIA provider and the Gemini -> Groq -> NVIDIA -> degraded chain (offline; HTTP is mocked)."""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agents import tool_calling
from app.core.config import settings
from llm.fallback_provider import GroqProvider
from llm.gemini_provider import GeminiProvider, LLMResponse
from llm.model_router import ModelRouter
from llm.nvidia_provider import NvidiaProvider, OpenAICompatibleClient
from llm.provider_errors import (
    AuthenticationError,
    DegradedModeError,
    InvalidResponseError,
    ModelNotFoundError,
    ProviderStatus,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from llm.usage_tracker import track_session_usage, usage_tracker


@pytest.fixture(autouse=True)
def _fresh_tracker():
    usage_tracker.reset()
    yield


# ---------------------------------------------------------------------------
# NVIDIA provider over a mocked HTTP endpoint
# ---------------------------------------------------------------------------

def _http(monkeypatch, status=200, body=None, headers=None, raise_exc=None):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(status, json=body if body is not None else {}, headers=_headers,
                              request=httpx.Request("POST", url))

    _headers = headers or {}
    monkeypatch.setattr("llm.nvidia_provider.httpx.post", fake_post)
    return calls


def _provider():
    return NvidiaProvider(api_key="nvapi-test", model="openai/gpt-oss-120b")


def test_text_generation_uses_openai_compatible_endpoint(monkeypatch):
    calls = _http(monkeypatch, body={
        "choices": [{"message": {"content": "Hello from NVIDIA", "tool_calls": None}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })
    response = _provider().generate("Say hello", system_prompt="sys")
    assert response.content == "Hello from NVIDIA"
    assert response.provider == "nvidia" and response.total_tokens == 8
    call = calls[0]
    assert call["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer nvapi-test"
    assert call["json"]["model"] == "openai/gpt-oss-120b"
    assert "max_tokens" not in call["json"]  # unset options are omitted, not sent as null


def test_tool_call_parsing_and_per_request_timeout(monkeypatch):
    calls = _http(monkeypatch, body={"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read_research_paper",
                                                      "arguments": json.dumps({"paper_id": "p2"})}}]}}]})
    spec = tool_calling.TOOL_SPECS["read_research_paper"]
    response = _provider().generate_tool_call("prompt", [spec], timeout=7.5)
    assert response.structured_data.name == "read_research_paper"
    assert response.structured_data.arguments == {"paper_id": "p2"}
    assert calls[0]["json"]["tool_choice"] == "required"
    assert calls[0]["timeout"] == 7.5


@pytest.mark.parametrize("status, headers, error", [
    (401, {}, AuthenticationError),
    (403, {}, AuthenticationError),
    (402, {}, QuotaExceededError),
    (404, {}, ModelNotFoundError),
    (429, {"Retry-After": "12"}, RateLimitError),
    (503, {}, ProviderUnavailableError),
])
def test_http_errors_are_classified(monkeypatch, status, headers, error):
    _http(monkeypatch, status=status, body={"detail": "x"}, headers=headers)
    with pytest.raises(error) as info:
        _provider().generate("q")
    assert info.value.provider == "nvidia"
    if status == 429:
        assert info.value.retry_after == 12.0


def test_timeout_is_a_provider_unavailable_error(monkeypatch):
    _http(monkeypatch, raise_exc=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderUnavailableError, match="timeout"):
        _provider().generate("q")


def test_missing_key_means_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", None)
    provider = NvidiaProvider()
    assert provider.status == ProviderStatus.AUTHENTICATION_ERROR
    with pytest.raises(AuthenticationError, match="NVIDIA_API_KEY"):
        provider.generate("q")


# ---------------------------------------------------------------------------
# Router chain: Gemini -> Groq -> NVIDIA -> degraded
# ---------------------------------------------------------------------------

def _mock(cls, name, side_effect=None):
    mock = MagicMock(spec=cls)
    mock.name, mock.model, mock.status = name, f"{name}-model", ProviderStatus.AVAILABLE
    if side_effect is not None:
        mock.generate.side_effect = side_effect
    else:
        mock.generate.return_value = LLMResponse(content=f"from {name}", provider=name, model="m",
                                                 latency_seconds=0.0)
    return mock


def _chain(gemini_effect=None, groq_effect=None, nvidia_effect=None):
    gemini = _mock(GeminiProvider, "gemini", gemini_effect)
    groq = _mock(GroqProvider, "groq", groq_effect)
    nvidia = _mock(NvidiaProvider, "nvidia", nvidia_effect)
    router = ModelRouter(gemini_provider=gemini, fallback_provider=groq, extra_fallback_providers=[nvidia],
                         initial_backoff_seconds=0.0)
    return router, gemini, groq, nvidia


def test_nvidia_takes_over_when_gemini_and_groq_fail():
    router, gemini, groq, nvidia = _chain(
        gemini_effect=QuotaExceededError("quota", provider="gemini"),
        groq_effect=RateLimitError("429", provider="groq", retry_after=10.0),
    )
    with track_session_usage("nvidia-chain") as usage:
        assert router.generate("q").provider == "nvidia"
    assert not router.is_degraded_mode()
    snap = usage.snapshot()
    assert snap["llm"]["nvidia"]["successes"] == 1
    assert snap["fallbacks"] == 2  # gemini -> groq, groq -> nvidia


def test_cooling_groq_is_skipped_straight_to_nvidia():
    router, gemini, groq, nvidia = _chain(
        gemini_effect=RateLimitError("429", provider="gemini", retry_after=20.0),
        groq_effect=RateLimitError("429", provider="groq", retry_after=20.0),
    )
    router.generate("q1")
    assert router.generate("q2").provider == "nvidia"
    assert groq.generate.call_count == 1 and gemini.generate.call_count == 1  # both skipped the second time
    assert nvidia.generate.call_count == 2


def test_nvidia_is_not_called_when_groq_answers():
    router, _, groq, nvidia = _chain(gemini_effect=ProviderUnavailableError("503 overloaded", provider="gemini"))
    assert router.generate("q").provider == "groq"
    nvidia.generate.assert_not_called()


def test_all_three_hard_failures_enter_degraded_mode():
    router, *_ = _chain(
        gemini_effect=QuotaExceededError("quota", provider="gemini"),
        groq_effect=ProviderUnavailableError("500 internal", provider="groq"),
        nvidia_effect=ProviderUnavailableError("500 internal", provider="nvidia"),
    )
    with pytest.raises(DegradedModeError, match="groq, nvidia"):
        router.generate("q")
    assert router.is_degraded_mode()


def test_rate_limited_chain_is_temporary_not_degraded():
    router, *_ = _chain(
        gemini_effect=RateLimitError("429", provider="gemini", retry_after=5.0),
        groq_effect=RateLimitError("429", provider="groq", retry_after=5.0),
        nvidia_effect=ProviderUnavailableError("500 internal", provider="nvidia"),
    )
    with pytest.raises(DegradedModeError, match="rate limited"):
        router.generate("q")
    assert not router.is_degraded_mode()


def test_budgeted_proposal_tries_nvidia_within_budget_only():
    gemini = _mock(GeminiProvider, "gemini")
    groq = _mock(GroqProvider, "groq")
    nvidia = _mock(NvidiaProvider, "nvidia")
    gemini.generate_tool_call.side_effect = InvalidResponseError("no call", provider="gemini")

    def slow_groq(*args, **kwargs):
        time.sleep(0.3)
        raise InvalidResponseError("no call", provider="groq")

    groq.generate_tool_call.side_effect = slow_groq
    router = ModelRouter(gemini_provider=gemini, fallback_provider=groq, extra_fallback_providers=[nvidia])
    with pytest.raises(InvalidResponseError):
        router.generate_tool_call("p", [], single_shot=True, timeout=3.2)  # < 3 s left after Groq
    nvidia.generate_tool_call.assert_not_called()
    assert not router.is_degraded_mode()


def test_nvidia_joins_the_default_chain_only_with_a_key(monkeypatch):
    gemini, groq = _mock(GeminiProvider, "gemini"), _mock(GroqProvider, "groq")
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", None)
    assert [p.name for p in ModelRouter(gemini_provider=gemini, fallback_provider=groq).fallbacks] == ["groq"]
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", "nvapi-test")
    assert [p.name for p in ModelRouter(gemini_provider=gemini, fallback_provider=groq).fallbacks] == [
        "groq", "nvidia"]

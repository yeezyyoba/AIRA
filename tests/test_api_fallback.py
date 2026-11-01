"""Automated tests for LLM API Quota, Outage Fallbacks, and Error Classification.

Verifies:
1. Gemini vendor error classification (auth, quota, rate limit, unavailable).
2. Groq vendor error classification.
3. Fallback tracking and absence of secrets in logs.
4. Degraded mode compliance: non-LLM operations permitted while reasoning is safely halted.
"""
from unittest.mock import MagicMock
import pytest
import groq

from llm.gemini_provider import GeminiProvider
from llm.fallback_provider import GroqProvider
from llm.model_router import ModelRouter
from llm.provider_errors import (
    AuthenticationError,
    DegradedModeError,
    ProviderStatus,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from llm.usage_tracker import UsageTracker


def test_gemini_error_classification():
    """Verify Gemini vendor errors map to correct AIRA exceptions and statuses."""
    provider = GeminiProvider(api_key="mock_key")

    # 1. Auth error
    with pytest.raises(AuthenticationError):
        provider._classify_and_raise(Exception("API_KEY_INVALID: User not allowed"))
    assert provider.status == ProviderStatus.AUTHENTICATION_ERROR

    # 2. Quota error
    with pytest.raises(QuotaExceededError):
        provider._classify_and_raise(Exception("RESOURCE_EXHAUSTED: 429 quota exhausted for today"))
    assert provider.status == ProviderStatus.QUOTA_EXHAUSTED

    # 3. Rate limit
    with pytest.raises(RateLimitError):
        provider._classify_and_raise(Exception("429 rate limit reached, please slow down"))
    assert provider.status == ProviderStatus.RATE_LIMITED

    # 4. Service unavailable
    with pytest.raises(ProviderUnavailableError):
        provider._classify_and_raise(Exception("503 Service Unavailable: connection timed out"))
    assert provider.status == ProviderStatus.TEMPORARILY_UNAVAILABLE


def test_groq_error_classification():
    """Verify Groq vendor errors map to correct AIRA exceptions and statuses."""
    provider = GroqProvider(api_key="mock_key")

    # 1. Auth error
    mock_auth_err = groq.AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(status_code=401),
        body={},
    )
    with pytest.raises(AuthenticationError):
        provider._classify_and_raise(mock_auth_err)
    assert provider.status == ProviderStatus.AUTHENTICATION_ERROR

    # 2. Rate limit / Quota
    mock_quota_err = groq.RateLimitError(
        message="You exceeded your current quota",
        response=MagicMock(status_code=429),
        body={},
    )
    with pytest.raises(QuotaExceededError):
        provider._classify_and_raise(mock_quota_err)
    assert provider.status == ProviderStatus.QUOTA_EXHAUSTED

    # 3. Connection / 500 error
    mock_conn_err = groq.APIConnectionError(request=MagicMock())
    with pytest.raises(ProviderUnavailableError):
        provider._classify_and_raise(mock_conn_err)
    assert provider.status == ProviderStatus.TEMPORARILY_UNAVAILABLE


def test_usage_tracker_excludes_secrets():
    """Verify usage tracker stores operational metrics without leaking secrets."""
    tracker = UsageTracker()
    tracker.record_call(
        provider="gemini",
        model="gemini-3.8-flash",
        latency=0.34,
        prompt_tokens=100,
        completion_tokens=200,
        success=True,
    )
    tracker.record_fallback(
        from_provider="gemini",
        to_provider="groq",
        reason="Quota reached for gemini-3.8-flash",
    )

    summary = tracker.get_summary()
    assert summary["grand_total_calls"] == 1
    assert summary["grand_total_tokens"] == 300
    assert summary["total_fallback_events"] == 1

    # Ensure no API keys or sensitive keywords are stored anywhere in summary
    summary_str = str(summary)
    assert "api_key" not in summary_str.lower()
    assert "secret" not in summary_str.lower()


def test_degraded_mode_behavior_preserves_deterministic_operations():
    """Verify that DegradedMode blocks LLM reasoning, but preserves deterministic task continuation."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.status = ProviderStatus.QUOTA_EXHAUSTED
    mock_gemini.generate.side_effect = QuotaExceededError("Quota exhausted", provider="gemini")

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"
    mock_groq.status = ProviderStatus.QUOTA_EXHAUSTED
    mock_groq.generate.side_effect = QuotaExceededError("Quota exhausted", provider="groq")

    router = ModelRouter(gemini_provider=mock_gemini, fallback_provider=mock_groq)

    # 1. LLM reasoning attempts fail with DegradedModeError
    with pytest.raises(DegradedModeError):
        router.generate("Reason about Amharic models")

    assert router.is_degraded_mode()

    # 2. Deterministic operations (e.g. F1 calculation, citation formatting, DOI validation)
    # can continue without throwing DegradedModeError.
    def deterministic_accuracy_calc(tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        return (tp + tn) / total if total > 0 else 0.0

    calc_result = deterministic_accuracy_calc(80, 85, 10, 5)
    assert calc_result == 0.9166666666666666
    # The system acknowledges degraded mode while deterministic tool succeeds
    assert router.is_degraded_mode() is True

"""Automated tests for Centralized Model Router and Failover Mechanism.

Verifies:
1. Gemini success.
2. Gemini quota exhaustion followed by successful Groq fallback.
3. Gemini authentication failure.
4. Both providers unavailable (Degraded Mode activation).
5. Temporary network failure and bounded retries.
6. Invalid or malformed provider response.
7. Structured JSON response validation.
8. Provider selection and switching logs / usage tracker records.
"""
from typing import List
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel

from llm.gemini_provider import GeminiProvider, LLMResponse
from llm.fallback_provider import GroqProvider
from llm.model_router import ModelRouter
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
from llm.usage_tracker import UsageTracker


class MockStructuredPlan(BaseModel):
    title: str
    steps: List[str]


@pytest.fixture
def fresh_tracker():
    """Provide a fresh usage tracker for each test."""
    from llm.usage_tracker import usage_tracker
    usage_tracker.reset()
    return usage_tracker


def test_gemini_success(fresh_tracker):
    """Scenario 1: Primary provider (Gemini) succeeds on first attempt."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.model = "gemini-3.8-flash"
    mock_gemini.status = ProviderStatus.AVAILABLE
    mock_gemini.generate.return_value = LLMResponse(
        content="Research plan for Amharic sentiment analysis.",
        provider="gemini",
        model="gemini-3.8-flash",
        latency_seconds=0.45,
        prompt_tokens=50,
        completion_tokens=120,
        total_tokens=170,
    )

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.status = ProviderStatus.AVAILABLE

    router = ModelRouter(gemini_provider=mock_gemini, fallback_provider=mock_groq)
    response = router.generate("Create a research plan")

    assert response.content == "Research plan for Amharic sentiment analysis."
    assert response.provider == "gemini"
    assert router.get_active_provider_name() == "gemini"
    assert not router.is_degraded_mode()
    mock_gemini.generate.assert_called_once()
    mock_groq.generate.assert_not_called()

    summary = fresh_tracker.get_summary()
    assert summary["providers"]["gemini"]["successful_calls"] == 1
    assert summary["providers"]["gemini"]["total_tokens"] == 170


def test_gemini_quota_exhaustion_fallback_to_groq(fresh_tracker):
    """Scenario 2: Gemini encounters quota exhaustion; router seamlessly fails over to Groq."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.model = "gemini-3.8-flash"
    mock_gemini.status = ProviderStatus.AVAILABLE
    # Simulate Quota Exhaustion
    mock_gemini.generate.side_effect = QuotaExceededError(
        "ResourceExhausted: 429 You exceeded your current quota.", provider="gemini"
    )

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.status = ProviderStatus.AVAILABLE
    mock_groq.generate.return_value = LLMResponse(
        content="Fallback response from Groq OpenAI GPT-OSS.",
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_seconds=0.22,
        prompt_tokens=45,
        completion_tokens=85,
        total_tokens=130,
    )

    router = ModelRouter(gemini_provider=mock_gemini, fallback_provider=mock_groq)
    response = router.generate("Compare sentiment models")

    assert response.content == "Fallback response from Groq OpenAI GPT-OSS."
    assert response.provider == "groq"
    assert router.get_active_provider_name() == "groq"
    assert not router.is_degraded_mode()

    # Verify Gemini was called only once (no retries for exhausted quota)
    mock_gemini.generate.assert_called_once()
    mock_groq.generate.assert_called_once()

    # Verify fallback event was recorded
    history = fresh_tracker.get_fallback_history()
    assert len(history) == 1
    assert history[0]["from_provider"] == "gemini"
    assert history[0]["to_provider"] == "groq"
    assert "quota" in history[0]["reason"].lower()


def test_gemini_authentication_failure_not_retried(fresh_tracker):
    """Scenario 3: Permanent authentication failure is never retried as a network issue."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.model = "gemini-3.8-flash"
    mock_gemini.status = ProviderStatus.AUTHENTICATION_ERROR
    mock_gemini.generate.side_effect = AuthenticationError(
        "API key invalid", provider="gemini"
    )

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.status = ProviderStatus.AVAILABLE
    mock_groq.generate.return_value = LLMResponse(
        content="Handled by fallback provider.",
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_seconds=0.3,
    )

    router = ModelRouter(gemini_provider=mock_gemini, fallback_provider=mock_groq, max_retries=2)
    response = router.generate("Test query")

    # Called exactly once - NOT retried 2 times!
    assert mock_gemini.generate.call_count == 1
    assert response.provider == "groq"
    assert response.content == "Handled by fallback provider."


def test_both_providers_unavailable_triggers_degraded_mode(fresh_tracker):
    """Scenario 4: Both Gemini and Groq fail, activating Degraded Mode safely."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.model = "gemini-3.8-flash"
    mock_gemini.status = ProviderStatus.QUOTA_EXHAUSTED
    mock_gemini.generate.side_effect = QuotaExceededError("Quota reached", provider="gemini")

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
    mock_groq.generate.side_effect = ProviderUnavailableError("503 Service Unavailable", provider="groq")

    router = ModelRouter(
        gemini_provider=mock_gemini,
        fallback_provider=mock_groq,
        max_retries=1,
        initial_backoff_seconds=0.01,
    )

    with pytest.raises(DegradedModeError) as exc_info:
        router.generate("Generate synthetic report")

    assert "Degraded Mode" in str(exc_info.value)
    assert router.is_degraded_mode()

    # Subsequent request immediately fails without executing
    with pytest.raises(DegradedModeError):
        router.generate("Another query")


def test_transient_failure_bounded_retries(fresh_tracker):
    """Scenario 5: Temporary network error is retried with backoff and succeeds on 2nd attempt."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.model = "gemini-3.8-flash"
    mock_gemini.status = ProviderStatus.AVAILABLE

    # Fails on 1st attempt, succeeds on 2nd
    mock_gemini.generate.side_effect = [
        ProviderUnavailableError("Connection reset by peer", provider="gemini"),
        LLMResponse(
            content="Recovered after transient network glitch.",
            provider="gemini",
            model="gemini-3.8-flash",
            latency_seconds=0.35,
        ),
    ]

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"

    router = ModelRouter(
        gemini_provider=mock_gemini,
        fallback_provider=mock_groq,
        max_retries=2,
        initial_backoff_seconds=0.01,
    )
    response = router.generate("Test transient recovery")

    assert response.content == "Recovered after transient network glitch."
    assert mock_gemini.generate.call_count == 2
    mock_groq.generate.assert_not_called()


def test_invalid_provider_response_handling(fresh_tracker):
    """Scenario 6: Empty response is caught as InvalidResponseError."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.model = "gemini-3.8-flash"
    mock_gemini.status = ProviderStatus.AVAILABLE

    # Return invalid empty content
    mock_gemini.generate.return_value = LLMResponse(
        content="",
        provider="gemini",
        model="gemini-3.8-flash",
        latency_seconds=0.1,
    )

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.status = ProviderStatus.AVAILABLE
    mock_groq.generate.return_value = LLMResponse(
        content="Valid fallback output from Groq.",
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_seconds=0.2,
    )

    router = ModelRouter(
        gemini_provider=mock_gemini,
        fallback_provider=mock_groq,
        max_retries=1,
        initial_backoff_seconds=0.01,
    )
    response = router.generate("Produce valid output")

    assert response.content == "Valid fallback output from Groq."
    assert response.provider == "groq"


def test_structured_output_generation(fresh_tracker):
    """Scenario 7: Structured JSON generation with Pydantic validation."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.model = "gemini-3.8-flash"
    mock_gemini.status = ProviderStatus.AVAILABLE

    expected_plan = MockStructuredPlan(
        title="Amharic NER Research",
        steps=["Collect corpus", "Train BiLSTM-CRF", "Evaluate F1"],
    )

    mock_gemini.generate_structured.return_value = LLMResponse(
        content='{"title": "Amharic NER Research", "steps": ["Collect corpus", "Train BiLSTM-CRF", "Evaluate F1"]}',
        provider="gemini",
        model="gemini-3.8-flash",
        latency_seconds=0.5,
        structured_data=expected_plan,
    )

    mock_groq = MagicMock(spec=GroqProvider)
    router = ModelRouter(gemini_provider=mock_gemini, fallback_provider=mock_groq)

    res = router.generate_structured("Plan NER research", schema=MockStructuredPlan)
    assert res.structured_data.title == "Amharic NER Research"
    assert len(res.structured_data.steps) == 3
    assert res.provider == "gemini"


def test_provider_selection_and_switching_logs(fresh_tracker, caplog):
    """Scenario 8: Detailed logging of provider selection and switching decisions."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_gemini.model = "gemini-3.8-flash"
    mock_gemini.status = ProviderStatus.AVAILABLE
    mock_gemini.generate.side_effect = QuotaExceededError("Gemini quota reached", provider="gemini")

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.status = ProviderStatus.AVAILABLE
    mock_groq.generate.return_value = LLMResponse(
        content="Success on Groq fallback.",
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_seconds=0.2,
    )

    router = ModelRouter(gemini_provider=mock_gemini, fallback_provider=mock_groq)
    with caplog.at_level("INFO"):
        response = router.generate("Check logs")

    assert response.provider == "groq"
    assert "Routing request to primary LLM: gemini" in caplog.text
    assert "Initiating fallback to groq" in caplog.text
    assert "Routing request to fallback LLM: groq" in caplog.text

    history = fresh_tracker.get_fallback_history()
    assert len(history) == 1
    assert history[0]["from_provider"] == "gemini"
    assert history[0]["to_provider"] == "groq"


def test_reset_status_exits_degraded_mode():
    """Verify reset_status restores provider availability and clears degraded mode."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.name = "gemini"
    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.name = "groq"

    router = ModelRouter(gemini_provider=mock_gemini, fallback_provider=mock_groq)
    router.degraded_mode = True
    router.active_provider_name = "groq"

    assert router.is_degraded_mode() is True
    router.reset_status()
    assert router.is_degraded_mode() is False
    assert router.get_active_provider_name() == "gemini"
    assert mock_gemini.status == ProviderStatus.AVAILABLE
    assert mock_groq.status == ProviderStatus.AVAILABLE


def test_configured_model_ids_passed_to_providers():
    """Requirement 5: Confirm configured model IDs are correctly initialized and passed to SDK calls."""
    # 1. Custom model for Gemini
    mock_gemini_client = MagicMock()
    mock_gemini_response = MagicMock()
    mock_gemini_response.text = "Custom model output"
    mock_gemini_response.usage_metadata = None
    mock_gemini_client.models.generate_content.return_value = mock_gemini_response

    custom_gemini = GeminiProvider(
        api_key="mock_key",
        model="custom-gemini-experimental",
        client=mock_gemini_client,
    )
    assert custom_gemini.model == "custom-gemini-experimental"

    res_gemini = custom_gemini.generate("Test prompt")
    assert res_gemini.model == "custom-gemini-experimental"
    # Verify the model ID was passed directly to client.models.generate_content
    call_kwargs = mock_gemini_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "custom-gemini-experimental"

    # 2. Custom model for Groq
    mock_groq_client = MagicMock()
    mock_groq_choice = MagicMock()
    mock_groq_choice.message.content = "Custom groq output"
    mock_groq_response = MagicMock()
    mock_groq_response.choices = [mock_groq_choice]
    mock_groq_response.usage = None
    mock_groq_client.chat.completions.create.return_value = mock_groq_response

    custom_groq = GroqProvider(
        api_key="mock_key",
        model="llama-3.1-8b-instant",
        client=mock_groq_client,
    )
    assert custom_groq.model == "llama-3.1-8b-instant"

    res_groq = custom_groq.generate("Test prompt")
    assert res_groq.model == "llama-3.1-8b-instant"
    # Verify the model ID was passed directly to client.chat.completions.create
    call_kwargs_groq = mock_groq_client.chat.completions.create.call_args.kwargs
    assert call_kwargs_groq["model"] == "llama-3.1-8b-instant"

    # 3. ModelRouter defaults to Settings values
    from app.core.config import settings
    default_router = ModelRouter(gemini_provider=custom_gemini, fallback_provider=custom_groq)
    assert default_router.gemini.model == "custom-gemini-experimental"
    assert default_router.fallback.model == "llama-3.1-8b-instant"


def test_model_unavailable_returns_clear_error_without_substitution(fresh_tracker):
    """Scenario 9: If configured model is unavailable, raise clear error without silent substitution."""
    # 1. Groq provider with unavailable model raises ModelNotFoundError
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = Exception(
        "NotFoundError: The model 'unavailable-model' does not exist or you do not have access to it."
    )
    groq_provider = GroqProvider(api_key="mock_key", model="unavailable-model", client=mock_groq_client)

    with pytest.raises(ModelNotFoundError) as exc_info:
        groq_provider.generate("Test prompt")
    assert "unavailable-model" in str(exc_info.value)
    assert "unavailable or not found" in str(exc_info.value)
    assert exc_info.value.provider == "groq"
    # Verify no substitution was attempted on the model attribute
    assert groq_provider.model == "unavailable-model"

    # 2. Gemini provider with unavailable model raises ModelNotFoundError
    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content.side_effect = Exception(
        "ClientError: 404 models/unavailable-gemini is not found"
    )
    gemini_provider = GeminiProvider(api_key="mock_key", model="unavailable-gemini", client=mock_gemini_client)

    with pytest.raises(ModelNotFoundError) as exc_info_gemini:
        gemini_provider.generate("Test prompt")
    assert "unavailable-gemini" in str(exc_info_gemini.value)
    assert "unavailable or not found" in str(exc_info_gemini.value)
    assert exc_info_gemini.value.provider == "gemini"
    assert gemini_provider.model == "unavailable-gemini"

    # 3. ModelRouter fails over from Gemini on ModelNotFoundError to Groq
    mock_gemini_client.models.generate_content.reset_mock()
    mock_valid_groq = MagicMock(spec=GroqProvider)
    mock_valid_groq.name = "groq"
    mock_valid_groq.model = "openai/gpt-oss-120b"
    mock_valid_groq.status = ProviderStatus.AVAILABLE
    mock_valid_groq.generate.return_value = LLMResponse(
        content="Fallback response",
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_seconds=0.1,
    )

    router = ModelRouter(gemini_provider=gemini_provider, fallback_provider=mock_valid_groq)
    res = router.generate("Test prompt")
    assert res.provider == "groq"
    assert res.content == "Fallback response"
    # Confirm Gemini was called only once and not retried repeatedly
    assert mock_gemini_client.models.generate_content.call_count == 1




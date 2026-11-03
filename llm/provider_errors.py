"""AIRA LLM Provider Error and Status Definitions.

Provides standardized exceptions and status enums across all LLM providers.
"""
import re
from enum import Enum
from typing import Any, Optional


class ProviderStatus(str, Enum):
    """Operational status of an LLM provider."""
    AVAILABLE = "AVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""
    def __init__(self, message: str, provider: str = "unknown", retry_after: Optional[float] = None):
        super().__init__(message)
        self.provider = provider
        # Seconds the provider asked us to wait before the next request, when it said so.
        self.retry_after = retry_after


class AuthenticationError(LLMProviderError):
    """Raised when API credentials or keys are invalid (permanent failure, do not retry)."""
    pass


class QuotaExceededError(LLMProviderError):
    """Raised when an API quota is exhausted (triggers immediate fallback, do not retry same provider)."""
    pass


class RateLimitError(LLMProviderError):
    """Raised when an API rate limit (HTTP 429) is encountered (retryable with backoff)."""
    pass


class ProviderUnavailableError(LLMProviderError):
    """Raised for transient server errors (5xx, timeouts, network issues)."""
    pass


class InvalidResponseError(LLMProviderError):
    """Raised when an LLM returns empty, blocked, or unparseable structured output."""
    pass


class DegradedModeError(LLMProviderError):
    """Raised when an LLM-dependent operation is attempted while in degraded mode (all LLMs unavailable)."""
    pass


class ModelNotFoundError(LLMProviderError):
    """Raised when the configured model does not exist or is unavailable for the API credentials."""
    pass


class BadRequestError(LLMProviderError):
    """Raised when the provider rejects the request itself (HTTP 400 / INVALID_ARGUMENT).

    Not retried against the same provider (the identical request would fail
    again), but the fallback provider may still accept it.
    """
    pass


_RETRY_HINT_PATTERNS = (
    # Gemini: "Please retry in 37.2s." / "'retryDelay': '37s'"
    re.compile(r"retry in\s+(?P<s>\d+(?:\.\d+)?)\s*s", re.IGNORECASE),
    re.compile(r"retrydelay['\"]?\s*[:=]\s*['\"]?(?P<s>\d+(?:\.\d+)?)s", re.IGNORECASE),
    # Groq: "Please try again in 1m2.5s" / "try again in 640ms" / "try again in 7.3s"
    re.compile(
        r"try again in\s+(?:(?P<m>\d+)m(?!s))?\s*(?:(?P<s>\d+(?:\.\d+)?)s|(?P<ms>\d+(?:\.\d+)?)ms)?",
        re.IGNORECASE,
    ),
)


def parse_retry_after_seconds(error: Any) -> Optional[float]:
    """Best-effort extraction of a provider's "wait N seconds" hint.

    Looks at a ``Retry-After`` response header (Groq/OpenAI-style SDK errors)
    and at the retry hints Gemini and Groq put in their error messages.
    Returns None when the provider gave no hint.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:  # pragma: no cover - exotic header containers
            value = None
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    message = str(error)
    for pattern in _RETRY_HINT_PATTERNS:
        match = pattern.search(message)
        if not match:
            continue
        groups = match.groupdict()
        minutes = float(groups.get("m") or 0)
        seconds = float(groups.get("s") or 0)
        millis = float(groups.get("ms") or 0)
        total = minutes * 60 + seconds + millis / 1000
        if total > 0:
            return total
    return None

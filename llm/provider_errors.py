"""AIRA LLM Provider Error and Status Definitions.

Provides standardized exceptions and status enums across all LLM providers.
"""
from enum import Enum


class ProviderStatus(str, Enum):
    """Operational status of an LLM provider."""
    AVAILABLE = "AVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message)
        self.provider = provider


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

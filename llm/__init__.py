"""AIRA LLM Package.

Exposes ModelRouter, GeminiProvider, GroqProvider, exceptions, and usage tracking.
"""
from llm.gemini_provider import GeminiProvider, LLMResponse
from llm.fallback_provider import GroqProvider
from llm.model_router import ModelRouter, model_router
from llm.provider_errors import (
    AuthenticationError,
    DegradedModeError,
    InvalidResponseError,
    LLMProviderError,
    ProviderStatus,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from llm.usage_tracker import UsageTracker, usage_tracker

__all__ = [
    "GeminiProvider",
    "GroqProvider",
    "ModelRouter",
    "model_router",
    "LLMResponse",
    "UsageTracker",
    "usage_tracker",
    "ProviderStatus",
    "LLMProviderError",
    "AuthenticationError",
    "QuotaExceededError",
    "RateLimitError",
    "ProviderUnavailableError",
    "InvalidResponseError",
    "DegradedModeError",
]

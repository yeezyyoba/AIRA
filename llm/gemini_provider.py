"""Google Gemini Primary LLM Provider.

Integrates with the official Google Gen AI SDK (google-genai) with robust
error classification for quotas, auth, rate limits, and network issues.
"""
from dataclasses import dataclass
import json
import time
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.core.config import settings
from llm.provider_errors import (
    AuthenticationError,
    InvalidResponseError,
    ModelNotFoundError,
    ProviderStatus,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from logging_config.logger import get_logger

logger = get_logger("gemini_provider")


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: str
    model: str
    latency_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    structured_data: Optional[Any] = None
    raw_response: Optional[Any] = None


class GeminiProvider:
    """Primary LLM provider using Google GenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.name = "gemini"
        self.status = ProviderStatus.AVAILABLE

        # Allow passing mock client directly for deterministic unit tests
        if client is not None:
            self._client = client
        elif self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.status = ProviderStatus.AUTHENTICATION_ERROR
                self._client = None
        else:
            self.status = ProviderStatus.AUTHENTICATION_ERROR
            self._client = None

    def _ensure_authenticated(self) -> None:
        """Verify client readiness before attempting calls."""
        if not self.api_key and self._client is None:
            self.status = ProviderStatus.AUTHENTICATION_ERROR
            raise AuthenticationError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in .env.",
                provider=self.name,
            )

    def _classify_and_raise(self, error: Exception) -> None:
        """Classify vendor-specific errors into standardized AIRA exceptions."""
        err_msg = str(error).lower()
        err_type = type(error).__name__.lower()

        logger.error(f"Gemini API error [{err_type}]: {error}")

        if (
            "api_key_invalid" in err_msg
            or "invalid_argument" in err_msg
            or "permission_denied" in err_msg
            or "unauthenticated" in err_msg
            or "401" in err_msg
            or "403" in err_msg
        ):
            self.status = ProviderStatus.AUTHENTICATION_ERROR
            raise AuthenticationError(f"Gemini authentication failed: {error}", provider=self.name)

        if "not_found" in err_msg or "404" in err_msg:
            self.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
            raise ModelNotFoundError(
                f"Gemini model '{self.model}' is unavailable or not found: {error}",
                provider=self.name,
            )

        if "resource_exhausted" in err_msg or "quota" in err_msg:
            self.status = ProviderStatus.QUOTA_EXHAUSTED
            raise QuotaExceededError(f"Gemini quota exhausted: {error}", provider=self.name)

        if "429" in err_msg or "rate limit" in err_msg:
            self.status = ProviderStatus.RATE_LIMITED
            raise RateLimitError(f"Gemini rate limit exceeded: {error}", provider=self.name)

        if (
            "unavailable" in err_msg
            or "deadline_exceeded" in err_msg
            or "timeout" in err_msg
            or "connection" in err_msg
            or "503" in err_msg
            or "500" in err_msg
        ):
            self.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
            raise ProviderUnavailableError(f"Gemini service temporarily unavailable: {error}", provider=self.name)

        # Default fallback to ProviderUnavailableError for unknown transient issues
        self.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
        raise ProviderUnavailableError(f"Gemini error: {error}", provider=self.name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate unstructured text from Gemini."""
        self._ensure_authenticated()

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None,
        )

        start_time = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            self._classify_and_raise(e)

        latency = time.perf_counter() - start_time

        # Validate response content
        text_content = getattr(response, "text", None)
        if not text_content:
            logger.warning("Gemini returned empty text or candidate was filtered.")
            raise InvalidResponseError(
                "Gemini returned an empty or filtered response.",
                provider=self.name,
            )

        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)

        self.status = ProviderStatus.AVAILABLE
        return LLMResponse(
            content=text_content,
            provider=self.name,
            model=self.model,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=response,
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate structured Pydantic-validated JSON from Gemini."""
        self._ensure_authenticated()

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
            response_mime_type="application/json",
            response_schema=schema,
        )

        start_time = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            self._classify_and_raise(e)

        latency = time.perf_counter() - start_time
        raw_text = getattr(response, "text", "")

        if not raw_text:
            raise InvalidResponseError("Gemini returned empty structured content.", provider=self.name)

        try:
            validated_obj = schema.model_validate_json(raw_text)
        except Exception as e:
            logger.error(f"Failed to validate structured JSON against {schema.__name__}: {e}")
            raise InvalidResponseError(
                f"Gemini output could not be parsed into {schema.__name__}: {e}",
                provider=self.name,
            )

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)

        self.status = ProviderStatus.AVAILABLE
        return LLMResponse(
            content=raw_text,
            provider=self.name,
            model=self.model,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            structured_data=validated_obj,
            raw_response=response,
        )

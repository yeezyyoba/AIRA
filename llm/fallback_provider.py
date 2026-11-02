"""Groq Independent Fallback LLM Provider.

Integrates with Groq SDK with equivalent error classification for quotas,
authentication, rate limits, and network issues.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
import groq

from app.core.config import settings
from llm.gemini_provider import LLMResponse, strip_json_fences
from llm.provider_errors import (
    AuthenticationError,
    BadRequestError,
    InvalidResponseError,
    ModelNotFoundError,
    ProviderStatus,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from logging_config.logger import get_logger

logger = get_logger("groq_provider")


class GroqProvider:
    """Independent fallback LLM provider using Groq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.name = "groq"
        self.status = ProviderStatus.AVAILABLE

        # Allow passing mock client directly for deterministic unit tests
        if client is not None:
            self._client = client
        elif self.api_key:
            try:
                # The router owns retries; disable the SDK's own retry loop so a
                # failing call is not silently repeated up to 9 times.
                self._client = groq.Groq(
                    api_key=self.api_key,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                    max_retries=0,
                )
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
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
                "Groq API key is not configured. Set GROQ_API_KEY in .env.",
                provider=self.name,
            )

    def _classify_and_raise(self, error: Exception) -> None:
        """Classify vendor-specific errors into standardized AIRA exceptions."""
        err_msg = str(error).lower()
        err_type = type(error).__name__.lower()

        logger.error(f"Groq API error [{err_type}]: {error}")

        if isinstance(error, groq.AuthenticationError) or "401" in err_msg or "unauthorized" in err_msg:
            self.status = ProviderStatus.AUTHENTICATION_ERROR
            raise AuthenticationError(f"Groq authentication failed: {error}", provider=self.name)

        if (
            isinstance(error, getattr(groq, "NotFoundError", Exception))
            or "model_not_found" in err_msg
            or "does not exist" in err_msg
            or "404" in err_msg
        ):
            self.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
            raise ModelNotFoundError(
                f"Groq model '{self.model}' is unavailable or not found: {error}",
                provider=self.name,
            )

        if isinstance(error, groq.RateLimitError) or "429" in err_msg:
            if "quota" in err_msg or "exceeded your current quota" in err_msg:
                self.status = ProviderStatus.QUOTA_EXHAUSTED
                raise QuotaExceededError(f"Groq quota exhausted: {error}", provider=self.name)
            self.status = ProviderStatus.RATE_LIMITED
            raise RateLimitError(f"Groq rate limit exceeded: {error}", provider=self.name)

        if (
            isinstance(error, (groq.APIConnectionError, groq.InternalServerError, groq.APITimeoutError))
            or "503" in err_msg
            or "500" in err_msg
            or "timeout" in err_msg
            or "connection" in err_msg
        ):
            self.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
            raise ProviderUnavailableError(f"Groq service temporarily unavailable: {error}", provider=self.name)

        if isinstance(error, getattr(groq, "BadRequestError", ())) or re.search(r"\b400\b", err_msg):
            raise BadRequestError(f"Groq rejected the request: {error}", provider=self.name)

        self.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
        raise ProviderUnavailableError(f"Groq error: {error}", provider=self.name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate unstructured text from Groq."""
        self._ensure_authenticated()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            self._classify_and_raise(e)

        latency = time.perf_counter() - start_time

        if not response.choices or not response.choices[0].message.content:
            logger.warning("Groq returned empty completion choice.")
            raise InvalidResponseError(
                "Groq returned an empty response.",
                provider=self.name,
            )

        text_content = response.choices[0].message.content

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            total_tokens = getattr(response.usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

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
        """Generate structured Pydantic-validated JSON from Groq."""
        self._ensure_authenticated()

        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        instruction = (
            f"You must respond with valid JSON strictly conforming to this schema:\n"
            f"{schema_json}\n"
            f"Do not include explanation or markdown formatting outside the JSON object."
        )

        sys_prompt = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        start_time = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            self._classify_and_raise(e)

        latency = time.perf_counter() - start_time

        if not response.choices or not response.choices[0].message.content:
            raise InvalidResponseError("Groq returned empty structured content.", provider=self.name)

        raw_text = response.choices[0].message.content

        try:
            validated_obj = schema.model_validate_json(strip_json_fences(raw_text))
        except Exception as e:
            logger.error(f"Failed to validate Groq structured JSON against {schema.__name__}: {e}")
            raise InvalidResponseError(
                f"Groq output could not be parsed into {schema.__name__}: {e}",
                provider=self.name,
            )

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            total_tokens = getattr(response.usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

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

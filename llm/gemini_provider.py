"""Google Gemini Primary LLM Provider.

Integrates with the official Google Gen AI SDK (google-genai) with robust
error classification for quotas, auth, rate limits, and network issues.
"""
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.core.config import settings
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


@dataclass
class ToolCall:
    """A model's proposed function call. Nothing has been executed yet."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None


def tool_call_response(call: ToolCall, provider: str, model: str, latency: float, raw: Any,
                       prompt_tokens: int = 0, completion_tokens: int = 0) -> LLMResponse:
    """Wrap a parsed tool call; ``content`` is the call as JSON so it is never empty."""
    return LLMResponse(
        content=json.dumps({"name": call.name, "arguments": call.arguments}, default=str),
        provider=provider,
        model=model,
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        structured_data=call,
        raw_response=raw,
    )


def strip_json_fences(raw_text: str) -> str:
    """Remove a surrounding Markdown code fence from model JSON output, if present."""
    text = (raw_text or "").strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else text


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
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=int(settings.LLM_TIMEOUT_SECONDS * 1000)),
                )
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

        def has_code(code: str) -> bool:
            return re.search(rf"\b{code}\b", err_msg) is not None

        if (
            "api_key_invalid" in err_msg
            or "api key not valid" in err_msg
            or "permission_denied" in err_msg
            or "unauthenticated" in err_msg
            or has_code("401")
            or has_code("403")
        ):
            self.status = ProviderStatus.AUTHENTICATION_ERROR
            raise AuthenticationError(f"Gemini authentication failed: {error}", provider=self.name)

        if "not_found" in err_msg or has_code("404"):
            self.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
            raise ModelNotFoundError(
                f"Gemini model '{self.model}' is unavailable or not found: {error}",
                provider=self.name,
            )

        if "resource_exhausted" in err_msg or "quota" in err_msg:
            self.status = ProviderStatus.QUOTA_EXHAUSTED
            raise QuotaExceededError(f"Gemini quota exhausted: {error}", provider=self.name)

        if has_code("429") or "rate limit" in err_msg:
            self.status = ProviderStatus.RATE_LIMITED
            raise RateLimitError(f"Gemini rate limit exceeded: {error}", provider=self.name)

        if (
            "unavailable" in err_msg
            or "deadline_exceeded" in err_msg
            or "timeout" in err_msg
            or "connection" in err_msg
            or has_code("503")
            or has_code("500")
            or has_code("502")
            or has_code("504")
        ):
            self.status = ProviderStatus.TEMPORARILY_UNAVAILABLE
            raise ProviderUnavailableError(f"Gemini service temporarily unavailable: {error}", provider=self.name)

        # A rejected request (bad schema, oversize context, ...) is not an auth problem.
        if "invalid_argument" in err_msg or has_code("400"):
            raise BadRequestError(f"Gemini rejected the request: {error}", provider=self.name)

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
            validated_obj = schema.model_validate_json(strip_json_fences(raw_text))
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

    def generate_tool_call(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Ask Gemini to propose exactly one call to one of ``tools`` (native function calling).

        Mode ANY forces a function call; automatic function calling is disabled
        so the SDK never executes anything: the application validates and runs it.
        """
        self._ensure_authenticated()
        declarations = [
            types.FunctionDeclaration(
                name=spec["name"],
                description=spec.get("description", ""),
                parameters_json_schema=spec.get("parameters") or {"type": "object", "properties": {}},
            )
            for spec in tools
        ]
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
            tools=[types.Tool(function_declarations=declarations)],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[spec["name"] for spec in tools],
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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

        calls = getattr(response, "function_calls", None) or []
        if not calls:
            raise InvalidResponseError("Gemini returned no function call.", provider=self.name)
        first = calls[0]
        name = getattr(first, "name", None)
        args = getattr(first, "args", None)
        if args is None:
            args = {}
        if not name or not isinstance(args, dict):
            raise InvalidResponseError("Gemini returned a malformed function call.", provider=self.name)

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        completion_tokens = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        self.status = ProviderStatus.AVAILABLE
        return tool_call_response(
            ToolCall(name=name, arguments=dict(args), call_id=getattr(first, "id", None)),
            self.name, self.model, latency, response, prompt_tokens, completion_tokens,
        )

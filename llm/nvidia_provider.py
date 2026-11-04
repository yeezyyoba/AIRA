"""NVIDIA (build.nvidia.com) LLM provider: third provider in the fallback chain.

NVIDIA's hosted models are served through an OpenAI-compatible endpoint
(``https://integrate.api.nvidia.com/v1``). This provider reuses everything in
``GroqProvider`` (text, structured JSON and tool-call parsing, error
classification); only the HTTP client differs, so no extra SDK is needed.

Configure with ``NVIDIA_API_KEY`` and ``NVIDIA_MODEL`` in ``.env``. Without a
key the provider is simply not added to the router's chain.
"""
import json
from types import SimpleNamespace
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings
from llm.fallback_provider import GroqProvider
from llm.provider_errors import (
    AuthenticationError,
    ProviderStatus,
    QuotaExceededError,
    parse_retry_after_seconds,
)
from logging_config.logger import get_logger

logger = get_logger("nvidia_provider")


class OpenAICompatibleHTTPError(Exception):
    """An HTTP error from an OpenAI-compatible endpoint.

    The message starts with the status code ("HTTP 429: ...") so the shared
    error classification recognises it, and ``response`` keeps the headers so
    a ``Retry-After`` hint is honoured.
    """

    def __init__(self, status_code: int, body: str, response: Optional[httpx.Response] = None):
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code
        self.response = response


def _to_namespace(value: Any) -> Any:
    """JSON dicts -> attribute access, matching the SDK objects GroqProvider reads."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


class _Completions:
    def __init__(self, client: "OpenAICompatibleClient"):
        self._client = client

    def create(self, timeout: Optional[float] = None, **payload: Any) -> Any:
        return self._client.post_chat(payload, timeout=timeout)


class OpenAICompatibleClient:
    """Minimal ``client.chat.completions.create(...)`` over httpx (no retries: the router owns them)."""

    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=_Completions(self))

    def post_chat(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Any:
        # Omit unset options (e.g. max_tokens=None) rather than sending JSON null.
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"},
                json=payload,
                timeout=timeout if timeout is not None else self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"timeout contacting NVIDIA API: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(f"connection error contacting NVIDIA API: {exc}") from exc
        if response.status_code != 200:
            raise OpenAICompatibleHTTPError(response.status_code, response.text, response)
        try:
            return _to_namespace(response.json())
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleHTTPError(502, f"invalid JSON from NVIDIA API: {exc}", response) from exc


class NvidiaProvider(GroqProvider):
    """Third LLM provider: NVIDIA-hosted models via the OpenAI-compatible API."""

    display_name = "NVIDIA"
    key_setting_name = "NVIDIA_API_KEY"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        base_url: Optional[str] = None,
    ):
        # Deliberately not calling GroqProvider.__init__: that builds a Groq SDK client.
        self.api_key = api_key or settings.NVIDIA_API_KEY
        self.model = model or settings.NVIDIA_MODEL
        self.name = "nvidia"
        self.status = ProviderStatus.AVAILABLE
        if client is not None:
            self._client = client
        elif self.api_key:
            self._client = OpenAICompatibleClient(
                base_url or settings.NVIDIA_BASE_URL, self.api_key, float(settings.LLM_TIMEOUT_SECONDS)
            )
        else:
            self.status = ProviderStatus.AUTHENTICATION_ERROR
            self._client = None

    def _classify_and_raise(self, error: Exception) -> None:
        """NVIDIA-specific codes first (403 = bad key, 402 = credits used up), then the shared rules."""
        status = getattr(error, "status_code", None)
        if status == 403:
            self.status = ProviderStatus.AUTHENTICATION_ERROR
            logger.error(f"NVIDIA API error [forbidden]: {error}")
            raise AuthenticationError(f"NVIDIA authentication failed: {error}", provider=self.name)
        if status == 402:
            self.status = ProviderStatus.QUOTA_EXHAUSTED
            logger.error(f"NVIDIA API error [credits exhausted]: {error}")
            raise QuotaExceededError(
                f"NVIDIA quota exhausted (no inference credits left): {error}",
                provider=self.name,
                retry_after=parse_retry_after_seconds(error),
            )
        super()._classify_and_raise(error)

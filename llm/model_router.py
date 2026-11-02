"""AIRA Centralized Model Router.

Single entry point for all LLM calls. Routes to Google Gemini primarily,
detects failures, handles bounded retries for transient errors, switches to Groq
on quota exhaustion or outages, and manages graceful Degraded Mode.
"""
import time
from typing import Any, Optional, Type
from pydantic import BaseModel

from app.core.config import settings
from llm.gemini_provider import GeminiProvider, LLMResponse
from llm.fallback_provider import GroqProvider
from llm.provider_errors import (
    AuthenticationError,
    BadRequestError,
    DegradedModeError,
    InvalidResponseError,
    ModelNotFoundError,
    ProviderStatus,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from llm.usage_tracker import usage_tracker
from logging_config.logger import get_logger

logger = get_logger("model_router")


class ModelRouter:
    """Centralized LLM router with automatic failover and degraded mode management."""

    def __init__(
        self,
        gemini_provider: Optional[GeminiProvider] = None,
        fallback_provider: Optional[GroqProvider] = None,
        max_retries: int = 2,
        initial_backoff_seconds: float = 0.5,
    ):
        self.gemini = gemini_provider or GeminiProvider()
        self.fallback = fallback_provider or GroqProvider()
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.active_provider_name: str = "gemini"
        self.degraded_mode: bool = False
        # Monotonic timestamps of when degraded mode / primary quota exhaustion
        # began. None means "set externally": stay in that state until reset.
        self._degraded_since: Optional[float] = None
        self._primary_quota_since: Optional[float] = None
        self.cooldown_seconds: float = float(settings.LLM_DEGRADED_COOLDOWN_SECONDS)

    def _cooldown_elapsed(self, since: Optional[float]) -> bool:
        return since is not None and (time.monotonic() - since) >= self.cooldown_seconds

    def is_degraded_mode(self) -> bool:
        """Return True if all LLM providers are currently exhausted or unavailable.

        Degraded mode entered by the router expires after a cooldown so a
        transient outage does not disable reasoning until the process restarts.
        """
        if self.degraded_mode and self._cooldown_elapsed(self._degraded_since):
            logger.info("Degraded-mode cooldown elapsed; probing LLM providers again.")
            self.reset_status()
        return self.degraded_mode

    def get_active_provider_name(self) -> str:
        """Return name of the provider currently selected for requests."""
        return self.active_provider_name

    def reset_status(self) -> None:
        """Reset provider statuses and exit degraded mode (e.g. after config update)."""
        self.gemini.status = ProviderStatus.AVAILABLE
        self.fallback.status = ProviderStatus.AVAILABLE
        self.active_provider_name = "gemini"
        self.degraded_mode = False
        self._degraded_since = None
        self._primary_quota_since = None
        logger.info("ModelRouter status reset: providers set to AVAILABLE.")

    def _execute_with_retry(
        self,
        provider: Any,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a provider call with bounded exponential backoff on transient errors.

        Permanent errors (AuthenticationError, QuotaExceededError) are NOT retried.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                method = getattr(provider, method_name)
                response: LLMResponse = method(*args, **kwargs)

                # Validate response before returning
                if not response.content or not isinstance(response.content, str):
                    raise InvalidResponseError(
                        f"Empty or non-string response content from {provider.name}.",
                        provider=provider.name,
                    )

                usage_tracker.record_call(
                    provider=response.provider,
                    model=response.model,
                    latency=response.latency_seconds,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    success=True,
                )
                return response

            except AuthenticationError as auth_err:
                # Permanent failure: Do not retry authentication errors!
                usage_tracker.record_call(
                    provider=provider.name,
                    model=provider.model,
                    latency=0.0,
                    success=False,
                )
                usage_tracker.record_error(provider.name, "AuthenticationError")
                logger.error(f"Permanent authentication failure for {provider.name}. Not retrying.")
                raise auth_err

            except QuotaExceededError as quota_err:
                # Quota exhausted: Do not retry the same provider!
                usage_tracker.record_call(
                    provider=provider.name,
                    model=provider.model,
                    latency=0.0,
                    success=False,
                )
                usage_tracker.record_error(provider.name, "QuotaExceededError")
                logger.warning(f"Quota exhausted for {provider.name}. Skipping retries to trigger fallback.")
                raise quota_err

            except ModelNotFoundError as model_err:
                # Configured model not found / unavailable: Do not retry!
                usage_tracker.record_call(
                    provider=provider.name,
                    model=provider.model,
                    latency=0.0,
                    success=False,
                )
                usage_tracker.record_error(provider.name, "ModelNotFoundError")
                logger.error(f"Configured model unavailable for {provider.name}: {model_err}. Skipping retries.")
                raise model_err

            except BadRequestError as bad_request:
                # The identical request would be rejected again: do not retry.
                usage_tracker.record_call(
                    provider=provider.name,
                    model=provider.model,
                    latency=0.0,
                    success=False,
                )
                usage_tracker.record_error(provider.name, "BadRequestError")
                logger.error(f"{provider.name} rejected the request: {bad_request}. Not retrying.")
                raise bad_request

            except (RateLimitError, ProviderUnavailableError, InvalidResponseError) as transient_err:
                last_exception = transient_err
                usage_tracker.record_error(provider.name, type(transient_err).__name__)

                if attempt < self.max_retries:
                    backoff = min(self.initial_backoff_seconds * (2 ** attempt), 2.0)
                    logger.warning(
                        f"{provider.name} transient error [{type(transient_err).__name__}]. "
                        f"Retrying ({attempt + 1}/{self.max_retries}) after {backoff:.2f}s backoff..."
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"{provider.name} exceeded max retries ({self.max_retries}). "
                        f"Final error: {transient_err}"
                    )

            except Exception as unhandled_err:
                usage_tracker.record_error(provider.name, type(unhandled_err).__name__)
                logger.error(f"Unexpected error in {provider.name}: {unhandled_err}")
                last_exception = unhandled_err
                break

        usage_tracker.record_call(
            provider=provider.name,
            model=provider.model,
            latency=0.0,
            success=False,
        )
        if last_exception:
            raise last_exception
        raise ProviderUnavailableError(f"Provider {provider.name} failed after retries.", provider=provider.name)

    def _route_request(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> LLMResponse:
        """Central routing loop: Try Gemini -> On eligible failure switch to Groq -> On failure enter Degraded Mode."""
        # Check if already in degraded mode (expires after the cooldown)
        if self.is_degraded_mode():
            logger.error("LLM reasoning requested while ModelRouter is in DEGRADED MODE.")
            raise DegradedModeError(
                "All configured LLM providers (Gemini and Groq) are currently unavailable. "
                "The system is operating in Degraded Mode: deterministic operations "
                "(PDF extraction, academic searches, DOI validation, citation formatting, "
                "and metric computation) remain available, but reasoning is paused.",
                provider="router",
            )

        # 1. Attempt Primary Provider (Gemini) if not already marked unavailable
        if self.gemini.status == ProviderStatus.QUOTA_EXHAUSTED and self._cooldown_elapsed(self._primary_quota_since):
            logger.info("Primary quota cooldown elapsed; retrying the primary provider.")
            self.gemini.status = ProviderStatus.AVAILABLE
            self._primary_quota_since = None
        if self.gemini.status != ProviderStatus.QUOTA_EXHAUSTED:
            try:
                logger.info(f"Routing request to primary LLM: {self.gemini.name} ({self.gemini.model})")
                self.active_provider_name = self.gemini.name
                return self._execute_with_retry(self.gemini, method_name, *args, **kwargs)
            except (QuotaExceededError, RateLimitError, ProviderUnavailableError, InvalidResponseError,
                    ModelNotFoundError, BadRequestError) as gemini_err:
                if isinstance(gemini_err, QuotaExceededError):
                    self._primary_quota_since = time.monotonic()
                logger.warning(
                    f"Primary provider {self.gemini.name} failed: {gemini_err}. "
                    f"Initiating fallback to {self.fallback.name}."
                )
                usage_tracker.record_fallback(
                    from_provider=self.gemini.name,
                    to_provider=self.fallback.name,
                    reason=str(gemini_err),
                )
            except AuthenticationError as auth_err:
                logger.error(
                    f"Primary provider {self.gemini.name} auth failed: {auth_err}. "
                    f"Falling back to {self.fallback.name}."
                )
                usage_tracker.record_fallback(
                    from_provider=self.gemini.name,
                    to_provider=self.fallback.name,
                    reason=f"Auth error: {auth_err}",
                )
            except Exception as unexpected_err:
                # Any other primary failure (SDK bug, malformed object) still
                # deserves a fallback attempt rather than crashing the caller.
                logger.error(
                    f"Primary provider {self.gemini.name} raised unexpected "
                    f"{type(unexpected_err).__name__}: {unexpected_err}. Falling back."
                )
                usage_tracker.record_fallback(
                    from_provider=self.gemini.name,
                    to_provider=self.fallback.name,
                    reason=f"Unexpected error: {unexpected_err}",
                )

        # 2. Attempt Fallback Provider (Groq)
        try:
            logger.info(f"Routing request to fallback LLM: {self.fallback.name} ({self.fallback.model})")
            self.active_provider_name = self.fallback.name
            return self._execute_with_retry(self.fallback, method_name, *args, **kwargs)
        except Exception as fallback_err:
            logger.critical(
                f"Fallback provider {self.fallback.name} also failed: {fallback_err}. "
                "Entering DEGRADED MODE."
            )
            self.degraded_mode = True
            self._degraded_since = time.monotonic()
            usage_tracker.record_error("router", "DegradedModeActivated")
            raise DegradedModeError(
                f"Both Gemini and {self.fallback.name} failed. "
                f"Primary error was handled; fallback error was: {fallback_err}. "
                "AIRA has entered Degraded Mode.",
                provider="router",
            ) from fallback_err

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Route and execute an unstructured text generation request."""
        return self._route_request(
            "generate",
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Route and execute a structured JSON generation request validated by a Pydantic schema."""
        return self._route_request(
            "generate_structured",
            prompt,
            schema,
            system_prompt=system_prompt,
            temperature=temperature,
        )


# Global ModelRouter singleton
model_router = ModelRouter()

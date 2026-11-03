"""AIRA Centralized Model Router.

Single entry point for all LLM calls. Routes to Google Gemini primarily,
detects failures, handles bounded retries for transient errors, switches to Groq
on quota exhaustion or outages, and manages graceful Degraded Mode.
"""
import time
from typing import Any, Dict, List, Optional, Type
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
    parse_retry_after_seconds,
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
        # How long the current primary quota pause lasts (a provider's retry
        # hint can make it shorter than the full degraded cooldown).
        self._primary_quota_seconds: Optional[float] = None
        self.cooldown_seconds: float = float(settings.LLM_DEGRADED_COOLDOWN_SECONDS)
        # Per-provider "do not call before" monotonic deadlines, set after a
        # 429 / overload so we stop hammering a provider that just refused us.
        self._cooling_until: Dict[str, float] = {}

    def _cooldown_elapsed(self, since: Optional[float]) -> bool:
        return since is not None and (time.monotonic() - since) >= self.cooldown_seconds

    # ------------------------------------------------------------------
    # Per-provider cooldowns
    # ------------------------------------------------------------------
    def cooling_remaining(self, provider_name: str) -> float:
        """Seconds until ``provider_name`` may be called again (0 when it may)."""
        until = self._cooling_until.get(provider_name)
        if until is None:
            return 0.0
        remaining = until - time.monotonic()
        if remaining <= 0:
            self._cooling_until.pop(provider_name, None)
            return 0.0
        return remaining

    def _start_cooldown(self, provider_name: str, seconds: float, reason: str) -> None:
        seconds = max(0.0, float(seconds))
        if seconds <= 0:
            return
        until = time.monotonic() + seconds
        # Never shorten a cooldown that is already longer.
        self._cooling_until[provider_name] = max(until, self._cooling_until.get(provider_name, 0.0))
        usage_tracker.record_cooldown(provider_name, seconds, reason)
        logger.warning(f"{provider_name} paused for {seconds:.1f}s after {reason}; requests go to the other provider.")

    @staticmethod
    def _hinted_cooldown(error: Exception, default_seconds: float) -> float:
        """Cooldown for a refused request: the provider's own hint (capped) or a default."""
        hint = getattr(error, "retry_after", None)
        if hint is None:
            hint = parse_retry_after_seconds(error)
        cap = float(settings.LLM_RETRY_HINT_MAX_SECONDS)
        if hint is None or hint <= 0:
            return min(default_seconds, cap)
        return min(float(hint), cap)

    @staticmethod
    def _is_overload_or_timeout(error: Exception) -> bool:
        """503 'high demand' / overloaded, or 504 / deadline / timeout.

        Retrying these seconds later against the same provider mostly burns the
        caller's time (and, for timeouts, a whole extra request), so the router
        falls back instead.
        """
        msg = str(error).lower()
        return any(token in msg for token in (
            "503", "504", "overloaded", "high demand", "deadline", "timed out", "timeout",
        ))

    @staticmethod
    def _is_timeout(error: Exception) -> bool:
        msg = str(error).lower()
        return any(token in msg for token in ("504", "deadline", "timed out", "timeout"))

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
        self._primary_quota_seconds = None
        self._cooling_until.clear()
        logger.info("ModelRouter status reset: providers set to AVAILABLE.")

    def _execute_with_retry(
        self,
        provider: Any,
        method_name: str,
        *args: Any,
        max_retries: Optional[int] = None,
        deadline: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a provider call with bounded exponential backoff on transient errors.

        Permanent errors (AuthenticationError, QuotaExceededError) are NOT retried.
        Rate limits (429) and overload / timeout errors are not retried against
        the same provider either: the provider is put in a short cooldown (its
        own retry hint when it gives one) and the router falls back. Only
        other transient failures (connection reset, empty response) are retried.

        ``deadline`` is a ``time.monotonic()`` value; no retry starts after it.
        """
        last_exception: Optional[Exception] = None
        retries = self.max_retries if max_retries is None else max(0, int(max_retries))

        for attempt in range(retries + 1):
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
                usage_tracker.record_rate_limited(provider.name)
                if provider is not self.gemini:
                    # The primary's quota pause is tracked by the router itself;
                    # any other provider gets a plain cooldown.
                    self._start_cooldown(
                        provider.name,
                        self._hinted_cooldown(quota_err, self.cooldown_seconds)
                        if (getattr(quota_err, "retry_after", None) or parse_retry_after_seconds(quota_err))
                        else self.cooldown_seconds,
                        "quota exhaustion",
                    )
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

            except RateLimitError as rate_err:
                # 429: retrying the same provider a second later just earns another 429.
                last_exception = rate_err
                usage_tracker.record_error(provider.name, "RateLimitError")
                usage_tracker.record_rate_limited(provider.name)
                self._start_cooldown(
                    provider.name,
                    self._hinted_cooldown(rate_err, float(settings.LLM_RATE_LIMIT_COOLDOWN_SECONDS)),
                    "a rate limit (429)",
                )
                break

            except (ProviderUnavailableError, InvalidResponseError) as transient_err:
                last_exception = transient_err
                usage_tracker.record_error(provider.name, type(transient_err).__name__)

                if isinstance(transient_err, ProviderUnavailableError) and self._is_overload_or_timeout(transient_err):
                    if self._is_timeout(transient_err):
                        usage_tracker.record_timeout(provider.name)
                    else:
                        self._start_cooldown(
                            provider.name,
                            self._hinted_cooldown(transient_err, float(settings.LLM_OVERLOAD_COOLDOWN_SECONDS)),
                            "an overload response (503)",
                        )
                    logger.warning(
                        f"{provider.name} overloaded or timed out [{type(transient_err).__name__}]; "
                        "not retrying the same provider."
                    )
                    break

                backoff = min(self.initial_backoff_seconds * (2 ** attempt), 2.0)
                out_of_time = deadline is not None and time.monotonic() + backoff >= deadline
                if attempt < retries and not out_of_time:
                    logger.warning(
                        f"{provider.name} transient error [{type(transient_err).__name__}]. "
                        f"Retrying ({attempt + 1}/{retries}) after {backoff:.2f}s backoff..."
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"{provider.name} gave up after {attempt + 1} attempt(s)"
                        f"{' (time budget spent)' if out_of_time else ''}. Final error: {transient_err}"
                    )
                    break

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

    def _call_provider(
        self,
        provider: Any,
        method_name: str,
        args: tuple,
        kwargs: Dict[str, Any],
        max_retries: Optional[int],
        deadline: Optional[float],
    ) -> LLMResponse:
        call_kwargs = dict(kwargs)
        if deadline is not None:
            # The provider's own HTTP timeout is the remaining budget, so the
            # request ends when the caller stops waiting instead of running on
            # in a background thread.
            call_kwargs["timeout"] = max(1.0, deadline - time.monotonic())
        return self._execute_with_retry(
            provider, method_name, *args, max_retries=max_retries, deadline=deadline, **call_kwargs
        )

    def _primary_quota_elapsed(self) -> bool:
        if self._primary_quota_since is None:
            return False
        window = self._primary_quota_seconds if self._primary_quota_seconds is not None else self.cooldown_seconds
        return (time.monotonic() - self._primary_quota_since) >= window

    def _route_request(
        self,
        method_name: str,
        *args: Any,
        single_shot: bool = False,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Central routing loop: Try Gemini -> On eligible failure switch to Groq -> On failure enter Degraded Mode.

        ``single_shot`` disables same-provider retries. ``timeout`` is a total
        time budget in seconds for the whole request (both providers): each
        provider call gets the remaining budget as its HTTP timeout, and the
        fallback is skipped when too little budget is left.
        """
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

        deadline = time.monotonic() + float(timeout) if timeout else None
        max_retries = 0 if single_shot else None

        # 1. Attempt Primary Provider (Gemini) unless it is out of quota or cooling down
        if self.gemini.status == ProviderStatus.QUOTA_EXHAUSTED and self._primary_quota_elapsed():
            logger.info("Primary quota cooldown elapsed; retrying the primary provider.")
            self.gemini.status = ProviderStatus.AVAILABLE
            self._primary_quota_since = None
            self._primary_quota_seconds = None
        primary_wait = self.cooling_remaining(self.gemini.name)
        primary_skipped = self.gemini.status == ProviderStatus.QUOTA_EXHAUSTED or primary_wait > 0
        if primary_skipped:
            logger.info(
                f"Skipping primary LLM {self.gemini.name}: "
                + ("quota exhausted" if self.gemini.status == ProviderStatus.QUOTA_EXHAUSTED
                   else f"cooling down for {primary_wait:.0f}s more")
                + f"; using {self.fallback.name}."
            )
        else:
            try:
                logger.info(f"Routing request to primary LLM: {self.gemini.name} ({self.gemini.model})")
                self.active_provider_name = self.gemini.name
                return self._call_provider(self.gemini, method_name, args, kwargs, max_retries, deadline)
            except (QuotaExceededError, RateLimitError, ProviderUnavailableError, InvalidResponseError,
                    ModelNotFoundError, BadRequestError) as gemini_err:
                if isinstance(gemini_err, QuotaExceededError):
                    self.gemini.status = ProviderStatus.QUOTA_EXHAUSTED
                    self._primary_quota_since = time.monotonic()
                    hint = getattr(gemini_err, "retry_after", None) or parse_retry_after_seconds(gemini_err)
                    # A short hint means a per-minute limit; no hint (daily quota) means the full cooldown.
                    self._primary_quota_seconds = (
                        min(float(hint), float(settings.LLM_RETRY_HINT_MAX_SECONDS)) if hint else None
                    )
                    usage_tracker.record_cooldown(
                        self.gemini.name,
                        self._primary_quota_seconds or self.cooldown_seconds,
                        "quota exhausted",
                    )
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
        if deadline is not None and deadline - time.monotonic() < float(settings.LLM_MIN_FALLBACK_BUDGET_SECONDS):
            # The caller's budget is spent: starting another request now would
            # only produce an answer nobody waits for.
            raise ProviderUnavailableError(
                f"Time budget of {timeout:.0f}s spent before the fallback provider could be tried.",
                provider="router",
            )
        fallback_wait = self.cooling_remaining(self.fallback.name)
        if fallback_wait > 0:
            # Temporary: not Degraded Mode, the cooldowns expire on their own.
            logger.warning(
                f"Fallback provider {self.fallback.name} is cooling down for {fallback_wait:.0f}s more; "
                "no LLM provider can take this request right now."
            )
            raise DegradedModeError(
                f"LLM providers are temporarily paused after rate limits ({self.fallback.name} "
                f"resumes in about {fallback_wait:.0f}s). Reasoning steps fall back to rules meanwhile.",
                provider="router",
            )
        try:
            logger.info(f"Routing request to fallback LLM: {self.fallback.name} ({self.fallback.model})")
            self.active_provider_name = self.fallback.name
            return self._call_provider(self.fallback, method_name, args, kwargs, max_retries, deadline)
        except Exception as fallback_err:
            if deadline is not None:
                # A budgeted single request (tool proposal) failing is not
                # evidence that both providers are down for two minutes.
                raise
            if isinstance(fallback_err, RateLimitError) and self.cooling_remaining(
                self.fallback.name
            ) > 0:
                # The fallback is in a short, known cooldown; do not escalate
                # to the long Degraded Mode pause for a per-minute limit.
                raise DegradedModeError(
                    f"Both Gemini and {self.fallback.name} are rate limited; requests resume in about "
                    f"{self.cooling_remaining(self.fallback.name):.0f}s. Last error: {fallback_err}",
                    provider="router",
                ) from fallback_err
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

    def generate_tool_call(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        single_shot: bool = False,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Route a function-calling request; ``structured_data`` is the proposed ``ToolCall``.

        Same Groq fallback / cooldown handling as ``generate``. A response
        without a tool call, or with malformed arguments, is an
        ``InvalidResponseError``. Tool proposals pass ``single_shot=True`` and a
        ``timeout`` budget: one attempt per provider, each bounded by what is
        left of the budget, and no fallback once the budget is spent.
        """
        return self._route_request(
            "generate_tool_call",
            prompt,
            tools,
            system_prompt=system_prompt,
            temperature=temperature,
            single_shot=single_shot,
            timeout=timeout,
        )


# Global ModelRouter singleton
model_router = ModelRouter()

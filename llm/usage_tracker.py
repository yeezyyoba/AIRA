"""AIRA LLM Token & Quota Usage Tracker.

Tracks token consumption, call counts, latency, and provider fallback events.
Secrets and API keys are strictly excluded.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Dict, Iterator, List, Optional
from logging_config.logger import get_logger

logger = get_logger("usage_tracker")


@dataclass
class FallbackEvent:
    """Record of a model router fallback switch."""
    timestamp: str
    from_provider: str
    to_provider: str
    reason: str


@dataclass
class ProviderUsage:
    """Aggregated usage metrics for a specific LLM provider."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_seconds: float = 0.0


def _llm_bucket() -> Dict[str, Any]:
    return {
        "calls": 0, "successes": 0, "failures": 0, "rate_limited": 0, "timeouts": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "cooldowns": 0,
    }


def _api_bucket() -> Dict[str, int]:
    return {"requests": 0, "successes": 0, "failures": 0, "rate_limited": 0}


class SessionUsage:
    """External API usage of one research session (thread-safe).

    Counts real requests only: every number here corresponds to a request that
    was actually sent (or refused by a cooldown, which is counted separately).
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self._lock = threading.Lock()
        self.llm: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, int] = {"calls": 0, "successes": 0, "failures": 0, "rate_limited": 0, "texts": 0}
        self.academic: Dict[str, Dict[str, int]] = {}
        self.fallbacks = 0

    def _llm(self, provider: str) -> Dict[str, Any]:
        return self.llm.setdefault(provider, _llm_bucket())

    def add_llm_call(self, provider: str, success: bool, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            b = self._llm(provider)
            b["calls"] += 1
            b["successes" if success else "failures"] += 1
            b["prompt_tokens"] += prompt_tokens
            b["completion_tokens"] += completion_tokens

    def add_llm_event(self, provider: str, key: str) -> None:
        with self._lock:
            self._llm(provider)[key] += 1

    def add_fallback(self) -> None:
        with self._lock:
            self.fallbacks += 1

    def add_embedding_call(self, success: bool, texts: int, rate_limited: bool = False) -> None:
        with self._lock:
            self.embeddings["calls"] += 1
            self.embeddings["successes" if success else "failures"] += 1
            self.embeddings["texts"] += texts
            if rate_limited:
                self.embeddings["rate_limited"] += 1

    def add_academic_request(self, api: str, success: bool, rate_limited: bool = False) -> None:
        with self._lock:
            b = self.academic.setdefault(api, _api_bucket())
            b["requests"] += 1
            b["successes" if success else "failures"] += 1
            if rate_limited:
                b["rate_limited"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            llm = {name: dict(b) for name, b in self.llm.items()}
            return {
                "llm": llm,
                "llm_totals": {
                    key: sum(b[key] for b in llm.values())
                    for key in ("calls", "successes", "failures", "rate_limited", "timeouts")
                },
                "fallbacks": self.fallbacks,
                "embeddings": dict(self.embeddings),
                "academic": {name: dict(b) for name, b in self.academic.items()},
            }

    def summary_line(self) -> str:
        snap = self.snapshot()
        parts = []
        for name, b in sorted(snap["llm"].items()):
            parts.append(
                f"{name} {b['calls']} calls ({b['failures']} failed, {b['rate_limited']} rate-limited, "
                f"{b['timeouts']} timeouts)"
            )
        e = snap["embeddings"]
        parts.append(f"embeddings {e['calls']} calls ({e['failures']} failed, {e['rate_limited']} rate-limited)")
        for name, b in sorted(snap["academic"].items()):
            parts.append(f"{name} {b['requests']} requests ({b['rate_limited']} rate-limited)")
        return "API usage: " + "; ".join(parts)


_current_session_usage: ContextVar[Optional[SessionUsage]] = ContextVar("aira_session_usage", default=None)
_session_usage_registry: Dict[str, SessionUsage] = {}
_registry_lock = threading.Lock()


def current_session_usage() -> Optional[SessionUsage]:
    return _current_session_usage.get()


@contextmanager
def track_session_usage(session_id: str) -> Iterator[SessionUsage]:
    """Attribute every API call made inside this context (and its threads/tasks) to ``session_id``.

    ``asyncio.to_thread`` and new tasks copy the context, so calls made from
    worker threads are attributed too.
    """
    usage = get_session_usage(session_id) or SessionUsage(session_id)
    with _registry_lock:
        _session_usage_registry[session_id] = usage
        # Keep the registry bounded in long-running servers.
        while len(_session_usage_registry) > 500:
            _session_usage_registry.pop(next(iter(_session_usage_registry)))
    token = _current_session_usage.set(usage)
    try:
        yield usage
    finally:
        _current_session_usage.reset(token)


def get_session_usage(session_id: str) -> Optional[SessionUsage]:
    with _registry_lock:
        return _session_usage_registry.get(session_id)


class UsageTracker:
    """Thread-safe tracker for LLM token usage, request latency, and failover events."""

    def __init__(self):
        self._lock = threading.Lock()
        self._provider_metrics: Dict[str, ProviderUsage] = {}
        self._fallback_history: List[FallbackEvent] = []
        self._error_counts: Dict[str, int] = {}

    def _get_provider_metrics(self, provider: str) -> ProviderUsage:
        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = ProviderUsage()
        return self._provider_metrics[provider]

    def record_call(
        self,
        provider: str,
        model: str,
        latency: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
    ) -> None:
        """Record an LLM request execution and its token consumption."""
        with self._lock:
            metrics = self._get_provider_metrics(provider)
            metrics.total_calls += 1
            if success:
                metrics.successful_calls += 1
            else:
                metrics.failed_calls += 1
            metrics.prompt_tokens += prompt_tokens
            metrics.completion_tokens += completion_tokens
            metrics.total_tokens += (prompt_tokens + completion_tokens)
            metrics.total_latency_seconds += latency
        session = current_session_usage()
        if session is not None:
            session.add_llm_call(provider, success, prompt_tokens, completion_tokens)

        logger.info(
            f"LLM Call Recorded | Provider: {provider} | Model: {model} | "
            f"Success: {success} | Latency: {latency:.2f}s | Tokens: {prompt_tokens + completion_tokens}"
        )

    def record_fallback(self, from_provider: str, to_provider: str, reason: str) -> None:
        """Record a failover event from one provider to another."""
        timestamp = datetime.now(timezone.utc).isoformat()
        event = FallbackEvent(
            timestamp=timestamp,
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
        )
        with self._lock:
            self._fallback_history.append(event)
        session = current_session_usage()
        if session is not None:
            session.add_fallback()

        logger.warning(
            f"Provider Failover Triggered: {from_provider} -> {to_provider} | Reason: {reason}"
        )

    def record_error(self, provider: str, error_type: str) -> None:
        """Record error occurrence by provider and category."""
        key = f"{provider}:{error_type}"
        with self._lock:
            self._error_counts[key] = self._error_counts.get(key, 0) + 1

    def record_rate_limited(self, provider: str) -> None:
        """A provider answered 429 / quota exhausted."""
        session = current_session_usage()
        if session is not None:
            session.add_llm_event(provider, "rate_limited")

    def record_timeout(self, provider: str) -> None:
        """A provider request timed out (504 / deadline / client timeout)."""
        self.record_error(provider, "Timeout")
        session = current_session_usage()
        if session is not None:
            session.add_llm_event(provider, "timeouts")

    def record_cooldown(self, provider: str, seconds: float, reason: str) -> None:
        """The router paused ``provider`` for ``seconds``."""
        self.record_error(provider, "Cooldown")
        session = current_session_usage()
        if session is not None:
            session.add_llm_event(provider, "cooldowns")

    def get_fallback_history(self) -> List[Dict[str, Any]]:
        """Return list of recorded fallback events."""
        with self._lock:
            return [
                {
                    "timestamp": e.timestamp,
                    "from_provider": e.from_provider,
                    "to_provider": e.to_provider,
                    "reason": e.reason,
                }
                for e in self._fallback_history
            ]

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated summary of token usage and request counts."""
        with self._lock:
            providers_summary = {}
            grand_total_tokens = 0
            grand_total_calls = 0

            for p_name, usage in self._provider_metrics.items():
                avg_latency = (
                    usage.total_latency_seconds / usage.total_calls
                    if usage.total_calls > 0
                    else 0.0
                )
                providers_summary[p_name] = {
                    "total_calls": usage.total_calls,
                    "successful_calls": usage.successful_calls,
                    "failed_calls": usage.failed_calls,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "average_latency_seconds": round(avg_latency, 3),
                }
                grand_total_tokens += usage.total_tokens
                grand_total_calls += usage.total_calls

            return {
                "grand_total_calls": grand_total_calls,
                "grand_total_tokens": grand_total_tokens,
                "total_fallback_events": len(self._fallback_history),
                "providers": providers_summary,
                "error_counts": dict(self._error_counts),
            }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._provider_metrics.clear()
            self._fallback_history.clear()
            self._error_counts.clear()


# Global usage tracker instance
usage_tracker = UsageTracker()

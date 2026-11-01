"""AIRA LLM Token & Quota Usage Tracker.

Tracks token consumption, call counts, latency, and provider fallback events.
Secrets and API keys are strictly excluded.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional
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

        logger.warning(
            f"Provider Failover Triggered: {from_provider} -> {to_provider} | Reason: {reason}"
        )

    def record_error(self, provider: str, error_type: str) -> None:
        """Record error occurrence by provider and category."""
        key = f"{provider}:{error_type}"
        with self._lock:
            self._error_counts[key] = self._error_counts.get(key, 0) + 1

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

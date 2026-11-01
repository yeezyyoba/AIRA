#!/usr/bin/env python3
"""Manual live test script for AIRA Phase 2 Groq fallback mechanism.

Usage:
    python scripts/manual_test_groq_fallback.py

Verifies:
1. Controlled simulation of a Gemini quota exhaustion (zero real Gemini API calls).
2. Production ModelRouter catching the quota error and triggering automatic failover.
3. Genuine, live API call to Groq using the configured GROQ_API_KEY.
4. Validation of the live response and usage tracker failover history.
5. Strict protection of secrets and credentials.
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

# Load configuration from .env
load_dotenv(ROOT_DIR / ".env")

from app.core.config import settings
from llm import (
    AuthenticationError,
    DegradedModeError,
    GeminiProvider,
    GroqProvider,
    ModelRouter,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    usage_tracker,
)
from logging_config.logger import setup_logging

setup_logging()


class SimulatedQuotaExhaustedGemini(GeminiProvider):
    """Controlled test double that simulates Gemini quota exhaustion.

    Guarantees that no real Gemini API requests are made and no actual quota is consumed.
    """

    def generate(self, *args, **kwargs):
        raise QuotaExceededError(
            "SIMULATED: 429 ResourceExhausted: You have exceeded your current quota for gemini-3.8-flash.",
            provider=self.name,
        )

    def generate_structured(self, *args, **kwargs):
        raise QuotaExceededError(
            "SIMULATED: 429 ResourceExhausted: You have exceeded your current quota for gemini-3.8-flash.",
            provider=self.name,
        )


def run_groq_fallback_test() -> None:
    print("\n" + "=" * 70)
    print("AIRA Phase 2: Live Groq Fallback & Failover Verification Test")
    print("=" * 70)

    # 1. Environment & configuration inspection (without printing secrets)
    raw_key = (settings.GROQ_API_KEY or "").strip()
    groq_key_present = bool(raw_key)

    print(f"Fallback Provider Configured : {settings.FALLBACK_PROVIDER.capitalize()}")
    print(f"Configured Groq Model        : {settings.GROQ_MODEL}")
    print(f"GROQ_API_KEY Status          : {'CONFIGURED ✅' if groq_key_present else 'MISSING ❌'}")
    print("-" * 70)

    if not groq_key_present:
        print("❌ ERROR: GROQ_API_KEY is not configured in your environment or .env file.")
        print("   Please add GROQ_API_KEY=your_key to .env and run this script again.\n")
        sys.exit(1)

    # 2. Reset telemetry for clean test reporting
    usage_tracker.reset()

    # 3. Initialize production ModelRouter with controlled Gemini test double and genuine Groq provider
    print("\nStep 1: Initializing production ModelRouter...")
    print("        - Primary Provider : Controlled Gemini Double (Injects 429 Quota Error)")
    print(f"        - Fallback Provider: Genuine Production GroqProvider ({settings.GROQ_MODEL})")

    simulated_gemini = SimulatedQuotaExhaustedGemini()
    genuine_groq = GroqProvider(api_key=raw_key, model=settings.GROQ_MODEL)

    router = ModelRouter(
        gemini_provider=simulated_gemini,
        fallback_provider=genuine_groq,
    )

    # 5. Define prompt
    prompt = "Explain what machine learning is in two sentences."
    print(f"\nStep 2: Dispatching request with prompt:")
    print(f"        \"{prompt}\"")
    print("-" * 70)
    print("Routing initiated: Gemini will fail with simulated quota error -> Groq will take over...")

    # 6. Execute request through production router
    try:
        response = router.generate(
            prompt=prompt,
            temperature=0.3,
        )

        print("\n" + "=" * 70)
        print("EXECUTION STATUS: SUCCESSFUL FAILOVER & COMPLETION ✅")
        print("=" * 70)
        print(f"Selected Provider : {response.provider} (Expected: groq)")
        print(f"Model Employed    : {response.model}")
        print(f"Execution Latency : {response.latency_seconds:.2f} seconds")
        print(
            f"Token Consumption : {response.total_tokens} total tokens "
            f"({response.prompt_tokens} prompt, {response.completion_tokens} completion)"
        )
        print("-" * 70)
        print("Genuine Groq Response Content:")
        print(response.content.strip())
        print("=" * 70)

        # 7. Verify Telemetry & Failover Records
        print("\nStep 3: Verifying UsageTracker Telemetry & Fallback History:")
        fallback_events = usage_tracker.get_fallback_history()
        summary = usage_tracker.get_summary()

        if fallback_events:
            event = fallback_events[0]
            print(f"  [✓] Failover Event Recorded : {event['from_provider']} -> {event['to_provider']}")
            print(f"  [✓] Recorded Failover Reason: {event['reason']}")
        else:
            print("  [!] WARNING: No fallback event found in usage tracker.")

        print(f"  [✓] Groq Successful Calls   : {summary['providers'].get('groq', {}).get('successful_calls', 0)}")
        print(f"  [✓] Groq Total Tokens Logged: {summary['providers'].get('groq', {}).get('total_tokens', 0)}")
        print(f"  [✓] Total Failover Events   : {summary['total_fallback_events']}")
        print("=" * 70 + "\n")

    except AuthenticationError as e:
        print("\n" + "=" * 70)
        print("AUTHENTICATION ERROR ON GROQ ❌")
        print("=" * 70)
        print(f"Provider : {e.provider}")
        print(f"Details  : {e}")
        print("Advice   : Please verify that your GROQ_API_KEY in .env is valid and active.")
        print("=" * 70 + "\n")

    except QuotaExceededError as e:
        print("\n" + "=" * 70)
        print("GROQ QUOTA ERROR ⚠️")
        print("=" * 70)
        print(f"Provider : {e.provider}")
        print(f"Details  : {e}")
        print("=" * 70 + "\n")

    except RateLimitError as e:
        print("\n" + "=" * 70)
        print("GROQ RATE LIMITED ⚠️")
        print("=" * 70)
        print(f"Provider : {e.provider}")
        print(f"Details  : {e}")
        print("=" * 70 + "\n")

    except ProviderUnavailableError as e:
        print("\n" + "=" * 70)
        print("GROQ SERVICE UNAVAILABLE ⚠️")
        print("=" * 70)
        print(f"Provider : {e.provider}")
        print(f"Details  : {e}")
        print("=" * 70 + "\n")

    except DegradedModeError as e:
        print("\n" + "=" * 70)
        print("DEGRADED MODE ACTIVATED 🛡️")
        print("=" * 70)
        print(f"Details  : {e}")
        print("=" * 70 + "\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("UNEXPECTED ERROR ❌")
        print("=" * 70)
        print(f"Error Type: {type(e).__name__}")
        print(f"Details   : {e}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    run_groq_fallback_test()

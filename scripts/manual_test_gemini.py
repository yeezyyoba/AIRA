#!/usr/bin/env python3
"""Manual live test script for Phase 2 Gemini LLM integration.

Usage:
    python scripts/manual_test_gemini.py

Sends a live prompt to Gemini using the existing ModelRouter implementation,
verifying real-world API connectivity, response extraction, and error handling.
Secrets and API keys are strictly protected and never printed.
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

# Load credentials from .env
load_dotenv(ROOT_DIR / ".env")

from app.core.config import settings
from llm import (
    AuthenticationError,
    DegradedModeError,
    InvalidResponseError,
    ModelRouter,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from logging_config.logger import setup_logging

setup_logging()


def run_live_test() -> None:
    print("\n" + "=" * 65)
    print("AIRA Phase 2: Live Gemini API Manual Test")
    print("=" * 65)

    # 1. Inspect configuration (without exposing secrets)
    gemini_key_present = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip())
    groq_key_present = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())

    print(f"Configured Primary Model : {settings.GEMINI_MODEL}")
    print(f"Gemini API Key Detected  : {'YES (Configured)' if gemini_key_present else 'NO (Missing from .env)'}")
    print(f"Fallback Provider/Model  : {settings.FALLBACK_PROVIDER} ({settings.GROQ_MODEL})")
    print(f"Groq API Key Detected    : {'YES (Configured)' if groq_key_present else 'NO (Missing from .env)'}")
    print("-" * 65)

    if not gemini_key_present:
        print("⚠️  NOTICE: GEMINI_API_KEY is not set in your .env file.")
        print("    To run a genuine live test:")
        print("    1. Create or open the .env file in the project root.")
        print("    2. Add: GEMINI_API_KEY=your_actual_google_api_key")
        print("    3. Re-run: python scripts/manual_test_gemini.py\n")

    # 2. Initialize ModelRouter using existing production implementation
    print("Initializing ModelRouter...")
    router = ModelRouter()

    # 3. Define prompt
    prompt = "Explain what an API is in two sentences."
    print(f"Sending prompt           : \"{prompt}\"")
    print("Waiting for response from Gemini...\n")

    # 4. Execute request and catch standardized errors
    try:
        response = router.generate(
            prompt=prompt,
            temperature=0.3,
        )

        print("=" * 65)
        print("STATUS: SUCCESS ✅")
        print("=" * 65)
        print(f"Selected Provider : {response.provider}")
        print(f"Model Used        : {response.model}")
        print(f"Latency           : {response.latency_seconds:.2f} seconds")
        print(f"Token Usage       : {response.total_tokens} total tokens "
              f"({response.prompt_tokens} prompt, {response.completion_tokens} completion)")
        print("-" * 65)
        print("Returned Response :")
        print(response.content.strip())
        print("=" * 65 + "\n")

    except AuthenticationError as e:
        print("=" * 65)
        print("STATUS: AUTHENTICATION ERROR ❌")
        print("=" * 65)
        print(f"Provider : {e.provider}")
        print(f"Details  : {e}")
        print("Advice   : Please verify that your GEMINI_API_KEY in .env is valid and active.")
        print("=" * 65 + "\n")

    except QuotaExceededError as e:
        print("=" * 65)
        print("STATUS: QUOTA EXHAUSTED ⚠️")
        print("=" * 65)
        print(f"Provider : {e.provider}")
        print(f"Details  : {e}")
        print("Advice   : Your Gemini quota has been reached for this model/tier.")
        print("=" * 65 + "\n")

    except RateLimitError as e:
        print("=" * 65)
        print("STATUS: RATE LIMITED ⚠️")
        print("=" * 65)
        print(f"Provider : {e.provider}")
        print(f"Details  : {e}")
        print("Advice   : Too many requests sent per minute. Please try again in a few moments.")
        print("=" * 65 + "\n")

    except ProviderUnavailableError as e:
        print("=" * 65)
        print("STATUS: NETWORK / SERVICE UNAVAILABLE ⚠️")
        print("=" * 65)
        print(f"Provider : {e.provider}")
        print(f"Details  : {e}")
        print("Advice   : Check your internet connection or Google API service status.")
        print("=" * 65 + "\n")

    except DegradedModeError as e:
        print("=" * 65)
        print("STATUS: DEGRADED MODE ACTIVATED 🛡️")
        print("=" * 65)
        print(f"Details  : {e}")
        print("=" * 65 + "\n")

    except Exception as e:
        print("=" * 65)
        print("STATUS: UNEXPECTED ERROR ❌")
        print("=" * 65)
        print(f"Error Type: {type(e).__name__}")
        print(f"Details   : {e}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    run_live_test()

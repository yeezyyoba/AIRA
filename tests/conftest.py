"""Shared pytest configuration: isolation from real services and real data.

This module is imported by pytest before any test module, so the environment
variables below take effect before ``app.core.config.settings`` is created.
Environment variables take precedence over values in ``.env``.

Guarantees for every test run:
* No LLM provider client is constructed (API keys are blanked), so no test can
  call Gemini or Groq, regardless of what the developer's ``.env`` contains.
* Embeddings use the deterministic mock provider.
* All data directories (processed papers, Chroma, reports) live in a
  throwaway temporary directory, so tests never pollute ``data/``.
* Outbound network connections are refused. Tests must mock external APIs.
  Loopback and Unix sockets stay allowed for in-process servers.
"""
import ipaddress
import os
import socket
import tempfile

import pytest

_TEST_ROOT = tempfile.mkdtemp(prefix="aira_tests_")

os.environ.update({
    "GEMINI_API_KEY": "",
    "GROQ_API_KEY": "",
    "OPENALEX_API_KEY": "",
    "SEMANTIC_SCHOLAR_API_KEY": "",
    "LANGSMITH_TRACING": "false",
    "ANONYMIZED_TELEMETRY": "False",
    "EMBEDDING_PROVIDER": "mock",
    "ENVIRONMENT": "test",
    "PROCESSED_DIRECTORY": os.path.join(_TEST_ROOT, "processed"),
    "PAPERS_DIRECTORY": os.path.join(_TEST_ROOT, "papers"),
    "REPORTS_DIRECTORY": os.path.join(_TEST_ROOT, "reports"),
    "CHROMA_PERSIST_DIRECTORY": os.path.join(_TEST_ROOT, "chroma"),
    "DATABASE_URL": f"sqlite:///{os.path.join(_TEST_ROOT, 'aira.db')}",
    # Tests must not depend on the developer's .env tuning.
    "MAX_AGENT_STEPS": "150",
    "MAX_RETRIES": "2",
    "MAX_SEARCH_RESULTS": "20",
    "MAX_PAPERS_TO_EXTRACT": "5",
    "MAX_PLAN_REVISIONS": "1",
    # Keep retry backoff short so failure-path tests stay fast.
    "ACADEMIC_INITIAL_BACKOFF_SECONDS": "0.01",
})


class NetworkAccessBlocked(RuntimeError):
    """Raised when a test attempts a real outbound network connection."""


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _is_local(address) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX paths and other non-IP addresses
    host = address[0]
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guarded_connect(self, address):
    if not _is_local(address):
        raise NetworkAccessBlocked(f"Test attempted a real network connection to {address!r}.")
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if not _is_local(address):
        raise NetworkAccessBlocked(f"Test attempted a real network connection to {address!r}.")
    return _real_connect_ex(self, address)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def offline_llm_router():
    """A deterministic stand-in for the ModelRouter (no provider calls)."""
    from unittest.mock import MagicMock

    from llm.gemini_provider import LLMResponse

    router = MagicMock()
    router.is_degraded_mode.return_value = False
    router.generate = MagicMock(return_value=LLMResponse(
        content=(
            "The papers adopt different architectures; the comparison table records each "
            "paper's reported values without imputation."
        ),
        model="offline-test-model",
        provider="offline",
        latency_seconds=0.0,
    ))
    return router

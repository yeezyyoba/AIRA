"""Per-session live event log.

Research runs take minutes, so the frontend streams progress while the graph
executes. Events come from two places:

* graph progress (node started/finished, the node's own observable log
  entries), published explicitly by the research endpoint, and
* ordinary ``aira.*`` logger output (tool calls, LLM fallbacks, PDF
  extraction...), captured by :class:`SessionLogHandler` and attributed to the
  session whose context emitted it.

Everything lives in memory and is bounded; it is a progress feed, not an
audit trail (the graph state keeps ``observable_logs`` for that).
"""
from __future__ import annotations

import contextvars
import logging
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Set while a research graph runs so log records can be tied to their session.
current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "aira_current_session_id", default=None
)

_MAX_EVENTS_PER_SESSION = 2000
_MAX_SESSIONS = 50
_MAX_MESSAGE_CHARS = 600


class _SessionLog:
    __slots__ = ("events", "next_seq", "running")

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.next_seq = 1
        self.running = False


class SessionEventBus:
    """Thread-safe, bounded, in-memory event buffer keyed by session id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, _SessionLog]" = OrderedDict()

    def _get(self, session_id: str, create: bool = False) -> Optional[_SessionLog]:
        log = self._sessions.get(session_id)
        if log is None and create:
            log = _SessionLog()
            self._sessions[session_id] = log
            # Evict the oldest finished sessions beyond the cap.
            while len(self._sessions) > _MAX_SESSIONS:
                victim = next((sid for sid, s in self._sessions.items() if not s.running), None)
                if victim is None:
                    break
                del self._sessions[victim]
        return log

    def start(self, session_id: str) -> None:
        with self._lock:
            log = self._get(session_id, create=True)
            log.running = True
            self._sessions.move_to_end(session_id)

    def finish(self, session_id: str) -> None:
        with self._lock:
            log = self._get(session_id)
            if log:
                log.running = False

    def is_running(self, session_id: str) -> bool:
        with self._lock:
            log = self._get(session_id)
            return bool(log and log.running)

    def running_sessions(self) -> List[str]:
        with self._lock:
            return [sid for sid, s in self._sessions.items() if s.running]

    def publish(
        self,
        session_id: str,
        message: str,
        *,
        kind: str = "log",
        level: str = "INFO",
        source: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not session_id:
            return
        text = str(message)
        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[: _MAX_MESSAGE_CHARS - 1] + "…"
        with self._lock:
            log = self._get(session_id, create=True)
            event = {
                "seq": log.next_seq,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "level": level,
                "source": source,
                "message": text,
            }
            if data:
                event["data"] = data
            log.next_seq += 1
            log.events.append(event)
            if len(log.events) > _MAX_EVENTS_PER_SESSION:
                del log.events[: len(log.events) - _MAX_EVENTS_PER_SESSION]

    def events_since(self, session_id: str, after: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            log = self._get(session_id)
            if not log:
                return []
            return [e for e in log.events if e["seq"] > after]


event_bus = SessionEventBus()


class SessionLogHandler(logging.Handler):
    """Forward ``aira.*`` log records to the live log of the session that emitted them."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            session_id = current_session_id.get()
            if session_id is None:
                # Work handed to plain thread pools loses the context variable.
                # If exactly one run is active the record can only belong to it.
                running = event_bus.running_sessions()
                if len(running) != 1 or record.name.startswith(("aira.api", "aira.app")):
                    return
                session_id = running[0]
            source = record.name[5:] if record.name.startswith("aira.") else record.name
            event_bus.publish(
                session_id,
                record.getMessage(),
                kind="log",
                level=record.levelname,
                source=source,
            )
        except Exception:  # never let the live log break real logging
            self.handleError(record)


_handler_installed = False


def install_session_log_handler(level: int = logging.INFO) -> None:
    """Attach the session handler to the ``aira`` logger once."""
    global _handler_installed
    if _handler_installed:
        return
    handler = SessionLogHandler(level=level)
    logging.getLogger("aira").addHandler(handler)
    _handler_installed = True

"""Tool 6: Research File Manager (Category: System & Environment).

Saves, lists, and loads generated research reports inside the configured
reports directory. Every path is derived from a sanitized identifier and
re-checked to resolve inside the reports directory, so identifiers coming
from LLM output, API clients, or paper metadata cannot escape it.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional

from app.core.config import settings
from logging_config.logger import get_logger

logger = get_logger("file_manager")

ALLOWED_FORMATS = ("json", "md")
_MAX_ID_LENGTH = 120


class FileManagerError(RuntimeError):
    """Raised when a report cannot be safely written or read."""


def _reports_dir(base_dir: Optional[str] = None) -> Path:
    directory = Path(base_dir or settings.REPORTS_DIRECTORY).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sanitize_identifier(identifier: str) -> str:
    """Turn an arbitrary identifier into a safe single path component."""
    if not identifier or not isinstance(identifier, str):
        raise FileManagerError("Report identifier must be a non-empty string.")
    clean = re.sub(r"[^\w\-]", "_", identifier.strip())[:_MAX_ID_LENGTH]
    clean = clean.strip("_") or "report"
    return clean


def _safe_path(directory: Path, identifier: str, extension: str) -> Path:
    if extension not in ALLOWED_FORMATS:
        raise FileManagerError(f"Unsupported report format '{extension}'.")
    candidate = (directory / f"{sanitize_identifier(identifier)}.{extension}").resolve()
    if candidate.parent != directory:
        raise FileManagerError("Resolved report path escapes the reports directory.")
    return candidate


def _atomic_write(path: Path, content: str) -> None:
    """Write via a temporary file so a crash never leaves a truncated report."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def save_report(
    report: Dict[str, Any],
    markdown: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Persist a report as JSON (always) and Markdown (when provided).

    Returns a mapping of format -> absolute file path.
    """
    if not isinstance(report, dict):
        raise FileManagerError("Report must be a dictionary.")
    report_id = report.get("report_id")
    if not report_id:
        raise FileManagerError("Report has no report_id; refusing to save an unidentifiable file.")

    directory = _reports_dir(base_dir)
    paths: Dict[str, str] = {}

    json_path = _safe_path(directory, str(report_id), "json")
    payload = dict(report)
    payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    _atomic_write(json_path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    paths["json"] = str(json_path)

    if markdown:
        md_path = _safe_path(directory, str(report_id), "md")
        _atomic_write(md_path, markdown)
        paths["md"] = str(md_path)

    logger.info("Saved report '%s' (%s).", report_id, ", ".join(paths))
    return paths


def load_report(report_id: str, base_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a saved report's JSON, or None if it does not exist."""
    directory = _reports_dir(base_dir)
    path = _safe_path(directory, report_id, "json")
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Saved report '%s' could not be read: %s", report_id, exc)
        return None
    return data if isinstance(data, dict) else None


def list_saved_reports(base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return lightweight summaries of every saved report, newest first."""
    directory = _reports_dir(base_dir)
    summaries: List[Dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable report file %s: %s", path.name, exc)
            continue
        if not isinstance(data, dict) or not data.get("report_id"):
            continue
        summaries.append({
            "report_id": data.get("report_id"),
            "research_question": data.get("research_question"),
            "papers_analyzed": data.get("papers_analyzed", []),
            "created_at": data.get("created_at"),
            "is_degraded": bool(data.get("is_degraded", False)),
            "report_status": data.get("report_status", "COMPLETE"),
        })
    summaries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return summaries

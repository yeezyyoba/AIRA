"""Extracted Academic Paper Repository.

Persists and retrieves structured paper extractions from local disk storage (data/processed/),
enabling indexing and semantic retrieval without re-reading PDFs.
Guards against path traversal and verifies content integrity.
"""
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional

from app.core.config import settings
from logging_config.logger import get_logger
from validation.schemas import PaperExtractionResultSchema

logger = get_logger("paper_repository")


def _sanitize_paper_id(paper_id: str) -> str:
    """Sanitize paper identifier to prevent directory traversal attacks."""
    if not paper_id or not isinstance(paper_id, str):
        raise ValueError("paper_id must be a non-empty string")

    # Take filename component only and replace path separators and special characters
    clean = Path(paper_id).name
    clean = re.sub(r"[^\w\.\-]", "_", clean)
    if not clean or clean.startswith("."):
        clean = f"paper_{clean.lstrip('.')}"
    return clean


class PaperRepository:
    """Local repository for managing extracted academic papers."""

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        self.storage_dir = Path(storage_dir or settings.PROCESSED_DIRECTORY)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, PaperExtractionResultSchema] = {}

    def _get_file_path(self, paper_id: str) -> Path:
        clean_id = _sanitize_paper_id(paper_id)
        return self.storage_dir / f"{clean_id}.json"

    def save_paper(self, result: PaperExtractionResultSchema) -> Path:
        """Persist extracted paper result to disk and in-memory cache."""
        clean_id = _sanitize_paper_id(result.paper_id)
        file_path = self._get_file_path(clean_id)

        try:
            data = result.model_dump()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self._memory_cache[clean_id] = result
            logger.info(f"Saved extracted paper '{clean_id}' to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to persist extracted paper '{clean_id}': {e}")
            raise IOError(f"Failed to save extracted paper '{clean_id}': {e}") from e

    def get_paper(self, paper_id: str) -> Optional[PaperExtractionResultSchema]:
        """Retrieve an extracted paper by ID from memory cache or disk."""
        clean_id = _sanitize_paper_id(paper_id)

        # Check memory cache first
        if clean_id in self._memory_cache:
            return self._memory_cache[clean_id]

        file_path = self._get_file_path(clean_id)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            result = PaperExtractionResultSchema(**data)
            self._memory_cache[clean_id] = result
            return result
        except Exception as e:
            logger.error(f"Failed to load extracted paper '{clean_id}' from {file_path}: {e}")
            return None

    def has_paper(self, paper_id: str) -> bool:
        """Check if an extracted paper exists in the repository."""
        clean_id = _sanitize_paper_id(paper_id)
        if clean_id in self._memory_cache:
            return True
        return self._get_file_path(clean_id).exists()

    def list_papers(self) -> List[str]:
        """List all paper IDs stored in the repository."""
        paper_ids = set(self._memory_cache.keys())
        for f in self.storage_dir.glob("*.json"):
            paper_ids.add(f.stem)
        return sorted(list(paper_ids))

    def delete_paper(self, paper_id: str) -> bool:
        """Delete an extracted paper from storage and cache."""
        clean_id = _sanitize_paper_id(paper_id)
        self._memory_cache.pop(clean_id, None)

        file_path = self._get_file_path(clean_id)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted extracted paper file {file_path}")
            return True
        return False


# Global default paper repository instance
paper_repository = PaperRepository()

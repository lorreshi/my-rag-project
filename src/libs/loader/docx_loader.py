"""Docx Loader — converts .docx to canonical Markdown using MarkItDown.

Responsibilities:
- Parse .docx → Markdown text via MarkItDown.
- MVP does NOT extract embedded images (degrades to no images).
- On MarkItDown failure: log a warning and degrade gracefully — return a
  Document with empty text (best-effort) and complete metadata, never raising
  a fatal error (mirrors Property 7: degradation does not block ingestion).
- Populate metadata: source_path, doc_type, doc_hash, images (empty list).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class DocxLoader(BaseLoader):
    """Load .docx files using MarkItDown → canonical Markdown.

    The MarkItDown instance can be injected for testing (mocking); by default a
    real ``MarkItDown()`` is created.
    """

    def __init__(self, markitdown: Any = None):
        self._markitdown = markitdown if markitdown is not None else MarkItDown()

    def load(self, path: str) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Docx file not found: {path}")
        if p.suffix.lower() != ".docx":
            raise ValueError(f"DocxLoader only supports .docx files, got: {p.suffix}")

        # Compute document hash for unique ID (over file bytes).
        doc_hash = self._compute_hash(p)

        # Parse .docx → Markdown. Degrade gracefully on failure (no fatal error).
        text = ""
        try:
            result = self._markitdown.convert(str(p))
            text = result.text_content or ""
        except Exception as exc:
            logger.warning(
                "MarkItDown conversion failed for %s: %s — degrading to empty text",
                path, exc,
            )
            text = ""

        # MVP: docx embedded images are not extracted — degrade to no images.
        metadata: dict[str, Any] = {
            "source_path": str(p),
            "doc_type": "docx",
            "doc_hash": doc_hash,
            "images": [],
        }

        doc_id = f"docx_{doc_hash[:12]}"
        return Document(id=doc_id, text=text, metadata=metadata)

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA256 hash of the file content."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @property
    def supported_extensions(self) -> list[str]:
        return [".docx"]

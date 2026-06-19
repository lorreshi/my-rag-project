"""Xlsx Loader — converts .xlsx to canonical Markdown tables using MarkItDown.

Responsibilities:
- Parse .xlsx → Markdown text via MarkItDown. MarkItDown renders each worksheet
  as a separate Markdown table, prefixed with a ``## {sheet_name}`` H2 heading
  (sheet order preserved). Downstream ``TableAwareSplitter`` (T9) relies on the
  ``## {sheet_name}`` markers to attribute each table block to its sheet, so the
  loader guarantees those markers are present and normalized to H2.
- MVP does NOT extract embedded images (degrades to no images).
- On MarkItDown failure: log a warning and degrade gracefully — return a
  Document with empty text (best-effort) and complete metadata, never raising
  a fatal error (mirrors Property 7: degradation does not block ingestion).
- Populate metadata: source_path, doc_type, doc_hash, images (empty list).
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Matches a sheet heading rendered by MarkItDown at any heading level
# (e.g. "# Sheet1", "### Sheet1") so it can be normalized to H2 ("## Sheet1").
_SHEET_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(?P<name>.+?)[ \t]*$", re.MULTILINE)


class XlsxLoader(BaseLoader):
    """Load .xlsx files using MarkItDown → canonical Markdown tables.

    Each worksheet is emitted as a Markdown table preceded by a
    ``## {sheet_name}`` H2 heading, in stable workbook order. The MarkItDown
    instance can be injected for testing (mocking); by default a real
    ``MarkItDown()`` is created.
    """

    def __init__(self, markitdown: Any = None):
        self._markitdown = markitdown if markitdown is not None else MarkItDown()

    def load(self, path: str) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Xlsx file not found: {path}")
        if p.suffix.lower() != ".xlsx":
            raise ValueError(f"XlsxLoader only supports .xlsx files, got: {p.suffix}")

        # Compute document hash for unique ID (over file bytes).
        doc_hash = self._compute_hash(p)

        # Parse .xlsx → Markdown tables. Degrade gracefully on failure
        # (e.g. missing optional deps), never raising a fatal error.
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

        # Guarantee sheet markers are H2 so TableAwareSplitter can attribute
        # each table block to its sheet via "## {sheet_name}".
        text = self._normalize_sheet_headings(text)

        # MVP: xlsx has no extracted images — degrade to no images.
        metadata: dict[str, Any] = {
            "source_path": str(p),
            "doc_type": "xlsx",
            "doc_hash": doc_hash,
            "images": [],
        }

        doc_id = f"xlsx_{doc_hash[:12]}"
        return Document(id=doc_id, text=text, metadata=metadata)

    @staticmethod
    def _normalize_sheet_headings(text: str) -> str:
        """Normalize any sheet heading to an H2 ``## {sheet_name}`` marker.

        MarkItDown already prefixes each sheet with ``## {sheet_name}``; this
        keeps that contract robust if a heading is emitted at a different level,
        ensuring the downstream splitter consistently sees H2 sheet markers.
        Markdown table separator rows (e.g. ``| --- |``) are left untouched —
        they never match a single heading token.
        """
        if not text:
            return text

        def _to_h2(match: re.Match[str]) -> str:
            return f"## {match.group('name').strip()}"

        return _SHEET_HEADING_RE.sub(_to_h2, text)

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
        return [".xlsx"]

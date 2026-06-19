"""Markdown Loader — reads Markdown files directly into a Document.

Responsibilities:
- Read the raw Markdown text as-is (UTF-8). The file is already Markdown, so it
  is NOT passed through MarkItDown.
- Parse inline image links (``![alt](path)``) into ImageRef entries. Local image
  files that exist are included in ``metadata.images``; broken/missing links are
  logged as warnings and skipped (graceful degradation, never blocking).
- Populate metadata: source_path, doc_type, doc_hash, images.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from src.core.types import Document, ImageRef
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Matches Markdown inline image syntax: ![alt](path)
# Captures the alt text and the link target separately. The link target stops
# at whitespace (to ignore optional "title") or the closing paren.
_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)(?:\s+[^)]*)?\)")


class MarkdownLoader(BaseLoader):
    """Load Markdown (.md/.markdown) files directly as Documents."""

    def load(self, path: str) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")
        if p.suffix.lower() not in (".md", ".markdown"):
            raise ValueError(
                f"MarkdownLoader only supports .md/.markdown files, got: {p.suffix}"
            )

        # Read raw Markdown text (already canonical Markdown — no MarkItDown).
        text = p.read_text(encoding="utf-8")

        doc_hash = self._compute_hash(p)

        # Parse inline image links → ImageRef (skip broken/missing local files).
        image_refs = self._extract_images(text, p)

        metadata: dict[str, Any] = {
            "source_path": str(p),
            "doc_type": "markdown",
            "doc_hash": doc_hash,
            "images": [ref.to_dict() for ref in image_refs],
        }

        doc_id = f"md_{doc_hash[:12]}"
        return Document(id=doc_id, text=text, metadata=metadata)

    def _extract_images(self, text: str, source: Path) -> list[ImageRef]:
        """Parse ``![alt](path)`` links into ImageRefs.

        Local image files that exist are included; missing/broken links are
        logged and skipped. Image paths are resolved relative to the Markdown
        file's directory when not absolute.
        """
        base_dir = source.parent
        image_refs: list[ImageRef] = []

        for index, match in enumerate(_IMAGE_PATTERN.finditer(text)):
            link = match.group("path")

            # Resolve the image path relative to the Markdown file's directory.
            img_path = Path(link)
            if not img_path.is_absolute():
                img_path = base_dir / img_path

            if not img_path.exists():
                logger.warning(
                    "Markdown image link not found, skipping: %s (in %s)",
                    link, source,
                )
                continue

            image_id = f"md_img_{index}"
            ref = ImageRef(
                id=image_id,
                path=str(img_path),
                text_offset=match.start(),
                text_length=match.end() - match.start(),
            )
            image_refs.append(ref)

        return image_refs

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
        return [".md", ".markdown"]

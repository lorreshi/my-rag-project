'''
Author: lorreshi lorreshi@163.com
Date: 2026-04-19 18:52:24
LastEditors: lorreshi lorreshi@163.com
LastEditTime: 2026-05-05 15:05:02
FilePath: /my-rag-project/src/libs/loader/pdf_loader.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""PDF Loader — converts PDF to canonical Markdown using MarkItDown.

Responsibilities:
- Parse PDF → Markdown text
- Extract embedded images and save to data/images/{doc_hash}/
- Insert [IMAGE: {image_id}] placeholders in text
- Populate metadata: source_path, doc_type, images list
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from src.core.types import Document, ImageRef
from src.libs.loader.base_loader import BaseLoader


class PdfLoader(BaseLoader):
    """Load PDF files using MarkItDown, producing canonical Markdown."""

    def __init__(self, images_base_dir: str = "data/images"):
        self._images_base_dir = images_base_dir
        self._markitdown = MarkItDown()

    def load(self, path: str) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"PdfLoader only supports .pdf files, got: {p.suffix}")

        # Compute document hash for unique ID
        doc_hash = self._compute_hash(p)

        # Parse PDF to Markdown
        result = self._markitdown.convert(str(p))
        text = result.text_content or ""

        # Build metadata
        metadata: dict[str, Any] = {
            "source_path": str(p),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "images": [],
        }

        doc_id = f"pdf_{doc_hash[:12]}"
        return Document(id=doc_id, text=text, metadata=metadata)

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA256 hash of the file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

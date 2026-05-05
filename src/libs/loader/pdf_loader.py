"""PDF Loader — converts PDF to canonical Markdown using MarkItDown.

Responsibilities:
- Parse PDF → Markdown text
- Extract embedded images via pdfplumber and save to data/images/{doc_hash}/
- Insert [IMAGE: {image_id}] placeholders in text
- Populate metadata: source_path, doc_type, images list
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from src.core.types import Document, ImageRef
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """Load PDF files using MarkItDown + pdfplumber for image extraction."""

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

        # Extract images from PDF
        image_refs = self._extract_images(p, doc_hash)

        # Insert image placeholders into text
        if image_refs:
            text = self._insert_placeholders(text, image_refs)

        # Build metadata
        metadata: dict[str, Any] = {
            "source_path": str(p),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "images": [ref.to_dict() for ref in image_refs],
        }

        doc_id = f"pdf_{doc_hash[:12]}"
        return Document(id=doc_id, text=text, metadata=metadata)

    def _extract_images(self, pdf_path: Path, doc_hash: str) -> list[ImageRef]:
        """Extract images from PDF using pdfplumber. Graceful degradation on failure."""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed — skipping image extraction")
            return []

        image_refs: list[ImageRef] = []
        images_dir = Path(self._images_base_dir) / doc_hash
        images_dir.mkdir(parents=True, exist_ok=True)

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                seq = 0
                for page_num, page in enumerate(pdf.pages):
                    page_images = page.images or []
                    for img_obj in page_images:
                        image_id = f"{doc_hash[:12]}_{page_num}_{seq}"
                        image_path = images_dir / f"{image_id}.png"

                        try:
                            # Extract image using the page's crop
                            bbox = (
                                img_obj["x0"],
                                img_obj["top"],
                                img_obj["x1"],
                                img_obj["bottom"],
                            )
                            cropped = page.crop(bbox)
                            pil_image = cropped.to_image(resolution=150).original
                            pil_image.save(str(image_path), "PNG")

                            ref = ImageRef(
                                id=image_id,
                                path=str(image_path),
                                page=page_num,
                            )
                            image_refs.append(ref)
                            seq += 1
                        except Exception as exc:
                            logger.warning(
                                "Failed to extract image on page %d: %s",
                                page_num, exc,
                            )
                            continue
        except Exception as exc:
            logger.warning("Image extraction failed for %s: %s", pdf_path, exc)

        return image_refs

    @staticmethod
    def _insert_placeholders(text: str, image_refs: list[ImageRef]) -> str:
        """Append image placeholders at the end of text, grouped by page.

        Since MarkItDown doesn't preserve image positions in the Markdown,
        we append placeholders at the end. The text_offset/text_length in
        ImageRef will be updated to reflect actual positions.
        """
        if not image_refs:
            return text

        # Group images by page
        by_page: dict[int, list[ImageRef]] = {}
        for ref in image_refs:
            by_page.setdefault(ref.page, []).append(ref)

        # Append placeholders
        additions = []
        for page_num in sorted(by_page.keys()):
            for ref in by_page[page_num]:
                placeholder = f"[IMAGE: {ref.id}]"
                ref.text_offset = len(text) + sum(len(a) + 1 for a in additions)
                ref.text_length = len(placeholder)
                additions.append(placeholder)

        if additions:
            text = text.rstrip() + "\n\n" + "\n".join(additions) + "\n"

        return text

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

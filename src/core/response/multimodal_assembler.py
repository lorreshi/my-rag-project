"""MultimodalAssembler — append image content to MCP responses.

When retrieval results reference images (via metadata ``image_refs`` /
``images``), this assembler resolves each image_id to a file path, reads the
bytes, and appends an MCP ImageContent item (base64 + mimeType) to the
response ``content`` array.

Image resolution is delegated to an image_resolver exposing
``get_path(image_id) -> str | None`` (e.g. SQLiteImageStorage). Failures to
resolve or read an image are skipped gracefully (never break the response).
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class MultimodalAssembler:
    """Append base64 image content for image-referencing results."""

    def __init__(self, image_resolver: Any, max_images: int = 5):
        """Initialize.

        Args:
            image_resolver: Object with ``get_path(image_id) -> str | None``.
            max_images: Cap on the number of images appended (latency/size).
        """
        self._resolver = image_resolver
        self._max_images = max_images

    def assemble(
        self,
        response: dict[str, Any],
        results: list["RetrievalResult"],
    ) -> dict[str, Any]:
        """Append ImageContent items to *response* for images in *results*.

        Args:
            response: An existing MCP tool result (with a ``content`` list).
            results: The retrieval results whose images should be attached.

        Returns:
            The same response dict with image content appended (mutated + returned).
        """
        content = response.setdefault("content", [])
        appended = 0
        seen: set[str] = set()

        for result in results:
            for image_id in self._image_ids(result):
                if appended >= self._max_images:
                    return response
                if image_id in seen:
                    continue
                seen.add(image_id)
                item = self._build_image_content(image_id)
                if item is not None:
                    content.append(item)
                    appended += 1

        return response

    @staticmethod
    def _image_ids(result: "RetrievalResult") -> list[str]:
        """Extract referenced image ids from a result's metadata."""
        meta = result.metadata or {}
        refs = meta.get("image_refs")
        if isinstance(refs, list) and refs:
            return [str(r) for r in refs]
        # Fall back to images list of dicts
        images = meta.get("images")
        if isinstance(images, list):
            return [img.get("id", "") for img in images if isinstance(img, dict) and img.get("id")]
        return []

    def _build_image_content(self, image_id: str) -> dict[str, Any] | None:
        """Resolve an image_id to an MCP ImageContent dict, or None on failure."""
        try:
            path = self._resolver.get_path(image_id)
        except Exception as exc:
            logger.warning("Image resolver failed for %s: %s", image_id, exc)
            return None

        if not path:
            logger.warning("No path for image_id %s", image_id)
            return None

        p = Path(path)
        if not p.exists():
            logger.warning("Image file missing: %s", path)
            return None

        try:
            data = base64.b64encode(p.read_bytes()).decode("utf-8")
        except OSError as exc:
            logger.warning("Failed to read image %s: %s", path, exc)
            return None

        return {
            "type": "image",
            "data": data,
            "mimeType": _MIME_BY_SUFFIX.get(p.suffix.lower(), "image/png"),
        }

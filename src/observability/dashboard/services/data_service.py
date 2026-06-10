"""DataService — read-side facade for the data browser page (G3).

Wraps a DocumentManager (and optionally an image storage) to provide the data
the dashboard needs: document listing, chunk details, and image lookups.
Pure data layer — no Streamlit imports, unit-testable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.ingestion.document_manager import DocumentManager
    from src.ingestion.storage.image_storage import BaseImageStorage


class DataService:
    """Facade over DocumentManager + ImageStorage for the data browser."""

    def __init__(
        self,
        document_manager: "DocumentManager",
        image_storage: "BaseImageStorage | None" = None,
    ):
        self._dm = document_manager
        self._images = image_storage

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        """Return document summaries as plain dicts."""
        return [d.to_dict() for d in self._dm.list_documents(collection)]

    def get_chunks(self, source_path: str) -> list[dict[str, Any]]:
        """Return all chunks for a document (id/text/metadata dicts)."""
        return self._dm.get_document_detail(source_path).chunks

    def list_collections(self) -> list[str]:
        """Return the distinct collection names present in the store."""
        names: set[str] = set()
        for doc in self._dm.list_documents():
            names.add(doc.collection)
        return sorted(names)

    def get_image_path(self, image_id: str) -> str | None:
        """Resolve an image id to a file path, if image storage is available."""
        if self._images is None:
            return None
        getter = getattr(self._images, "get_path", None)
        if callable(getter):
            return getter(image_id)
        return None

    def chunk_images(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Return resolvable image entries (id + path) referenced by a chunk."""
        meta = chunk.get("metadata", {}) or {}
        refs = meta.get("image_refs", []) or []
        entries: list[dict[str, Any]] = []
        for image_id in refs:
            path = self.get_image_path(image_id)
            entries.append({"id": image_id, "path": path})
        return entries

"""IngestionService — dashboard-facing ingestion helpers (G4).

Wraps saving an uploaded file to a collection's documents directory and running
the pipeline. Kept Streamlit-free so the file-handling logic is unit-testable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.ingestion.pipeline import IngestionResult

logger = logging.getLogger(__name__)


class IngestionService:
    """Save uploads and drive the ingestion pipeline for the dashboard."""

    def __init__(self, pipeline: Any, documents_base_dir: str = "data/documents"):
        self._pipeline = pipeline
        self._base = Path(documents_base_dir)

    def save_upload(self, filename: str, data: bytes, collection: str = "default") -> str:
        """Persist uploaded bytes under data/documents/{collection}/ and return the path."""
        safe_name = Path(filename).name  # strip any path components
        if not safe_name:
            raise ValueError("Invalid filename")
        dest_dir = self._base / collection
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_name
        dest.write_bytes(data)
        return str(dest)

    def ingest(
        self,
        path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> "IngestionResult":
        """Run the ingestion pipeline for *path*."""
        return self._pipeline.run(
            path, collection=collection, force=force, on_progress=on_progress
        )

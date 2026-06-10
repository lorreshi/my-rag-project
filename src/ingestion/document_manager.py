"""DocumentManager — cross-store document lifecycle management.

Coordinates the four storage backends that hold a document's data:
- Chroma (vector store): chunk vectors + payload, keyed by metadata.source_path
- BM25Indexer: inverted index entries
- ImageStorage: extracted image files + id->path mapping
- FileIntegrity: ingestion history (SHA256 records)

Provides list / detail / delete / stats used by the dashboard (G3/G4).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.ingestion.storage.image_storage import BaseImageStorage
    from src.libs.loader.file_integrity import FileIntegrityChecker
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Summary info for an ingested document."""

    source_path: str
    collection: str = "default"
    chunk_count: int = 0
    image_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
        }


@dataclass
class DocumentDetail:
    """Detailed info for a single document."""

    source_path: str
    chunks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass
class DeleteResult:
    """Outcome of a delete_document operation."""

    source_path: str
    chunks_deleted: int = 0
    bm25_removed: bool = False
    images_deleted: int = 0
    integrity_removed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "chunks_deleted": self.chunks_deleted,
            "bm25_removed": self.bm25_removed,
            "images_deleted": self.images_deleted,
            "integrity_removed": self.integrity_removed,
        }


@dataclass
class CollectionStats:
    """Aggregate statistics for a collection (or all collections)."""

    collection: str
    document_count: int = 0
    chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
        }


class DocumentManager:
    """Coordinate document lifecycle across the storage backends."""

    def __init__(
        self,
        vector_store: "BaseVectorStore",
        bm25_indexer: "BM25Indexer | None" = None,
        image_storage: "BaseImageStorage | None" = None,
        file_integrity: "FileIntegrityChecker | None" = None,
    ):
        self._store = vector_store
        self._bm25 = bm25_indexer
        self._images = image_storage
        self._integrity = file_integrity

    # ------------------------------------------------------------------
    # Listing / detail
    # ------------------------------------------------------------------

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """List ingested documents, grouped by source_path.

        Aggregates chunk records from the vector store by their
        ``metadata.source_path`` (and optional ``collection``).
        """
        records = self._all_records(collection)
        grouped: dict[str, DocumentInfo] = {}
        for rec in records:
            meta = rec.get("metadata", {}) or {}
            source = meta.get("source_path", "unknown")
            coll = meta.get("collection", collection or "default")
            info = grouped.get(source)
            if info is None:
                info = DocumentInfo(source_path=source, collection=coll)
                grouped[source] = info
            info.chunk_count += 1
            if meta.get("image_refs"):
                info.image_count += len(meta["image_refs"])
        return sorted(grouped.values(), key=lambda d: d.source_path)

    def get_document_detail(self, source_path: str) -> DocumentDetail:
        """Return all chunks for a document identified by source_path."""
        records = self._store.get_by_metadata({"source_path": source_path})
        return DocumentDetail(source_path=source_path, chunks=records)

    # ------------------------------------------------------------------
    # Deletion (coordinated across 4 stores)
    # ------------------------------------------------------------------

    def delete_document(
        self, source_path: str, collection: str = "default"
    ) -> DeleteResult:
        """Delete a document's data across all storage backends."""
        result = DeleteResult(source_path=source_path)

        # 1. Chroma — delete chunk vectors by source_path
        try:
            result.chunks_deleted = self._store.delete_by_metadata(
                {"source_path": source_path}
            )
        except Exception as exc:
            logger.warning("Chroma delete failed for %s: %s", source_path, exc)

        # 2. BM25 — remove document entries
        if self._bm25 is not None:
            try:
                remover = getattr(self._bm25, "remove_document", None)
                if callable(remover):
                    remover(source_path)
                    result.bm25_removed = True
            except Exception as exc:
                logger.warning("BM25 remove failed for %s: %s", source_path, exc)

        # 3. ImageStorage — delete by doc hash if available
        if self._images is not None:
            try:
                doc_hash = self._derive_doc_hash(source_path)
                if doc_hash:
                    result.images_deleted = self._images.delete_by_doc_hash(doc_hash)
            except Exception as exc:
                logger.warning("Image delete failed for %s: %s", source_path, exc)

        # 4. FileIntegrity — remove processing record so file can re-ingest
        if self._integrity is not None:
            try:
                file_hash = self._integrity.compute_sha256(source_path)
                self._integrity.remove_record(file_hash)
                result.integrity_removed = True
            except Exception as exc:
                logger.warning("Integrity remove failed for %s: %s", source_path, exc)

        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """Return document + chunk counts for a collection (or all)."""
        docs = self.list_documents(collection)
        chunk_count = sum(d.chunk_count for d in docs)
        return CollectionStats(
            collection=collection or "all",
            document_count=len(docs),
            chunk_count=chunk_count,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _all_records(self, collection: str | None) -> list[dict[str, Any]]:
        filter_ = {"collection": collection} if collection else {}
        try:
            return self._store.get_by_metadata(filter_)
        except Exception as exc:
            logger.warning("Failed to list records: %s", exc)
            return []

    @staticmethod
    def _derive_doc_hash(source_path: str) -> str:
        """Compute the SHA256 doc hash of a file (used for image grouping)."""
        import hashlib
        from pathlib import Path

        p = Path(source_path)
        if not p.exists():
            return ""
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

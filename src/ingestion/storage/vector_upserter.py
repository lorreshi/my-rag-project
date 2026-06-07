"""VectorUpserter — deterministic IDs + idempotent vector store writes.

Receives chunks paired with their dense vectors, generates a stable, content-
derived ``vector_id`` for each, and writes them to a BaseVectorStore via upsert.

Determinism contract:
- ``vector_id = sha256(source_path + "::" + chunk_index + "::" + content_hash8)``
  where ``content_hash8`` is the first 8 hex chars of sha256(chunk.text).
- Same source_path + index + content  -> same id (re-ingest is idempotent).
- Any content change                  -> different id.

Because the vector store upsert is keyed by id, re-ingesting unchanged content
overwrites the same record (no duplicates).
"""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from src.core.types import Chunk
from src.libs.vector_store.base_vector_store import VectorRecord

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class VectorUpserter:
    """Generate stable IDs and upsert dense vectors idempotently."""

    def __init__(self, vector_store: "BaseVectorStore"):
        if vector_store is None:
            raise ValueError("VectorUpserter requires a BaseVectorStore instance")
        self._store = vector_store

    @staticmethod
    def make_vector_id(chunk: Chunk) -> str:
        """Compute a deterministic vector id for a chunk.

        Uses source_path + chunk_index + content hash so that identical content
        at the same logical position maps to the same id.
        """
        source_path = chunk.metadata.get("source_path", chunk.source_ref or "")
        chunk_index = chunk.metadata.get("chunk_index", "")
        content_hash8 = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
        key = f"{source_path}::{chunk_index}::{content_hash8}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

    def upsert(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        """Build records with stable IDs and upsert them, preserving order.

        Args:
            chunks: Chunks to store.
            dense_vectors: Dense vectors aligned with chunks (same length/order).
            trace: Optional trace context.

        Returns:
            List of generated vector IDs, in input order.

        Raises:
            ValueError: if chunks and dense_vectors lengths differ.
        """
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and dense_vectors "
                f"({len(dense_vectors)}) must have equal length"
            )

        if trace:
            trace.start_stage("vector_upserter")

        if not chunks:
            if trace:
                trace.end_stage(details={"upserted": 0})
            return []

        records: list[VectorRecord] = []
        ids: list[str] = []
        for chunk, vector in zip(chunks, dense_vectors):
            vid = self.make_vector_id(chunk)
            ids.append(vid)
            records.append(
                VectorRecord(
                    id=vid,
                    vector=vector,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                )
            )

        count = self._store.upsert(records, trace=trace)
        logger.info("Upserted %d vectors", count)

        if trace:
            trace.end_stage(details={"upserted": count})

        return ids

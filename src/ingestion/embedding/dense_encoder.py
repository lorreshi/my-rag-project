"""DenseEncoder — batch dense vectorization of chunks.

Wraps a BaseEmbedding provider to encode a list of Chunks into dense vectors.
Handles batching to limit request size and preserves input order so that
output[i] corresponds to chunks[i].
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.types import Chunk

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext
    from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class DenseEncoder:
    """Encode chunk text into dense embedding vectors."""

    def __init__(self, embedding: "BaseEmbedding", batch_size: int = 32):
        """Initialize DenseEncoder.

        Args:
            embedding: A BaseEmbedding provider instance.
            batch_size: Number of texts per embedding request (must be > 0).
        """
        if embedding is None:
            raise ValueError("DenseEncoder requires a BaseEmbedding instance")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._embedding = embedding
        self._batch_size = batch_size

    def encode(
        self,
        chunks: list[Chunk],
        trace: "TraceContext | None" = None,
    ) -> list[list[float]]:
        """Encode chunks into dense vectors.

        Args:
            chunks: Chunks to encode (uses chunk.text).
            trace: Optional trace context.

        Returns:
            List of dense vectors, one per chunk, in the same order.

        Raises:
            RuntimeError: if the provider returns a vector count that does not
                match the number of input chunks.
        """
        if trace:
            trace.start_stage("dense_encoder")

        if not chunks:
            if trace:
                trace.end_stage(details={"count": 0})
            return []

        texts = [c.text for c in chunks]
        vectors: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            batch_vectors = self._embedding.embed(batch, trace=trace)
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding provider returned {len(batch_vectors)} vectors "
                    f"for a batch of {len(batch)} texts"
                )
            vectors.extend(batch_vectors)

        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Dense encoding produced {len(vectors)} vectors for "
                f"{len(chunks)} chunks"
            )

        dim = len(vectors[0]) if vectors else 0
        if trace:
            trace.end_stage(details={"count": len(vectors), "dimension": dim})

        return vectors

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (may be 0 until first call)."""
        return self._embedding.dimension

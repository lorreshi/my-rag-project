"""BatchProcessor — orchestrates dense + sparse encoding over batches.

Splits chunks into fixed-size batches and drives both the DenseEncoder and the
SparseEncoder, preserving input order. Per-batch timing is recorded for trace
observability (Phase F will consume it).

The result merges each chunk with its dense vector and sparse statistics into
an EncodedChunk, ready for the storage stage (VectorUpserter / BM25Indexer).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

from src.core.types import Chunk

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext
    from src.ingestion.embedding.dense_encoder import DenseEncoder
    from src.ingestion.embedding.sparse_encoder import SparseEncoder, SparseVector

logger = logging.getLogger(__name__)


@dataclass
class EncodedChunk:
    """A chunk paired with its dense vector and sparse statistics."""

    chunk: Chunk
    dense_vector: list[float] = field(default_factory=list)
    sparse_vector: "SparseVector | None" = None


def chunk_batches(items: list, batch_size: int):
    """Yield successive batch_size-sized slices of items (order preserved)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class BatchProcessor:
    """Drive dense + sparse encoding over batches of chunks."""

    def __init__(
        self,
        dense_encoder: "DenseEncoder | None" = None,
        sparse_encoder: "SparseEncoder | None" = None,
        batch_size: int = 32,
    ):
        """Initialize BatchProcessor.

        Args:
            dense_encoder: Optional DenseEncoder (dense path disabled if None).
            sparse_encoder: Optional SparseEncoder (sparse path disabled if None).
            batch_size: Number of chunks per batch (must be > 0).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._dense = dense_encoder
        self._sparse = sparse_encoder
        self._batch_size = batch_size
        self.batch_timings: list[float] = []

    def process(
        self,
        chunks: list[Chunk],
        trace: "TraceContext | None" = None,
    ) -> list[EncodedChunk]:
        """Encode all chunks in batches, preserving order.

        Returns:
            List of EncodedChunk aligned with the input order.
        """
        if trace:
            trace.start_stage("batch_processor")

        self.batch_timings = []
        encoded: list[EncodedChunk] = []
        num_batches = 0

        for batch in chunk_batches(chunks, self._batch_size):
            num_batches += 1
            start_t = perf_counter()

            dense_vectors: list[list[float]] = []
            sparse_vectors: list = []

            if self._dense is not None:
                dense_vectors = self._dense.encode(batch, trace=trace)
            if self._sparse is not None:
                sparse_vectors = self._sparse.encode(batch, trace=trace)

            for i, chunk in enumerate(batch):
                encoded.append(
                    EncodedChunk(
                        chunk=chunk,
                        dense_vector=dense_vectors[i] if dense_vectors else [],
                        sparse_vector=sparse_vectors[i] if sparse_vectors else None,
                    )
                )

            self.batch_timings.append(perf_counter() - start_t)

        if trace:
            trace.end_stage(
                details={
                    "num_batches": num_batches,
                    "num_chunks": len(chunks),
                    "batch_size": self._batch_size,
                }
            )

        return encoded

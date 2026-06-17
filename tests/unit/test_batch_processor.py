"""Unit tests for BatchProcessor — batching, ordering, and orchestration."""
from __future__ import annotations

import pytest

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.batch_processor import (
    BatchProcessor,
    EncodedChunk,
    chunk_batches,
)
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.libs.tokenizer import JiebaTokenizer


class FakeEmbedding:
    def __init__(self, dim: int = 3):
        self._dim = dim

    def embed(self, texts, trace=None):
        return [[float(len(t))] * self._dim for t in texts]

    @property
    def provider_name(self):
        return "fake"

    @property
    def dimension(self):
        return self._dim


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", text=f"text number {i}", metadata={}, source_ref="doc")
        for i in range(n)
    ]


class TestBatching:
    def test_5_chunks_batch_size_2_makes_3_batches(self):
        dense = DenseEncoder(FakeEmbedding(), batch_size=2)
        bp = BatchProcessor(dense_encoder=dense, batch_size=2)
        bp.process(_chunks(5))
        assert len(bp.batch_timings) == 3

    def test_chunk_batches_helper(self):
        batches = list(chunk_batches(list(range(5)), 2))
        assert batches == [[0, 1], [2, 3], [4]]

    def test_batch_size_validation(self):
        with pytest.raises(ValueError):
            BatchProcessor(batch_size=0)

    def test_chunk_batches_invalid_size(self):
        with pytest.raises(ValueError):
            list(chunk_batches([1, 2, 3], 0))


class TestOrdering:
    def test_output_count_matches_input(self):
        dense = DenseEncoder(FakeEmbedding(), batch_size=2)
        sparse = SparseEncoder()
        bp = BatchProcessor(dense, sparse, batch_size=2)
        result = bp.process(_chunks(5))
        assert len(result) == 5
        assert all(isinstance(e, EncodedChunk) for e in result)

    def test_order_stable(self):
        dense = DenseEncoder(FakeEmbedding(), batch_size=2)
        bp = BatchProcessor(dense_encoder=dense, batch_size=2)
        chunks = _chunks(5)
        result = bp.process(chunks)
        for original, enc in zip(chunks, result):
            assert enc.chunk.id == original.id

    def test_dense_vector_aligned(self):
        dense = DenseEncoder(FakeEmbedding(dim=2), batch_size=2)
        bp = BatchProcessor(dense_encoder=dense, batch_size=2)
        chunks = _chunks(3)
        result = bp.process(chunks)
        for enc in result:
            assert enc.dense_vector[0] == float(len(enc.chunk.text))

    def test_sparse_vector_aligned(self):
        sparse = SparseEncoder(tokenizer=JiebaTokenizer(stopwords=set()))
        bp = BatchProcessor(sparse_encoder=sparse, batch_size=2)
        result = bp.process(_chunks(3))
        for enc in result:
            assert enc.sparse_vector is not None
            assert enc.sparse_vector.chunk_id == enc.chunk.id


class TestOptionalPaths:
    def test_dense_only(self):
        dense = DenseEncoder(FakeEmbedding(), batch_size=2)
        bp = BatchProcessor(dense_encoder=dense, batch_size=2)
        result = bp.process(_chunks(3))
        assert all(e.dense_vector for e in result)
        assert all(e.sparse_vector is None for e in result)

    def test_sparse_only(self):
        sparse = SparseEncoder()
        bp = BatchProcessor(sparse_encoder=sparse, batch_size=2)
        result = bp.process(_chunks(3))
        assert all(e.dense_vector == [] for e in result)
        assert all(e.sparse_vector is not None for e in result)

    def test_both_paths(self):
        dense = DenseEncoder(FakeEmbedding(), batch_size=2)
        sparse = SparseEncoder()
        bp = BatchProcessor(dense, sparse, batch_size=2)
        result = bp.process(_chunks(4))
        assert all(e.dense_vector for e in result)
        assert all(e.sparse_vector is not None for e in result)

    def test_empty_chunks(self):
        dense = DenseEncoder(FakeEmbedding(), batch_size=2)
        bp = BatchProcessor(dense_encoder=dense, batch_size=2)
        assert bp.process([]) == []


class TestTrace:
    def test_trace_records_stage(self):
        dense = DenseEncoder(FakeEmbedding(), batch_size=2)
        bp = BatchProcessor(dense_encoder=dense, batch_size=2)
        trace = TraceContext(trace_type="ingestion")
        bp.process(_chunks(5), trace=trace)
        # find the batch_processor stage (encoders also record stages)
        stages = {s.name: s for s in trace.stages}
        assert "batch_processor" in stages
        assert stages["batch_processor"].details["num_batches"] == 3
        assert stages["batch_processor"].details["num_chunks"] == 5

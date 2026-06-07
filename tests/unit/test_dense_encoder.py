"""Unit tests for DenseEncoder — batching and contract guarantees.

Uses a fake deterministic embedding provider. No real API calls.
"""
from __future__ import annotations

import pytest

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.dense_encoder import DenseEncoder


class FakeEmbedding:
    """Deterministic fake embedding provider for testing."""

    def __init__(self, dim: int = 4):
        self._dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts, trace=None):
        self.calls.append(list(texts))
        # Deterministic vector based on text length
        return [[float(len(t))] * self._dim for t in texts]

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return self._dim


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", text=f"chunk text {i}", metadata={}, source_ref="doc")
        for i in range(n)
    ]


class TestEncoding:
    def test_vector_count_matches_chunks(self):
        enc = DenseEncoder(FakeEmbedding(), batch_size=2)
        vectors = enc.encode(_chunks(5))
        assert len(vectors) == 5

    def test_dimension_consistent(self):
        enc = DenseEncoder(FakeEmbedding(dim=8), batch_size=3)
        vectors = enc.encode(_chunks(4))
        assert all(len(v) == 8 for v in vectors)

    def test_order_preserved(self):
        fake = FakeEmbedding(dim=2)
        enc = DenseEncoder(fake, batch_size=2)
        chunks = _chunks(3)
        vectors = enc.encode(chunks)
        # vector value encodes text length; verify per-chunk correspondence
        for chunk, vec in zip(chunks, vectors):
            assert vec[0] == float(len(chunk.text))

    def test_batching_splits_correctly(self):
        fake = FakeEmbedding()
        enc = DenseEncoder(fake, batch_size=2)
        enc.encode(_chunks(5))
        # 5 chunks, batch_size 2 -> batches of [2, 2, 1]
        assert [len(b) for b in fake.calls] == [2, 2, 1]

    def test_single_batch_when_large_batch_size(self):
        fake = FakeEmbedding()
        enc = DenseEncoder(fake, batch_size=100)
        enc.encode(_chunks(5))
        assert len(fake.calls) == 1
        assert len(fake.calls[0]) == 5

    def test_empty_chunks(self):
        enc = DenseEncoder(FakeEmbedding())
        assert enc.encode([]) == []


class TestContractViolations:
    def test_mismatched_vector_count_raises(self):
        class BadEmbedding(FakeEmbedding):
            def embed(self, texts, trace=None):
                return [[0.0] * self._dim]  # always returns 1 vector

        enc = DenseEncoder(BadEmbedding(), batch_size=10)
        with pytest.raises(RuntimeError):
            enc.encode(_chunks(3))

    def test_requires_embedding(self):
        with pytest.raises(ValueError):
            DenseEncoder(None)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            DenseEncoder(FakeEmbedding(), batch_size=0)


class TestTrace:
    def test_trace_records_stage(self):
        enc = DenseEncoder(FakeEmbedding(dim=4), batch_size=2)
        trace = TraceContext(trace_type="ingestion")
        enc.encode(_chunks(3), trace=trace)
        assert trace.stages[0].name == "dense_encoder"
        assert trace.stages[0].details["count"] == 3
        assert trace.stages[0].details["dimension"] == 4

    def test_trace_empty(self):
        enc = DenseEncoder(FakeEmbedding())
        trace = TraceContext(trace_type="ingestion")
        enc.encode([], trace=trace)
        assert trace.stages[0].details["count"] == 0

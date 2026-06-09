"""Unit tests for DenseRetriever — embedding + vector store orchestration."""
from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.core.query_engine.dense_retriever import DenseRetriever
from src.libs.vector_store.base_vector_store import BaseVectorStore, QueryResult


class FakeEmbedding:
    def __init__(self, dim: int = 4):
        self._dim = dim
        self.embed_calls: list[list[str]] = []

    def embed(self, texts, trace=None):
        self.embed_calls.append(list(texts))
        return [[0.1] * self._dim for _ in texts]

    @property
    def provider_name(self):
        return "fake"

    @property
    def dimension(self):
        return self._dim


class FakeVectorStore(BaseVectorStore):
    def __init__(self, results=None):
        self._results = results or []
        self.query_calls: list[dict] = []

    def upsert(self, records, trace=None):
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None):
        self.query_calls.append({"vector": vector, "top_k": top_k, "filters": filters})
        return self._results[:top_k]

    def delete_by_metadata(self, filter, trace=None):
        return 0

    @property
    def backend_name(self):
        return "fake"


def _qr(rid, score, text="", meta=None):
    return QueryResult(id=rid, score=score, text=text, metadata=meta or {})


class TestRetrieve:
    def test_returns_retrieval_results(self):
        store = FakeVectorStore([
            _qr("c0", 0.9, "alpha", {"source_path": "a.pdf"}),
            _qr("c1", 0.8, "beta", {"source_path": "b.pdf"}),
        ])
        r = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=store)
        results = r.retrieve("some query", top_k=10)
        assert len(results) == 2
        assert all(isinstance(x, RetrievalResult) for x in results)

    def test_result_fields_populated(self):
        store = FakeVectorStore([_qr("c0", 0.95, "hello text", {"page": 3})])
        r = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=store)
        result = r.retrieve("q")[0]
        assert result.chunk_id == "c0"
        assert result.score == 0.95
        assert result.text == "hello text"
        assert result.metadata["page"] == 3

    def test_embedding_called_with_query(self):
        emb = FakeEmbedding()
        store = FakeVectorStore([])
        r = DenseRetriever(embedding_client=emb, vector_store=store)
        r.retrieve("my query text")
        assert emb.embed_calls == [["my query text"]]

    def test_top_k_passed_to_store(self):
        store = FakeVectorStore([])
        r = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=store)
        r.retrieve("q", top_k=5)
        assert store.query_calls[0]["top_k"] == 5

    def test_filters_passed_to_store(self):
        store = FakeVectorStore([])
        r = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=store)
        r.retrieve("q", filters={"collection": "docs"})
        assert store.query_calls[0]["filters"] == {"collection": "docs"}

    def test_empty_query_returns_empty(self):
        emb = FakeEmbedding()
        store = FakeVectorStore([_qr("c0", 0.9)])
        r = DenseRetriever(embedding_client=emb, vector_store=store)
        assert r.retrieve("") == []
        assert r.retrieve("   ") == []
        # embedding should not be called for empty query
        assert emb.embed_calls == []

    def test_no_hits_returns_empty(self):
        store = FakeVectorStore([])
        r = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=store)
        assert r.retrieve("q") == []


class TestConstruction:
    def test_requires_components_or_settings(self):
        with pytest.raises(ValueError):
            DenseRetriever()

    def test_injected_components_used(self):
        emb = FakeEmbedding()
        store = FakeVectorStore([])
        r = DenseRetriever(embedding_client=emb, vector_store=store)
        assert r._embedding is emb
        assert r._store is store


class TestSerialization:
    def test_result_to_dict(self):
        store = FakeVectorStore([_qr("c0", 0.5, "txt", {"k": "v"})])
        r = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=store)
        d = r.retrieve("q")[0].to_dict()
        assert d == {"chunk_id": "c0", "score": 0.5, "text": "txt", "metadata": {"k": "v"}}


class TestTrace:
    def test_trace_records_stage(self):
        store = FakeVectorStore([_qr("c0", 0.9, "t")])
        r = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=store)
        trace = TraceContext(trace_type="query")
        r.retrieve("q", trace=trace)
        stages = {s.name: s for s in trace.stages}
        assert "dense_retrieval" in stages
        assert stages["dense_retrieval"].details["count"] == 1

    def test_trace_empty_query(self):
        r = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=FakeVectorStore())
        trace = TraceContext(trace_type="query")
        r.retrieve("", trace=trace)
        assert trace.stages[0].details["reason"] == "empty_query"

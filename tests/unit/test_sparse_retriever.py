"""Unit tests for SparseRetriever — BM25 query + text/metadata resolution."""
from __future__ import annotations

import pytest

from src.core.types import Chunk, RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    """In-memory store supporting get_by_ids."""

    def __init__(self, records: dict[str, dict]):
        # records: {id: {"text":..., "metadata":...}}
        self._records = records
        self.get_calls: list[list[str]] = []

    def upsert(self, records, trace=None):
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None):
        return []

    def delete_by_metadata(self, filter, trace=None):
        return 0

    def get_by_ids(self, ids):
        self.get_calls.append(list(ids))
        return [
            {"id": i, "text": self._records[i]["text"], "metadata": self._records[i]["metadata"]}
            for i in ids if i in self._records
        ]

    @property
    def backend_name(self):
        return "fake"


def _build_index(tmp_path, corpus: dict[str, str]) -> BM25Indexer:
    """corpus: {chunk_id: text}. Build a BM25 index over it."""
    enc = SparseEncoder(stopwords=set())
    chunks = [Chunk(id=cid, text=txt, metadata={}, source_ref="d") for cid, txt in corpus.items()]
    svs = enc.encode(chunks)
    idx = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    idx.build(svs)
    return idx


@pytest.fixture
def corpus():
    return {
        "c0": "machine learning models",
        "c1": "deep learning networks",
        "c2": "vector database search",
    }


@pytest.fixture
def store(corpus):
    return FakeVectorStore({
        cid: {"text": txt, "metadata": {"source_path": f"{cid}.pdf"}}
        for cid, txt in corpus.items()
    })


class TestRetrieve:
    def test_returns_retrieval_results(self, tmp_path, corpus, store):
        idx = _build_index(tmp_path, corpus)
        r = SparseRetriever(bm25_indexer=idx, vector_store=store)
        results = r.retrieve(["learning"])
        assert all(isinstance(x, RetrievalResult) for x in results)
        ids = {x.chunk_id for x in results}
        assert ids == {"c0", "c1"}

    def test_text_and_metadata_resolved(self, tmp_path, corpus, store):
        idx = _build_index(tmp_path, corpus)
        r = SparseRetriever(bm25_indexer=idx, vector_store=store)
        results = r.retrieve(["database"])
        assert results[0].chunk_id == "c2"
        assert results[0].text == "vector database search"
        assert results[0].metadata["source_path"] == "c2.pdf"

    def test_scores_populated(self, tmp_path, corpus, store):
        idx = _build_index(tmp_path, corpus)
        r = SparseRetriever(bm25_indexer=idx, vector_store=store)
        results = r.retrieve(["learning"])
        assert all(x.score > 0 for x in results)

    def test_top_k_limits(self, tmp_path, corpus, store):
        idx = _build_index(tmp_path, corpus)
        r = SparseRetriever(bm25_indexer=idx, vector_store=store)
        results = r.retrieve(["learning", "vector", "database"], top_k=1)
        assert len(results) <= 1

    def test_empty_keywords(self, tmp_path, corpus, store):
        idx = _build_index(tmp_path, corpus)
        r = SparseRetriever(bm25_indexer=idx, vector_store=store)
        assert r.retrieve([]) == []

    def test_no_match(self, tmp_path, corpus, store):
        idx = _build_index(tmp_path, corpus)
        r = SparseRetriever(bm25_indexer=idx, vector_store=store)
        assert r.retrieve(["nonexistent"]) == []

    def test_missing_chunk_skipped(self, tmp_path, corpus):
        """BM25 hit not present in store is skipped safely."""
        idx = _build_index(tmp_path, corpus)
        # store missing c1
        partial = FakeVectorStore({
            "c0": {"text": "machine learning models", "metadata": {}},
            "c2": {"text": "vector database search", "metadata": {}},
        })
        r = SparseRetriever(bm25_indexer=idx, vector_store=partial)
        results = r.retrieve(["learning"])
        ids = {x.chunk_id for x in results}
        assert "c1" not in ids
        assert "c0" in ids


class TestConstruction:
    def test_requires_store_or_settings(self, tmp_path, corpus):
        idx = _build_index(tmp_path, corpus)
        with pytest.raises(ValueError):
            SparseRetriever(bm25_indexer=idx)

    def test_auto_load_missing_index_no_crash(self, tmp_path, store):
        # No index built in this dir; should warn, not crash, and return empty.
        r = SparseRetriever(
            vector_store=store, index_dir=str(tmp_path / "empty"), auto_load=True
        )
        assert r.retrieve(["anything"]) == []


class TestGetByIds:
    def test_chroma_store_has_get_by_ids(self):
        from src.libs.vector_store.chroma_store import ChromaStore
        assert hasattr(ChromaStore, "get_by_ids")


class TestTrace:
    def test_trace_records_stage(self, tmp_path, corpus, store):
        idx = _build_index(tmp_path, corpus)
        r = SparseRetriever(bm25_indexer=idx, vector_store=store)
        trace = TraceContext(trace_type="query")
        r.retrieve(["learning"], trace=trace)
        stages = {s.name: s for s in trace.stages}
        assert "sparse_retrieval" in stages
        assert stages["sparse_retrieval"].details["count"] == 2

    def test_trace_no_keywords(self, tmp_path, corpus, store):
        idx = _build_index(tmp_path, corpus)
        r = SparseRetriever(bm25_indexer=idx, vector_store=store)
        trace = TraceContext(trace_type="query")
        r.retrieve([], trace=trace)
        assert trace.stages[0].details["reason"] == "no_keywords"

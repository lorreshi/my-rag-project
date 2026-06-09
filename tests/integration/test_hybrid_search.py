"""Integration tests for HybridSearch (D5).

Wires the real QueryProcessor + Fusion with fake Dense/Sparse retrievers (and
a fake embedding/store underneath where useful) to verify orchestration,
filtering, and single-route degradation.
"""
from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.core.query_engine.fusion import ReciprocalRankFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor


class FakeRetriever:
    """Configurable fake for dense/sparse retrievers."""

    def __init__(self, results=None, raises=False, kind="dense"):
        self._results = results or []
        self._raises = raises
        self._kind = kind
        self.called_with = None

    def retrieve(self, query_or_keywords, top_k=20, filters=None, trace=None):
        self.called_with = {"arg": query_or_keywords, "top_k": top_k, "filters": filters}
        if self._raises:
            raise RuntimeError(f"{self._kind} boom")
        return self._results


def _r(cid, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=0.0, text=text, metadata=meta or {})


def _hybrid(dense, sparse):
    return HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=ReciprocalRankFusion(k=60),
    )


class TestOrchestration:
    def test_returns_topk(self):
        dense = FakeRetriever([_r("a"), _r("b"), _r("c")], kind="dense")
        sparse = FakeRetriever([_r("b"), _r("d")], kind="sparse")
        hs = _hybrid(dense, sparse)
        results = hs.search("vector search query", top_k=2)
        assert len(results) == 2
        assert all(isinstance(x, RetrievalResult) for x in results)

    def test_both_routes_called(self):
        dense = FakeRetriever([_r("a")], kind="dense")
        sparse = FakeRetriever([_r("b")], kind="sparse")
        hs = _hybrid(dense, sparse)
        hs.search("machine learning")
        assert dense.called_with is not None
        assert sparse.called_with is not None

    def test_dense_gets_normalized_query(self):
        dense = FakeRetriever([_r("a")], kind="dense")
        sparse = FakeRetriever([], kind="sparse")
        hs = _hybrid(dense, sparse)
        hs.search("  How to   configure Azure  ")
        # dense receives the normalized query text
        assert dense.called_with["arg"] == "How to configure Azure"

    def test_sparse_gets_keywords(self):
        dense = FakeRetriever([], kind="dense")
        sparse = FakeRetriever([_r("a")], kind="sparse")
        hs = _hybrid(dense, sparse)
        hs.search("configure Azure endpoint")
        # sparse receives a keyword list (stopwords removed)
        assert isinstance(sparse.called_with["arg"], list)
        assert "azure" in sparse.called_with["arg"]

    def test_fusion_merges_results(self):
        dense = FakeRetriever([_r("a"), _r("b")], kind="dense")
        sparse = FakeRetriever([_r("a"), _r("c")], kind="sparse")
        hs = _hybrid(dense, sparse)
        results = hs.search("q", top_k=10)
        ids = {x.chunk_id for x in results}
        assert ids == {"a", "b", "c"}
        # 'a' appears in both -> should rank first
        assert results[0].chunk_id == "a"


class TestFilters:
    def test_post_filter_drops_mismatch(self):
        dense = FakeRetriever([
            _r("a", meta={"doc_type": "pdf"}),
            _r("b", meta={"doc_type": "html"}),
        ], kind="dense")
        sparse = FakeRetriever([], kind="sparse")
        hs = _hybrid(dense, sparse)
        results = hs.search("q", filters={"doc_type": "pdf"})
        ids = {x.chunk_id for x in results}
        assert ids == {"a"}

    def test_missing_key_included(self):
        dense = FakeRetriever([
            _r("a", meta={"doc_type": "pdf"}),
            _r("b", meta={}),  # missing doc_type -> lenient include
        ], kind="dense")
        sparse = FakeRetriever([], kind="sparse")
        hs = _hybrid(dense, sparse)
        results = hs.search("q", filters={"doc_type": "pdf"})
        ids = {x.chunk_id for x in results}
        assert ids == {"a", "b"}

    def test_filters_forwarded_to_dense(self):
        dense = FakeRetriever([_r("a", meta={"collection": "c"})], kind="dense")
        sparse = FakeRetriever([], kind="sparse")
        hs = _hybrid(dense, sparse)
        hs.search("q", filters={"collection": "c"})
        assert dense.called_with["filters"] == {"collection": "c"}


class TestDegradation:
    def test_dense_failure_degrades_to_sparse(self):
        dense = FakeRetriever(raises=True, kind="dense")
        sparse = FakeRetriever([_r("s1"), _r("s2")], kind="sparse")
        hs = _hybrid(dense, sparse)
        results = hs.search("q")
        ids = {x.chunk_id for x in results}
        assert ids == {"s1", "s2"}

    def test_sparse_failure_degrades_to_dense(self):
        dense = FakeRetriever([_r("d1"), _r("d2")], kind="dense")
        sparse = FakeRetriever(raises=True, kind="sparse")
        hs = _hybrid(dense, sparse)
        results = hs.search("q")
        ids = {x.chunk_id for x in results}
        assert ids == {"d1", "d2"}

    def test_both_empty_returns_empty(self):
        dense = FakeRetriever([], kind="dense")
        sparse = FakeRetriever([], kind="sparse")
        hs = _hybrid(dense, sparse)
        assert hs.search("q") == []


class TestTrace:
    def test_trace_records_stage(self):
        dense = FakeRetriever([_r("a")], kind="dense")
        sparse = FakeRetriever([_r("b")], kind="sparse")
        hs = _hybrid(dense, sparse)
        trace = TraceContext(trace_type="query")
        hs.search("q", trace=trace)
        stages = {s.name: s for s in trace.stages}
        assert "hybrid_search" in stages
        assert stages["hybrid_search"].details["final_count"] == 2

"""T15 integration — single-route + QueryTransform degradation.

Validates:
- Property 7 (优雅降级保持): if the dense OR sparse route raises, the query
  still returns the surviving route's results instead of failing.
- Property 10 (QueryTransform 失败降级): when a multi_query/HyDE LLM call
  fails, the dense side degrades to the single original query and the query
  still returns normally (``degraded=True`` recorded in trace).
"""
from __future__ import annotations

from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.core.query_engine.fusion import ReciprocalRankFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.query_transform import MultiQueryTransform, HyDETransform


class FakeRetriever:
    def __init__(self, results=None, raises=False, kind="dense"):
        self._results = results or []
        self._raises = raises
        self._kind = kind
        self.calls = 0

    def retrieve(self, query_or_keywords, top_k=20, filters=None, **kwargs):
        self.calls += 1
        if self._raises:
            raise RuntimeError(f"{self._kind} boom")
        return list(self._results)


class _RaisingLLM:
    def chat(self, messages, trace=None):
        raise RuntimeError("llm down")


class _FixedLLM:
    """Returns two deterministic rewrite variants."""

    def chat(self, messages, trace=None):
        class _Resp:
            content = "azure 设置方法\n如何配置 azure 端点"

        return _Resp()


def _r(cid, meta=None):
    return RetrievalResult(chunk_id=cid, score=0.0, text="", metadata=meta or {})


def _hybrid(dense, sparse, transform=None):
    return HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=ReciprocalRankFusion(k=60),
        query_transform=transform,
    )


class TestSingleRouteDegradation:
    def test_dense_failure_degrades_to_sparse(self):
        dense = FakeRetriever(raises=True, kind="dense")
        sparse = FakeRetriever([_r("s1"), _r("s2")], kind="sparse")
        hs = _hybrid(dense, sparse)
        ids = {x.chunk_id for x in hs.search("q")}
        assert ids == {"s1", "s2"}

    def test_sparse_failure_degrades_to_dense(self):
        dense = FakeRetriever([_r("d1"), _r("d2")], kind="dense")
        sparse = FakeRetriever(raises=True, kind="sparse")
        hs = _hybrid(dense, sparse)
        ids = {x.chunk_id for x in hs.search("q")}
        assert ids == {"d1", "d2"}

    def test_both_failures_return_empty_not_raise(self):
        dense = FakeRetriever(raises=True, kind="dense")
        sparse = FakeRetriever(raises=True, kind="sparse")
        hs = _hybrid(dense, sparse)
        assert hs.search("q") == []


class TestQueryTransformDegradation:
    def test_multi_query_llm_failure_degrades_single(self):
        dense = FakeRetriever([_r("a")], kind="dense")
        sparse = FakeRetriever([_r("b")], kind="sparse")
        transform = MultiQueryTransform(_RaisingLLM(), n=3)
        hs = _hybrid(dense, sparse, transform=transform)

        trace = TraceContext(trace_type="query")
        results = hs.search("q", trace=trace)
        # Query still returns; degraded recorded; only one dense list ran.
        assert {x.chunk_id for x in results} == {"a", "b"}
        stage = {s.name: s for s in trace.stages}["hybrid_search"]
        assert stage.details["dense_lists"] == 1
        assert stage.details["query_transform_degraded"] is True
        assert dense.calls == 1

    def test_hyde_llm_failure_degrades_single(self):
        dense = FakeRetriever([_r("a")], kind="dense")
        sparse = FakeRetriever([], kind="sparse")
        transform = HyDETransform(_RaisingLLM(), augment=True)
        hs = _hybrid(dense, sparse, transform=transform)

        trace = TraceContext(trace_type="query")
        results = hs.search("q", trace=trace)
        assert {x.chunk_id for x in results} == {"a"}
        stage = {s.name: s for s in trace.stages}["hybrid_search"]
        assert stage.details["query_transform_degraded"] is True
        assert dense.calls == 1

    def test_multi_query_success_runs_multiple_dense_lists(self):
        dense = FakeRetriever([_r("a")], kind="dense")
        sparse = FakeRetriever([_r("b")], kind="sparse")
        transform = MultiQueryTransform(_FixedLLM(), n=3, max_concurrency=4)
        hs = _hybrid(dense, sparse, transform=transform)

        trace = TraceContext(trace_type="query")
        hs.search("azure 配置", trace=trace)
        stage = {s.name: s for s in trace.stages}["hybrid_search"]
        # original + 2 fixed variants -> 3 dense lists, no degradation.
        assert stage.details["dense_lists"] == 3
        assert stage.details["query_transform_degraded"] is False
        assert dense.calls == 3

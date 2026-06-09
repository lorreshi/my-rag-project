"""Unit tests for Core Reranker — orchestration + fallback to fusion order."""
from __future__ import annotations

from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.core.query_engine.reranker import Reranker
from src.libs.reranker.base_reranker import (
    BaseReranker,
    NoneReranker,
    RerankCandidate,
)


def _r(cid, score=0.0, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=score, text=text or cid, metadata=meta or {})


class ReverseReranker(BaseReranker):
    """Reranker that reverses candidate order (deterministic, for testing)."""

    def rerank(self, query, candidates, trace=None):
        rev = list(reversed(candidates))
        for i, c in enumerate(rev):
            c.score = float(len(rev) - i)
        return rev

    @property
    def backend_name(self):
        return "reverse"


class FailingReranker(BaseReranker):
    """Reranker that raises."""

    def rerank(self, query, candidates, trace=None):
        raise RuntimeError("backend exploded")

    @property
    def backend_name(self):
        return "failing"


class SelfReportFailReranker(BaseReranker):
    """Reranker that returns candidates but flags has_failed."""

    def __init__(self):
        self.has_failed = False

    def rerank(self, query, candidates, trace=None):
        self.has_failed = True
        return list(candidates)

    @property
    def backend_name(self):
        return "selfreport"


class TestReranking:
    def test_reorders_with_backend(self):
        rr = Reranker(backend=ReverseReranker(), top_m=30)
        cands = [_r("a"), _r("b"), _r("c")]
        results = rr.rerank("q", cands)
        assert [x.chunk_id for x in results] == ["c", "b", "a"]

    def test_not_fallback_on_success(self):
        rr = Reranker(backend=ReverseReranker())
        results = rr.rerank("q", [_r("a"), _r("b")])
        assert all(x.metadata["rerank_fallback"] is False for x in results)

    def test_backend_recorded(self):
        rr = Reranker(backend=ReverseReranker())
        results = rr.rerank("q", [_r("a")])
        assert results[0].metadata["rerank_backend"] == "reverse"

    def test_text_metadata_preserved(self):
        rr = Reranker(backend=ReverseReranker())
        cands = [_r("a", text="alpha", meta={"source_path": "a.pdf"})]
        results = rr.rerank("q", cands)
        assert results[0].text == "alpha"
        assert results[0].metadata["source_path"] == "a.pdf"

    def test_top_m_limits_backend_input(self):
        rr = Reranker(backend=ReverseReranker(), top_m=2)
        cands = [_r("a"), _r("b"), _r("c"), _r("d")]
        results = rr.rerank("q", cands)
        # only first 2 reranked (reversed -> b,a), tail c,d appended in order
        assert [x.chunk_id for x in results] == ["b", "a", "c", "d"]

    def test_top_k_cut(self):
        rr = Reranker(backend=ReverseReranker())
        results = rr.rerank("q", [_r("a"), _r("b"), _r("c")], top_k=2)
        assert len(results) == 2


class TestFallback:
    def test_exception_falls_back_to_fusion_order(self):
        rr = Reranker(backend=FailingReranker())
        cands = [_r("a"), _r("b"), _r("c")]
        results = rr.rerank("q", cands)
        # order preserved (fusion order)
        assert [x.chunk_id for x in results] == ["a", "b", "c"]
        assert all(x.metadata["rerank_fallback"] is True for x in results)

    def test_self_reported_failure_triggers_fallback(self):
        rr = Reranker(backend=SelfReportFailReranker())
        cands = [_r("a"), _r("b")]
        results = rr.rerank("q", cands)
        assert all(x.metadata["rerank_fallback"] is True for x in results)
        assert [x.chunk_id for x in results] == ["a", "b"]

    def test_fallback_does_not_break_query(self):
        rr = Reranker(backend=FailingReranker())
        results = rr.rerank("q", [_r("a")])
        assert len(results) == 1


class TestNoneBackend:
    def test_none_reranker_identity(self):
        rr = Reranker(backend=NoneReranker())
        cands = [_r("a"), _r("b"), _r("c")]
        results = rr.rerank("q", cands)
        assert [x.chunk_id for x in results] == ["a", "b", "c"]

    def test_none_not_marked_fallback(self):
        rr = Reranker(backend=NoneReranker())
        results = rr.rerank("q", [_r("a")])
        assert results[0].metadata["rerank_fallback"] is False
        assert results[0].metadata["rerank_backend"] == "none"


class TestEdges:
    def test_empty_candidates(self):
        rr = Reranker(backend=ReverseReranker())
        assert rr.rerank("q", []) == []

    def test_default_backend_is_none(self):
        rr = Reranker()
        results = rr.rerank("q", [_r("a")])
        assert results[0].metadata["rerank_backend"] == "none"


class TestTrace:
    def test_trace_success(self):
        rr = Reranker(backend=ReverseReranker())
        trace = TraceContext(trace_type="query")
        rr.rerank("q", [_r("a"), _r("b")], trace=trace)
        stages = {s.name: s for s in trace.stages}
        assert "rerank" in stages
        assert stages["rerank"].details["fallback"] is False

    def test_trace_fallback(self):
        rr = Reranker(backend=FailingReranker())
        trace = TraceContext(trace_type="query")
        rr.rerank("q", [_r("a")], trace=trace)
        stages = {s.name: s for s in trace.stages}
        assert stages["rerank"].details["fallback"] is True

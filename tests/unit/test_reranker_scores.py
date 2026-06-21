"""Tests for T9: Reranker head/tail score-scale unification.

Covers:
- Merged list (reranked head + fusion tail) is monotonically non-increasing.
- Head segment is placed before the tail segment.
- Original scores preserved in metadata (raw_score + score_source).
- Fallback / NoneReranker paths stay on a single (already monotonic) scale.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""
from __future__ import annotations

import pytest

from src.core.query_engine.reranker import Reranker
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker


def _r(cid, score=0.0, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=score, text=text or cid, metadata=meta or {})


class SortedScoreReranker(BaseReranker):
    """Realistic backend: returns candidates with descending relevance scores."""

    def __init__(self, name="cross_encoder"):
        self._name = name

    def rerank(self, query, candidates, trace=None):
        # Assign descending scores in the given order (simulate sorted output).
        n = len(candidates)
        for i, c in enumerate(candidates):
            c.score = float(n - i)  # n, n-1, ... 1
        return list(candidates)

    @property
    def backend_name(self):
        return self._name


def _is_non_increasing(values: list[float]) -> bool:
    return all(values[i] >= values[i + 1] for i in range(len(values) - 1))


@pytest.mark.unit
class TestScoreUnification:
    def test_merged_list_is_monotonic(self):
        rr = Reranker(backend=SortedScoreReranker(), top_m=2)
        # tail carries RRF-style fusion scores (could even exceed head numerically)
        cands = [_r("a", 0.1), _r("b", 0.1), _r("c", 0.9), _r("d", 0.8)]
        results = rr.rerank("q", cands)
        scores = [r.score for r in results]
        assert _is_non_increasing(scores)

    def test_head_before_tail(self):
        rr = Reranker(backend=SortedScoreReranker(), top_m=2)
        cands = [_r("a"), _r("b"), _r("c"), _r("d")]
        results = rr.rerank("q", cands)
        # head = first 2 (reranked, order preserved), tail = c, d
        assert [r.chunk_id for r in results] == ["a", "b", "c", "d"]

    def test_tail_strictly_below_head_min(self):
        rr = Reranker(backend=SortedScoreReranker(), top_m=2)
        cands = [_r("a"), _r("b"), _r("c"), _r("d")]
        results = rr.rerank("q", cands)
        head_scores = [r.score for r in results[:2]]
        tail_scores = [r.score for r in results[2:]]
        assert min(head_scores) > max(tail_scores)

    def test_raw_score_and_source_recorded(self):
        rr = Reranker(backend=SortedScoreReranker(name="cross_encoder"), top_m=2)
        cands = [_r("a"), _r("b"), _r("c", 0.5), _r("d", 0.4)]
        results = rr.rerank("q", cands)
        head, tail = results[:2], results[2:]
        for r in head:
            assert r.metadata["score_source"] == "cross_encoder"
            assert "raw_score" in r.metadata
        for r in tail:
            assert r.metadata["score_source"] == "rrf"
            assert "raw_score" in r.metadata
        # tail raw_score keeps the original fusion score
        assert tail[0].metadata["raw_score"] == pytest.approx(0.5)
        assert tail[1].metadata["raw_score"] == pytest.approx(0.4)

    def test_no_tail_keeps_head_scores(self):
        rr = Reranker(backend=SortedScoreReranker(), top_m=30)
        cands = [_r("a"), _r("b"), _r("c")]
        results = rr.rerank("q", cands)
        # all in head -> descending backend scores 3,2,1
        assert [r.score for r in results] == [3.0, 2.0, 1.0]
        assert _is_non_increasing([r.score for r in results])


@pytest.mark.unit
class TestMonotonicOnNonRerankPaths:
    def test_none_reranker_scores_monotonic(self):
        rr = Reranker(backend=NoneReranker())
        cands = [_r("a", 0.9), _r("b", 0.5), _r("c", 0.1)]
        results = rr.rerank("q", cands)
        assert _is_non_increasing([r.score for r in results])

    def test_fallback_preserves_fusion_scores(self):
        class Boom(BaseReranker):
            def rerank(self, query, candidates, trace=None):
                raise RuntimeError("boom")

            @property
            def backend_name(self):
                return "boom"

        rr = Reranker(backend=Boom())
        cands = [_r("a", 0.9), _r("b", 0.5), _r("c", 0.1)]
        results = rr.rerank("q", cands)
        assert [r.score for r in results] == [0.9, 0.5, 0.1]
        assert all(r.metadata["rerank_fallback"] is True for r in results)

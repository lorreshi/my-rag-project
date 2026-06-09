"""Unit tests for ReciprocalRankFusion (RRF)."""
from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.core.query_engine.fusion import ReciprocalRankFusion


def _r(cid, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=0.0, text=text, metadata=meta or {})


class TestFusion:
    def test_single_list_preserves_order(self):
        rrf = ReciprocalRankFusion(k=60)
        lst = [_r("a"), _r("b"), _r("c")]
        fused = rrf.fuse([lst])
        assert [x.chunk_id for x in fused] == ["a", "b", "c"]

    def test_doc_in_both_lists_ranks_higher(self):
        rrf = ReciprocalRankFusion(k=60)
        dense = [_r("a"), _r("b"), _r("c")]
        sparse = [_r("b"), _r("d"), _r("a")]
        fused = rrf.fuse([dense, sparse])
        # 'a' (ranks 1 & 3) and 'b' (ranks 2 & 1) appear in both -> top 2
        top_two = {fused[0].chunk_id, fused[1].chunk_id}
        assert top_two == {"a", "b"}

    def test_rrf_score_computed(self):
        rrf = ReciprocalRankFusion(k=60)
        fused = rrf.fuse([[_r("a")]])
        # single list, rank 1 -> 1/(60+1)
        assert fused[0].score == pytest.approx(1.0 / 61)

    def test_score_accumulates_across_lists(self):
        rrf = ReciprocalRankFusion(k=60)
        fused = rrf.fuse([[_r("a")], [_r("a")]])
        # a at rank 1 in both -> 2 * 1/61
        assert fused[0].score == pytest.approx(2.0 / 61)

    def test_deterministic(self):
        rrf = ReciprocalRankFusion(k=60)
        dense = [_r("a"), _r("b"), _r("c")]
        sparse = [_r("c"), _r("b"), _r("a")]
        r1 = rrf.fuse([dense, sparse])
        r2 = rrf.fuse([dense, sparse])
        assert [x.chunk_id for x in r1] == [x.chunk_id for x in r2]
        assert [x.score for x in r1] == [x.score for x in r2]

    def test_tie_broken_by_chunk_id(self):
        rrf = ReciprocalRankFusion(k=60)
        # symmetric -> all same score; order by chunk_id
        dense = [_r("b"), _r("a")]
        sparse = [_r("a"), _r("b")]
        fused = rrf.fuse([dense, sparse])
        assert [x.chunk_id for x in fused] == ["a", "b"]

    def test_k_configurable(self):
        small_k = ReciprocalRankFusion(k=1)
        big_k = ReciprocalRankFusion(k=1000)
        s = small_k.fuse([[_r("a")]])[0].score
        b = big_k.fuse([[_r("a")]])[0].score
        assert s > b  # smaller k -> larger contribution

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            ReciprocalRankFusion(k=0)

    def test_top_k_limit(self):
        rrf = ReciprocalRankFusion(k=60)
        lst = [_r(c) for c in "abcde"]
        fused = rrf.fuse([lst], top_k=2)
        assert len(fused) == 2

    def test_empty_lists(self):
        rrf = ReciprocalRankFusion(k=60)
        assert rrf.fuse([]) == []
        assert rrf.fuse([[], []]) == []

    def test_payload_preserved(self):
        rrf = ReciprocalRankFusion(k=60)
        dense = [_r("a", text="alpha text", meta={"source_path": "a.pdf"})]
        fused = rrf.fuse([dense])
        assert fused[0].text == "alpha text"
        assert fused[0].metadata["source_path"] == "a.pdf"

    def test_payload_from_first_occurrence(self):
        rrf = ReciprocalRankFusion(k=60)
        dense = [_r("a", text="from dense")]
        sparse = [_r("a", text="from sparse")]
        fused = rrf.fuse([dense, sparse])
        assert fused[0].text == "from dense"


class TestTrace:
    def test_trace_records_stage(self):
        rrf = ReciprocalRankFusion(k=60)
        trace = TraceContext(trace_type="query")
        rrf.fuse([[_r("a"), _r("b")], [_r("b")]], trace=trace)
        stage = trace.stages[0]
        assert stage.name == "fusion"
        assert stage.details["algorithm"] == "rrf"
        assert stage.details["num_lists"] == 2
        assert stage.details["k"] == 60

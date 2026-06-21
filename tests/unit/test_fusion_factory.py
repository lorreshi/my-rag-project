"""Tests for T4: WeightedSumFusion + FusionFactory.

Validates: Requirements 4.1, 4.2, 4.7
"""
from __future__ import annotations

import pytest

from src.core.query_engine.fusion import (
    BaseFusion,
    ReciprocalRankFusion,
    WeightedSumFusion,
)
from src.core.query_engine.fusion_factory import FusionFactory
from src.core.settings import RetrievalConfig, Settings
from src.core.types import RetrievalResult


def _r(cid, score=0.0):
    return RetrievalResult(chunk_id=cid, score=score, text=cid, metadata={})


def _settings(algorithm="rrf", weights=None, rrf_k=60):
    s = Settings()
    s.retrieval = RetrievalConfig(
        fusion_algorithm=algorithm,
        fusion_weights=weights or {},
        rrf_k=rrf_k,
    )
    return s


@pytest.mark.unit
class TestFusionFactory:
    def test_creates_rrf(self):
        fusion = FusionFactory.create(_settings("rrf"))
        assert isinstance(fusion, ReciprocalRankFusion)
        assert isinstance(fusion, BaseFusion)

    def test_creates_weighted_sum(self):
        fusion = FusionFactory.create(_settings("weighted_sum"))
        assert isinstance(fusion, WeightedSumFusion)

    def test_rrf_k_forwarded(self):
        fusion = FusionFactory.create(_settings("rrf", rrf_k=10))
        assert fusion._k == 10  # noqa: SLF001 (white-box check)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion_algorithm"):
            FusionFactory.create(_settings("does_not_exist"))

    def test_missing_retrieval_defaults_to_rrf(self):
        class _S:
            pass

        fusion = FusionFactory.create(_S())  # type: ignore[arg-type]
        assert isinstance(fusion, ReciprocalRankFusion)

    def test_case_insensitive(self):
        fusion = FusionFactory.create(_settings("WEIGHTED_SUM"))
        assert isinstance(fusion, WeightedSumFusion)


@pytest.mark.unit
class TestWeightedSumFusion:
    def test_single_list_all_equal_normalizes_to_one(self):
        # single item -> span 0 -> normalized 1.0
        out = WeightedSumFusion().fuse([[_r("a", 0.42)]])
        assert out[0].score == pytest.approx(1.0)

    def test_min_max_normalization(self):
        # scores 1,2,3 -> normalized 0, 0.5, 1.0
        lst = [_r("a", 3.0), _r("b", 2.0), _r("c", 1.0)]
        out = {r.chunk_id: r.score for r in WeightedSumFusion().fuse([lst])}
        assert out["a"] == pytest.approx(1.0)
        assert out["b"] == pytest.approx(0.5)
        assert out["c"] == pytest.approx(0.0)

    def test_weighted_combination_across_lists(self):
        dense = [_r("x", 10.0), _r("y", 0.0)]   # norm: x=1, y=0
        sparse = [_r("y", 5.0), _r("x", 1.0)]   # norm: y=1, x=0
        fusion = WeightedSumFusion(weights={"dense": 2.0, "sparse": 1.0})
        out = {r.chunk_id: r.score for r in fusion.fuse([dense, sparse])}
        # x = 2*1 + 1*0 = 2 ; y = 2*0 + 1*1 = 1
        assert out["x"] == pytest.approx(2.0)
        assert out["y"] == pytest.approx(1.0)

    def test_order_invariant(self):
        dense = [_r("x", 10.0), _r("y", 1.0)]
        sparse = [_r("y", 5.0), _r("z", 2.0)]
        fusion = WeightedSumFusion()
        a = fusion.fuse([dense, sparse])
        b = fusion.fuse([sparse, dense])
        assert {r.chunk_id: round(r.score, 9) for r in a} == {
            r.chunk_id: round(r.score, 9) for r in b
        }

    def test_tie_broken_by_chunk_id(self):
        # symmetric single-item lists -> both normalize to 1.0 (equal score)
        out = WeightedSumFusion().fuse([[_r("b", 1.0)], [_r("a", 1.0)]])
        assert [r.chunk_id for r in out] == ["a", "b"]

    def test_empty(self):
        assert WeightedSumFusion().fuse([]) == []
        assert WeightedSumFusion().fuse([[], []]) == []

    def test_top_k(self):
        lst = [_r(c, float(i)) for i, c in enumerate("abcde")]
        out = WeightedSumFusion().fuse([lst], top_k=2)
        assert len(out) == 2

    def test_payload_preserved_first_occurrence(self):
        dense = [RetrievalResult(chunk_id="a", score=1.0, text="from dense", metadata={"s": 1})]
        sparse = [RetrievalResult(chunk_id="a", score=1.0, text="from sparse", metadata={"s": 2})]
        out = WeightedSumFusion().fuse([dense, sparse])
        assert out[0].text == "from dense"
        assert out[0].metadata["s"] == 1

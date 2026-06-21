"""Tests for T3: BaseFusion abstraction + weighted RRF.

Covers:
- BaseFusion is the abstract base; ReciprocalRankFusion implements it.
- Backward compatibility: no weights / equal weights == plain RRF (Property 3).
- Fusion is order-invariant across input lists (Property 13).
- Per-route weighting (dict by route name, positional sequence) changes scores
  by the documented formula: contribution = w_i / (k + rank).

Validates: Requirements 4.2, 4.4, 4.5, 4.6, 4.8
"""
from __future__ import annotations

import pytest

from src.core.query_engine.fusion import BaseFusion, ReciprocalRankFusion
from src.core.types import RetrievalResult


def _r(cid: str, text: str = "", meta: dict | None = None) -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, score=0.0, text=text, metadata=meta or {})


@pytest.mark.unit
class TestBaseFusionContract:
    def test_rrf_is_base_fusion(self):
        assert isinstance(ReciprocalRankFusion(), BaseFusion)

    def test_base_fusion_is_abstract(self):
        with pytest.raises(TypeError):
            BaseFusion()  # type: ignore[abstract]


@pytest.mark.unit
class TestBackwardCompatibility:
    """Property 3: weights=None or equal weights == plain RRF, item by item."""

    def _lists(self):
        dense = [_r("a"), _r("b"), _r("c")]
        sparse = [_r("b"), _r("d"), _r("a")]
        return [dense, sparse]

    def test_no_weights_matches_plain_rrf(self):
        plain = ReciprocalRankFusion(k=60)
        out = plain.fuse(self._lists())
        # Compare against the explicit RRF formula.
        expected_scores = {}
        for results in self._lists():
            for rank, item in enumerate(results, start=1):
                expected_scores[item.chunk_id] = (
                    expected_scores.get(item.chunk_id, 0.0) + 1.0 / (60 + rank)
                )
        for item in out:
            assert item.score == pytest.approx(expected_scores[item.chunk_id])

    def test_equal_dict_weights_identical_to_unweighted(self):
        unweighted = ReciprocalRankFusion(k=60)
        equal = ReciprocalRankFusion(k=60, weights={"dense": 1.0, "sparse": 1.0})
        a = unweighted.fuse(self._lists())
        b = equal.fuse(self._lists())
        assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
        for x, y in zip(a, b):
            assert x.score == pytest.approx(y.score)

    def test_equal_positional_weights_identical(self):
        unweighted = ReciprocalRankFusion(k=60)
        equal = ReciprocalRankFusion(k=60, weights=[1.0, 1.0])
        a = unweighted.fuse(self._lists())
        b = equal.fuse(self._lists())
        for x, y in zip(a, b):
            assert x.chunk_id == y.chunk_id
            assert x.score == pytest.approx(y.score)


@pytest.mark.unit
class TestOrderInvariance:
    """Property 13: fused result is independent of input-list order."""

    def test_list_order_does_not_change_result(self):
        dense = [_r("a"), _r("b"), _r("c")]
        sparse = [_r("c"), _r("d"), _r("a")]
        rrf = ReciprocalRankFusion(k=60)
        forward = rrf.fuse([dense, sparse])
        backward = rrf.fuse([sparse, dense])
        assert [x.chunk_id for x in forward] == [x.chunk_id for x in backward]
        fwd = {x.chunk_id: x.score for x in forward}
        bwd = {x.chunk_id: x.score for x in backward}
        for cid in fwd:
            assert fwd[cid] == pytest.approx(bwd[cid])

    def test_single_dense_list_equiv_to_baseline_two_list_when_sparse_empty(self):
        """N=1 dense list + empty sparse == baseline single-list fusion."""
        dense = [_r("a"), _r("b")]
        rrf = ReciprocalRankFusion(k=60)
        one = rrf.fuse([dense])
        with_empty = rrf.fuse([dense, []])
        assert [x.chunk_id for x in one] == [x.chunk_id for x in with_empty]
        for x, y in zip(one, with_empty):
            assert x.score == pytest.approx(y.score)


@pytest.mark.unit
class TestWeightingBehaviour:
    def test_dict_weight_scales_route_contribution(self):
        # 'x' only in dense (rank 1), 'y' only in sparse (rank 1).
        dense = [_r("x")]
        sparse = [_r("y")]
        weighted = ReciprocalRankFusion(k=60, weights={"dense": 2.0, "sparse": 1.0})
        out = {r.chunk_id: r.score for r in weighted.fuse([dense, sparse])}
        assert out["x"] == pytest.approx(2.0 / 61)
        assert out["y"] == pytest.approx(1.0 / 61)

    def test_higher_dense_weight_promotes_dense_only_doc(self):
        # 'x' dense rank 1; 'y' sparse rank 1. Boosting dense ranks 'x' first.
        dense = [_r("x")]
        sparse = [_r("y")]
        weighted = ReciprocalRankFusion(k=60, weights={"dense": 5.0, "sparse": 1.0})
        out = weighted.fuse([dense, sparse])
        assert out[0].chunk_id == "x"

    def test_positional_weights_map_by_index(self):
        dense = [_r("x")]
        sparse = [_r("y")]
        weighted = ReciprocalRankFusion(k=60, weights=[3.0, 1.0])
        out = {r.chunk_id: r.score for r in weighted.fuse([dense, sparse])}
        assert out["x"] == pytest.approx(3.0 / 61)
        assert out["y"] == pytest.approx(1.0 / 61)

    def test_missing_route_name_defaults_to_one(self):
        # Only 'dense' weighted; extra list index falls back to weight 1.0.
        dense = [_r("x")]
        sparse = [_r("y")]
        weighted = ReciprocalRankFusion(k=60, weights={"dense": 2.0})
        out = {r.chunk_id: r.score for r in weighted.fuse([dense, sparse])}
        assert out["x"] == pytest.approx(2.0 / 61)
        assert out["y"] == pytest.approx(1.0 / 61)  # default weight

    def test_invalid_k_still_rejected(self):
        with pytest.raises(ValueError):
            ReciprocalRankFusion(k=0)

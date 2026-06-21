"""Tests for T13: MMR diversity re-ranking.

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
"""
from __future__ import annotations

import pytest

from src.core.query_engine.diversity import apply_mmr, mmr_rerank
from src.core.types import RetrievalResult


def _r(cid, text=""):
    return RetrievalResult(chunk_id=cid, score=1.0, text=text or cid, metadata={})


@pytest.mark.unit
class TestMMRRerank:
    def test_lambda_one_is_identity(self):
        results = [_r("a"), _r("b"), _r("c")]
        vectors = {"a": [1, 0], "b": [1, 0], "c": [0, 1]}  # a,b identical
        out = mmr_rerank(results, [1, 0], vectors, lambda_=1.0)
        assert [r.chunk_id for r in out] == ["a", "b", "c"]

    def test_redundancy_suppressed(self):
        # a and b identical vectors; c is different. With diversity weight, the
        # second pick should be the diverse 'c', not the near-duplicate 'b'.
        results = [_r("a"), _r("b"), _r("c")]
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
        out = mmr_rerank(results, [1.0, 0.5], vectors, lambda_=0.5)
        assert out[0].chunk_id == "a"
        assert out[1].chunk_id == "c"  # diverse pick beats near-duplicate b

    def test_missing_vector_degrades_to_order(self):
        results = [_r("a"), _r("b")]
        out = mmr_rerank(results, [1, 0], {"a": [1, 0]}, lambda_=0.5)  # b missing
        assert [r.chunk_id for r in out] == ["a", "b"]

    def test_empty(self):
        assert mmr_rerank([], [1, 0], {}, lambda_=0.5) == []

    def test_top_k_cap(self):
        results = [_r(c) for c in "abcd"]
        vectors = {c: [i, 1] for i, c in enumerate("abcd")}
        out = mmr_rerank(results, [1, 1], vectors, lambda_=0.5, top_k=2)
        assert len(out) == 2


@pytest.mark.unit
class TestApplyMMR:
    def test_uses_metadata_vector_when_present(self):
        a = RetrievalResult(chunk_id="a", score=1.0, text="a", metadata={"dense_vector": [1.0, 0.0]})
        b = RetrievalResult(chunk_id="b", score=1.0, text="b", metadata={"dense_vector": [0.0, 1.0]})

        def boom_embed(texts):
            # only the query needs embedding here; texts == [query]
            assert texts == ["q"]
            return [[1.0, 0.1]]

        out = apply_mmr([a, b], "q", embed_fn=boom_embed, lambda_=0.5)
        assert {r.chunk_id for r in out} == {"a", "b"}

    def test_embeds_missing_texts(self):
        a = _r("a", text="alpha")
        b = _r("b", text="beta")

        def embed(texts):
            # [query, alpha, beta]
            mapping = {"q": [1.0, 0.0], "alpha": [1.0, 0.0], "beta": [0.0, 1.0]}
            return [mapping[t] for t in texts]

        out = apply_mmr([a, b], "q", embed_fn=embed, lambda_=0.5)
        assert [r.chunk_id for r in out][0] == "a"

    def test_embed_failure_degrades(self):
        def boom(texts):
            raise RuntimeError("embed down")

        results = [_r("a"), _r("b")]
        out = apply_mmr(results, "q", embed_fn=boom, lambda_=0.5)
        assert [r.chunk_id for r in out] == ["a", "b"]

    def test_empty_results(self):
        assert apply_mmr([], "q", embed_fn=lambda t: [], lambda_=0.5) == []

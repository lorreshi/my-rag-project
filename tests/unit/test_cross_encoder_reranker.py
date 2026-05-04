"""Tests for Cross-Encoder Reranker (B7.8). Uses mock scorer."""
from __future__ import annotations

import pytest

from src.libs.reranker.base_reranker import RerankCandidate
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.reranker_factory import RerankerFactory
from src.core.settings import Settings, RerankConfig

# Trigger registration
import src.libs.reranker.cross_encoder_reranker as mod  # noqa: F401


def _candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="first doc", score=0.0),
        RerankCandidate(id="b", text="second doc", score=0.0),
        RerankCandidate(id="c", text="third doc", score=0.0),
    ]


def _deterministic_scorer(query: str, text: str) -> float:
    """Score = position of first char of text in alphabet / 26."""
    return ord(text[0].lower()) / 26.0 if text else 0.0


@pytest.mark.unit
class TestCrossEncoderReranker:

    def test_rerank_sorts_by_score(self):
        # 't' > 's' > 'f' in alphabet, so third > second > first
        rr = CrossEncoderReranker(scorer=_deterministic_scorer)
        result = rr.rerank("query", _candidates())
        assert [c.id for c in result] == ["c", "b", "a"]

    def test_scores_from_scorer(self):
        rr = CrossEncoderReranker(scorer=lambda q, t: len(t) / 100.0)
        result = rr.rerank("query", _candidates())
        for c in result:
            assert c.score == pytest.approx(len(c.text) / 100.0)

    def test_empty_candidates(self):
        rr = CrossEncoderReranker(scorer=_deterministic_scorer)
        assert rr.rerank("query", []) == []

    def test_fallback_on_scorer_error(self):
        def failing_scorer(q, t):
            raise RuntimeError("model crashed")

        rr = CrossEncoderReranker(scorer=failing_scorer)
        result = rr.rerank("query", _candidates())
        # Should return original order
        assert [c.id for c in result] == ["a", "b", "c"]
        assert rr.has_failed is True

    def test_has_failed_false_on_success(self):
        rr = CrossEncoderReranker(scorer=_deterministic_scorer)
        rr.rerank("query", _candidates())
        assert rr.has_failed is False

    def test_has_failed_resets_between_calls(self):
        call_count = 0

        def sometimes_fails(q, t):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise RuntimeError("fail")
            return 0.5

        rr = CrossEncoderReranker(scorer=sometimes_fails)
        # First call: scorer fails on call_count=1
        rr.rerank("query", _candidates())
        assert rr.has_failed is True

        # Reset counter so second call fully succeeds
        call_count = 100
        rr.rerank("query", _candidates())
        assert rr.has_failed is False

    def test_backend_name(self):
        rr = CrossEncoderReranker(scorer=_deterministic_scorer)
        assert rr.backend_name == "cross_encoder"

    def test_single_candidate(self):
        rr = CrossEncoderReranker(scorer=_deterministic_scorer)
        cands = [RerankCandidate(id="x", text="only one", score=0.0)]
        result = rr.rerank("query", cands)
        assert len(result) == 1
        assert result[0].id == "x"


@pytest.mark.unit
class TestCrossEncoderFactory:

    def test_factory_creates_cross_encoder(self):
        settings = Settings(rerank=RerankConfig(backend="cross_encoder"))
        rr = RerankerFactory.create(settings)
        assert isinstance(rr, CrossEncoderReranker)
        assert rr.backend_name == "cross_encoder"

    def test_factory_fallback_scorer_works(self):
        """Without sentence-transformers installed, factory uses word-overlap scorer."""
        settings = Settings(rerank=RerankConfig(backend="cross_encoder"))
        rr = RerankerFactory.create(settings)
        cands = [
            RerankCandidate(id="a", text="python programming language", score=0.0),
            RerankCandidate(id="b", text="java coffee beans", score=0.0),
        ]
        result = rr.rerank("python programming", cands)
        # "a" has more word overlap with query
        assert result[0].id == "a"

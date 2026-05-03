"""Tests for Reranker abstract interface, NoneReranker, and factory (B5)."""

import pytest

from src.libs.reranker.base_reranker import (
    BaseReranker,
    NoneReranker,
    RerankCandidate,
)
from src.libs.reranker.reranker_factory import (
    RerankerFactory,
    register_backend,
    _REGISTRY,
)
from src.core.settings import Settings, RerankConfig


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    # Always keep 'none' registered
    _REGISTRY["none"] = lambda _s: NoneReranker()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="first", score=0.9),
        RerankCandidate(id="b", text="second", score=0.8),
        RerankCandidate(id="c", text="third", score=0.7),
    ]


@pytest.mark.unit
class TestBaseRerankerInterface:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseReranker()


@pytest.mark.unit
class TestNoneReranker:

    def test_preserves_order(self):
        cands = _candidates()
        result = NoneReranker().rerank("query", cands)
        assert [c.id for c in result] == ["a", "b", "c"]

    def test_preserves_scores(self):
        cands = _candidates()
        result = NoneReranker().rerank("query", cands)
        assert [c.score for c in result] == [0.9, 0.8, 0.7]


    def test_empty_candidates(self):
        assert NoneReranker().rerank("query", []) == []

    def test_returns_copy(self):
        cands = _candidates()
        result = NoneReranker().rerank("query", cands)
        assert result is not cands  # should be a new list

    def test_backend_name(self):
        assert NoneReranker().backend_name == "none"


@pytest.mark.unit
class TestRerankerFactory:

    def test_create_none_backend(self):
        rr = RerankerFactory.create(Settings(rerank=RerankConfig(backend="none")))
        assert isinstance(rr, NoneReranker)

    def test_create_case_insensitive(self):
        rr = RerankerFactory.create(Settings(rerank=RerankConfig(backend="NONE")))
        assert isinstance(rr, NoneReranker)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker backend 'nope'"):
            RerankerFactory.create(Settings(rerank=RerankConfig(backend="nope")))

    def test_register_custom_backend(self):
        class FakeReranker(BaseReranker):
            def rerank(self, query, candidates, trace=None):
                return list(reversed(candidates))

            @property
            def backend_name(self):
                return "fake"

        register_backend("fake", lambda s: FakeReranker())
        rr = RerankerFactory.create(Settings(rerank=RerankConfig(backend="fake")))
        result = rr.rerank("q", _candidates())
        assert [c.id for c in result] == ["c", "b", "a"]

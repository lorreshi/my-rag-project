"""Tests for Embedding abstract interface and factory (B2)."""

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import (
    EmbeddingFactory,
    register_provider,
    _REGISTRY,
)
from src.core.settings import Settings, EmbeddingConfig


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------

class FakeEmbedding(BaseEmbedding):
    """Deterministic stub that returns fixed-dimension vectors."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    def embed(self, texts, trace=None):
        # Return a stable vector: [len(text)/100, 0.1, 0.2, ...]
        return [
            [len(t) / 100.0] + [i * 0.1 for i in range(1, self._dim)]
            for t in texts
        ]

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _make_settings(provider: str = "fake", model: str = "fake-embed") -> Settings:
    return Settings(embedding=EmbeddingConfig(provider=provider, model=model))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaseEmbeddingInterface:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseEmbedding()  # type: ignore[abstract]

    def test_fake_embed_returns_correct_shape(self):
        emb = FakeEmbedding(dim=4)
        vecs = emb.embed(["hello", "world"])
        assert len(vecs) == 2
        assert all(len(v) == 4 for v in vecs)

    def test_fake_embed_deterministic(self):
        emb = FakeEmbedding(dim=3)
        v1 = emb.embed(["test"])
        v2 = emb.embed(["test"])
        assert v1 == v2

    def test_fake_embed_empty_input(self):
        emb = FakeEmbedding()
        assert emb.embed([]) == []

    def test_fake_provider_name(self):
        assert FakeEmbedding().provider_name == "fake"

    def test_fake_dimension(self):
        assert FakeEmbedding(dim=8).dimension == 8


@pytest.mark.unit
class TestEmbeddingFactory:

    def test_create_registered_provider(self):
        register_provider("fake", lambda s: FakeEmbedding())
        emb = EmbeddingFactory.create(_make_settings("fake"))
        assert isinstance(emb, FakeEmbedding)

    def test_create_case_insensitive(self):
        register_provider("fake", lambda s: FakeEmbedding())
        emb = EmbeddingFactory.create(_make_settings("FAKE"))
        assert isinstance(emb, FakeEmbedding)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider 'nope'"):
            EmbeddingFactory.create(_make_settings("nope"))

    def test_unknown_provider_lists_available(self):
        register_provider("alpha", lambda s: FakeEmbedding())
        register_provider("beta", lambda s: FakeEmbedding())
        with pytest.raises(ValueError, match="alpha, beta"):
            EmbeddingFactory.create(_make_settings("nope"))

    def test_factory_passes_settings(self):
        def _build(s):
            # Use model name length as dimension for testing
            return FakeEmbedding(dim=len(s.embedding.model))

        register_provider("fake", _build)
        emb = EmbeddingFactory.create(_make_settings("fake", "abc"))
        assert emb.dimension == 3

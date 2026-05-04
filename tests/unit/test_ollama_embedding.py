"""Tests for Ollama Embedding provider (B7.4). All HTTP calls are mocked."""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.core.settings import Settings, EmbeddingConfig

# Trigger registration
import src.libs.embedding.ollama_embedding  # noqa: F401


def _mock_response(vectors: list[list[float]]):
    data = {"embeddings": vectors, "model": "nomic-embed-text"}
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _settings(**overrides) -> Settings:
    defaults = dict(provider="ollama", model="nomic-embed-text")
    defaults.update(overrides)
    return Settings(embedding=EmbeddingConfig(**defaults))


@pytest.mark.unit
class TestOllamaEmbedding:

    @patch("src.libs.embedding.ollama_embedding.urlopen")
    def test_embed_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([[0.1, 0.2], [0.3, 0.4]])
        emb = EmbeddingFactory.create(_settings())
        vecs = emb.embed(["hello", "world"])
        assert len(vecs) == 2
        assert vecs[0] == [0.1, 0.2]

    @patch("src.libs.embedding.ollama_embedding.urlopen")
    def test_empty_input(self, mock_urlopen):
        emb = EmbeddingFactory.create(_settings())
        assert emb.embed([]) == []
        mock_urlopen.assert_not_called()

    @patch("src.libs.embedding.ollama_embedding.urlopen")
    def test_dimension_cached(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([[0.1, 0.2, 0.3, 0.4]])
        emb = EmbeddingFactory.create(_settings())
        assert emb.dimension == 0
        emb.embed(["test"])
        assert emb.dimension == 4

    @patch("src.libs.embedding.ollama_embedding.urlopen")
    def test_provider_name(self, mock_urlopen):
        emb = EmbeddingFactory.create(_settings())
        assert emb.provider_name == "ollama"

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            EmbeddingFactory.create(_settings(model=""))

    @patch("src.libs.embedding.ollama_embedding.urlopen")
    def test_connection_error(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        emb = EmbeddingFactory.create(_settings())
        with pytest.raises(RuntimeError, match=r"\[ollama-embed\] Connection failed"):
            emb.embed(["test"])

    @patch("src.libs.embedding.ollama_embedding.urlopen")
    def test_custom_base_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([[0.1]])
        emb = EmbeddingFactory.create(_settings(base_url="http://myhost:9999"))
        emb.embed(["test"])
        req = mock_urlopen.call_args[0][0]
        assert "myhost:9999" in req.full_url

    @patch("src.libs.embedding.ollama_embedding.urlopen")
    def test_http_error(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=None, fp=None
        )
        emb = EmbeddingFactory.create(_settings())
        with pytest.raises(RuntimeError, match=r"\[ollama-embed\] HTTP 404"):
            emb.embed(["test"])

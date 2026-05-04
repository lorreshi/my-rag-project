"""Smoke tests for OpenAI / Azure Embedding providers (B7.3).
All HTTP calls are mocked.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.core.settings import Settings, EmbeddingConfig

# Trigger registrations
import src.libs.embedding.openai_embedding  # noqa: F401
import src.libs.embedding.azure_embedding   # noqa: F401


def _mock_response(vectors: list[list[float]]):
    data = {
        "data": [
            {"index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _settings(provider: str, **overrides) -> Settings:
    defaults = dict(
        provider=provider,
        model="text-embedding-3-small",
        api_key="sk-test",
        azure_endpoint="https://my.openai.azure.com",
    )
    defaults.update(overrides)
    return Settings(embedding=EmbeddingConfig(**defaults))


@pytest.mark.unit
class TestOpenAIEmbedding:

    @patch("src.libs.embedding.openai_embedding.urlopen")
    def test_embed_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([[0.1, 0.2], [0.3, 0.4]])
        emb = EmbeddingFactory.create(_settings("openai"))
        vecs = emb.embed(["hello", "world"])
        assert len(vecs) == 2
        assert vecs[0] == [0.1, 0.2]

    @patch("src.libs.embedding.openai_embedding.urlopen")
    def test_empty_input(self, mock_urlopen):
        emb = EmbeddingFactory.create(_settings("openai"))
        assert emb.embed([]) == []
        mock_urlopen.assert_not_called()

    @patch("src.libs.embedding.openai_embedding.urlopen")
    def test_dimension_cached(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([[0.1, 0.2, 0.3]])
        emb = EmbeddingFactory.create(_settings("openai"))
        assert emb.dimension == 0  # unknown before first call
        emb.embed(["test"])
        assert emb.dimension == 3

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            EmbeddingFactory.create(_settings("openai", api_key=""))

    @patch("src.libs.embedding.openai_embedding.urlopen")
    def test_provider_name(self, mock_urlopen):
        emb = EmbeddingFactory.create(_settings("openai"))
        assert emb.provider_name == "openai"

    @patch("src.libs.embedding.openai_embedding.urlopen")
    def test_http_error(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="http://x", code=401, msg="Unauthorized", hdrs=None, fp=None
        )
        emb = EmbeddingFactory.create(_settings("openai"))
        with pytest.raises(RuntimeError, match=r"\[openai\] HTTP 401"):
            emb.embed(["test"])


@pytest.mark.unit
class TestAzureEmbedding:

    @patch("src.libs.embedding.openai_embedding.urlopen")
    def test_embed_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([[0.5, 0.6]])
        emb = EmbeddingFactory.create(_settings("azure"))
        vecs = emb.embed(["hello"])
        assert vecs == [[0.5, 0.6]]

    @patch("src.libs.embedding.openai_embedding.urlopen")
    def test_provider_name(self, mock_urlopen):
        emb = EmbeddingFactory.create(_settings("azure"))
        assert emb.provider_name == "azure"

    def test_empty_endpoint_raises(self):
        with pytest.raises(ValueError, match="azure_endpoint"):
            EmbeddingFactory.create(_settings("azure", azure_endpoint=""))

    @patch("src.libs.embedding.openai_embedding.urlopen")
    def test_url_contains_deployment(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([[0.1]])
        emb = EmbeddingFactory.create(_settings("azure", model="my-deploy"))
        emb.embed(["test"])
        req = mock_urlopen.call_args[0][0]
        assert "my-deploy" in req.full_url
        assert "api-version=" in req.full_url

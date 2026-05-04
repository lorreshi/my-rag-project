"""Azure OpenAI Embedding implementation.

Reuses OpenAIEmbedding core logic, overrides URL and auth header.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.embedding_factory import register_provider


class AzureEmbedding(OpenAIEmbedding):
    """Azure OpenAI Embeddings provider."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        model: str = "text-embedding-ada-002",
        api_version: str = "2024-06-01",
    ):
        if not api_key:
            raise ValueError("Azure Embedding requires a non-empty api_key")
        if not azure_endpoint:
            raise ValueError("Azure Embedding requires a non-empty azure_endpoint")
        # Don't call super().__init__ with api_key validation since we handle it
        self._api_key = api_key
        self._model = model
        self._base_url = ""  # not used directly
        self._endpoint = azure_endpoint.rstrip("/")
        self._api_version = api_version
        self._dim: int | None = None

    def embed(self, texts, trace=None):
        if not texts:
            return []

        url = (
            f"{self._endpoint}/openai/deployments/{self._model}"
            f"/embeddings?api-version={self._api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }
        payload = {"input": texts}

        data = self._post(url, headers, payload)
        items = sorted(data["data"], key=lambda x: x["index"])
        vectors = [item["embedding"] for item in items]
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    @property
    def provider_name(self) -> str:
        return "azure"


def _create_azure(settings: Any) -> AzureEmbedding:
    cfg = settings.embedding
    return AzureEmbedding(
        api_key=cfg.api_key,
        azure_endpoint=cfg.azure_endpoint,
        model=cfg.model,
        api_version=cfg.api_version,
    )


register_provider("azure", _create_azure)

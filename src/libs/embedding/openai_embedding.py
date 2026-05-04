"""OpenAI Embedding implementation.

Uses the OpenAI Embeddings API. Serves as the base for Azure Embedding.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import register_provider


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embeddings provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
    ):
        if not api_key:
            raise ValueError("OpenAI Embedding requires a non-empty api_key")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dim: int | None = None  # cached after first call

    def embed(self, texts, trace=None):
        if not texts:
            return []

        url = f"{self._base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {"model": self._model, "input": texts}

        data = self._post(url, headers, payload)
        # Sort by index to guarantee order
        items = sorted(data["data"], key=lambda x: x["index"])
        vectors = [item["embedding"] for item in items]
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        """HTTP POST helper. Separated for subclass override."""
        try:
            body = json.dumps(payload).encode("utf-8")
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"[{self.provider_name}] HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"[{self.provider_name}] Connection failed: {exc.reason}"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        if self._dim is None:
            return 0  # unknown until first embed() call
        return self._dim


def _create_openai(settings: Any) -> OpenAIEmbedding:
    cfg = settings.embedding
    return OpenAIEmbedding(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url or "https://api.openai.com/v1",
    )


register_provider("openai", _create_openai)

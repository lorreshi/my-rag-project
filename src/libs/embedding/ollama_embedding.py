"""Ollama Embedding implementation.

Calls the local Ollama HTTP API for embedding generation.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import register_provider

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaEmbedding(BaseEmbedding):
    """Ollama local Embedding provider."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = _DEFAULT_BASE_URL,
    ):
        if not model:
            raise ValueError("Ollama Embedding requires a non-empty model name")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dim: int | None = None

    def embed(self, texts, trace=None):
        if not texts:
            return []

        # Ollama /api/embed supports batch input
        url = f"{self._base_url}/api/embed"
        headers = {"Content-Type": "application/json"}
        payload = {"model": self._model, "input": texts}

        try:
            body = json.dumps(payload).encode("utf-8")
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"[ollama-embed] HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"[ollama-embed] Connection failed — is Ollama running at "
                f"{self._base_url}? ({exc.reason})"
            ) from exc

        vectors = data.get("embeddings", [])
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def dimension(self) -> int:
        if self._dim is None:
            return 0
        return self._dim


def _create_ollama(settings: Any) -> OllamaEmbedding:
    cfg = settings.embedding
    return OllamaEmbedding(
        model=cfg.model,
        base_url=cfg.base_url or _DEFAULT_BASE_URL,
    )


register_provider("ollama", _create_ollama)

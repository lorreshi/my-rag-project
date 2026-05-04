"""Ollama LLM implementation.

Calls the local Ollama HTTP API (default http://localhost:11434).
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import register_provider

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaLLM(BaseLLM):
    """Ollama local Chat provider."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = _DEFAULT_BASE_URL,
        temperature: float = 0.0,
    ):
        if not model:
            raise ValueError("Ollama LLM requires a non-empty model name")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        if not messages:
            raise ValueError("messages must not be empty")

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        url = f"{self._base_url}/api/chat"
        headers = {"Content-Type": "application/json"}

        try:
            body = json.dumps(payload).encode("utf-8")
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"[ollama] HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"[ollama] Connection failed — is Ollama running at "
                f"{self._base_url}? ({exc.reason})"
            ) from exc

        message = data.get("message", {})
        content = message.get("content", "")
        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            usage={},
            raw=data,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"


def _create_ollama(settings: Any) -> OllamaLLM:
    cfg = settings.llm
    return OllamaLLM(
        model=cfg.model,
        base_url=cfg.base_url or _DEFAULT_BASE_URL,
        temperature=cfg.temperature,
    )


register_provider("ollama", _create_ollama)

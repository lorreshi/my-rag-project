"""OpenAI LLM implementation.

Uses the OpenAI Chat Completions API. Serves as the base for all
OpenAI-compatible providers (Azure, DeepSeek, etc.).
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import register_provider


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completions provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        if not api_key:
            raise ValueError("OpenAI LLM requires a non-empty api_key")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        if not messages:
            raise ValueError("messages must not be empty")

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"[{self.provider_name}] HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"[{self.provider_name}] Connection failed: {exc.reason}"
            ) from exc

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            usage=usage,
            raw=data,
        )

    @property
    def provider_name(self) -> str:
        return "openai"


def _create_openai(settings: Any) -> OpenAILLM:
    cfg = settings.llm
    return OpenAILLM(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url or "https://api.openai.com/v1",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


register_provider("openai", _create_openai)

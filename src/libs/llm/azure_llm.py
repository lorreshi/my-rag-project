"""Azure OpenAI LLM implementation.

Uses the Azure OpenAI Chat Completions API. Inherits core logic from
OpenAILLM but overrides URL construction and authentication header.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import register_provider


class AzureLLM(BaseLLM):
    """Azure OpenAI Chat Completions provider."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        model: str = "gpt-4o",
        api_version: str = "2024-06-01",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        if not api_key:
            raise ValueError("Azure LLM requires a non-empty api_key")
        if not azure_endpoint:
            raise ValueError("Azure LLM requires a non-empty azure_endpoint")
        self._api_key = api_key
        self._endpoint = azure_endpoint.rstrip("/")
        self._model = model
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        if not messages:
            raise ValueError("messages must not be empty")

        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        url = (
            f"{self._endpoint}/openai/deployments/{self._model}"
            f"/chat/completions?api-version={self._api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"[azure] HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"[azure] Connection failed: {exc.reason}"
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
        return "azure"


def _create_azure(settings: Any) -> AzureLLM:
    cfg = settings.llm
    return AzureLLM(
        api_key=cfg.api_key,
        azure_endpoint=cfg.azure_endpoint,
        model=cfg.model,
        api_version=cfg.api_version,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


register_provider("azure", _create_azure)

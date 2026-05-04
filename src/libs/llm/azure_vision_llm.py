"""Azure Vision LLM implementation (GPT-4o / GPT-4-Vision).

Sends multimodal requests (text + base64 image) to Azure OpenAI.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from src.libs.llm.base_llm import ChatResponse
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.llm_factory import register_vision_provider


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision provider."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        model: str = "gpt-4o",
        api_version: str = "2024-06-01",
        max_image_size: int = 2048,
    ):
        if not api_key:
            raise ValueError("Azure Vision LLM requires a non-empty api_key")
        if not azure_endpoint:
            raise ValueError("Azure Vision LLM requires a non-empty azure_endpoint")
        self._api_key = api_key
        self._endpoint = azure_endpoint.rstrip("/")
        self._model = model
        self._api_version = api_version
        self._max_image_size = max_image_size

    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        trace=None,
    ) -> ChatResponse:
        if not text:
            raise ValueError("text prompt must not be empty")

        image_b64 = self.image_to_base64(image)

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
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
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"[azure-vision] HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"[azure-vision] Connection failed: {exc.reason}"
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


def _create_azure_vision(settings: Any) -> AzureVisionLLM:
    cfg = settings.vision_llm
    return AzureVisionLLM(
        api_key=cfg.api_key,
        azure_endpoint=cfg.azure_endpoint,
        model=cfg.model or "gpt-4o",
        api_version=cfg.api_version,
        max_image_size=cfg.max_image_size,
    )


register_vision_provider("azure", _create_azure_vision)

"""DeepSeek LLM implementation.

DeepSeek uses an OpenAI-compatible API, so we reuse OpenAILLM with a
different default base_url and provider_name.
"""
from __future__ import annotations

from typing import Any

from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.llm_factory import register_provider

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekLLM(OpenAILLM):
    """DeepSeek Chat Completions provider (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = _DEEPSEEK_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"


def _create_deepseek(settings: Any) -> DeepSeekLLM:
    cfg = settings.llm
    return DeepSeekLLM(
        api_key=cfg.api_key,
        model=cfg.model or "deepseek-chat",
        base_url=cfg.base_url or _DEEPSEEK_BASE_URL,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


register_provider("deepseek", _create_deepseek)

"""LLM abstract base class.

All LLM providers must implement this interface so the rest of the codebase
can call ``llm.chat(messages)`` without knowing which backend is in use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResponse:
    """Unified response from any LLM provider."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None  # provider-specific raw response


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Send a chat completion request and return the response.

        Args:
            messages: Conversation history as a list of ChatMessage.

        Returns:
            ChatResponse with at least the ``content`` field populated.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable provider identifier (e.g. 'azure')."""
        ...

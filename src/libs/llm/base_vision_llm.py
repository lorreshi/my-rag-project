"""Vision LLM abstract base class.

Extends the LLM abstraction to support multimodal input (text + image).
Used by ImageCaptioner in the Ingestion Transform phase.
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from src.libs.llm.base_llm import ChatResponse

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


class BaseVisionLLM(ABC):
    """Abstract base class for Vision LLM providers."""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        trace: "TraceContext | None" = None,
    ) -> ChatResponse:
        """Send a multimodal request with text and an image.

        Args:
            text: The text prompt / instruction.
            image: Either a file path (str) or raw image bytes.
            trace: Optional trace context for observability.

        Returns:
            ChatResponse with the model's description / analysis.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable provider identifier."""
        ...

    @staticmethod
    def image_to_base64(image: str | bytes) -> str:
        """Convert an image path or bytes to a base64-encoded string."""
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")

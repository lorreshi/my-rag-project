"""ImageCaptioner — optional Vision-LLM image captioning with graceful degradation.

Strategy:
1. When a Vision LLM is enabled and a chunk references images (via
   metadata["images"] / image_refs), generate a text caption for each image
   and stitch it into the chunk so it becomes searchable ("search text, find image").
2. When disabled, unavailable, or on error, the chunk keeps its image_refs
   untouched and is flagged with `has_unprocessed_images = True`. Ingestion is
   never blocked.

Captions are appended to chunk.text (default, maximizes retrieval coverage) and
also recorded in chunk.metadata["image_captions"] (image_id -> caption).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext
    from src.libs.llm.base_vision_llm import BaseVisionLLM

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = "config/prompts/image_captioning.txt"


class ImageCaptioner(BaseTransform):
    """Generate captions for images referenced by chunks (optional)."""

    def __init__(
        self,
        vision_llm: "BaseVisionLLM | None" = None,
        use_vision: bool = True,
        prompt_path: str | None = None,
    ):
        """Initialize ImageCaptioner.

        Args:
            vision_llm: Optional Vision LLM instance. If None, captioning is off.
            use_vision: Config switch to enable/disable captioning.
            prompt_path: Path to captioning prompt template; default if not given.
        """
        self._vision_llm = vision_llm
        self._enabled = use_vision and (vision_llm is not None)
        self._prompt = self._load_prompt(prompt_path)

    def transform(
        self,
        chunks: list[Chunk],
        trace: "TraceContext | None" = None,
    ) -> list[Chunk]:
        """Caption images for each chunk that references them."""
        if trace:
            trace.start_stage("image_captioner")

        stats = {"captioned": 0, "unprocessed": 0, "images_total": 0, "errors": 0}

        for chunk in chunks:
            images = self._chunk_images(chunk)
            if not images:
                continue

            stats["images_total"] += len(images)

            if not self._enabled:
                self._mark_unprocessed(chunk)
                stats["unprocessed"] += 1
                continue

            try:
                captions = self._caption_images(chunk, images, trace)
                if captions:
                    self._apply_captions(chunk, captions)
                    stats["captioned"] += 1
                    # If some images failed, still flag remainder
                    if len(captions) < len(images):
                        chunk.metadata["has_unprocessed_images"] = True
                        stats["unprocessed"] += 1
                else:
                    self._mark_unprocessed(chunk)
                    stats["unprocessed"] += 1
            except Exception as exc:
                logger.warning(
                    "ImageCaptioner failed on chunk %s: %s — leaving images unprocessed",
                    chunk.id, exc,
                )
                stats["errors"] += 1
                self._mark_unprocessed(chunk)
                stats["unprocessed"] += 1

        if trace:
            trace.end_stage(details=stats)

        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_images(chunk: Chunk) -> list[dict[str, Any]]:
        """Return image dicts referenced by this chunk (from metadata)."""
        images = chunk.metadata.get("images", [])
        if not isinstance(images, list):
            return []
        # Normalize to dicts (ImageRef may be stored as dict already)
        result: list[dict[str, Any]] = []
        for img in images:
            if isinstance(img, dict):
                result.append(img)
            elif hasattr(img, "to_dict"):
                result.append(img.to_dict())
        return result

    def _caption_images(
        self,
        chunk: Chunk,
        images: list[dict[str, Any]],
        trace: "TraceContext | None",
    ) -> dict[str, str]:
        """Generate captions for each image. Returns {image_id: caption}.

        Individual image failures are skipped (logged), not fatal.
        """
        captions: dict[str, str] = {}
        for img in images:
            image_id = img.get("id", "")
            path = img.get("path", "")
            if not path or not Path(path).exists():
                logger.warning(
                    "Image path missing for %s (chunk %s), skipping",
                    image_id, chunk.id,
                )
                continue
            try:
                response = self._vision_llm.chat_with_image(  # type: ignore[union-attr]
                    text=self._prompt, image=path, trace=trace
                )
                caption = (response.content or "").strip()
                if caption:
                    captions[image_id] = caption
            except Exception as exc:
                logger.warning(
                    "Caption generation failed for image %s: %s", image_id, exc
                )
                continue
        return captions

    @staticmethod
    def _apply_captions(chunk: Chunk, captions: dict[str, str]) -> None:
        """Stitch captions into chunk text and metadata."""
        chunk.metadata["image_captions"] = captions
        # Append captions to text for retrieval coverage
        appended = "\n\n".join(
            f"[IMAGE CAPTION {img_id}]: {cap}" for img_id, cap in captions.items()
        )
        if appended:
            chunk.text = chunk.text.rstrip() + "\n\n" + appended
        chunk.metadata["has_unprocessed_images"] = False

    @staticmethod
    def _mark_unprocessed(chunk: Chunk) -> None:
        """Flag the chunk as having images that were not captioned."""
        chunk.metadata["has_unprocessed_images"] = True

    def _load_prompt(self, prompt_path: str | None = None) -> str:
        """Load captioning prompt from file, with hardcoded fallback."""
        path = Path(prompt_path) if prompt_path else Path(_DEFAULT_PROMPT_PATH)
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                return content
        return (
            "You are an expert at describing images in technical documents. "
            "Provide a detailed, factual description of this image, including any "
            "text, chart data, or diagram structure it contains."
        )

    @property
    def name(self) -> str:
        return "image_captioner"

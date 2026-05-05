"""DocumentChunker — adapter between libs.splitter and Ingestion Pipeline.

Converts a Document object into a list of Chunk objects, adding business
logic on top of the raw text splitting: ID generation, metadata inheritance,
image reference distribution, and source tracing.
"""
from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from src.core.types import Document, Chunk, ImageRef
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings

# Regex to find [IMAGE: {image_id}] placeholders
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


class DocumentChunker:
    """Split a Document into Chunks with full business metadata."""

    def __init__(self, settings: "Settings", splitter_type: str = "recursive"):
        self._splitter = SplitterFactory.create(settings, splitter_type)

    def split_document(self, document: Document) -> list[Chunk]:
        """Split a Document into a list of Chunk objects.

        Args:
            document: The Document to split.

        Returns:
            List of Chunk objects with IDs, metadata, and source references.
        """
        # Use the configured splitter to get raw text chunks
        raw_chunks = self._splitter.split_text(document.text)

        # Build image lookup from document metadata
        doc_images = document.images  # List[ImageRef]
        image_map: dict[str, ImageRef] = {img.id: img for img in doc_images}

        chunks: list[Chunk] = []
        for index, chunk_text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(document.id, index, chunk_text)
            metadata = self._inherit_metadata(document, index, chunk_text, image_map)

            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                start_offset=0,  # Approximate; exact offset tracking is optional
                end_offset=len(chunk_text),
                source_ref=document.id,
            )
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
        """Generate a unique, deterministic chunk ID.

        Format: {doc_id}_{index:04d}_{hash_8chars}
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    @staticmethod
    def _inherit_metadata(
        document: Document,
        chunk_index: int,
        chunk_text: str,
        image_map: dict[str, ImageRef],
    ) -> dict:
        """Build chunk metadata: inherit from document + add chunk-specific fields."""
        # Start with a copy of document metadata (excluding document-level images)
        meta = {k: v for k, v in document.metadata.items() if k != "images"}
        meta["chunk_index"] = chunk_index

        # Scan for image placeholders in this chunk's text
        found_ids = _IMAGE_PLACEHOLDER_RE.findall(chunk_text)
        found_ids = [fid.strip() for fid in found_ids]

        if found_ids:
            # Only include images that this chunk actually references
            chunk_images = [
                image_map[fid].to_dict()
                for fid in found_ids
                if fid in image_map
            ]
            meta["images"] = chunk_images
            meta["image_refs"] = [fid for fid in found_ids if fid in image_map]

        return meta

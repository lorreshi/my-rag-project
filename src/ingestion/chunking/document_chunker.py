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
        # Use the configured splitter to get rich split pieces (text + metadata)
        pieces = self._splitter.split(document.text)

        # Build image lookup from document metadata
        doc_images = document.images  # List[ImageRef]
        image_map: dict[str, ImageRef] = {img.id: img for img in doc_images}

        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            chunk_text = piece.text
            chunk_id = self._generate_chunk_id(document.id, index, chunk_text)
            metadata = self._inherit_metadata(
                document, index, chunk_text, image_map, piece.metadata
            )

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

    # Fields the chunker computes itself; a SplitPiece's metadata must never
    # overwrite these (source_ref lives on the Chunk field, not in metadata).
    _PROTECTED_KEYS = ("chunk_index", "image_refs", "images")

    @classmethod
    def _inherit_metadata(
        cls,
        document: Document,
        chunk_index: int,
        chunk_text: str,
        image_map: dict[str, ImageRef],
        piece_metadata: dict | None = None,
    ) -> dict:
        """Build chunk metadata: inherit from document + add chunk-specific fields.

        Merge order (later wins, except protected keys are guarded):
        1. Document metadata (excluding document-level ``images``).
        2. Per-chunk structured metadata from the SplitPiece (e.g. ``sheet_name``,
           row range) — but NOT allowed to overwrite the chunker-owned keys
           ``chunk_index``/``image_refs``/``images``.
        3. Chunker-computed fields (``chunk_index`` and, if present, image fields).
        """
        # Start with a copy of document metadata (excluding document-level images)
        meta = {k: v for k, v in document.metadata.items() if k != "images"}

        # Merge per-chunk structured metadata, excluding protected keys so the
        # piece can never clobber chunker-owned fields.
        if piece_metadata:
            for key, value in piece_metadata.items():
                if key in cls._PROTECTED_KEYS:
                    continue
                meta[key] = value

        # Chunker-owned fields (set last so they always win).
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

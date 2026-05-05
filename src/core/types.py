"""Core data types / contracts.

Defines Document, Chunk, and ChunkRecord — the shared data structures used
across ingestion, retrieval, and MCP layers. These are the single source of
truth for data shape throughout the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageRef:
    """Reference to an extracted image within a document.

    Attributes:
        id: Global unique image identifier (format: {doc_hash}_{page}_{seq}).
        path: Storage path (convention: data/images/{collection}/{image_id}.png).
        page: Page number in the source document (0-indexed, optional).
        text_offset: Start position of the placeholder in Document.text.
        text_length: Length of the placeholder string.
        position: Physical position info (optional, e.g. PDF coordinates).
    """

    id: str
    path: str = ""
    page: int = 0
    text_offset: int = 0
    text_length: int = 0
    position: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "page": self.page,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageRef":
        return cls(
            id=data["id"],
            path=data.get("path", ""),
            page=data.get("page", 0),
            text_offset=data.get("text_offset", 0),
            text_length=data.get("text_length", 0),
            position=data.get("position", {}),
        )


@dataclass
class Document:
    """A parsed document — output of the Loader stage.

    Attributes:
        id: Unique document identifier (typically based on file hash).
        text: Full document text in canonical Markdown format.
        metadata: Document-level metadata. Must contain at least 'source_path'.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_path(self) -> str:
        return self.metadata.get("source_path", "")

    @property
    def images(self) -> list[ImageRef]:
        """Return image references from metadata."""
        raw = self.metadata.get("images", [])
        return [ImageRef.from_dict(r) if isinstance(r, dict) else r for r in raw]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            id=data["id"],
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Chunk:
    """A text chunk — output of the Splitter stage.

    Attributes:
        id: Unique chunk identifier (format: {doc_id}_{index:04d}_{hash}).
        text: Chunk text content.
        metadata: Inherited from Document + chunk-specific fields.
        start_offset: Character offset in the original Document.text.
        end_offset: End character offset in the original Document.text.
        source_ref: Reference back to the parent Document ID.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: str = ""

    @property
    def image_refs(self) -> list[str]:
        """Return image IDs referenced by this chunk."""
        return self.metadata.get("image_refs", [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            start_offset=data.get("start_offset", 0),
            end_offset=data.get("end_offset", 0),
            source_ref=data.get("source_ref", ""),
        )


@dataclass
class ChunkRecord:
    """A chunk ready for storage — output of the Embedding stage.

    Extends Chunk with vector representations for retrieval.

    Attributes:
        id: Same as Chunk.id.
        text: Chunk text content.
        metadata: Full metadata (inherited + enriched).
        dense_vector: Dense embedding vector (list of floats).
        sparse_vector: Sparse vector representation (dict of token -> weight).
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_vector: list[float] = field(default_factory=list)
    sparse_vector: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        return cls(
            id=data["id"],
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            dense_vector=data.get("dense_vector", []),
            sparse_vector=data.get("sparse_vector", {}),
        )

"""MCP tool: get_document_summary.

Returns title / summary / tags for a document identified by ``doc_id``.
Information is sourced from chunk metadata stored in the vector store (the
MetadataEnricher writes title/summary/tags onto chunks during ingestion).

A document's summary is derived from its chunks:
- title:   the title of the first chunk (chunk_index 0) if present.
- summary: the first non-empty chunk summary.
- tags:    the union of chunk tags (deduped, order-preserving).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.mcp_server.protocol_handler import InvalidParams

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger("mcp-server")

TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = "Get the title, summary, and tags for a document by its doc_id."
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string", "description": "The document identifier."},
    },
    "required": ["doc_id"],
}


class GetDocumentSummaryTool:
    """Resolve a document summary from stored chunk metadata."""

    def __init__(self, document_lookup: "DocumentLookup"):
        """Initialize.

        Args:
            document_lookup: An object exposing
                ``get_chunks_by_doc_id(doc_id) -> list[dict]`` where each dict
                has ``metadata``. This abstracts over the vector store so the
                tool is easily testable.
        """
        self._lookup = document_lookup

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_id = arguments.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise InvalidParams("'doc_id' is required and must be a non-empty string")

        chunks = self._lookup.get_chunks_by_doc_id(doc_id)
        if not chunks:
            raise InvalidParams(f"Document not found: {doc_id}")

        title, summary, tags = self._aggregate(chunks)

        text_lines = [f"### {title or doc_id}", ""]
        if summary:
            text_lines.append(summary)
            text_lines.append("")
        if tags:
            text_lines.append("**标签**: " + ", ".join(tags))
        markdown = "\n".join(text_lines)

        return {
            "content": [{"type": "text", "text": markdown}],
            "structuredContent": {
                "doc_id": doc_id,
                "title": title,
                "summary": summary,
                "tags": tags,
                "chunk_count": len(chunks),
            },
            "isError": False,
        }

    @staticmethod
    def _aggregate(chunks: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
        """Derive (title, summary, tags) from a document's chunk metadata."""
        title = ""
        summary = ""
        tags: list[str] = []
        seen_tags: set[str] = set()

        # Prefer chunk_index 0 for the title; fall back to first available.
        def _meta(c):
            return c.get("metadata", {}) or {}

        ordered = sorted(chunks, key=lambda c: _meta(c).get("chunk_index", 1_000_000))

        for c in ordered:
            meta = _meta(c)
            if not title and meta.get("title"):
                title = meta["title"]
            if not summary and meta.get("summary"):
                summary = meta["summary"]
            for tag in meta.get("tags", []) or []:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    tags.append(tag)

        return title, summary, tags

    def register(self, handler) -> None:
        """Register this tool with a ProtocolHandler."""
        handler.register_tool(
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            input_schema=INPUT_SCHEMA,
            handler=self.__call__,
        )


class DocumentLookup:
    """Look up a document's chunks by doc_id via a vector store.

    Uses the vector store's metadata query. Chunks are expected to carry a
    ``doc_id`` / ``source_ref`` field in metadata identifying their document.
    """

    def __init__(self, vector_store: "BaseVectorStore", id_field: str = "doc_id"):
        self._store = vector_store
        self._id_field = id_field

    def get_chunks_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        # Best-effort: rely on delete_by_metadata-style filtering is not ideal;
        # vector stores expose get via metadata through a dedicated method if
        # available. Here we use a generic 'get_by_metadata' if present.
        getter = getattr(self._store, "get_by_metadata", None)
        if callable(getter):
            return getter({self._id_field: doc_id})
        return []

"""CitationGenerator — build structured citations from retrieval results.

Produces a list of Citation objects with stable 1-based numbering, suitable for
both human-readable Markdown references ([1], [2], ...) and the structured
``citations`` payload returned to MCP clients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.types import RetrievalResult


@dataclass
class Citation:
    """A single citation referencing a source chunk."""

    id: int
    source: str
    chunk_id: str
    score: float = 0.0
    page: Any = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "text": self.text,
        }


class CitationGenerator:
    """Generate citations from retrieval results."""

    def __init__(self, snippet_length: int = 200):
        self._snippet_length = snippet_length

    def generate(self, results: list["RetrievalResult"]) -> list[Citation]:
        """Build a numbered citation list from retrieval results.

        Args:
            results: Ranked retrieval results (order is preserved as citation order).

        Returns:
            List of Citation, numbered from 1.
        """
        citations: list[Citation] = []
        for i, r in enumerate(results, start=1):
            meta = r.metadata or {}
            source = meta.get("source_path", meta.get("source", "unknown"))
            page = meta.get("page", meta.get("page_num", ""))
            citations.append(
                Citation(
                    id=i,
                    source=source,
                    chunk_id=r.chunk_id,
                    score=r.score,
                    page=page,
                    text=self._snippet(r.text),
                )
            )
        return citations

    def _snippet(self, text: str) -> str:
        """Trim text to a snippet for the citation payload."""
        s = (text or "").strip().replace("\n", " ")
        if len(s) > self._snippet_length:
            return s[: self._snippet_length].rsplit(" ", 1)[0] + "…"
        return s

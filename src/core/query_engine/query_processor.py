"""QueryProcessor — keyword extraction + filter parsing.

Transforms a raw user query into a ProcessedQuery containing:
- keywords: tokenized, stopword-filtered terms for sparse (BM25) retrieval.
- filters: a generic metadata filter dict (collection / doc_type / etc.).
- normalized_query: the (lightly) cleaned query text for dense embedding.

Tokenization mirrors SparseEncoder so that query keywords align with the BM25
index vocabulary (same casing, same CJK-per-char behavior).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# Same tokenizer contract as SparseEncoder: ASCII word runs + per-char CJK.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")

_DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "with", "as", "by", "at",
    "this", "that", "it", "from", "how", "what", "why", "when", "where",
    "do", "does", "can", "i", "we", "you",
}


@dataclass
class ProcessedQuery:
    """Result of query processing."""

    raw_query: str
    normalized_query: str
    keywords: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "keywords": self.keywords,
            "filters": self.filters,
        }


class QueryProcessor:
    """Extract keywords and parse filters from a raw query."""

    def __init__(
        self,
        lowercase: bool = True,
        stopwords: set[str] | None = None,
    ):
        self._lowercase = lowercase
        self._stopwords = (
            _DEFAULT_STOPWORDS if stopwords is None else stopwords
        )

    def process(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> ProcessedQuery:
        """Process a raw query into keywords + filters.

        Args:
            query: The raw user query.
            filters: Optional pre-supplied metadata filters (merged/normalized).
            trace: Optional trace context.

        Returns:
            ProcessedQuery. ``keywords`` may be empty if the query is empty or
            consists only of stopwords; callers should handle that gracefully.
        """
        if trace:
            trace.start_stage("query_processing")

        normalized = self._normalize(query)
        keywords = self._extract_keywords(normalized)
        parsed_filters = self._parse_filters(filters)

        result = ProcessedQuery(
            raw_query=query,
            normalized_query=normalized,
            keywords=keywords,
            filters=parsed_filters,
        )

        if trace:
            trace.end_stage(
                details={
                    "num_keywords": len(keywords),
                    "has_filters": bool(parsed_filters),
                }
            )

        return result

    def _normalize(self, query: str) -> str:
        """Collapse whitespace; preserve content for embedding."""
        if not query:
            return ""
        return re.sub(r"\s+", " ", query).strip()

    def _extract_keywords(self, text: str) -> list[str]:
        """Tokenize and drop stopwords, preserving first-seen order (deduped)."""
        if not text:
            return []
        raw = _TOKEN_RE.findall(text)
        if self._lowercase:
            raw = [t.lower() for t in raw]
        seen: set[str] = set()
        keywords: list[str] = []
        for t in raw:
            if t in self._stopwords or t in seen:
                continue
            seen.add(t)
            keywords.append(t)
        return keywords

    def _parse_filters(self, filters: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize the filters structure. Always returns a dict.

        Drops keys with None values; leaves the rest as-is. This is the generic
        filter contract consumed by retrievers / hybrid search.
        """
        if not filters:
            return {}
        return {k: v for k, v in filters.items() if v is not None}

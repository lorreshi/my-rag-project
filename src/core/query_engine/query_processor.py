"""QueryProcessor — keyword extraction + filter parsing.

Transforms a raw user query into a ProcessedQuery containing:
- keywords: tokenized, stopword-filtered terms for sparse (BM25) retrieval.
- filters: a generic metadata filter dict (collection / doc_type / etc.).
- normalized_query: the (lightly) cleaned query text for dense embedding.

Tokenization is delegated to a shared ``BaseTokenizer`` (the same component the
ingestion-side ``SparseEncoder`` uses) so query keywords align with the BM25
index vocabulary. Lowercasing and stopword filtering are the tokenizer's
responsibility; this processor only dedupes the tokens while preserving
first-seen order.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.libs.tokenizer import BaseTokenizer, JiebaTokenizer

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


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

    def __init__(self, tokenizer: BaseTokenizer | None = None):
        """Initialize QueryProcessor.

        Args:
            tokenizer: Shared BM25 tokenizer. When ``None``, defaults to
                ``JiebaTokenizer()`` (matching ``TokenizerFactory``'s default
                and the ingestion-side ``SparseEncoder``), so
                ``QueryProcessor()`` remains usable without arguments.
        """
        self._tokenizer: BaseTokenizer = tokenizer or JiebaTokenizer()

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
                    "method": "keyword_extraction",
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
        """Tokenize via the shared tokenizer, deduped preserving first-seen order.

        Lowercasing and stopword filtering happen inside the tokenizer, so the
        BM25 vocabulary stays aligned with the ingestion side. This method only
        removes duplicates while keeping the first occurrence's position.
        """
        if not text:
            return []
        seen: set[str] = set()
        keywords: list[str] = []
        for t in self._tokenizer.tokenize(text):
            if t in seen:
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

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

from src.libs.tokenizer import BaseTokenizer, JiebaTokenizer, normalize_text

if TYPE_CHECKING:
    from src.core.query_engine.filter_extractor import BaseFilterExtractor
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Result of query processing."""

    raw_query: str
    normalized_query: str
    keywords: list[str] = field(default_factory=list)
    expanded_keywords: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "keywords": self.keywords,
            "expanded_keywords": self.expanded_keywords,
            "filters": self.filters,
        }


class QueryProcessor:
    """Extract keywords and parse filters from a raw query."""

    def __init__(
        self,
        tokenizer: BaseTokenizer | None = None,
        nfkc: bool = True,
        casefold: bool = True,
        to_simplified: bool = False,
        filter_extractor: "BaseFilterExtractor | None" = None,
        synonym_map: dict[str, list[str]] | None = None,
    ):
        """Initialize QueryProcessor.

        Args:
            tokenizer: Shared BM25 tokenizer. When ``None``, defaults to
                ``JiebaTokenizer()`` (matching ``TokenizerFactory``'s default
                and the ingestion-side ``SparseEncoder``), so
                ``QueryProcessor()`` remains usable without arguments.
            nfkc: Apply NFKC normalization to the dense ``normalized_query``.
            casefold: Apply case folding to the dense ``normalized_query``.
            to_simplified: Apply traditional->simplified to ``normalized_query``
                (needs OpenCC; degrades to no-op when missing).
            filter_extractor: Optional, opt-in extractor that parses structured
                constraints from the query text into ``filters``. When ``None``
                (default) no extraction happens (behaviour unchanged).

        The normalization flags MUST match those used by ``tokenizer`` so the
        dense query text and the sparse BM25 keywords share one canonical form.
        ``TokenizerFactory`` / ``HybridSearch.from_settings`` wire both from the
        same ``settings.retrieval`` values.
        """
        self._tokenizer: BaseTokenizer = tokenizer or JiebaTokenizer()
        self._nfkc = nfkc
        self._casefold = casefold
        self._to_simplified = to_simplified
        self._filter_extractor = filter_extractor
        self._synonyms = synonym_map or {}

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
        expanded = self._expand_keywords(keywords)
        parsed_filters = self._parse_filters(filters, query)

        result = ProcessedQuery(
            raw_query=query,
            normalized_query=normalized,
            keywords=keywords,
            expanded_keywords=expanded,
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
        """Apply deterministic normalization, then collapse whitespace.

        Serves the dense side (``normalized_query`` feeds embedding). Uses the
        same ``normalize_text`` pipeline (NFKC + casefold + optional t2s) as the
        shared tokenizer, so dense text and sparse BM25 keywords share one
        canonical form. Whitespace collapsing is dense-only and does not affect
        tokenization.
        """
        if not query:
            return ""
        text = normalize_text(
            query,
            nfkc=self._nfkc,
            casefold=self._casefold,
            to_simplified=self._to_simplified,
        )
        return re.sub(r"\s+", " ", text).strip()

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

    def _expand_keywords(self, keywords: list[str]) -> list[str]:
        """Original keywords + synonym/alias tokens (OR-expansion for BM25).

        Dedup preserving first-seen order, with the original keywords kept as
        the prefix (so they retain their natural BM25 weight). Alias strings are
        run through the shared tokenizer so they align with the BM25 vocabulary.
        Returns a copy of ``keywords`` unchanged when no synonym map is set.
        """
        if not self._synonyms or not keywords:
            return list(keywords)
        seen: set[str] = set(keywords)
        expanded: list[str] = list(keywords)
        for kw in keywords:
            for alias in self._synonyms.get(kw, []):
                for tok in self._tokenizer.tokenize(alias):
                    if tok not in seen:
                        seen.add(tok)
                        expanded.append(tok)
        return expanded

    def _parse_filters(
        self, filters: dict[str, Any] | None, query: str = ""
    ) -> dict[str, Any]:
        """Merge extracted + external filters. Always returns a dict.

        Precedence: external (pre-supplied) filters win over extracted ones —
        the extractor only fills keys the caller did not provide. Keys with
        None values are dropped. With no extractor (default) this reduces to
        "external filters minus None values" (unchanged behaviour).
        """
        merged: dict[str, Any] = {}
        if self._filter_extractor is not None and query:
            try:
                merged.update(self._filter_extractor.extract(query))
            except Exception:  # extractor must not raise; be defensive anyway
                merged = {}
        if filters:
            merged.update({k: v for k, v in filters.items() if v is not None})
        return {k: v for k, v in merged.items() if v is not None}

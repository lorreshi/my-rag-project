"""SparseEncoder — BM25 term statistics for chunks.

Tokenizes chunk text and computes per-chunk term frequencies and document
length. This is the input contract for the BM25Indexer (C11), which later
computes IDF and builds the inverted index.

This module intentionally does NOT compute IDF (a corpus-global statistic);
it only emits per-document (per-chunk) term statistics so it can run
incrementally and stream into the indexer.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.types import Chunk

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# Tokenizer: ASCII word runs (len>=2) plus individual CJK characters.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")

_DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "with", "as", "by", "at",
    "this", "that", "it", "from",
}


@dataclass
class SparseVector:
    """Per-chunk BM25 term statistics.

    Attributes:
        chunk_id: ID of the source chunk.
        term_freqs: Mapping of token -> raw term frequency within the chunk.
        doc_length: Total number of tokens in the chunk (for BM25 length norm).
    """

    chunk_id: str
    term_freqs: dict[str, int] = field(default_factory=dict)
    doc_length: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "term_freqs": self.term_freqs,
            "doc_length": self.doc_length,
        }


class SparseEncoder:
    """Compute BM25 term statistics for chunks."""

    def __init__(
        self,
        lowercase: bool = True,
        stopwords: set[str] | None = None,
    ):
        """Initialize SparseEncoder.

        Args:
            lowercase: Whether to lowercase tokens before counting.
            stopwords: Optional stopword set; defaults to a small English set.
        """
        self._lowercase = lowercase
        self._stopwords = (
            _DEFAULT_STOPWORDS if stopwords is None else stopwords
        )

    def encode(
        self,
        chunks: list[Chunk],
        trace: "TraceContext | None" = None,
    ) -> list[SparseVector]:
        """Encode chunks into per-chunk sparse term statistics.

        Args:
            chunks: Chunks to encode (uses chunk.text).
            trace: Optional trace context.

        Returns:
            List of SparseVector, one per chunk, in the same order. Chunks with
            empty/whitespace text yield an empty SparseVector (no terms,
            doc_length 0).
        """
        if trace:
            trace.start_stage("sparse_encoder")

        results: list[SparseVector] = []
        empty_count = 0

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            if not tokens:
                empty_count += 1
                results.append(SparseVector(chunk_id=chunk.id))
                continue
            term_freqs = dict(Counter(tokens))
            results.append(
                SparseVector(
                    chunk_id=chunk.id,
                    term_freqs=term_freqs,
                    doc_length=len(tokens),
                )
            )

        if trace:
            trace.end_stage(
                details={"count": len(results), "empty": empty_count}
            )

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into BM25 terms, dropping stopwords."""
        if not text or not text.strip():
            return []
        raw = _TOKEN_RE.findall(text)
        if self._lowercase:
            raw = [t.lower() for t in raw]
        return [t for t in raw if t not in self._stopwords]

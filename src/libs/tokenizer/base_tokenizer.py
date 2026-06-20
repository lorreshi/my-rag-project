"""Tokenizer abstract base class.

Defines the single tokenization contract shared by both the ingestion side
(``SparseEncoder``) and the query side (``QueryProcessor``) so that the BM25
vocabulary stays aligned across indexing and retrieval.

All implementations must return tokens that are already lowercased and
stopword-filtered, so callers can treat the output as the final BM25 term
sequence without further normalization.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract base class for BM25 tokenization strategies."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize *text* into a BM25 term sequence.

        Args:
            text: The raw text to tokenize.

        Returns:
            List of tokens, already lowercased and stopword-filtered, in
            document order. Empty/whitespace-only text yields an empty list.
        """
        ...

"""Tokenizer Factory — configuration-driven tokenizer instantiation.

Reads ``retrieval.tokenizer`` (``jieba`` by default, or ``regex`` to fall back
to the legacy character-level behavior) and returns a shared ``BaseTokenizer``.
Both the ingestion side (``SparseEncoder``) and the query side
(``QueryProcessor``) create their tokenizer through this factory so that the
BM25 vocabulary stays aligned.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.libs.tokenizer.base_tokenizer import BaseTokenizer
from src.libs.tokenizer.jieba_tokenizer import DEFAULT_STOPWORDS, JiebaTokenizer

if TYPE_CHECKING:
    from src.core.settings import Settings

# Legacy character-level contract: ASCII word runs whole + single CJK chars.
_REGEX_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


class RegexTokenizer(BaseTokenizer):
    """Legacy character-level tokenizer (fallback / comparison).

    Reproduces the old behavior: ASCII letter/digit runs are kept whole while
    each CJK character becomes its own token.
    """

    def __init__(
        self,
        lowercase: bool = True,
        stopwords: set[str] | None = None,
    ):
        self._lowercase = lowercase
        self._stopwords = (
            DEFAULT_STOPWORDS if stopwords is None else stopwords
        )

    def tokenize(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        raw = _REGEX_TOKEN_RE.findall(text)
        if self._lowercase:
            raw = [t.lower() for t in raw]
        return [t for t in raw if t not in self._stopwords]


class TokenizerFactory:
    """Create a BaseTokenizer based on ``settings.retrieval.tokenizer``."""

    @staticmethod
    def create(settings: "Settings") -> BaseTokenizer:
        """Instantiate the configured tokenizer.

        Defaults to ``jieba`` when the ``retrieval.tokenizer`` field is absent
        so missing config never raises.

        Raises:
            ValueError: if an unknown tokenizer name is configured.
        """
        retrieval = getattr(settings, "retrieval", None)
        name = getattr(retrieval, "tokenizer", None) or "jieba"
        key = str(name).lower()

        if key == "jieba":
            return JiebaTokenizer()
        if key == "regex":
            return RegexTokenizer()

        raise ValueError(
            f"Unknown tokenizer '{key}'. Available: jieba, regex"
        )

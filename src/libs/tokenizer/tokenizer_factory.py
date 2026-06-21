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
from src.libs.tokenizer.normalize import normalize_text

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
        nfkc: bool = True,
        to_simplified: bool = False,
    ):
        self._lowercase = lowercase
        self._stopwords = (
            DEFAULT_STOPWORDS if stopwords is None else stopwords
        )
        self._nfkc = nfkc
        self._to_simplified = to_simplified

    def tokenize(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        # Shared deterministic normalization (symmetric with ingestion side).
        text = normalize_text(
            text,
            nfkc=self._nfkc,
            casefold=self._lowercase,
            to_simplified=self._to_simplified,
        )
        if not text.strip():
            return []
        raw = _REGEX_TOKEN_RE.findall(text)
        # Already case-folded via normalize_text above.
        return [t for t in raw if t not in self._stopwords]


def _normalization_flags(settings: "Settings") -> dict:
    """Read the shared normalization toggles from settings.retrieval."""
    retrieval = getattr(settings, "retrieval", None)
    return {
        "nfkc": getattr(retrieval, "enable_nfkc", True),
        "lowercase": getattr(retrieval, "normalize_casefold", True),
        "to_simplified": getattr(retrieval, "normalize_to_simplified", False),
    }


class TokenizerFactory:
    """Create a BaseTokenizer based on ``settings.retrieval.tokenizer``."""

    @staticmethod
    def create(settings: "Settings") -> BaseTokenizer:
        """Instantiate the configured tokenizer.

        Defaults to ``jieba`` when the ``retrieval.tokenizer`` field is absent
        so missing config never raises. Shared normalization flags
        (``enable_nfkc`` / ``normalize_casefold`` / ``normalize_to_simplified``)
        are forwarded so the index and query sides stay symmetric.

        Raises:
            ValueError: if an unknown tokenizer name is configured.
        """
        retrieval = getattr(settings, "retrieval", None)
        name = getattr(retrieval, "tokenizer", None) or "jieba"
        key = str(name).lower()
        flags = _normalization_flags(settings)

        if key == "jieba":
            return JiebaTokenizer(**flags)
        if key == "regex":
            return RegexTokenizer(**flags)

        raise ValueError(
            f"Unknown tokenizer '{key}'. Available: jieba, regex"
        )

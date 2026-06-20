"""RegexTokenizer — legacy character-level fallback.

Reproduces the original behavior previously duplicated inside SparseEncoder
and QueryProcessor: ASCII word/number runs plus individual CJK characters
(per-character, not word-level). Selected via ``retrieval.tokenizer=regex``
for comparison / graceful degradation.
"""
from __future__ import annotations

import re

from src.libs.tokenizer.base_tokenizer import BaseTokenizer, DEFAULT_STOPWORDS
from src.libs.tokenizer.tokenizer_factory import register_tokenizer

# ASCII word runs plus single CJK characters (legacy char-level behavior).
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


class RegexTokenizer(BaseTokenizer):
    """Character-level tokenizer (legacy behavior)."""

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
        raw = _TOKEN_RE.findall(text)
        if self._lowercase:
            raw = [t.lower() for t in raw]
        return [t for t in raw if t not in self._stopwords]


def _create_regex(settings) -> RegexTokenizer:
    return RegexTokenizer()


register_tokenizer("regex", _create_regex)

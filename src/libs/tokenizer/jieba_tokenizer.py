"""JiebaTokenizer — default shared tokenizer for BM25 vocabulary.

Chinese segments are cut at the word level via ``jieba.cut`` (so Chinese text
is no longer matched character-by-character), while runs of ASCII letters /
digits are extracted with a word regex. All tokens are lowercased and
stopword-filtered, so the output is the final BM25 term sequence.

The stopword set merges the small English sets that previously lived
(duplicated) in ``SparseEncoder`` and ``QueryProcessor`` with a set of common
Chinese function words, since jieba now emits Chinese words that warrant
filtering.
"""
from __future__ import annotations

import re

import jieba

from src.libs.tokenizer.base_tokenizer import BaseTokenizer

# Runs of ASCII letters / digits (used for the non-CJK portions of the text).
_ASCII_RE = re.compile(r"[A-Za-z0-9]+")
# Maximal runs of CJK characters (handed to jieba for word-level cutting).
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

# Merged English stopwords from the former SparseEncoder + QueryProcessor sets.
_ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "with", "as", "by", "at",
    "this", "that", "it", "from", "how", "what", "why", "when", "where",
    "do", "does", "can", "i", "we", "you",
}

# Common Chinese function words now that jieba emits word-level tokens.
_CHINESE_STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "有", "个", "也", "这", "那", "就",
    "都", "而", "及", "与", "着", "之", "用", "于", "你", "我们", "他", "她",
    "它", "吗", "呢", "啊", "把", "被", "让", "给", "但", "并", "或", "如果",
    "因为", "所以", "以及", "对", "中", "等", "为", "上", "下",
}

# Default merged stopword set shared by the jieba and regex tokenizers.
DEFAULT_STOPWORDS: set[str] = _ENGLISH_STOPWORDS | _CHINESE_STOPWORDS


class JiebaTokenizer(BaseTokenizer):
    """Word-level Chinese tokenizer with ASCII word handling.

    - Chinese runs go through ``jieba.cut`` (word-level segmentation).
    - ASCII letter/digit runs are extracted via ``[A-Za-z0-9]+``.
    - All tokens are lowercased (when enabled) and stopword-filtered.
    """

    def __init__(
        self,
        lowercase: bool = True,
        stopwords: set[str] | None = None,
    ):
        """Initialize JiebaTokenizer.

        Args:
            lowercase: Whether to lowercase tokens before filtering.
            stopwords: Optional stopword set; defaults to the merged
                Chinese + English set.
        """
        self._lowercase = lowercase
        self._stopwords = (
            DEFAULT_STOPWORDS if stopwords is None else stopwords
        )

    def tokenize(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        raw: list[str] = []
        pos = 0
        for match in _CJK_RUN_RE.finditer(text):
            # Non-CJK gap before this CJK run -> ASCII word extraction.
            if match.start() > pos:
                raw.extend(_ASCII_RE.findall(text[pos:match.start()]))
            # CJK run -> jieba word-level cut.
            for token in jieba.cut(match.group()):
                token = token.strip()
                if token:
                    raw.append(token)
            pos = match.end()
        # Trailing non-CJK tail.
        if pos < len(text):
            raw.extend(_ASCII_RE.findall(text[pos:]))

        if self._lowercase:
            raw = [t.lower() for t in raw]
        return [t for t in raw if t not in self._stopwords]

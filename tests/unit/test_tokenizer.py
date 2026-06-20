"""Unit tests for the shared tokenizer (T1).

Covers:
- Chinese word-level segmentation (not per-character).
- Mixed Chinese/English/digit tokenization.
- Stopword filtering (Chinese + English).
- ``regex`` fallback to legacy character-level behavior.
- ``TokenizerFactory`` default (jieba) and explicit selection.

Validates: Requirements 2.1, 2.3
"""
from __future__ import annotations

import pytest

from src.core.settings import RetrievalConfig, Settings
from src.libs.tokenizer import BaseTokenizer, JiebaTokenizer, TokenizerFactory
from src.libs.tokenizer.tokenizer_factory import RegexTokenizer


class TestJiebaTokenizer:
    def test_chinese_word_level_not_single_char(self):
        """Chinese sentences are cut into words, not single characters."""
        tok = JiebaTokenizer()
        tokens = tok.tokenize("自然语言处理是人工智能的重要方向")
        # At least one multi-character word token must be present.
        assert any(len(t) > 1 for t in tokens)
        # Known multi-char words should appear (word-level segmentation).
        assert "自然语言" in tokens
        assert "人工智能" in tokens

    def test_mixed_chinese_english_digits(self):
        """Mixed text splits CJK via jieba and ASCII runs via regex."""
        tok = JiebaTokenizer()
        tokens = tok.tokenize("使用RAG技术结合BM25和Embedding")
        # ASCII runs kept whole and lowercased.
        assert "rag" in tokens
        assert "bm25" in tokens
        assert "embedding" in tokens
        # Chinese words present at word level.
        assert "技术" in tokens
        assert "结合" in tokens

    def test_stopwords_filtered(self):
        """Both Chinese and English stopwords are removed."""
        tok = JiebaTokenizer()
        tokens = tok.tokenize("这是一个测试 the cat")
        assert "the" not in tokens   # English stopword
        assert "是" not in tokens     # Chinese stopword
        assert "cat" in tokens
        assert "测试" in tokens

    def test_lowercase_applied(self):
        tok = JiebaTokenizer()
        assert tok.tokenize("Hello WORLD Foo") == ["hello", "world", "foo"]

    def test_empty_and_whitespace(self):
        tok = JiebaTokenizer()
        assert tok.tokenize("") == []
        assert tok.tokenize("   \n\t ") == []

    def test_custom_stopwords_override(self):
        tok = JiebaTokenizer(stopwords={"foo"})
        tokens = tok.tokenize("foo bar the")
        assert "foo" not in tokens
        # With a custom set, default stopwords like "the" are no longer filtered.
        assert "the" in tokens
        assert "bar" in tokens

    def test_is_base_tokenizer(self):
        assert isinstance(JiebaTokenizer(), BaseTokenizer)


class TestRegexTokenizer:
    def test_char_level_cjk(self):
        """Legacy behavior: each CJK char is its own token; ASCII kept whole."""
        tok = RegexTokenizer()
        tokens = tok.tokenize("机器学习ABC123")
        assert tokens == ["机", "器", "学", "习", "abc123"]

    def test_stopwords_filtered(self):
        tok = RegexTokenizer()
        tokens = tok.tokenize("the dog")
        assert "the" not in tokens
        assert "dog" in tokens

    def test_empty(self):
        assert RegexTokenizer().tokenize("") == []


class TestTokenizerFactory:
    def _settings(self, tokenizer: str | None = None) -> Settings:
        retrieval = RetrievalConfig()
        if tokenizer is not None:
            retrieval.tokenizer = tokenizer  # type: ignore[attr-defined]
        return Settings(retrieval=retrieval)

    def test_default_is_jieba_when_field_absent(self):
        """Missing retrieval.tokenizer defaults to jieba (no error)."""
        tok = TokenizerFactory.create(self._settings())
        assert isinstance(tok, JiebaTokenizer)

    def test_explicit_jieba(self):
        tok = TokenizerFactory.create(self._settings("jieba"))
        assert isinstance(tok, JiebaTokenizer)

    def test_regex_fallback(self):
        tok = TokenizerFactory.create(self._settings("regex"))
        assert isinstance(tok, RegexTokenizer)
        # Confirms legacy character-level behavior is selected.
        assert tok.tokenize("机器ABC") == ["机", "器", "abc"]

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            TokenizerFactory.create(self._settings("unknown"))

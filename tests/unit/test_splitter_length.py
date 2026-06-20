"""Tests for pluggable length functions + token size_unit (T7).

Validates: Requirements 4.3, 4.4 (Property 12)

Property 12: Token 度量生效 —— size_unit=token 时切分块的 token 数不超过
chunk_size（结构不可分的兜底块除外）；同一文本在 char 与 token 两种度量下
产出不同的分块边界。
"""
from __future__ import annotations

import pytest

import tiktoken

# Trigger registration / import side-effects.
import src.libs.splitter.recursive_splitter as mod  # noqa: F401
from src.libs.splitter.length import char_length, token_length
from src.libs.splitter.recursive_splitter import RecursiveSplitter


_ENCODING = "cl100k_base"
_enc = tiktoken.get_encoding(_ENCODING)


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


@pytest.mark.unit
class TestLengthFunctions:
    """char_length / token_length basic counting correctness."""

    def test_char_length_matches_len(self):
        assert char_length("") == 0
        assert char_length("hello") == 5
        assert char_length("中文测试") == 4

    def test_token_length_matches_tiktoken(self):
        counter = token_length(_ENCODING)
        for text in ["hello world", "这是一个中文测试句子。", "mixed 中英文 text 123"]:
            assert counter(text) == _token_count(text)

    def test_token_length_caches_encoding(self):
        # The factory should return a callable that works repeatedly without
        # re-fetching the encoding each call (smoke check: repeated calls work).
        counter = token_length(_ENCODING)
        first = counter("repeat me")
        second = counter("repeat me")
        assert first == second == _token_count("repeat me")

    def test_token_length_default_encoding(self):
        counter = token_length()  # default cl100k_base
        assert counter("hello") == _token_count("hello")


@pytest.mark.unit
class TestTokenSizeUnit:
    """Requirement 4.3 / Property 12 — token-measured chunks respect chunk_size."""

    def test_each_chunk_within_token_budget(self):
        # English + Chinese mixed long text.
        text = (
            "RecursiveSplitter measures size by tokens when size_unit is token. "
            "这是一段中文内容，用来测试 token 度量是否生效。"
            "Tokens differ from characters, especially for CJK text. "
            "每个分块的 token 数都应当不超过设定的 chunk_size。"
        ) * 6
        chunk_size = 32
        s = RecursiveSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            size_unit="token",
            token_encoding=_ENCODING,
        )
        chunks = s.split_text(text)
        assert len(chunks) > 1

        # Every chunk fits the token budget, except indivisible fallback blocks
        # (a chunk that is a single character cannot be split further).
        for c in chunks:
            tokens = _token_count(c)
            assert tokens <= chunk_size or len(c) == 1, (
                f"chunk exceeds token budget: tokens={tokens}, chunk={c!r}"
            )

    def test_explicit_length_fn_overrides_size_unit(self):
        # length_fn wins over size_unit.
        s = RecursiveSplitter(
            chunk_size=10,
            chunk_overlap=0,
            length_fn=char_length,
            size_unit="token",
        )
        assert s._length is char_length

    def test_char_default_uses_len(self):
        s = RecursiveSplitter(chunk_size=10, chunk_overlap=0)
        assert s._length is len


@pytest.mark.unit
class TestCharVsTokenBoundaries:
    """Property 12 — same text yields different boundaries under char vs token."""

    def test_boundaries_differ_for_mixed_text(self):
        text = (
            "This is a paragraph mixing English and 中文内容。"
            "We want char and token measurement to produce different splits. "
            "因为中文字符的 token 数与字符数尺度不同，分块边界自然不同。"
            "More English words here to make the text long enough for splitting. "
            "再补充一些中文句子，确保文本足够长以触发多次切分。"
        ) * 4

        char_splitter = RecursiveSplitter(
            chunk_size=80, chunk_overlap=0, size_unit="char"
        )
        token_splitter = RecursiveSplitter(
            chunk_size=80, chunk_overlap=0, size_unit="token", token_encoding=_ENCODING
        )

        char_chunks = char_splitter.split_text(text)
        token_chunks = token_splitter.split_text(text)

        # Both produce multiple chunks but with different boundaries.
        assert len(char_chunks) > 1
        assert len(token_chunks) > 1
        assert char_chunks != token_chunks


@pytest.mark.unit
class TestTiktokenMissing:
    """Requirement 4.3 — tiktoken missing gives a readable error."""

    def test_missing_tiktoken_raises_readable_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No module named 'tiktoken'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(RuntimeError) as exc_info:
            token_length()

        msg = str(exc_info.value)
        assert "tiktoken" in msg
        assert "install" in msg.lower()

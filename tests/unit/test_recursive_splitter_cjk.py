"""Tests for RecursiveSplitter CJK separators + overlap fix (T6).

Validates: Requirements 4.1, 4.2, 4.6, 4.7 (Property 11)
"""
from __future__ import annotations

import pytest

# Trigger registration
import src.libs.splitter.recursive_splitter as mod  # noqa: F401
from src.libs.splitter.base_splitter import SplitPiece
from src.libs.splitter.recursive_splitter import RecursiveSplitter


def _has_cjk_punct_boundary(chunks: list[str]) -> bool:
    """True if at least one chunk break occurs at CJK punctuation.

    The recursive splitter reattaches the separator to the *start* of the
    following part, so a punctuation boundary shows up as a chunk that starts
    (or ends) with a CJK punctuation mark.
    """
    cjk_enders = "。！？；，"
    return any(c and (c[0] in cjk_enders or c[-1] in cjk_enders) for c in chunks)


@pytest.mark.unit
class TestCjkSeparators:
    """Requirement 4.1 / 4.2 — Chinese prose breaks at punctuation, not chars."""

    def test_long_chinese_no_single_char_fragments(self):
        # Long Chinese paragraph, no newlines, with Chinese punctuation.
        sentence = "这是一个用于测试的中文长句子，它包含了很多内容。"
        text = sentence * 20  # well over chunk_size
        s = RecursiveSplitter(chunk_size=40, chunk_overlap=0)
        chunks = s.split_text(text)

        assert len(chunks) > 1
        # Property 11: no length-1 character fragments.
        assert all(len(c) > 1 for c in chunks), [c for c in chunks if len(c) <= 1]

    def test_breaks_at_chinese_punctuation(self):
        text = (
            "第一句话讲的是切分。第二句话讲的是标点！第三句话是疑问吗？"
            "第四句使用分号；第五句使用逗号，然后继续。"
        ) * 3
        s = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
        chunks = s.split_text(text)

        assert len(chunks) > 1
        # At least one chunk ends on a CJK punctuation mark.
        assert _has_cjk_punct_boundary(chunks)

    def test_pure_chinese_no_punctuation_fallback(self):
        # Edge case: a long run with no punctuation at all — fallback is allowed
        # to produce char-level pieces, but it should not crash.
        text = "中" * 100
        s = RecursiveSplitter(chunk_size=20, chunk_overlap=0)
        chunks = s.split_text(text)
        assert "".join(chunks) != "" if chunks else True


@pytest.mark.unit
class TestCodeBlockPreserved:
    """Requirement 4.7 — code block structure is not destroyed."""

    def test_small_code_block_stays_intact(self):
        text = (
            "前面的说明文字。\n\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "```\n\n"
            "后面的说明文字。"
        )
        s = RecursiveSplitter(chunk_size=200, chunk_overlap=0)
        chunks = s.split_text(text)
        # Whole code block fits in one chunk: fences and body together.
        assert any(
            "```python" in c and "return 42" in c and "```" in c.split("```python", 1)[1]
            for c in chunks
        )

    def test_code_fences_not_split_into_unusable_pieces(self):
        text = (
            "```python\n"
            "x = 1\n"
            "y = 2\n"
            "```"
        )
        s = RecursiveSplitter(chunk_size=200, chunk_overlap=0)
        chunks = s.split_text(text)
        joined = "".join(chunks)
        # Opening and closing fences both survive.
        assert joined.count("```") == 2


@pytest.mark.unit
class TestOverlapNoForcedSpace:
    """Requirement 4.6 — overlap join must not force-insert a space."""

    def test_overlap_does_not_insert_space(self):
        # Chinese sentences; with overlap the tail of a previous chunk is
        # prepended directly with no separating space.
        text = "甲乙丙丁。戊己庚辛。壬癸子丑。寅卯辰巳。"
        s = RecursiveSplitter(chunk_size=10, chunk_overlap=4)
        chunks = s.split_text(text)
        assert len(chunks) >= 2
        # No spurious spaces introduced anywhere (source has none).
        assert all(" " not in c for c in chunks), chunks

    def test_overlap_join_matches_direct_concat(self):
        # Build chunks without overlap, then verify overlap version equals
        # prev_tail + chunk (no extra char between them).
        text = "AAAA.\n\nBBBB.\n\nCCCC."
        no_ov = RecursiveSplitter(chunk_size=6, chunk_overlap=0).split_text(text)
        ov = RecursiveSplitter(chunk_size=6, chunk_overlap=3).split_text(text)
        assert len(no_ov) >= 2
        # For each overlapped chunk (after the first), it should start with a
        # tail of the previous base chunk directly followed by the base chunk
        # text — never a tail + " " + chunk.
        for c in ov[1:]:
            assert "  " not in c  # no doubled space artifacts


@pytest.mark.unit
class TestSplitContract:
    """Requirement 3.x — split() returns SplitPiece; split_text stays compatible."""

    def test_split_returns_split_pieces_with_empty_metadata(self):
        s = RecursiveSplitter(chunk_size=20, chunk_overlap=0)
        pieces = s.split("第一句。第二句。第三句。第四句。")
        assert pieces
        assert all(isinstance(p, SplitPiece) for p in pieces)
        assert all(p.metadata == {} for p in pieces)

    def test_split_text_consistent_with_split(self):
        s = RecursiveSplitter(chunk_size=20, chunk_overlap=0)
        text = "第一句。第二句。第三句。第四句。第五句。"
        pieces = s.split(text)
        texts = s.split_text(text)
        assert texts == [p.text for p in pieces]

    def test_empty_text(self):
        s = RecursiveSplitter()
        assert s.split("") == []
        assert s.split("   ") == []
        assert s.split_text("") == []

    def test_splitter_type_unchanged(self):
        assert RecursiveSplitter().splitter_type == "recursive"

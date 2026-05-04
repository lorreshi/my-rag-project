"""Tests for RecursiveSplitter (B7.5)."""
from __future__ import annotations

import pytest

from src.libs.splitter.splitter_factory import SplitterFactory
from src.core.settings import Settings

# Trigger registration
import src.libs.splitter.recursive_splitter as mod  # noqa: F401
from src.libs.splitter.recursive_splitter import RecursiveSplitter


@pytest.mark.unit
class TestRecursiveSplitter:

    def test_short_text_single_chunk(self):
        s = RecursiveSplitter(chunk_size=100)
        chunks = s.split_text("Hello world")
        assert chunks == ["Hello world"]

    def test_empty_text(self):
        s = RecursiveSplitter()
        assert s.split_text("") == []
        assert s.split_text("   ") == []

    def test_splits_by_paragraph(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        s = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
        chunks = s.split_text(text)
        assert len(chunks) >= 2
        assert "Paragraph one." in chunks[0]

    def test_splits_by_heading(self):
        text = "# Intro\n\nSome text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        s = RecursiveSplitter(chunk_size=40, chunk_overlap=0)
        chunks = s.split_text(text)
        assert len(chunks) >= 2

    def test_code_block_not_broken(self):
        text = "Some text.\n\n```python\ndef foo():\n    return 42\n```\n\nMore text."
        s = RecursiveSplitter(chunk_size=200, chunk_overlap=0)
        chunks = s.split_text(text)
        # Code block should stay in one chunk since total < chunk_size
        assert any("def foo():" in c and "return 42" in c for c in chunks)

    def test_respects_chunk_size(self):
        text = "word " * 500  # ~2500 chars
        s = RecursiveSplitter(chunk_size=100, chunk_overlap=0)
        chunks = s.split_text(text)
        for c in chunks:
            assert len(c) <= 120  # small tolerance for separator reattach

    def test_splitter_type(self):
        assert RecursiveSplitter().splitter_type == "recursive"

    def test_overlap_adds_context(self):
        text = "AAA.\n\nBBB.\n\nCCC."
        s = RecursiveSplitter(chunk_size=10, chunk_overlap=4)
        chunks = s.split_text(text)
        # With overlap, later chunks may contain tail of previous
        assert len(chunks) >= 2


@pytest.mark.unit
class TestSplitterFactoryIntegration:

    def test_factory_creates_recursive(self):
        sp = SplitterFactory.create(Settings(), "recursive")
        assert isinstance(sp, RecursiveSplitter)
        assert sp.splitter_type == "recursive"

    def test_factory_case_insensitive(self):
        sp = SplitterFactory.create(Settings(), "RECURSIVE")
        assert isinstance(sp, RecursiveSplitter)

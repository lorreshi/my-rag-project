"""Tests for T8: size configuration + per-collection override.

Covers:
- SplitterConfig defaults and parsing from a raw dict (incl. overrides/by_doc_type).
- DocumentChunker honouring per-collection size overrides.
- Graceful fallback when the splitter section is missing.

Validates: Requirements 4.5, 6.1, 6.2, 6.3
"""
from __future__ import annotations

import pytest

from src.core.settings import Settings, SplitterConfig, _parse_raw
from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker


@pytest.mark.unit
class TestSplitterConfigDefaults:
    def test_default_values(self):
        cfg = SplitterConfig()
        assert cfg.type == "recursive"
        assert cfg.size_unit == "token"
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64
        assert cfg.token_encoding == "cl100k_base"
        assert cfg.by_doc_type == {}
        assert cfg.overrides == {}

    def test_settings_has_splitter_default(self):
        settings = Settings()
        assert isinstance(settings.splitter, SplitterConfig)
        assert settings.splitter.chunk_size == 512

    def test_independent_default_dicts(self):
        a = SplitterConfig()
        b = SplitterConfig()
        a.overrides["faq"] = {"chunk_size": 10}
        assert b.overrides == {}


@pytest.mark.unit
class TestSplitterConfigParsing:
    def test_parse_full_section(self):
        raw = {
            "llm": {"provider": "openai", "model": "m"},
            "embedding": {"provider": "openai", "model": "e"},
            "vector_store": {"backend": "chroma"},
            "splitter": {
                "type": "recursive",
                "size_unit": "char",
                "chunk_size": 800,
                "chunk_overlap": 100,
                "token_encoding": "cl100k_base",
                "by_doc_type": {"xlsx": "table_aware"},
                "overrides": {"faq": {"chunk_size": 256, "chunk_overlap": 32}},
            },
        }
        settings = _parse_raw(raw)
        sp = settings.splitter
        assert sp.size_unit == "char"
        assert sp.chunk_size == 800
        assert sp.chunk_overlap == 100
        assert sp.by_doc_type == {"xlsx": "table_aware"}
        assert sp.overrides == {"faq": {"chunk_size": 256, "chunk_overlap": 32}}

    def test_parse_missing_splitter_uses_defaults(self):
        raw = {
            "llm": {"provider": "openai", "model": "m"},
            "embedding": {"provider": "openai", "model": "e"},
            "vector_store": {"backend": "chroma"},
        }
        settings = _parse_raw(raw)
        assert isinstance(settings.splitter, SplitterConfig)
        assert settings.splitter.chunk_size == 512
        assert settings.splitter.overrides == {}

    def test_parse_partial_splitter_keeps_other_defaults(self):
        raw = {"splitter": {"chunk_size": 1000}}
        settings = _parse_raw(raw)
        assert settings.splitter.chunk_size == 1000
        # Unspecified fields fall back to defaults
        assert settings.splitter.chunk_overlap == 64
        assert settings.splitter.size_unit == "token"


def _make_doc(text: str) -> Document:
    return Document(
        id="doc_test",
        text=text,
        metadata={"source_path": "x.pdf", "doc_type": "pdf"},
    )


# A long Chinese paragraph (no newlines) so size differences produce different
# chunk counts. Uses CJK punctuation so the recursive splitter can break it.
_LONG_TEXT = "。".join(f"这是第{i}个句子内容用于测试切分大小配置" for i in range(40))


@pytest.mark.unit
class TestPerCollectionOverride:
    def _settings(self) -> Settings:
        settings = Settings()
        # Use char unit for deterministic, dependency-free size measurement.
        settings.splitter.size_unit = "char"
        settings.splitter.chunk_size = 400
        settings.splitter.chunk_overlap = 0
        settings.splitter.overrides = {
            "faq": {"chunk_size": 50, "chunk_overlap": 0},
        }
        return settings

    def test_default_collection_uses_default_size(self):
        chunker = DocumentChunker(self._settings())
        chunks = chunker.split_document(_make_doc(_LONG_TEXT))
        assert all(len(c.text) <= 400 for c in chunks)

    def test_override_collection_uses_smaller_size(self):
        chunker = DocumentChunker(self._settings())
        default_chunks = chunker.split_document(_make_doc(_LONG_TEXT), "default")
        faq_chunks = chunker.split_document(_make_doc(_LONG_TEXT), "faq")

        # Smaller override size produces more (and smaller) chunks.
        assert len(faq_chunks) > len(default_chunks)
        # Every faq chunk respects the override size (single tokens that cannot
        # be split further are the only exception, which this text avoids).
        assert all(len(c.text) <= 50 for c in faq_chunks)

    def test_unconfigured_collection_falls_back_to_default(self):
        chunker = DocumentChunker(self._settings())
        unknown = chunker.split_document(_make_doc(_LONG_TEXT), "no_such_collection")
        default_chunks = chunker.split_document(_make_doc(_LONG_TEXT), "default")
        assert len(unknown) == len(default_chunks)


@pytest.mark.unit
class TestMissingSplitterSectionRobust:
    def test_chunker_without_splitter_attr_does_not_raise(self):
        """A Settings-like object missing the splitter attribute must not crash."""

        class BareSettings:
            pass

        # Build a minimal settings stand-in lacking a `splitter` attribute by
        # deleting it from a real Settings instance.
        settings = Settings()
        delattr(settings, "splitter")
        chunker = DocumentChunker(settings)
        chunks = chunker.split_document(_make_doc(_LONG_TEXT))
        # Falls back to RecursiveSplitter defaults (token unit, size 512).
        assert len(chunks) >= 1

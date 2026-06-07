"""Unit tests for MetadataEnricher — rule baseline, LLM mode, and fallback.

Tests use mock LLM to isolate logic. No real API calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.metadata_enricher import MetadataEnricher


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_chunk(text: str, chunk_id: str = "test_0001_abc") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "test.pdf", "doc_type": "pdf"},
        source_ref="doc_test",
    )


_SAMPLE_TEXT = (
    "## Vector Databases\n\n"
    "Vector databases store high-dimensional embeddings and support efficient "
    "similarity search. They are a core component of retrieval augmented "
    "generation systems. Popular options include Chroma and Qdrant."
)


@pytest.fixture
def llm_json():
    """Mock LLM returning a clean JSON metadata object."""
    llm = MagicMock()
    llm.provider_name = "mock"
    response = MagicMock()
    response.content = (
        '{"title": "Vector Databases Overview", '
        '"summary": "Vector databases store embeddings for similarity search.", '
        '"tags": ["vector database", "embeddings", "similarity search", "rag"]}'
    )
    llm.chat.return_value = response
    return llm


@pytest.fixture
def llm_fenced_json():
    """Mock LLM that wraps JSON in markdown code fences."""
    llm = MagicMock()
    llm.provider_name = "mock"
    response = MagicMock()
    response.content = (
        "Here is the metadata:\n```json\n"
        '{"title": "Fenced Title", "summary": "A summary.", "tags": ["a", "b"]}'
        "\n```"
    )
    llm.chat.return_value = response
    return llm


@pytest.fixture
def failing_llm():
    llm = MagicMock()
    llm.provider_name = "failing_mock"
    llm.chat.side_effect = RuntimeError("API connection failed")
    return llm


@pytest.fixture
def llm_bad_json():
    """Mock LLM returning unparseable content."""
    llm = MagicMock()
    llm.provider_name = "bad_mock"
    response = MagicMock()
    response.content = "Sorry, I cannot do that."
    llm.chat.return_value = response
    return llm


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

class TestRuleBaseline:
    def test_fields_present_and_nonempty(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        chunk = _make_chunk(_SAMPLE_TEXT)
        result = enricher.transform([chunk])[0]
        assert result.metadata["title"]
        assert result.metadata["summary"]
        assert isinstance(result.metadata["tags"], list)
        assert len(result.metadata["tags"]) > 0

    def test_title_from_heading(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        chunk = _make_chunk(_SAMPLE_TEXT)
        result = enricher.transform([chunk])[0]
        assert result.metadata["title"] == "Vector Databases"

    def test_title_from_first_line_when_no_heading(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        chunk = _make_chunk("Retrieval augmented generation basics.\nMore detail here.")
        result = enricher.transform([chunk])[0]
        assert "Retrieval augmented generation" in result.metadata["title"]

    def test_summary_is_leading_sentences(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        chunk = _make_chunk(_SAMPLE_TEXT)
        result = enricher.transform([chunk])[0]
        assert "Vector databases store" in result.metadata["summary"]

    def test_tags_exclude_stopwords(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        chunk = _make_chunk(_SAMPLE_TEXT)
        result = enricher.transform([chunk])[0]
        tags = result.metadata["tags"]
        assert "the" not in tags
        assert "and" not in tags

    def test_tags_respect_max(self):
        enricher = MetadataEnricher(llm=None, use_llm=False, max_tags=3)
        chunk = _make_chunk(_SAMPLE_TEXT)
        result = enricher.transform([chunk])[0]
        assert len(result.metadata["tags"]) <= 3

    def test_empty_text(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        chunk = _make_chunk("")
        result = enricher.transform([chunk])[0]
        assert result.metadata["title"] == "Untitled"
        assert result.metadata["summary"] == ""
        assert result.metadata["tags"] == []

    def test_marked_rule(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        chunk = _make_chunk(_SAMPLE_TEXT)
        result = enricher.transform([chunk])[0]
        assert result.metadata["enriched_by"] == "rule"


# ---------------------------------------------------------------------------
# LLM mode (mocked)
# ---------------------------------------------------------------------------

class TestLLMMode:
    def test_llm_called(self, llm_json):
        enricher = MetadataEnricher(llm=llm_json, use_llm=True)
        enricher.transform([_make_chunk(_SAMPLE_TEXT)])
        llm_json.chat.assert_called_once()

    def test_llm_metadata_applied(self, llm_json):
        enricher = MetadataEnricher(llm=llm_json, use_llm=True)
        result = enricher.transform([_make_chunk(_SAMPLE_TEXT)])[0]
        assert result.metadata["title"] == "Vector Databases Overview"
        assert result.metadata["summary"].startswith("Vector databases store")
        assert "embeddings" in result.metadata["tags"]
        assert result.metadata["enriched_by"] == "llm"

    def test_llm_fenced_json_parsed(self, llm_fenced_json):
        enricher = MetadataEnricher(llm=llm_fenced_json, use_llm=True)
        result = enricher.transform([_make_chunk(_SAMPLE_TEXT)])[0]
        assert result.metadata["title"] == "Fenced Title"
        assert result.metadata["tags"] == ["a", "b"]
        assert result.metadata["enriched_by"] == "llm"

    def test_tags_normalized_and_deduped(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = (
            '{"title": "T", "summary": "S", '
            '"tags": ["RAG", "rag", "Embeddings", "  rag  "]}'
        )
        llm.chat.return_value = response
        enricher = MetadataEnricher(llm=llm, use_llm=True)
        result = enricher.transform([_make_chunk(_SAMPLE_TEXT)])[0]
        tags = result.metadata["tags"]
        assert tags.count("rag") == 1
        assert "embeddings" in tags

    def test_llm_disabled_uses_rule(self, llm_json):
        enricher = MetadataEnricher(llm=llm_json, use_llm=False)
        result = enricher.transform([_make_chunk(_SAMPLE_TEXT)])[0]
        llm_json.chat.assert_not_called()
        assert result.metadata["enriched_by"] == "rule"

    def test_short_text_skips_llm(self, llm_json):
        enricher = MetadataEnricher(llm=llm_json, use_llm=True)
        enricher.transform([_make_chunk("Short.")])
        llm_json.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback / degradation
# ---------------------------------------------------------------------------

class TestFallback:
    def test_llm_exception_falls_back_to_rule(self, failing_llm):
        enricher = MetadataEnricher(llm=failing_llm, use_llm=True)
        result = enricher.transform([_make_chunk(_SAMPLE_TEXT)])[0]
        assert result.metadata["enriched_by"] == "rule"
        assert result.metadata["enrich_fallback"] == "llm_failed"
        # rule baseline still produced valid metadata
        assert result.metadata["title"] == "Vector Databases"
        assert result.metadata["tags"]

    def test_bad_json_falls_back_to_rule(self, llm_bad_json):
        enricher = MetadataEnricher(llm=llm_bad_json, use_llm=True)
        result = enricher.transform([_make_chunk(_SAMPLE_TEXT)])[0]
        assert result.metadata["enriched_by"] == "rule"
        assert result.metadata["enrich_fallback"] == "llm_failed"

    def test_missing_title_falls_back(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = '{"summary": "no title here", "tags": ["x"]}'
        llm.chat.return_value = response
        enricher = MetadataEnricher(llm=llm, use_llm=True)
        result = enricher.transform([_make_chunk(_SAMPLE_TEXT)])[0]
        assert result.metadata["enriched_by"] == "rule"

    def test_multiple_chunks_isolated(self, failing_llm):
        enricher = MetadataEnricher(llm=failing_llm, use_llm=True)
        chunks = [
            _make_chunk(_SAMPLE_TEXT, "c1"),
            _make_chunk(_SAMPLE_TEXT, "c2"),
            _make_chunk(_SAMPLE_TEXT, "c3"),
        ]
        results = enricher.transform(chunks)
        assert len(results) == 3
        for r in results:
            assert r.metadata["enriched_by"] == "rule"
            assert r.metadata["title"]


# ---------------------------------------------------------------------------
# Trace + edge cases
# ---------------------------------------------------------------------------

class TestTraceAndEdges:
    def test_trace_records_stage(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        trace = TraceContext(trace_type="ingestion")
        enricher.transform([_make_chunk(_SAMPLE_TEXT)], trace=trace)
        assert len(trace.stages) == 1
        assert trace.stages[0].name == "metadata_enricher"
        assert "rule_enriched" in trace.stages[0].details

    def test_no_trace_ok(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        result = enricher.transform([_make_chunk(_SAMPLE_TEXT)], trace=None)
        assert len(result) == 1

    def test_empty_chunk_list(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        assert enricher.transform([]) == []

    def test_preserves_id_and_source_ref(self, llm_json):
        enricher = MetadataEnricher(llm=llm_json, use_llm=True)
        chunk = _make_chunk(_SAMPLE_TEXT, "my_id")
        chunk.source_ref = "doc_abc"
        result = enricher.transform([chunk])[0]
        assert result.id == "my_id"
        assert result.source_ref == "doc_abc"

    def test_preserves_existing_metadata(self):
        enricher = MetadataEnricher(llm=None, use_llm=False)
        chunk = _make_chunk(_SAMPLE_TEXT)
        chunk.metadata["custom"] = "keep"
        result = enricher.transform([chunk])[0]
        assert result.metadata["custom"] == "keep"

    def test_prompt_fallback_when_file_missing(self, tmp_path):
        enricher = MetadataEnricher(
            llm=None, use_llm=False, prompt_path=str(tmp_path / "nope.txt")
        )
        assert "{text}" in enricher._prompt_template

"""Unit tests for QueryProcessor — keyword extraction and filter parsing."""
from __future__ import annotations

from src.core.trace.trace_context import TraceContext
from src.core.query_engine.query_processor import ProcessedQuery, QueryProcessor


class TestKeywords:
    def test_extracts_keywords(self):
        qp = QueryProcessor()
        result = qp.process("How to configure Azure OpenAI endpoint")
        assert "azure" in result.keywords
        assert "openai" in result.keywords
        assert "configure" in result.keywords

    def test_keywords_nonempty_for_content_query(self):
        qp = QueryProcessor()
        result = qp.process("vector database similarity search")
        assert len(result.keywords) > 0

    def test_stopwords_removed(self):
        qp = QueryProcessor()
        result = qp.process("how do I configure the system")
        assert "how" not in result.keywords
        assert "the" not in result.keywords
        assert "configure" in result.keywords
        assert "system" in result.keywords

    def test_lowercased(self):
        qp = QueryProcessor()
        result = qp.process("Azure AZURE azure")
        assert result.keywords == ["azure"]

    def test_deduplicated_preserve_order(self):
        qp = QueryProcessor()
        result = qp.process("rerank fusion rerank dense")
        assert result.keywords == ["rerank", "fusion", "dense"]

    def test_punctuation_ignored(self):
        qp = QueryProcessor()
        result = qp.process("BM25, RRF; cross-encoder!")
        assert "bm25" in result.keywords
        assert "rrf" in result.keywords

    def test_cjk_tokenized(self):
        qp = QueryProcessor()
        result = qp.process("检索增强")
        assert "检" in result.keywords
        assert len(result.keywords) == 4

    def test_empty_query(self):
        qp = QueryProcessor()
        result = qp.process("")
        assert result.keywords == []

    def test_stopwords_only_query(self):
        qp = QueryProcessor()
        result = qp.process("how do i")
        assert result.keywords == []


class TestNormalization:
    def test_whitespace_collapsed(self):
        qp = QueryProcessor()
        result = qp.process("  hello    world  \n  test ")
        assert result.normalized_query == "hello world test"

    def test_raw_query_preserved(self):
        qp = QueryProcessor()
        result = qp.process("Some Query Text")
        assert result.raw_query == "Some Query Text"


class TestFilters:
    def test_filters_default_empty_dict(self):
        qp = QueryProcessor()
        result = qp.process("query")
        assert result.filters == {}
        assert isinstance(result.filters, dict)

    def test_filters_passed_through(self):
        qp = QueryProcessor()
        result = qp.process("query", filters={"collection": "docs", "doc_type": "pdf"})
        assert result.filters == {"collection": "docs", "doc_type": "pdf"}

    def test_none_values_dropped(self):
        qp = QueryProcessor()
        result = qp.process("query", filters={"collection": "docs", "doc_type": None})
        assert result.filters == {"collection": "docs"}

    def test_none_filters(self):
        qp = QueryProcessor()
        result = qp.process("query", filters=None)
        assert result.filters == {}


class TestSerialization:
    def test_to_dict(self):
        qp = QueryProcessor()
        result = qp.process("vector search", filters={"collection": "c"})
        d = result.to_dict()
        assert set(d.keys()) == {"raw_query", "normalized_query", "keywords", "filters"}
        assert d["filters"] == {"collection": "c"}

    def test_is_processed_query(self):
        qp = QueryProcessor()
        result = qp.process("test")
        assert isinstance(result, ProcessedQuery)


class TestTrace:
    def test_trace_records_stage(self):
        qp = QueryProcessor()
        trace = TraceContext(trace_type="query")
        qp.process("vector database search", filters={"collection": "c"}, trace=trace)
        stage = trace.stages[0]
        assert stage.name == "query_processing"
        assert stage.details["num_keywords"] == 3
        assert stage.details["has_filters"] is True

"""Unit tests for ResponseBuilder + CitationGenerator (E3)."""
from __future__ import annotations

from src.core.types import RetrievalResult
from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.response_builder import ResponseBuilder


def _r(cid, score, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=score, text=text or cid, metadata=meta or {})


class TestCitationGenerator:
    def test_numbered_from_one(self):
        gen = CitationGenerator()
        cites = gen.generate([_r("a", 0.9), _r("b", 0.8)])
        assert [c.id for c in cites] == [1, 2]

    def test_source_and_page_extracted(self):
        gen = CitationGenerator()
        cites = gen.generate([_r("a", 0.9, meta={"source_path": "doc.pdf", "page": 5})])
        assert cites[0].source == "doc.pdf"
        assert cites[0].page == 5

    def test_fallback_source(self):
        gen = CitationGenerator()
        cites = gen.generate([_r("a", 0.9, meta={})])
        assert cites[0].source == "unknown"

    def test_snippet_truncated(self):
        gen = CitationGenerator(snippet_length=10)
        long_text = "word " * 50
        cites = gen.generate([_r("a", 0.9, text=long_text)])
        assert cites[0].text.endswith("…")

    def test_to_dict_fields(self):
        gen = CitationGenerator()
        c = gen.generate([_r("a", 0.91234, meta={"source_path": "d.pdf", "page": 2})])[0]
        d = c.to_dict()
        assert set(d.keys()) == {"id", "source", "page", "chunk_id", "score", "text"}
        assert d["score"] == 0.9123

    def test_empty(self):
        assert CitationGenerator().generate([]) == []


class TestResponseBuilder:
    def test_content_is_markdown(self):
        rb = ResponseBuilder()
        resp = rb.build([_r("a", 0.9, text="alpha", meta={"source_path": "a.pdf"})], "q")
        assert resp["content"][0]["type"] == "text"
        assert "###" in resp["content"][0]["text"]

    def test_inline_citation_markers(self):
        rb = ResponseBuilder()
        resp = rb.build(
            [_r("a", 0.9, meta={"source_path": "a.pdf"}),
             _r("b", 0.8, meta={"source_path": "b.pdf"})],
            "q",
        )
        text = resp["content"][0]["text"]
        assert "[1]" in text
        assert "[2]" in text

    def test_structured_citations(self):
        rb = ResponseBuilder()
        resp = rb.build(
            [_r("a", 0.9, text="alpha", meta={"source_path": "a.pdf", "page": 3})], "q"
        )
        cites = resp["structuredContent"]["citations"]
        assert len(cites) == 1
        assert cites[0]["source"] == "a.pdf"
        assert cites[0]["page"] == 3
        assert cites[0]["chunk_id"] == "a"
        assert "score" in cites[0]

    def test_query_echoed(self):
        rb = ResponseBuilder()
        resp = rb.build([_r("a", 0.9)], "my question")
        assert resp["structuredContent"]["query"] == "my question"
        assert "my question" in resp["content"][0]["text"]

    def test_empty_results_friendly_message(self):
        rb = ResponseBuilder()
        resp = rb.build([], "q")
        assert resp["structuredContent"]["citations"] == []
        # friendly message, not empty
        assert resp["content"][0]["text"].strip() != ""
        assert "未找到" in resp["content"][0]["text"]

    def test_not_error_flag(self):
        rb = ResponseBuilder()
        resp = rb.build([_r("a", 0.9)], "q")
        assert resp["isError"] is False

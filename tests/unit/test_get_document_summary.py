"""Unit tests for get_document_summary tool (E5)."""
from __future__ import annotations

import pytest

from src.mcp_server.protocol_handler import InvalidParams, ProtocolHandler
from src.mcp_server.tools.get_document_summary import GetDocumentSummaryTool


class FakeLookup:
    """Fake document lookup returning preconfigured chunks."""

    def __init__(self, mapping):
        # mapping: {doc_id: [chunk_dict, ...]}
        self._mapping = mapping

    def get_chunks_by_doc_id(self, doc_id):
        return self._mapping.get(doc_id, [])


def _chunk(meta):
    return {"metadata": meta}


@pytest.fixture
def lookup():
    return FakeLookup({
        "doc_a": [
            _chunk({"chunk_index": 0, "title": "Intro to RAG", "summary": "Overview of RAG.", "tags": ["rag", "retrieval"]}),
            _chunk({"chunk_index": 1, "title": "Details", "summary": "More detail.", "tags": ["retrieval", "embeddings"]}),
        ],
    })


class TestSummary:
    def test_title_from_first_chunk(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        result = tool({"doc_id": "doc_a"})
        assert result["structuredContent"]["title"] == "Intro to RAG"

    def test_summary_from_first_chunk(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        result = tool({"doc_id": "doc_a"})
        assert result["structuredContent"]["summary"] == "Overview of RAG."

    def test_tags_unioned_deduped(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        result = tool({"doc_id": "doc_a"})
        tags = result["structuredContent"]["tags"]
        assert tags == ["rag", "retrieval", "embeddings"]

    def test_chunk_count(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        result = tool({"doc_id": "doc_a"})
        assert result["structuredContent"]["chunk_count"] == 2

    def test_markdown_content(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        result = tool({"doc_id": "doc_a"})
        text = result["content"][0]["text"]
        assert "Intro to RAG" in text
        assert "标签" in text

    def test_doc_id_in_structured(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        result = tool({"doc_id": "doc_a"})
        assert result["structuredContent"]["doc_id"] == "doc_a"


class TestOrdering:
    def test_title_prefers_chunk_index_zero(self):
        # chunks out of order; chunk_index 0 should win for title
        lookup = FakeLookup({
            "d": [
                _chunk({"chunk_index": 2, "title": "Later", "summary": "", "tags": []}),
                _chunk({"chunk_index": 0, "title": "First", "summary": "S0", "tags": []}),
            ]
        })
        tool = GetDocumentSummaryTool(lookup)
        result = tool({"doc_id": "d"})
        assert result["structuredContent"]["title"] == "First"


class TestErrors:
    def test_unknown_doc_id_raises(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        with pytest.raises(InvalidParams):
            tool({"doc_id": "missing"})

    def test_missing_doc_id_raises(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        with pytest.raises(InvalidParams):
            tool({})

    def test_empty_doc_id_raises(self, lookup):
        tool = GetDocumentSummaryTool(lookup)
        with pytest.raises(InvalidParams):
            tool({"doc_id": "  "})


class TestRegistration:
    def test_unknown_doc_returns_invalid_params_via_handler(self, lookup):
        handler = ProtocolHandler()
        GetDocumentSummaryTool(lookup).register(handler)
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "get_document_summary", "arguments": {"doc_id": "missing"}},
        })
        assert resp["error"]["code"] == -32602

    def test_existing_doc_via_handler(self, lookup):
        handler = ProtocolHandler()
        GetDocumentSummaryTool(lookup).register(handler)
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "get_document_summary", "arguments": {"doc_id": "doc_a"}},
        })
        assert resp["result"]["structuredContent"]["title"] == "Intro to RAG"

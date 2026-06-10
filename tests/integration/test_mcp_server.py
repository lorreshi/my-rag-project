"""Integration tests for the MCP server (E1+).

Drives the server as a subprocess over stdio to verify the transport contract:
stdout carries only JSON-RPC messages, logs go to stderr, and initialize works.
In-process tests cover the server loop with mocked streams.
"""
from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO

import pytest

from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.server import MCPServer


def _make_request(method, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


class TestStdioSubprocess:
    """Black-box: run server as a subprocess, feed stdin, read stdout."""

    def test_initialize_over_stdio(self):
        request = _make_request("initialize", {"protocolVersion": "2025-06-18"})
        proc = subprocess.run(
            [sys.executable, "-m", "src.mcp_server.server"],
            input=request + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        # stdout must contain exactly one valid JSON-RPC response
        out_lines = [l for l in proc.stdout.splitlines() if l.strip()]
        assert len(out_lines) == 1
        response = json.loads(out_lines[0])
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"]["serverInfo"]["name"] == "smart-knowledge-hub"

    def test_stdout_not_polluted_by_logs(self):
        request = _make_request("initialize", {})
        proc = subprocess.run(
            [sys.executable, "-m", "src.mcp_server.server"],
            input=request + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Every stdout line must be valid JSON (no log lines leaked)
        for line in proc.stdout.splitlines():
            if line.strip():
                json.loads(line)  # raises if a log line slipped in
        # Logs should appear on stderr
        assert "MCP server" in proc.stderr


class TestServerLoopInProcess:
    """White-box: drive the loop with in-memory streams."""

    def test_initialize_response(self):
        stdin = StringIO(_make_request("initialize", {}) + "\n")
        stdout = StringIO()
        MCPServer(stdin=stdin, stdout=stdout).serve_forever()
        out = [l for l in stdout.getvalue().splitlines() if l.strip()]
        resp = json.loads(out[0])
        assert resp["result"]["protocolVersion"]

    def test_blank_lines_ignored(self):
        stdin = StringIO("\n\n" + _make_request("initialize", {}) + "\n\n")
        stdout = StringIO()
        MCPServer(stdin=stdin, stdout=stdout).serve_forever()
        out = [l for l in stdout.getvalue().splitlines() if l.strip()]
        assert len(out) == 1

    def test_malformed_json_returns_parse_error(self):
        stdin = StringIO("{ this is not json\n")
        stdout = StringIO()
        MCPServer(stdin=stdin, stdout=stdout).serve_forever()
        resp = json.loads(stdout.getvalue().splitlines()[0])
        assert resp["error"]["code"] == -32700

    def test_notification_no_response(self):
        # notifications/initialized has no id -> no response written
        stdin = StringIO(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        stdout = StringIO()
        MCPServer(stdin=stdin, stdout=stdout).serve_forever()
        assert stdout.getvalue().strip() == ""

    def test_multiple_requests(self):
        stdin = StringIO(
            _make_request("initialize", {}, req_id=1) + "\n"
            + _make_request("tools/list", {}, req_id=2) + "\n"
        )
        stdout = StringIO()
        handler = ProtocolHandler()
        MCPServer(handler=handler, stdin=stdin, stdout=stdout).serve_forever()
        out = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
        assert len(out) == 2
        assert out[0]["id"] == 1
        assert out[1]["id"] == 2


# ---------------------------------------------------------------------------
# E3: query_knowledge_hub tool
# ---------------------------------------------------------------------------

from src.core.types import RetrievalResult
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool


class _FakeHybrid:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, query, top_k=10, filters=None, trace=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self._results)


def _qr(cid, score, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=score, text=text or cid, metadata=meta or {})


class TestQueryKnowledgeHubTool:
    def test_returns_markdown_with_citations(self):
        hybrid = _FakeHybrid([
            _qr("c0", 0.9, "alpha text", {"source_path": "a.pdf", "page": 1}),
            _qr("c1", 0.8, "beta text", {"source_path": "b.pdf", "page": 2}),
        ])
        tool = QueryKnowledgeHubTool(hybrid_search=hybrid, reranker=None)
        result = tool({"query": "test"})
        text = result["content"][0]["text"]
        assert "[1]" in text and "[2]" in text

    def test_structured_citations_fields(self):
        hybrid = _FakeHybrid([_qr("c0", 0.95, "txt", {"source_path": "a.pdf", "page": 7})])
        tool = QueryKnowledgeHubTool(hybrid_search=hybrid, reranker=None)
        result = tool({"query": "test"})
        cite = result["structuredContent"]["citations"][0]
        assert cite["source"] == "a.pdf"
        assert cite["page"] == 7
        assert cite["chunk_id"] == "c0"
        assert "score" in cite

    def test_no_results_friendly(self):
        tool = QueryKnowledgeHubTool(hybrid_search=_FakeHybrid([]), reranker=None)
        result = tool({"query": "nothing"})
        assert result["structuredContent"]["citations"] == []
        assert result["content"][0]["text"].strip() != ""

    def test_collection_forwarded(self):
        hybrid = _FakeHybrid([_qr("c0", 0.9)])
        tool = QueryKnowledgeHubTool(hybrid_search=hybrid, reranker=None)
        tool({"query": "q", "collection": "mycoll"})
        assert hybrid.calls[0]["filters"] == {"collection": "mycoll"}

    def test_missing_query_raises(self):
        from src.mcp_server.protocol_handler import InvalidParams
        tool = QueryKnowledgeHubTool(hybrid_search=_FakeHybrid([]), reranker=None)
        with pytest.raises(InvalidParams):
            tool({})

    def test_invalid_top_k_raises(self):
        from src.mcp_server.protocol_handler import InvalidParams
        tool = QueryKnowledgeHubTool(hybrid_search=_FakeHybrid([]), reranker=None)
        with pytest.raises(InvalidParams):
            tool({"query": "q", "top_k": -1})

    def test_reranker_applied(self):
        hybrid = _FakeHybrid([_qr("c0", 0.5), _qr("c1", 0.9)])

        class _Rev:
            def rerank(self, query, candidates, top_k=None, trace=None):
                out = list(reversed(candidates))
                return out[:top_k] if top_k else out

        tool = QueryKnowledgeHubTool(hybrid_search=hybrid, reranker=_Rev())
        result = tool({"query": "q"})
        # reversed order -> c1 first in citations
        assert result["structuredContent"]["citations"][0]["chunk_id"] == "c1"

    def test_registers_with_handler(self):
        handler = ProtocolHandler()
        tool = QueryKnowledgeHubTool(hybrid_search=_FakeHybrid([_qr("c0", 0.9)]), reranker=None)
        tool.register(handler)
        # tools/list shows it
        listed = handler.handle(json.loads(_make_request("tools/list")))
        names = [t["name"] for t in listed["result"]["tools"]]
        assert "query_knowledge_hub" in names
        # tools/call routes to it
        called = handler.handle(
            json.loads(_make_request("tools/call", {"name": "query_knowledge_hub", "arguments": {"query": "hi"}}))
        )
        assert "content" in called["result"]

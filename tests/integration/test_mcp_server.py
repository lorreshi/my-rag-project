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


# ---------------------------------------------------------------------------
# E6: multimodal (image) response assembly
# ---------------------------------------------------------------------------

import base64 as _base64

from src.core.response.multimodal_assembler import MultimodalAssembler


class _FakeResolver:
    def __init__(self, mapping):
        self._mapping = mapping  # {image_id: path}

    def get_path(self, image_id):
        return self._mapping.get(image_id)


_PNG_BYTES = b"\x89PNG\r\n\x1a\n fake png data"


def _img_file(tmp_path, name="i1.png"):
    p = tmp_path / name
    p.write_bytes(_PNG_BYTES)
    return str(p)


class TestMultimodalAssembler:
    def test_appends_image_content(self, tmp_path):
        path = _img_file(tmp_path)
        assembler = MultimodalAssembler(_FakeResolver({"i1": path}))
        response = {"content": [{"type": "text", "text": "answer"}]}
        results = [_qr("c0", 0.9, "txt", {"image_refs": ["i1"]})]
        out = assembler.assemble(response, results)
        image_items = [c for c in out["content"] if c["type"] == "image"]
        assert len(image_items) == 1

    def test_image_content_is_base64(self, tmp_path):
        path = _img_file(tmp_path)
        assembler = MultimodalAssembler(_FakeResolver({"i1": path}))
        response = {"content": []}
        results = [_qr("c0", 0.9, meta={"image_refs": ["i1"]})]
        out = assembler.assemble(response, results)
        img = [c for c in out["content"] if c["type"] == "image"][0]
        assert img["data"] == _base64.b64encode(_PNG_BYTES).decode("utf-8")

    def test_mime_type_png(self, tmp_path):
        path = _img_file(tmp_path, "pic.png")
        assembler = MultimodalAssembler(_FakeResolver({"i1": path}))
        out = assembler.assemble({"content": []}, [_qr("c0", 0.9, meta={"image_refs": ["i1"]})])
        img = [c for c in out["content"] if c["type"] == "image"][0]
        assert img["mimeType"] == "image/png"

    def test_mime_type_jpeg(self, tmp_path):
        path = _img_file(tmp_path, "pic.jpg")
        assembler = MultimodalAssembler(_FakeResolver({"i1": path}))
        out = assembler.assemble({"content": []}, [_qr("c0", 0.9, meta={"image_refs": ["i1"]})])
        img = [c for c in out["content"] if c["type"] == "image"][0]
        assert img["mimeType"] == "image/jpeg"

    def test_text_preserved_before_image(self, tmp_path):
        path = _img_file(tmp_path)
        assembler = MultimodalAssembler(_FakeResolver({"i1": path}))
        response = {"content": [{"type": "text", "text": "answer"}]}
        out = assembler.assemble(response, [_qr("c0", 0.9, meta={"image_refs": ["i1"]})])
        assert out["content"][0]["type"] == "text"
        assert out["content"][1]["type"] == "image"

    def test_no_images_no_change(self):
        assembler = MultimodalAssembler(_FakeResolver({}))
        response = {"content": [{"type": "text", "text": "answer"}]}
        out = assembler.assemble(response, [_qr("c0", 0.9, meta={})])
        assert len(out["content"]) == 1

    def test_missing_file_skipped(self):
        assembler = MultimodalAssembler(_FakeResolver({"i1": "/nonexistent.png"}))
        out = assembler.assemble({"content": []}, [_qr("c0", 0.9, meta={"image_refs": ["i1"]})])
        assert all(c["type"] != "image" for c in out["content"])

    def test_unresolved_id_skipped(self):
        assembler = MultimodalAssembler(_FakeResolver({}))
        out = assembler.assemble({"content": []}, [_qr("c0", 0.9, meta={"image_refs": ["missing"]})])
        assert all(c["type"] != "image" for c in out["content"])

    def test_dedup_images(self, tmp_path):
        path = _img_file(tmp_path)
        assembler = MultimodalAssembler(_FakeResolver({"i1": path}))
        results = [
            _qr("c0", 0.9, meta={"image_refs": ["i1"]}),
            _qr("c1", 0.8, meta={"image_refs": ["i1"]}),
        ]
        out = assembler.assemble({"content": []}, results)
        assert len([c for c in out["content"] if c["type"] == "image"]) == 1

    def test_max_images_cap(self, tmp_path):
        paths = {f"i{i}": _img_file(tmp_path, f"i{i}.png") for i in range(5)}
        assembler = MultimodalAssembler(_FakeResolver(paths), max_images=2)
        results = [_qr(f"c{i}", 0.9, meta={"image_refs": [f"i{i}"]}) for i in range(5)]
        out = assembler.assemble({"content": []}, results)
        assert len([c for c in out["content"] if c["type"] == "image"]) == 2

    def test_images_dict_fallback(self, tmp_path):
        path = _img_file(tmp_path)
        assembler = MultimodalAssembler(_FakeResolver({"i1": path}))
        results = [_qr("c0", 0.9, meta={"images": [{"id": "i1", "path": path}]})]
        out = assembler.assemble({"content": []}, results)
        assert len([c for c in out["content"] if c["type"] == "image"]) == 1
